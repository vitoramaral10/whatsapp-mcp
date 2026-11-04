//! WhatsApp MCP server: one process holding the WhatsApp session, the message
//! history, the MCP tools (streamable HTTP) and the small REST API other
//! services on the network use to send messages.

mod audio;
mod embed;
mod extract;
mod mcp;
mod media;
mod store;
mod transcribe;
mod wa;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use log::info;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use serde::Deserialize;
use serde_json::{json, Value};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    // `whatsapp-mcp healthcheck`: the image has no shell or curl.
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        let port = env("MCP_PORT", "8000");
        let ok = std::net::TcpStream::connect_timeout(&format!("127.0.0.1:{port}").parse()?, Duration::from_secs(3)).is_ok();
        std::process::exit(if ok { 0 } else { 1 });
    }
    env_logger::Builder::new()
        .parse_filters(&env("WHATSAPP_LOG", &format!("info,whatsapp_rust={}", env("WHATSAPP_LOG_LEVEL", "info").to_lowercase())))
        .format_timestamp_secs()
        .init();
    tokio::runtime::Builder::new_multi_thread()
        // Two workers are plenty: the work is I/O plus one transcription at a time.
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<()> {
    let store_dir = PathBuf::from(env("WHATSAPP_STORE_DIR", "/app/whatsapp-bridge/store"));
    tokio::fs::create_dir_all(&store_dir).await.context("create store dir")?;
    let store = Arc::new(store::Store::open(&store_dir.join("messages.db"))?);
    // Names and LID mappings the Go bridge had learned, carried over once.
    if !store.has_contacts() {
        match store.import_whatsmeow(&store_dir.join("whatsapp.db")) {
            Ok(n) if n > 0 => info!("imported {n} contacts and LID mappings from the old whatsmeow store"),
            Ok(_) => {}
            Err(e) => log::warn!("whatsmeow import skipped: {e:#}"),
        }
    }

    let allowed: Vec<String> = env("WHATSAPP_ALLOWED_CHATS", "")
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(|c| if c.contains('@') { c.to_string() } else { format!("{c}@s.whatsapp.net") })
        .collect();
    if !allowed.is_empty() {
        store.restrict(&allowed)?;
        info!("MCP tools limited to {} chats", allowed.len());
    }

    let wa = wa::Wa::new(Arc::clone(&store), store_dir.clone());
    let pair_phone = Some(env("WHATSAPP_PAIR_PHONE", "")).filter(|p| !p.is_empty());
    wa.start(&store_dir.join(env("WHATSAPP_SESSION_DB", "whatsapp-rs.db")), pair_phone).await?;
    lock_down(&store_dir);

    let whisper = env("WHISPER_URL", "");
    let transcriber = (!whisper.is_empty()).then(|| {
        transcribe::Transcriber::new(
            whisper,
            Duration::from_secs_f64(env("WHISPER_WAIT_S", "10").parse().unwrap_or(10.0)),
            Duration::from_secs_f64(env("WHISPER_TIMEOUT_S", "180").parse().unwrap_or(180.0)),
        )
    });
    let embed_url = env("EMBED_URL", "");
    let embedder = (!embed_url.is_empty()).then(|| embed::Embedder::new(embed_url, env("EMBED_MODEL", "bge-m3")));
    if let Some(e) = &embedder {
        tokio::spawn(Arc::clone(e).backfill(Arc::clone(&store)));
    }
    let read_only = matches!(env("WHATSAPP_READ_ONLY", "").to_lowercase().as_str(), "1" | "true" | "yes");
    let ctx = Arc::new(mcp::Ctx {
        store,
        wa: Arc::clone(&wa),
        paths: media::PathMap {
            container: store_dir.to_string_lossy().into_owned(),
            host: env("WHATSAPP_MEDIA_HOST_PATH", ""),
            send_dirs: std::env::split_paths(&env("WHATSAPP_SEND_DIRS", &store_dir.to_string_lossy())).collect(),
        },
        transcriber,
        max_inline: env("WHATSAPP_MEDIA_MAX_INLINE_BYTES", "8388608").parse().unwrap_or(8 << 20),
        embedder,
        read_only,
    });

    let mcp_ctx = Arc::clone(&ctx);
    let mcp_service = StreamableHttpService::new(
        move || Ok(mcp::Server::new(Arc::clone(&mcp_ctx))),
        LocalSessionManager::default().into(),
        // Clients reach this through the gateway by service name, not localhost.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    );
    let mcp_addr: SocketAddr = format!("{}:{}", env("MCP_HOST", "0.0.0.0"), env("MCP_PORT", "8000")).parse()?;
    let api_addr: SocketAddr = format!("0.0.0.0:{}", env("WHATSAPP_API_PORT", "8080")).parse()?;

    let mcp_app = Router::new().nest_service("/mcp", mcp_service);
    let api_app = Router::new()
        .route("/api/send", post(api_send))
        .route("/api/download", post(api_download))
        .with_state(Arc::clone(&ctx));

    info!("MCP on http://{mcp_addr}/mcp, REST API on http://{api_addr}/api");
    let mcp_listener = tokio::net::TcpListener::bind(mcp_addr).await?;
    let api_listener = tokio::net::TcpListener::bind(api_addr).await?;
    tokio::select! {
        r = axum::serve(mcp_listener, mcp_app) => r?,
        r = axum::serve(api_listener, api_app) => r?,
        _ = shutdown() => info!("shutting down"),
    }
    Ok(())
}

/// The store holds the session keys and the whole history: owner only. SQLite
/// gives the -wal and -shm files it creates later the database's own mode.
fn lock_down(store_dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let set = |p: &std::path::Path, mode: u32| {
        if let Err(e) = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)) {
            log::warn!("chmod {mode:o} {}: {e}", p.display());
        }
    };
    set(store_dir, 0o700);
    for entry in std::fs::read_dir(store_dir).into_iter().flatten().flatten() {
        if entry.path().is_file() && media::is_database(&entry.file_name().to_string_lossy()) {
            set(&entry.path(), 0o600);
        }
    }
}

async fn shutdown() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[derive(Deserialize)]
struct SendRequest {
    recipient: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    media_path: String,
}

/// The Go bridge's /api/send, kept for services that call it directly.
async fn api_send(State(ctx): State<Arc<mcp::Ctx>>, Json(req): Json<SendRequest>) -> (StatusCode, Json<Value>) {
    if req.recipient.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"success": false, "message": "Recipient is required"})));
    }
    if req.message.is_empty() && req.media_path.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"success": false, "message": "Message or media path is required"})));
    }
    let result = async {
        let file = if req.media_path.is_empty() {
            None
        } else {
            Some(media::outgoing(&ctx.paths, &req.media_path, "", "")?)
        };
        ctx.wa.send(&req.recipient, &req.message, file, false, None).await
    }
    .await;
    match result {
        Ok(m) => (StatusCode::OK, Json(json!({"success": true, "message": m}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"success": false, "message": format!("{e:#}")}))),
    }
}

#[derive(Deserialize)]
struct DownloadRequest {
    message_id: String,
    chat_jid: String,
}

async fn api_download(State(ctx): State<Arc<mcp::Ctx>>, Json(req): Json<DownloadRequest>) -> (StatusCode, Json<Value>) {
    if req.message_id.is_empty() || req.chat_jid.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"success": false, "message": "Message ID and Chat JID are required"})));
    }
    match ctx.wa.download(&req.message_id, &req.chat_jid).await {
        Ok((kind, path)) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "message": format!("Successfully downloaded {kind} media"),
                "filename": path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "path": path,
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"success": false, "message": format!("Failed to download media: {e:#}")})),
        ),
    }
}
