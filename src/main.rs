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
mod upload;
mod wa;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use log::{error, info, warn};
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager};
use serde::Deserialize;
use serde_json::{Value, json};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

/// Seconds from the environment; a value that is not a usable duration falls
/// back to `default` with a warning.
fn env_secs(name: &str, default: f64) -> Duration {
    let raw = env(name, "");
    match raw.parse::<f64>().ok().and_then(|s| Duration::try_from_secs_f64(s).ok()) {
        Some(d) => d,
        None => {
            if !raw.is_empty() {
                warn!("{name}={raw} is not a valid number of seconds; using {default}");
            }
            Duration::from_secs_f64(default)
        }
    }
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
            Err(e) => warn!("whatsmeow import skipped: {e:#}"),
        }
    }

    // A bad entry stops the start: skipping it could leave the list empty, which means every chat.
    let allowed = env("WHATSAPP_ALLOWED_CHATS", "")
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(|c| wa::recipient_jid(c).map(|j| j.to_string()).with_context(|| format!("WHATSAPP_ALLOWED_CHATS entry {c:?}")))
        .collect::<Result<Vec<_>>>()?;
    if !allowed.is_empty() {
        store.restrict(&allowed)?;
        info!("MCP tools limited to {} chats", allowed.len());
    }

    let wa = wa::Wa::new(Arc::clone(&store), store_dir.clone());
    let pair_phone = Some(env("WHATSAPP_PAIR_PHONE", "")).filter(|p| !p.is_empty());
    let mut bot = wa.start(&store_dir.join(env("WHATSAPP_SESSION_DB", "whatsapp-rs.db")), pair_phone).await?;
    lock_down(&store_dir);

    let whisper = env("WHISPER_URL", "");
    let transcriber = (!whisper.is_empty()).then(|| {
        transcribe::Transcriber::new(whisper, Arc::clone(&store), env_secs("WHISPER_WAIT_S", 10.0), env_secs("WHISPER_TIMEOUT_S", 180.0))
    });
    if let Some(t) = &transcriber {
        tokio::spawn(transcribe_incoming(Arc::clone(t), Arc::clone(&wa), Arc::clone(&store)));
    }
    let embed_url = env("EMBED_URL", "");
    let embedder = (!embed_url.is_empty()).then(|| embed::Embedder::new(embed_url, env("EMBED_MODEL", "bge-m3")));
    if let Some(e) = &embedder {
        tokio::spawn(Arc::clone(e).backfill(Arc::clone(&store)));
    }
    let read_only = matches!(env("WHATSAPP_READ_ONLY", "").to_lowercase().as_str(), "1" | "true" | "yes");
    // Read-only has no send tools to hand out upload URLs, so no listener either.
    let upload_url = env("WHATSAPP_UPLOAD_URL", "");
    let uploads = (!upload_url.is_empty() && !read_only).then(|| Arc::new(upload::Uploads::new(upload_url)));
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
        uploads,
    });

    let mcp_ctx = Arc::clone(&ctx);
    let mcp_service = StreamableHttpService::new(
        move || Ok(mcp::Server::new(Arc::clone(&mcp_ctx))),
        LocalSessionManager::default().into(),
        // Clients may reach this behind a proxy under any host name, not only localhost.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    );
    let mcp_addr: SocketAddr = format!("{}:{}", env("MCP_HOST", "0.0.0.0"), env("MCP_PORT", "8000")).parse()?;
    let api_host = env("WHATSAPP_API_HOST", &env("MCP_HOST", "0.0.0.0"));
    let api_addr: SocketAddr = format!("{api_host}:{}", env("WHATSAPP_API_PORT", "8080")).parse()?;

    let mcp_app = Router::new().nest_service("/mcp", mcp_service);
    let api_app = Router::new().route("/api/send", post(api_send)).route("/api/download", post(api_download)).with_state(Arc::clone(&ctx));

    info!("MCP on http://{mcp_addr}/mcp, REST API on http://{api_addr}/api");
    let mcp_listener = tokio::net::TcpListener::bind(mcp_addr).await?;
    let api_listener = tokio::net::TcpListener::bind(api_addr).await?;
    // Its own port, so publishing it exposes the upload URLs and nothing of /api.
    let upload_server = async {
        if ctx.uploads.is_none() {
            return std::future::pending::<Result<()>>().await;
        }
        let addr: SocketAddr =
            format!("{}:{}", env("WHATSAPP_UPLOAD_HOST", &env("MCP_HOST", "0.0.0.0")), env("WHATSAPP_UPLOAD_PORT", "8090")).parse()?;
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("upload URLs on http://{addr}/upload, handed out as {}/upload", env("WHATSAPP_UPLOAD_URL", "").trim_end_matches('/'));
        axum::serve(listener, upload::router(Arc::clone(&ctx))).await.context("upload server stopped")
    };
    // A logged-out session fails again on every restart, so a stopped client
    // keeps the history tools serving instead of exiting into a restart loop.
    let client_stopped = async {
        (&mut bot).await;
        error!(
            "WhatsApp client stopped (logged out or replaced by another session); history stays readable, but to send again delete the session database and pair again"
        );
        std::future::pending::<()>().await
    };
    let served = tokio::select! {
        r = axum::serve(mcp_listener, mcp_app) => r.context("MCP server stopped"),
        r = axum::serve(api_listener, api_app) => r.context("REST API stopped"),
        r = upload_server => r,
        () = client_stopped => unreachable!(),
        _ = stop_signal() => {
            info!("shutting down");
            Ok(())
        }
    };
    // Flushes the device snapshot, pending receipts and message secrets.
    bot.shutdown().await;
    served
}

/// The store holds the session keys and the whole history: owner only. SQLite
/// gives the -wal and -shm files it creates later the database's own mode.
fn lock_down(store_dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let set = |p: &std::path::Path, mode: u32| {
        if let Err(e) = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)) {
            warn!("chmod {mode:o} {}: {e}", p.display());
        }
    };
    set(store_dir, 0o700);
    for entry in std::fs::read_dir(store_dir).into_iter().flatten().flatten() {
        if entry.path().is_file() && media::is_database(&entry.file_name().to_string_lossy()) {
            set(&entry.path(), 0o600);
        }
    }
}

async fn stop_signal() {
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

/// Transcribes voice notes of the last days as they come, one at a time, so
/// the searches find what was said without anyone opening them. Audio the CDN
/// no longer has is tried once per run.
async fn transcribe_incoming(t: Arc<transcribe::Transcriber>, wa: Arc<wa::Wa>, store: Arc<store::Store>) {
    let mut failed = std::collections::HashSet::new();
    loop {
        let since = chrono::Utc::now() - chrono::Duration::days(3);
        if !wa.is_connected() {
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }
        // Enough rows that the notes already given up on cannot crowd out new ones.
        let todo = match store.untranscribed(since, 50 + failed.len() as u32) {
            Ok(rows) => rows.into_iter().filter(|m| !failed.contains(m)).collect(),
            Err(e) => {
                warn!("auto transcription: listing voice notes failed: {e:#}");
                Vec::new()
            }
        };
        for (id, chat) in todo {
            let outcome = match wa.download(&id, &chat).await {
                Ok((_, path)) => loop {
                    match t.transcription(&id, &chat, &path).await {
                        transcribe::Outcome::Pending => continue,
                        done => break done,
                    }
                },
                Err(e) => transcribe::Outcome::Error(format!("{e:#}")),
            };
            match outcome {
                transcribe::Outcome::Unreachable(e) => {
                    warn!("auto transcription paused, whisper unreachable: {e}");
                    break; // not the audio's fault: try it again on the next round
                }
                transcribe::Outcome::Error(_) if !wa.is_connected() => break, // likewise, once back online
                transcribe::Outcome::Error(e) => {
                    warn!("auto transcription of {id} failed: {e}");
                    failed.insert((id, chat));
                }
                _ => {}
            }
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}

fn failure(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"success": false, "message": message.into()})))
}

/// The Go bridge's /api/send, kept for services that call it directly. Bound
/// by the same read-only mode and chat allowlist as the MCP tools.
async fn api_send(State(ctx): State<Arc<mcp::Ctx>>, Json(req): Json<SendRequest>) -> (StatusCode, Json<Value>) {
    if ctx.read_only {
        return failure(StatusCode::FORBIDDEN, "Sending is disabled: the server is in read-only mode");
    }
    if req.recipient.is_empty() {
        return failure(StatusCode::BAD_REQUEST, "Recipient is required");
    }
    if req.message.is_empty() && req.media_path.is_empty() {
        return failure(StatusCode::BAD_REQUEST, "Message or media path is required");
    }
    match wa::recipient_jid(&req.recipient) {
        Ok(jid) if !ctx.store.allowed(&jid.to_string()) => {
            return failure(StatusCode::FORBIDDEN, format!("{} is not in the allowed chats", req.recipient));
        }
        Ok(_) => {}
        Err(e) => return failure(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
    let result = async {
        let file = if req.media_path.is_empty() {
            None
        } else {
            Some(media::outgoing(&ctx.paths, ctx.allowed_chat(), &req.media_path, "", "").await?)
        };
        ctx.wa.send(&req.recipient, &req.message, file, false, None).await
    }
    .await;
    match result {
        Ok(m) => (StatusCode::OK, Json(json!({"success": true, "message": m}))),
        Err(e) => failure(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

#[derive(Deserialize)]
struct DownloadRequest {
    message_id: String,
    chat_jid: String,
}

async fn api_download(State(ctx): State<Arc<mcp::Ctx>>, Json(req): Json<DownloadRequest>) -> (StatusCode, Json<Value>) {
    if req.message_id.is_empty() || req.chat_jid.is_empty() {
        return failure(StatusCode::BAD_REQUEST, "Message ID and Chat JID are required");
    }
    if !ctx.store.allowed(&req.chat_jid) {
        return failure(StatusCode::FORBIDDEN, "Chat is not in the allowed chats");
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
        Err(e) => failure(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to download media: {e:#}")),
    }
}
