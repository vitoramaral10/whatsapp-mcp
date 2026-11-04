//! The MCP tools. Names, parameters and output shapes match the Python server
//! they replace, so clients and the gateway see no change.

use std::sync::Arc;

use base64::Engine;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::*;
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::media::{self, PathMap};
use crate::store::{format_message, format_messages, parse_ts, MessageFilter, Store};
use crate::transcribe::{Outcome, Transcriber};
use crate::wa::Wa;

pub struct Ctx {
    pub store: Arc<Store>,
    pub wa: Arc<Wa>,
    pub paths: PathMap,
    pub transcriber: Option<Arc<Transcriber>>,
    /// Above this size download_media returns only the path.
    pub max_inline: u64,
}

#[derive(Clone)]
pub struct Server {
    ctx: Arc<Ctx>,
    tool_router: ToolRouter<Server>,
}

type R = Result<CallToolResult, ErrorData>;

fn text(t: impl Into<String>) -> R {
    Ok(CallToolResult::success(vec![ContentBlock::text(t)]))
}

fn json_out(v: impl serde::Serialize) -> R {
    text(serde_json::to_string(&v).unwrap_or_else(|_| "null".into()))
}

fn internal(e: anyhow::Error) -> ErrorData {
    ErrorData::internal_error(format!("{e:#}"), None)
}

fn d20() -> u32 {
    20
}
fn d1() -> u32 {
    1
}
fn d5() -> u32 {
    5
}
fn dtrue() -> bool {
    true
}
fn dlast() -> String {
    "last_active".into()
}

#[derive(Deserialize, JsonSchema)]
pub struct QueryArgs {
    /// Search term to match against contact names or phone numbers
    query: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ListMessagesArgs {
    /// Optional ISO-8601 formatted string to only return messages after this date
    after: Option<String>,
    /// Optional ISO-8601 formatted string to only return messages before this date
    before: Option<String>,
    /// Optional phone number to filter messages by sender
    sender_phone_number: Option<String>,
    /// Optional chat JID to filter messages by chat
    chat_jid: Option<String>,
    /// Optional search term to filter messages by content
    query: Option<String>,
    /// Maximum number of messages to return (default 20)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
    /// Whether to include messages before and after matches (default True)
    #[serde(default = "dtrue")]
    include_context: bool,
    /// Number of messages to include before each match (default 1)
    #[serde(default = "d1")]
    context_before: u32,
    /// Number of messages to include after each match (default 1)
    #[serde(default = "d1")]
    context_after: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct ListChatsArgs {
    /// Optional search term to filter chats by name or JID
    query: Option<String>,
    /// Maximum number of chats to return (default 20)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
    /// Whether to include the last message in each chat (default True)
    #[serde(default = "dtrue")]
    include_last_message: bool,
    /// Field to sort results by, either "last_active" or "name" (default "last_active")
    #[serde(default = "dlast")]
    sort_by: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GetChatArgs {
    /// The JID of the chat to retrieve
    chat_jid: String,
    /// Whether to include the last message (default True)
    #[serde(default = "dtrue")]
    include_last_message: bool,
}

#[derive(Deserialize, JsonSchema)]
pub struct PhoneArgs {
    /// The phone number to search for
    sender_phone_number: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ContactChatsArgs {
    /// The contact's JID to search for
    jid: String,
    /// Maximum number of chats to return (default 20)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct JidArgs {
    /// The JID of the contact to search for
    jid: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ContextArgs {
    /// The ID of the message to get context for
    message_id: String,
    /// Number of messages to include before the target message (default 5)
    #[serde(default = "d5")]
    before: u32,
    /// Number of messages to include after the target message (default 5)
    #[serde(default = "d5")]
    after: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct SendMessageArgs {
    /// The recipient - either a phone number with country code but no + or other symbols,
    /// or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
    recipient: String,
    /// The message text to send
    message: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct SendFileArgs {
    /// The recipient - either a phone number with country code but no + or other symbols,
    /// or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
    recipient: String,
    /// Path to a file already on the WhatsApp server (e.g. one returned by download_media).
    /// Leave empty when sending content_base64.
    #[serde(default)]
    media_path: String,
    /// The file content, base64-encoded. Use this to send a file from the client side.
    #[serde(default)]
    content_base64: String,
    /// Required with content_base64; its extension picks the media type and it is shown
    /// as the document name (e.g. "relatorio.pdf")
    #[serde(default)]
    filename: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct DownloadArgs {
    /// The ID of the message containing the media
    message_id: String,
    /// The JID of the chat containing the message
    chat_jid: String,
}

fn iso(raw: &Option<String>, name: &str) -> Result<Option<chrono::DateTime<chrono::Utc>>, ErrorData> {
    let Some(raw) = raw.as_deref().filter(|r| !r.is_empty()) else { return Ok(None) };
    let parsed = parse_ts(raw)
        .or_else(|| {
            // Naive ISO dates and datetimes are taken as UTC, as Python did.
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f")
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f"))
                .or_else(|_| chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").map(|d| d.and_hms_opt(0, 0, 0).unwrap()))
                .ok()
                .map(|n| n.and_utc())
        })
        .ok_or_else(|| ErrorData::invalid_params(format!("Invalid date format for '{name}': {raw}. Please use ISO-8601 format."), None))?;
    Ok(Some(parsed))
}

fn status(result: anyhow::Result<String>) -> R {
    let (success, message) = match result {
        Ok(m) => (true, m),
        Err(e) => (false, format!("{e:#}")),
    };
    json_out(json!({ "success": success, "message": message }))
}

#[tool_router]
impl Server {
    pub fn new(ctx: Arc<Ctx>) -> Self {
        Self { ctx, tool_router: Self::tool_router() }
    }

    #[tool(description = "Search WhatsApp contacts by name or phone number.")]
    async fn search_contacts(&self, Parameters(a): Parameters<QueryArgs>) -> R {
        json_out(self.ctx.store.search_contacts(&a.query).map_err(internal)?)
    }

    #[tool(description = "Get WhatsApp messages matching specified criteria with optional context.")]
    async fn list_messages(&self, Parameters(a): Parameters<ListMessagesArgs>) -> R {
        let s = &self.ctx.store;
        let filter = MessageFilter {
            after: iso(&a.after, "after")?,
            before: iso(&a.before, "before")?,
            sender: a.sender_phone_number.filter(|v| !v.is_empty()),
            chat_jid: a.chat_jid.filter(|v| !v.is_empty()),
            query: a.query.filter(|v| !v.is_empty()),
            limit: a.limit,
            page: a.page,
        };
        let found = s.list_messages(&filter).map_err(internal)?;
        if !a.include_context || found.is_empty() {
            return text(format_messages(s, &found));
        }
        let mut all = Vec::new();
        for m in found {
            match s.message_context(&m.id, a.context_before, a.context_after).map_err(internal)? {
                Some(ctx) => {
                    all.extend(ctx.before);
                    all.push(ctx.message);
                    all.extend(ctx.after);
                }
                None => all.push(m),
            }
        }
        text(format_messages(s, &all))
    }

    #[tool(description = "Get WhatsApp chats matching specified criteria.")]
    async fn list_chats(&self, Parameters(a): Parameters<ListChatsArgs>) -> R {
        json_out(
            self.ctx
                .store
                .list_chats(a.query.as_deref(), a.limit, a.page, a.include_last_message, a.sort_by == "name")
                .map_err(internal)?,
        )
    }

    #[tool(description = "Get WhatsApp chat metadata by JID.")]
    async fn get_chat(&self, Parameters(a): Parameters<GetChatArgs>) -> R {
        json_out(self.ctx.store.get_chat(&a.chat_jid, a.include_last_message).map_err(internal)?)
    }

    #[tool(description = "Get WhatsApp chat metadata by sender phone number.")]
    async fn get_direct_chat_by_contact(&self, Parameters(a): Parameters<PhoneArgs>) -> R {
        json_out(self.ctx.store.direct_chat_by_phone(&a.sender_phone_number).map_err(internal)?)
    }

    #[tool(description = "Get all WhatsApp chats involving the contact.")]
    async fn get_contact_chats(&self, Parameters(a): Parameters<ContactChatsArgs>) -> R {
        json_out(self.ctx.store.contact_chats(&a.jid, a.limit, a.page).map_err(internal)?)
    }

    #[tool(description = "Get most recent WhatsApp message involving the contact.")]
    async fn get_last_interaction(&self, Parameters(a): Parameters<JidArgs>) -> R {
        let s = &self.ctx.store;
        match s.last_interaction(&a.jid).map_err(internal)? {
            Some(m) => text(format_message(s, &m, true)),
            None => json_out(Value::Null),
        }
    }

    #[tool(description = "Get context around a specific WhatsApp message.")]
    async fn get_message_context(&self, Parameters(a): Parameters<ContextArgs>) -> R {
        match self.ctx.store.message_context(&a.message_id, a.before, a.after).map_err(internal)? {
            Some(ctx) => json_out(ctx),
            None => Err(ErrorData::invalid_params(format!("Message with ID {} not found", a.message_id), None)),
        }
    }

    #[tool(description = "Send a WhatsApp message to a person or group. For group chats use the JID.\n\nReturns a dictionary containing success status and a status message")]
    async fn send_message(&self, Parameters(a): Parameters<SendMessageArgs>) -> R {
        if a.recipient.is_empty() {
            return status(Err(anyhow::anyhow!("Recipient must be provided")));
        }
        status(self.ctx.wa.send(&a.recipient, &a.message, None, false).await)
    }

    #[tool(description = "Send a file such as a picture, raw audio, video or document via WhatsApp to the specified recipient. For group messages use the JID.\n\nReturns a dictionary containing success status and a status message")]
    async fn send_file(&self, Parameters(a): Parameters<SendFileArgs>) -> R {
        status(self.send_outgoing(&a, false).await)
    }

    #[tool(description = "Send any audio file as a WhatsApp audio message to the specified recipient. For group messages use the JID. Any format WhatsApp can play is converted to an Opus voice note.\n\nReturns a dictionary containing success status and a status message")]
    async fn send_audio_message(&self, Parameters(a): Parameters<SendFileArgs>) -> R {
        status(self.send_outgoing(&a, true).await)
    }

    #[tool(description = "Download media from a WhatsApp message and return its content.\n\nReturns a JSON text block with success status, message, file_path, mime_type and size_bytes (plus transcription, transcription_pending or transcription_error for audio), followed by the file itself: an image block for pictures, an embedded resource (base64 blob) for everything else. Files over the inline limit come back as path only.")]
    async fn download_media(&self, Parameters(a): Parameters<DownloadArgs>) -> R {
        let (_, path) = match self.ctx.wa.download(&a.message_id, &a.chat_jid).await {
            Ok(r) => r,
            Err(e) => {
                log::warn!("download {} failed: {e:#}", a.message_id);
                return json_out(json!({ "success": false, "message": "Failed to download media" }));
            }
        };
        let host_path = self.ctx.paths.to_host(&path.to_string_lossy());
        let mime = media::mime_of(&path);
        let size = tokio::fs::metadata(&path).await.map(|m| m.len()).unwrap_or(0);
        let mut info = json!({
            "success": true, "message": "Media downloaded successfully",
            "file_path": host_path, "mime_type": mime, "size_bytes": size,
        });
        if mime.starts_with("audio/") {
            if let Some(t) = &self.ctx.transcriber {
                match t.transcription(&path).await {
                    Outcome::Text(text) => info["transcription"] = text.into(),
                    Outcome::Pending => {
                        info["transcription_pending"] = "Still transcribing; call download_media again in about 30 seconds".into()
                    }
                    Outcome::Error(e) => info["transcription_error"] = e.into(),
                }
            }
        }
        if size > self.ctx.max_inline {
            info["message"] = format!("Media downloaded successfully; over the {}-byte inline limit, so only the path is returned", self.ctx.max_inline).into();
            return text(info.to_string());
        }
        let data = tokio::fs::read(&path).await.map_err(|e| internal(e.into()))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(data);
        let content = if mime.starts_with("image/") {
            ContentBlock::image(b64, mime)
        } else {
            ContentBlock::resource(ResourceContents::blob(b64, format!("file://{host_path}")).with_mime_type(mime))
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(info.to_string()), content]))
    }
}

impl Server {
    async fn send_outgoing(&self, a: &SendFileArgs, voice: bool) -> anyhow::Result<String> {
        if a.recipient.is_empty() {
            anyhow::bail!("Recipient must be provided");
        }
        let file = media::outgoing(&self.ctx.paths, &a.media_path, &a.content_base64, &a.filename)
            .map_err(|e| anyhow::anyhow!("Invalid file content: {e:#}"))?;
        self.ctx.wa.send(&a.recipient, "", Some(file), voice).await
    }
}

#[tool_handler]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}
