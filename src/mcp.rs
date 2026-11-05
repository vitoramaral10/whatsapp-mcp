//! The MCP tools. Names, parameters and output shapes match the Python server
//! they replace, so existing clients see no change.

use std::sync::Arc;

use base64::Engine;
use rmcp::handler::server::{router::tool::ToolRouter, tool::IntoCallToolResult, wrapper::Parameters};
use rmcp::model::*;
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::embed::Embedder;
use crate::media::{self, PathMap};
use crate::store::{MessageFilter, Store, format_message, format_messages, parse_ts};
use crate::transcribe::{Outcome, Transcriber};
use crate::wa::Wa;

pub struct Ctx {
    pub store: Arc<Store>,
    pub wa: Arc<Wa>,
    pub paths: PathMap,
    pub transcriber: Option<Arc<Transcriber>>,
    /// Above this size download_media returns only the path.
    pub max_inline: u64,
    /// Semantic half of search_messages; without it the search is full-text only.
    pub embedder: Option<Arc<Embedder>>,
    /// Hides the send tools: whatever a message says, the model cannot answer it.
    pub read_only: bool,
}

impl Ctx {
    /// The chat allowlist as a check media::outgoing can carry off-thread.
    pub fn allowed_chat(&self) -> impl Fn(&str) -> bool + Send + 'static {
        let store = Arc::clone(&self.store);
        move |jid| store.allowed(jid)
    }
}

/// Page sizes and context lengths are capped, so one call cannot pull the
/// whole history into a reply.
const MAX_LIMIT: u32 = 200;
const MAX_CONTEXT: u32 = 50;
const MAX_DIGEST: u32 = 100;

#[derive(Clone)]
pub struct Server {
    ctx: Arc<Ctx>,
    tool_router: ToolRouter<Server>,
}

/// Why a tool call failed. `Tool` reaches the caller as an `isError` result
/// with `{"success":false,"message":...}`; `Args` stays a JSON-RPC
/// invalid_params error, for arguments that are malformed.
#[derive(Debug)]
enum Fail {
    Tool(String),
    Args(ErrorData),
}

impl From<anyhow::Error> for Fail {
    fn from(e: anyhow::Error) -> Self {
        Fail::Tool(format!("{e:#}"))
    }
}

impl From<ErrorData> for Fail {
    fn from(e: ErrorData) -> Self {
        Fail::Args(e)
    }
}

impl IntoCallToolResult for Fail {
    fn into_call_tool_result(self) -> Result<CallToolResponse, ErrorData> {
        match self {
            Fail::Tool(message) => {
                let body = json!({ "success": false, "message": message }).to_string();
                Ok(CallToolResult::error(vec![ContentBlock::text(body)]).into())
            }
            Fail::Args(e) => Err(e),
        }
    }
}

fn fail<T>(message: impl Into<String>) -> Result<T, Fail> {
    Err(Fail::Tool(message.into()))
}

type R = Result<CallToolResult, Fail>;

fn text(t: impl Into<String>) -> R {
    Ok(CallToolResult::success(vec![ContentBlock::text(t)]))
}

fn json_out(v: impl serde::Serialize) -> R {
    text(serde_json::to_string(&v).unwrap_or_else(|_| "null".into()))
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

#[derive(Deserialize, JsonSchema, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum SortBy {
    /// Most recently active first
    #[default]
    LastActive,
    /// Alphabetical by chat name
    Name,
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
    /// Maximum number of messages to return (default 20, at most 200)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
    /// Whether to include messages before and after matches (default True)
    #[serde(default = "dtrue")]
    include_context: bool,
    /// Number of messages to include before each match (default 1, at most 50)
    #[serde(default = "d1")]
    context_before: u32,
    /// Number of messages to include after each match (default 1, at most 50)
    #[serde(default = "d1")]
    context_after: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct ListChatsArgs {
    /// Optional search term to filter chats by name or JID
    query: Option<String>,
    /// Maximum number of chats to return (default 20, at most 200)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
    /// Whether to include the last message in each chat (default True)
    #[serde(default = "dtrue")]
    include_last_message: bool,
    /// Field to sort results by, either "last_active" or "name" (default "last_active")
    #[serde(default)]
    sort_by: SortBy,
    /// Only chats with unread messages (default False)
    #[serde(default)]
    unread_only: bool,
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
    /// Maximum number of chats to return (default 20, at most 200)
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
    /// Optional JID of the chat containing the message; IDs are only unique within a chat
    #[serde(default)]
    chat_jid: Option<String>,
    /// Number of messages to include before the target message (default 5, at most 50)
    #[serde(default = "d5")]
    before: u32,
    /// Number of messages to include after the target message (default 5, at most 50)
    #[serde(default = "d5")]
    after: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct SendMessageArgs {
    /// The recipient - either a phone number with country code but no + or other symbols,
    /// or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
    recipient: String,
    /// The message text to send. To mention someone in a group, write @ followed by their
    /// phone number with country code (e.g. "@5511999999999"); WhatsApp shows their name.
    message: String,
    /// Optional ID of a message in this chat to reply to; it appears quoted above the reply
    #[serde(default)]
    reply_to_message_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct EditMessageArgs {
    /// The chat the message was sent to: a phone number with country code or a JID
    recipient: String,
    /// The ID of a text message you sent
    message_id: String,
    /// The new text
    message: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct DeleteMessageArgs {
    /// The chat the message was sent to: a phone number with country code or a JID
    recipient: String,
    /// The ID of a message you sent
    message_id: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ReactionArgs {
    /// The chat of the message: a phone number with country code or a JID
    recipient: String,
    /// The ID of the message to react to
    message_id: String,
    /// One emoji, e.g. "👍"; an empty string removes your reaction
    emoji: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ChatArgs {
    /// The chat: a phone number with country code or a JID
    recipient: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct DigestArgs {
    /// ISO-8601 start of the period (e.g. "2026-11-04T00:00:00-03:00" or "2026-11-04";
    /// without a timezone it is UTC)
    after: String,
    /// Optional ISO-8601 end of the period (default now)
    before: Option<String>,
    /// Maximum number of chats, busiest first (default 20, at most 100)
    #[serde(default = "d20")]
    max_chats: u32,
    /// Maximum number of messages shown per chat, the latest ones (default 30, at most 100)
    #[serde(default = "d30")]
    max_messages_per_chat: u32,
}

fn d30() -> u32 {
    30
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// What to look for, in any words: matches by meaning as well as by the words themselves
    query: String,
    /// Optional chat JID to search only that chat
    chat_jid: Option<String>,
    /// Optional phone number to only match messages from that sender
    sender_phone_number: Option<String>,
    /// Optional ISO-8601 formatted string to only return messages after this date
    after: Option<String>,
    /// Optional ISO-8601 formatted string to only return messages before this date
    before: Option<String>,
    /// Maximum number of messages to return (default 20, at most 200)
    #[serde(default = "d20")]
    limit: u32,
    /// Page number for pagination (default 0)
    #[serde(default)]
    page: u32,
}

#[derive(Deserialize, JsonSchema)]
pub struct SendFileArgs {
    /// The recipient - either a phone number with country code but no + or other symbols,
    /// or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
    recipient: String,
    /// Path to a file already on the WhatsApp server (e.g. one returned by download_media),
    /// up to 100 MB. Leave empty when sending content_base64.
    #[serde(default)]
    media_path: String,
    /// The file content, base64-encoded (a data: URL prefix is accepted). Use this to send
    /// a file from the client side. The whole request is capped at 4 MiB, so the file must
    /// be under about 3 MB; send larger files with media_path.
    #[serde(default)]
    content_base64: String,
    /// Required with content_base64; its extension picks the media type and it is shown
    /// as the document name (e.g. "report.pdf")
    #[serde(default)]
    filename: String,
    /// Optional text shown under a picture, video or document (not on voice notes or stickers)
    #[serde(default)]
    caption: String,
    /// Optional ID of a message in this chat to reply to
    #[serde(default)]
    reply_to_message_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SendAudioArgs {
    /// The recipient - either a phone number with country code but no + or other symbols,
    /// or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
    recipient: String,
    /// Path to an audio file already on the WhatsApp server (e.g. one returned by
    /// download_media), up to 100 MB. Leave empty when sending content_base64.
    #[serde(default)]
    media_path: String,
    /// The audio, base64-encoded (a data: URL prefix is accepted). The whole request is
    /// capped at 4 MiB, so the file must be under about 3 MB; send larger files with media_path.
    #[serde(default)]
    content_base64: String,
    /// Required with content_base64; its extension tells the decoder the format
    /// (e.g. "note.mp3")
    #[serde(default)]
    filename: String,
    /// Optional ID of a message in this chat to reply to
    #[serde(default)]
    reply_to_message_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct DownloadArgs {
    /// The ID of the message containing the media
    message_id: String,
    /// The JID of the chat containing the message
    chat_jid: String,
}

fn invalid(e: anyhow::Error) -> ErrorData {
    ErrorData::invalid_params(format!("{e:#}"), None)
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

fn status(result: Result<String, impl Into<Fail>>) -> R {
    json_out(json!({ "success": true, "message": result.map_err(Into::into)? }))
}

#[tool_router]
impl Server {
    pub fn new(ctx: Arc<Ctx>) -> Self {
        let mut tool_router = Self::tool_router();
        if ctx.read_only {
            for name in
                ["send_message", "send_file", "send_audio_message", "edit_message", "delete_message", "send_reaction", "mark_chat_read"]
            {
                tool_router.remove_route(name);
            }
        }
        Self { ctx, tool_router }
    }

    #[tool(description = "Search WhatsApp contacts by name or phone number.")]
    async fn search_contacts(&self, Parameters(a): Parameters<QueryArgs>) -> R {
        json_out(self.ctx.store.search_contacts(&a.query)?)
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
            limit: a.limit.min(MAX_LIMIT),
            page: a.page,
        };
        let found = s.list_messages(&filter)?;
        if !a.include_context || found.is_empty() {
            return text(format_messages(s, &found));
        }
        // Neighbouring matches share context: each message is shown once.
        let mut seen = std::collections::HashSet::new();
        let mut all = Vec::new();
        for m in found {
            let group =
                match s.message_context(&m.id, Some(&m.chat_jid), a.context_before.min(MAX_CONTEXT), a.context_after.min(MAX_CONTEXT))? {
                    Some(ctx) => ctx.before.into_iter().chain([ctx.message]).chain(ctx.after).collect(),
                    None => vec![m],
                };
            all.extend(group.into_iter().filter(|x| seen.insert((x.chat_jid.clone(), x.id.clone()))));
        }
        text(format_messages(s, &all))
    }

    #[tool(
        description = "Search WhatsApp messages by meaning and by words at once (semantic + full-text, ranked best match first). Prefer this over list_messages' query for questions like \"when did we talk about the trip\": it finds the messages even when they use other words."
    )]
    async fn search_messages(&self, Parameters(a): Parameters<SearchArgs>) -> R {
        let s = &self.ctx.store;
        let filter = MessageFilter {
            after: iso(&a.after, "after")?,
            before: iso(&a.before, "before")?,
            sender: a.sender_phone_number.filter(|v| !v.is_empty()),
            chat_jid: a.chat_jid.filter(|v| !v.is_empty()),
            query: None,
            limit: a.limit.min(MAX_LIMIT),
            page: a.page,
        };
        let qvec = match &self.ctx.embedder {
            Some(e) => match e.embed_one(&a.query).await {
                Ok(v) => Some(v),
                Err(err) => {
                    log::warn!("semantic search unavailable, full-text only: {err:#}");
                    None
                }
            },
            None => None,
        };
        text(format_messages(s, &s.search(&a.query, qvec.as_deref(), &filter)?))
    }

    #[tool(description = "Get WhatsApp chats matching specified criteria.")]
    async fn list_chats(&self, Parameters(a): Parameters<ListChatsArgs>) -> R {
        let s = &self.ctx.store;
        json_out(s.list_chats(
            a.query.as_deref(),
            a.limit.min(MAX_LIMIT),
            a.page,
            a.include_last_message,
            a.sort_by == SortBy::Name,
            a.unread_only,
        )?)
    }

    #[tool(description = "Get WhatsApp chat metadata by JID.")]
    async fn get_chat(&self, Parameters(a): Parameters<GetChatArgs>) -> R {
        match self.ctx.store.get_chat(&a.chat_jid, a.include_last_message)? {
            Some(chat) => json_out(chat),
            None => fail(format!("No chat with JID {}", a.chat_jid)),
        }
    }

    #[tool(description = "Get WhatsApp chat metadata by sender phone number.")]
    async fn get_direct_chat_by_contact(&self, Parameters(a): Parameters<PhoneArgs>) -> R {
        match self.ctx.store.direct_chat_by_phone(&a.sender_phone_number)? {
            Some(chat) => json_out(chat),
            None => fail(format!("No direct chat with phone number {}", a.sender_phone_number)),
        }
    }

    #[tool(description = "Get all WhatsApp chats involving the contact.")]
    async fn get_contact_chats(&self, Parameters(a): Parameters<ContactChatsArgs>) -> R {
        json_out(self.ctx.store.contact_chats(&a.jid, a.limit.min(MAX_LIMIT), a.page)?)
    }

    #[tool(description = "Get most recent WhatsApp message involving the contact.")]
    async fn get_last_interaction(&self, Parameters(a): Parameters<JidArgs>) -> R {
        let s = &self.ctx.store;
        match s.last_interaction(&a.jid)? {
            Some(m) => text(format_message(s, &m, true)),
            None => text(format!("No messages with {} found.", a.jid)),
        }
    }

    #[tool(description = "Get context around a specific WhatsApp message.")]
    async fn get_message_context(&self, Parameters(a): Parameters<ContextArgs>) -> R {
        let chat = a.chat_jid.as_deref().filter(|c| !c.is_empty());
        match self.ctx.store.message_context(&a.message_id, chat, a.before.min(MAX_CONTEXT), a.after.min(MAX_CONTEXT))? {
            Some(ctx) => json_out(ctx),
            None => fail(format!("No message with ID {}", a.message_id)),
        }
    }

    #[tool(
        description = "Send a WhatsApp message to a person or group. For group chats use the JID.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn send_message(&self, Parameters(a): Parameters<SendMessageArgs>) -> R {
        self.allowed_recipient(&a.recipient)?;
        status(self.ctx.wa.send(&a.recipient, &a.message, None, false, a.reply_to_message_id.as_deref()).await)
    }

    #[tool(
        description = "Edit the text of a WhatsApp text message you sent. WhatsApp only accepts edits in the first 15 minutes or so after sending.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn edit_message(&self, Parameters(a): Parameters<EditMessageArgs>) -> R {
        self.allowed_recipient(&a.recipient)?;
        status(self.ctx.wa.edit(&a.recipient, &a.message_id, &a.message).await)
    }

    #[tool(
        description = "Delete a WhatsApp message you sent, for everyone in the chat.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn delete_message(&self, Parameters(a): Parameters<DeleteMessageArgs>) -> R {
        self.allowed_recipient(&a.recipient)?;
        status(self.ctx.wa.delete(&a.recipient, &a.message_id).await)
    }

    #[tool(
        description = "React to a WhatsApp message with an emoji, or remove your reaction with an empty emoji.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn send_reaction(&self, Parameters(a): Parameters<ReactionArgs>) -> R {
        self.allowed_recipient(&a.recipient)?;
        status(self.ctx.wa.react(&a.recipient, &a.message_id, &a.emoji).await)
    }

    #[tool(
        description = "Mark every unread message of a WhatsApp chat as read: the senders see blue ticks and the chat shows as read on your phone.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn mark_chat_read(&self, Parameters(a): Parameters<ChatArgs>) -> R {
        self.allowed_recipient(&a.recipient)?;
        status(self.ctx.wa.mark_read(&a.recipient).await)
    }

    #[tool(
        description = "Everything said in a period, chat by chat (busiest first), with each chat's message and unread counts and its latest messages in order. Made for summaries such as \"what happened today\"."
    )]
    async fn get_digest(&self, Parameters(a): Parameters<DigestArgs>) -> R {
        let after = iso(&Some(a.after), "after")?.ok_or_else(|| ErrorData::invalid_params("after is required", None))?;
        let before = iso(&a.before, "before")?;
        let s = &self.ctx.store;
        let chats = s.active_chats(after, before, a.max_chats.min(MAX_DIGEST))?;
        if chats.is_empty() {
            return text("No messages in this period.");
        }
        let mut out = String::new();
        for (chat, n) in chats {
            let filter = MessageFilter {
                chat_jid: Some(chat.jid.clone()),
                after: Some(after),
                before,
                limit: a.max_messages_per_chat.min(MAX_DIGEST),
                ..Default::default()
            };
            let mut msgs = s.list_messages(&filter)?;
            msgs.reverse();
            let name = chat.name.as_deref().unwrap_or(&chat.jid);
            out.push_str(&format!("## {name} ({}) - {n} messages, {} unread\n", chat.jid, chat.unread_count));
            msgs.iter().for_each(|m| out.push_str(&format_message(s, m, false)));
            out.push('\n');
        }
        text(out)
    }

    #[tool(
        description = "Send a file via WhatsApp to the specified recipient: jpg, png and gif go as pictures, webp as a sticker (no caption), mp4, avi and mov as videos, .ogg/.opus audio as a voice note, and anything else (other audio included) as a document. For group messages use the JID. To send audio as a voice note in any format, use send_audio_message.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn send_file(&self, Parameters(a): Parameters<SendFileArgs>) -> R {
        status(self.send_outgoing(a, false).await)
    }

    #[tool(
        description = "Send an audio file as a WhatsApp voice note to the specified recipient. For group messages use the JID. MP3, AAC/M4A, WAV, FLAC, Ogg Vorbis and Ogg Opus are accepted, up to 30 minutes and 100 MB; anything but Ogg Opus is converted to Opus.\n\nReturns a dictionary containing success status and a status message"
    )]
    async fn send_audio_message(&self, Parameters(a): Parameters<SendAudioArgs>) -> R {
        let a = SendFileArgs {
            recipient: a.recipient,
            media_path: a.media_path,
            content_base64: a.content_base64,
            filename: a.filename,
            caption: String::new(),
            reply_to_message_id: a.reply_to_message_id,
        };
        status(self.send_outgoing(a, true).await)
    }

    #[tool(
        description = "Download media from a WhatsApp message and return its content.\n\nReturns a JSON text block with success status, message, file_path, mime_type and size_bytes (plus transcription, transcription_pending or transcription_error for audio), followed by the file itself: an image block for pictures, an embedded resource (base64 blob) for everything else. Files over the inline limit come back as path only."
    )]
    async fn download_media(&self, Parameters(a): Parameters<DownloadArgs>) -> R {
        // Normalized like a recipient; status@broadcast and the like are kept as given.
        let jid = crate::wa::recipient_jid(&a.chat_jid).map(|j| j.to_string()).unwrap_or_else(|_| a.chat_jid.trim().to_string());
        if !self.ctx.store.allowed(&jid) {
            return fail(format!("{} is not in the allowed chats", a.chat_jid));
        }
        let (_, path) = match self.ctx.wa.download(&a.message_id, &jid).await {
            Ok(r) => r,
            Err(e) => {
                log::warn!("download {} failed: {e:#}", a.message_id);
                return fail(format!("Failed to download media: {e:#}"));
            }
        };
        // The row the transcript belongs to: a LID chat is stored under its phone number.
        let chat = self.ctx.wa.canonical_jid(&jid).await;
        let host_path = self.ctx.paths.to_host(&path.to_string_lossy());
        let mime = media::mime_of(&path);
        let size = tokio::fs::metadata(&path).await.map(|m| m.len()).unwrap_or(0);
        let mut info = json!({
            "success": true, "message": "Media downloaded successfully",
            "file_path": host_path, "mime_type": mime, "size_bytes": size,
        });
        if mime.starts_with("audio/")
            && let Some(t) = &self.ctx.transcriber
        {
            match t.transcription(&a.message_id, &chat, &path).await {
                Outcome::Text(text) => info["transcription"] = text.into(),
                Outcome::Pending => {
                    info["transcription_pending"] = "Still transcribing; call download_media again in about 30 seconds".into()
                }
                Outcome::Error(e) | Outcome::Unreachable(e) => info["transcription_error"] = e.into(),
            }
        }
        if size > self.ctx.max_inline {
            info["message"] =
                format!("Media downloaded successfully; over the {}-byte inline limit, so only the path is returned", self.ctx.max_inline)
                    .into();
            return text(info.to_string());
        }
        let data = tokio::fs::read(&path).await.map_err(anyhow::Error::from)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(data);
        // Only the formats clients can render go as an image; svg, heic and the rest as a file.
        let content = if matches!(mime.as_str(), "image/jpeg" | "image/png" | "image/gif" | "image/webp") {
            ContentBlock::image(b64, mime)
        } else {
            ContentBlock::resource(ResourceContents::blob(b64, media::file_uri(&host_path)).with_mime_type(mime))
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(info.to_string()), content]))
    }
}

impl Server {
    /// Checks a recipient: malformed is invalid_params, outside the allowlist a tool error.
    fn allowed_recipient(&self, recipient: &str) -> Result<(), Fail> {
        let jid = crate::wa::recipient_jid(recipient).map_err(invalid)?;
        if !self.ctx.store.allowed(&jid.to_string()) {
            return fail(format!("{recipient} is not in the allowed chats"));
        }
        Ok(())
    }

    async fn send_outgoing(&self, a: SendFileArgs, voice: bool) -> Result<String, Fail> {
        self.allowed_recipient(&a.recipient)?;
        let file = media::outgoing(&self.ctx.paths, self.ctx.allowed_chat(), &a.media_path, &a.content_base64, &a.filename)
            .await
            .map_err(|e| e.context("Invalid file content"))?;
        Ok(self.ctx.wa.send(&a.recipient, &a.caption, Some(file), voice, a.reply_to_message_id.as_deref()).await?)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("whatsapp-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools over one personal WhatsApp account and its stored message history. Find a chat with \
                 search_contacts or list_chats, read it with list_messages, get_message_context or get_digest, \
                 and use search_messages for questions by meaning. Chats and recipients are JIDs \
                 (e.g. 5511999999999@s.whatsapp.net, or ...@g.us for groups); a phone number with country \
                 code also works for recipients. With a whisper server configured, download_media also returns the transcript of a voice note. \
                 Sending tools act as the account owner, so send only what the user asked for; they are \
                 absent in read-only mode, and an allowlist may limit which chats any tool can see.",
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(read_only: bool) -> (tempfile::TempDir, Server) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(&dir.path().join("messages.db")).unwrap());
        let wa = Wa::new(Arc::clone(&store), dir.path().to_path_buf());
        let ctx = Ctx { store, wa, paths: PathMap::default(), transcriber: None, max_inline: 0, embedder: None, read_only };
        (dir, Server::new(Arc::new(ctx)))
    }

    #[test]
    fn read_only_hides_exactly_the_tools_that_write() {
        let names = |s: &Server| {
            let mut n: Vec<String> = s.tool_router.list_all().into_iter().map(|t| t.name.to_string()).collect();
            n.sort();
            n
        };
        let (_a, full) = server(false);
        let (_b, ro) = server(true);
        assert_eq!(names(&full).len(), 18);
        assert!(names(&full).contains(&"search_messages".to_string()));
        let hidden: Vec<String> = names(&full).into_iter().filter(|n| !names(&ro).contains(n)).collect();
        assert_eq!(
            hidden,
            ["delete_message", "edit_message", "mark_chat_read", "send_audio_message", "send_file", "send_message", "send_reaction"]
        );
    }

    /// What the client gets: (isError, JSON body) or the JSON-RPC error.
    fn wire(r: R) -> Result<(bool, serde_json::Value), ErrorData> {
        let CallToolResponse::Complete(r) = r.into_call_tool_result()? else { panic!("not complete") };
        let body = r.content[0].as_text().map(|t| t.text.clone()).unwrap_or_default();
        Ok((r.is_error == Some(true), serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body))))
    }

    #[tokio::test]
    async fn failures_are_tool_errors_and_bad_arguments_protocol_errors() {
        let (_d, srv) = server(false);
        let args = serde_json::from_value(json!({ "chat_jid": "1@g.us" })).unwrap();
        let (is_error, body) = wire(srv.get_chat(Parameters(args)).await).unwrap();
        assert!(is_error);
        assert_eq!(body, json!({ "success": false, "message": "No chat with JID 1@g.us" }));
        // A send that fails (no client) is a tool error too.
        let args = serde_json::from_value(json!({ "recipient": "5511900000001", "message": "oi" })).unwrap();
        let (is_error, body) = wire(srv.send_message(Parameters(args)).await).unwrap();
        assert!(is_error && body["success"] == false, "{body}");
        let args = serde_json::from_value(json!({ "recipient": "not a number", "message": "oi" })).unwrap();
        assert_eq!(wire(srv.send_message(Parameters(args)).await).unwrap_err().code, ErrorCode::INVALID_PARAMS);
        let args = serde_json::from_value(json!({ "after": "yesterday" })).unwrap();
        assert_eq!(wire(srv.get_digest(Parameters(args)).await).unwrap_err().code, ErrorCode::INVALID_PARAMS);
        let args = serde_json::from_value(json!({ "jid": "1@s.whatsapp.net" })).unwrap();
        assert_eq!(
            wire(srv.get_last_interaction(Parameters(args)).await).unwrap(),
            (false, json!("No messages with 1@s.whatsapp.net found."))
        );
        assert_eq!(srv.get_info().server_info.name, "whatsapp-mcp");
    }

    #[tokio::test]
    async fn digest_takes_plain_dates_and_sort_by_only_its_two_values() {
        let (_d, srv) = server(true);
        srv.ctx.store.store_chat("55@s.whatsapp.net", "Ana", chrono::Utc::now()).unwrap();
        srv.ctx.store.store_message("A", "55@s.whatsapp.net", "55", "oi", chrono::Utc::now(), false, &Default::default()).unwrap();
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let args = serde_json::from_value(json!({ "after": today })).unwrap();
        let (is_error, body) = wire(srv.get_digest(Parameters(args)).await).unwrap();
        assert!(!is_error && body.as_str().unwrap().contains("## Ana"), "{body}");
        for ok in ["name", "last_active"] {
            assert!(serde_json::from_value::<ListChatsArgs>(json!({ "sort_by": ok })).is_ok());
        }
        assert!(serde_json::from_value::<ListChatsArgs>(json!({ "sort_by": "date" })).is_err());
    }

    #[tokio::test]
    async fn context_shared_by_neighbouring_matches_is_listed_once() {
        let (_d, srv) = server(true);
        let s = &srv.ctx.store;
        let t = chrono::Utc::now();
        s.store_chat("55@s.whatsapp.net", "Ana", t).unwrap();
        for (i, id) in ["A", "B", "C"].into_iter().enumerate() {
            s.store_message(id, "55@s.whatsapp.net", "55", "oi", t + chrono::Duration::seconds(i as i64), false, &Default::default())
                .unwrap();
        }
        let args = serde_json::from_value(json!({ "chat_jid": "55@s.whatsapp.net" })).unwrap();
        let out = serde_json::to_string(&srv.list_messages(Parameters(args)).await.unwrap()).unwrap();
        for id in ["A", "B", "C"] {
            assert_eq!(out.matches(&format!("Message ID: {id} ")).count(), 1, "{out}");
        }
    }
}
