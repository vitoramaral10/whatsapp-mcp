//! The WhatsApp side: session, events into the history store, send, download.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{TimeZone, Utc};
use log::{debug, info, warn};
use whatsapp_rust::download::{DownloadParams, MediaType};
use whatsapp_rust::prelude::*;
use whatsapp_rust::upload::UploadOptions;

use crate::store::{ContactField, Media, Store};
use crate::{audio, extract, media};

pub struct Wa {
    pub store: Arc<Store>,
    pub store_dir: PathBuf,
    client: OnceLock<Arc<Client>>,
}

impl Wa {
    pub fn new(store: Arc<Store>, store_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self { store, store_dir, client: OnceLock::new() })
    }

    fn client(&self) -> Result<&Arc<Client>> {
        self.client.get().ok_or_else(|| anyhow!("WhatsApp client is not started"))
    }

    /// Builds the client and runs it until the process stops.
    /// `pair_phone` asks for an 8-character pairing code for that number, to type
    /// on the phone instead of scanning a QR code from a second screen.
    pub async fn start(self: &Arc<Self>, session_db: &Path, pair_phone: Option<String>) -> Result<()> {
        let backend = SqliteStore::new(&session_db.to_string_lossy())
            .await
            .map_err(|e| anyhow!("open session store {}: {e}", session_db.display()))?;
        let this = Arc::clone(self);
        let mut builder = Bot::builder().with_backend(backend);
        if let Some(phone) = pair_phone {
            builder = builder.with_pair_code(whatsapp_rust::pair_code::PairCodeOptions { phone_number: phone, ..Default::default() });
        }
        let bot = builder
            .on_pair_code(|code, ttl| async move {
                println!("\nPairing code (valid {}s): {code}\nWhatsApp > Linked devices > Link with phone number\n", ttl.as_secs())
            })
            .on_qr_code(|code, ttl| async move { print_qr(&code, ttl) })
            .on_event(move |ev, _client| {
                let this = Arc::clone(&this);
                async move { this.on_event(&ev).await }
            })
            .build()
            .await
            .map_err(|e| anyhow!("start WhatsApp client: {e}"))?;
        let _ = self.client.set(bot.client());
        tokio::spawn(bot.run());
        Ok(())
    }

    async fn on_event(self: &Arc<Self>, ev: &Event) {
        match ev {
            Event::Messages(batch) => {
                for m in batch.messages.iter() {
                    if let Err(e) = self.on_message(&m.message, &m.info).await {
                        warn!("failed to store message {}: {e:#}", m.info.id);
                    }
                }
            }
            Event::HistorySync(h) => match h.get() {
                Some(sync) => self.on_history(sync).await,
                None => warn!("history sync chunk could not be decoded"),
            },
            Event::Connected(_) => {
                info!("connected to WhatsApp");
                let this = Arc::clone(self);
                tokio::spawn(async move {
                    match this.store.merge_lid_chats() {
                        Ok((n, total)) => info!("LID chat merge: {n} of {total} folded into phone-number rows"),
                        Err(e) => warn!("LID chat merge failed: {e:#}"),
                    }
                    this.refresh_all_names();
                });
            }
            Event::PushNameUpdate(p) => {
                let jid = self.canonical(&p.jid).await;
                self.save_contact(&jid, ContactField::PushName, &p.new_push_name);
            }
            Event::ContactUpdate(c) => {
                if let (Some(lid), Some(pn)) = (c.action.lid_jid.as_deref(), c.action.pn_jid.as_deref()) {
                    self.learn_lid(lid, pn);
                }
                let jid = self.canonical(&c.jid).await;
                self.save_contact(&jid, ContactField::FullName, c.action.full_name.as_deref().unwrap_or(""));
                self.save_contact(&jid, ContactField::FirstName, c.action.first_name.as_deref().unwrap_or(""));
            }
            Event::LoggedOut(l) => warn!("device logged out ({:?}); restart and scan the QR code again", l.reason),
            Event::PairSuccess(_) => info!("paired successfully"),
            _ => {}
        }
    }

    fn save_contact(&self, jid: &Jid, field: ContactField, value: &str) {
        if value.is_empty() {
            return;
        }
        let key = jid.to_non_ad().to_string();
        if let Err(e) = self.store.upsert_contact(&key, field, value) {
            warn!("failed to store contact {key}: {e:#}");
        }
        self.refresh_name(&jid.to_non_ad());
    }

    fn learn_lid(&self, lid: &str, pn: &str) {
        let user = |j: &str| j.split('@').next().unwrap_or_default().split(':').next().unwrap_or_default().to_string();
        let _ = self.store.put_lid(&user(lid), &user(pn));
    }

    /// The phone-number address for a LID, so one person is one chat row.
    /// Direct chats arrive under the LID while older history uses the number.
    async fn canonical(&self, jid: &Jid) -> Jid {
        if !jid.is_lid() {
            return jid.to_non_ad();
        }
        if let Some(pn) = self.store.pn_for_lid(&jid.user) {
            return Jid::new(pn.as_str(), Server::Pn);
        }
        if let Ok(client) = self.client() {
            if let Ok(Some(entry)) = client.get_lid_pn_entry(jid).await {
                let _ = self.store.put_lid(&entry.lid, &entry.phone_number);
                return Jid::new(&*entry.phone_number, Server::Pn);
            }
        }
        jid.to_non_ad()
    }

    async fn on_message(&self, msg: &wa::Message, info: &MessageInfo) -> Result<()> {
        // sender_alt carries the other address (LID <-> phone) of the sender.
        if let Some(alt) = &info.source.sender_alt {
            let (lid, pn) = if alt.is_lid() { (alt, &info.source.sender) } else { (&info.source.sender, alt) };
            if lid.is_lid() && pn.is_pn() {
                let _ = self.store.put_lid(&lid.user, &pn.user);
            }
        }
        let chat = self.canonical(&info.source.chat).await;
        let chat_jid = chat.to_string();
        let sender = self.canonical(&info.source.sender).await.user.to_string();
        if !info.source.is_from_me && !info.push_name.is_empty() {
            let _ = self.store.upsert_contact(&format!("{sender}@s.whatsapp.net"), ContactField::PushName, &info.push_name);
        }
        // Our own messages must not name the chat after us.
        let hint = if info.source.is_from_me { "" } else { sender.as_str() };
        let name = self.chat_name(&chat, None, hint).await;
        self.store.store_chat(&chat_jid, &name, info.timestamp)?;

        if let Some(change) = extract::change(msg) {
            return self.apply(change, &info.id, &chat_jid, &sender, info.timestamp, info.source.is_from_me);
        }
        let content = extract::text(msg);
        let media = extract::media(msg, info.timestamp);
        if content.is_empty() && media.kind.is_empty() {
            debug!("skipping message {} in {chat_jid}: no extractable content (fields: {})", info.id, extract::describe(msg));
            return Ok(());
        }
        self.store.store_message(&info.id, &chat_jid, &sender, &content, info.timestamp, info.source.is_from_me, &media)?;
        let dir = if info.source.is_from_me { "→" } else { "←" };
        if media.kind.is_empty() {
            info!("{dir} {sender}: {content}");
        } else {
            info!("{dir} {sender}: [{}: {}] {content}", media.kind, media.filename);
        }
        Ok(())
    }

    /// Edits and deletions rewrite the original; a reaction is its own line in
    /// the chat, quoting what it reacts to.
    fn apply(&self, change: extract::Change, id: &str, chat_jid: &str, sender: &str, ts: chrono::DateTime<Utc>, from_me: bool) -> Result<()> {
        use extract::Change;
        let (target, found) = match &change {
            Change::Edit { id, text } => (id, self.store.edit_message(id, chat_jid, text)?),
            Change::Revoke { id } => (id, self.store.revoke_message(id, chat_jid)?),
            Change::React { emoji, .. } if emoji.is_empty() => return Ok(()),
            Change::React { id: target, emoji } => {
                let quoted: String = self.store.message_content(target, chat_jid).unwrap_or_default().chars().take(80).collect();
                let content = if quoted.is_empty() { format!("[reacted {emoji}]") } else { format!("[reacted {emoji} to: {quoted}]") };
                self.store.store_message(id, chat_jid, sender, &content, ts, from_me, &Default::default())?;
                (target, true)
            }
        };
        if found {
            info!("{change:?} in {chat_jid}");
        } else {
            debug!("{change:?} in {chat_jid}: message {target} is not on file");
        }
        Ok(())
    }

    async fn on_history(&self, sync: &wa::HistorySync) {
        for m in &sync.phone_number_to_lid_mappings {
            if let (Some(pn), Some(lid)) = (&m.pn_jid, &m.lid_jid) {
                self.learn_lid(lid, pn);
            }
        }
        for p in &sync.pushnames {
            if let (Some(id), Some(name)) = (&p.id, &p.pushname) {
                if let Ok(jid) = id.parse::<Jid>() {
                    let jid = self.canonical(&jid).await;
                    let _ = self.store.upsert_contact(&jid.to_string(), ContactField::PushName, name);
                }
            }
        }
        let own = self.client().ok().and_then(|c| c.pn()).map(|j| j.user.to_string()).unwrap_or_default();
        let mut stored = 0usize;
        for conv in &sync.conversations {
            if let (Some(lid), Some(pn)) = (&conv.lid_jid, &conv.pn_jid) {
                self.learn_lid(lid, pn);
            }
            let Ok(jid) = conv.id.parse::<Jid>() else {
                warn!("history sync: bad chat JID {}", conv.id);
                continue;
            };
            let chat = self.canonical(&jid).await;
            let chat_jid = chat.to_string();
            let conv_name = conv.display_name.as_deref().or(conv.name.as_deref()).filter(|n| !n.is_empty());
            let name = self.chat_name(&chat, conv_name, "").await;
            for hm in &conv.messages {
                let Some(w) = hm.message.as_option() else { continue };
                let Some(m) = w.message.as_option() else { continue };
                let Some(ts) = w.message_timestamp.and_then(|t| Utc.timestamp_opt(t as i64, 0).single()) else { continue };
                let key = w.key.as_option();
                let from_me = key.and_then(|k| k.from_me).unwrap_or(false);
                let participant = key.and_then(|k| k.participant.clone()).or_else(|| w.participant.clone()).unwrap_or_default();
                let sender = if from_me {
                    own.clone()
                } else if !participant.is_empty() {
                    match participant.parse::<Jid>() {
                        Ok(p) => self.canonical(&p).await.user.to_string(),
                        Err(_) => participant.split('@').next().unwrap_or_default().to_string(),
                    }
                } else {
                    chat.user.to_string()
                };
                let id = key.and_then(|k| k.id.clone()).unwrap_or_default();
                if let Some(change) = extract::change(m) {
                    if self.store.store_chat(&chat_jid, &name, ts).is_ok() {
                        let _ = self.apply(change, &id, &chat_jid, &sender, ts, from_me);
                    }
                    continue;
                }
                let content = extract::text(m);
                let media = extract::media(m, ts);
                if content.is_empty() && media.kind.is_empty() {
                    continue;
                }
                if self.store.store_chat(&chat_jid, &name, ts).is_err() {
                    continue;
                }
                match self.store.store_message(&id, &chat_jid, &sender, &content, ts, from_me, &media) {
                    Ok(true) => stored += 1,
                    Ok(false) => {}
                    Err(e) => warn!("failed to store history message: {e:#}"),
                }
            }
        }
        info!("history sync: {} conversations, {stored} messages stored", sync.conversations.len());
    }

    /// Stored real name, else group subject / address book / push name, else
    /// the number. A stored name equal to the number is a placeholder and is
    /// looked up again, otherwise the number would stick forever.
    async fn chat_name(&self, chat: &Jid, conv_name: Option<&str>, sender: &str) -> String {
        let jid = chat.to_string();
        if let Some(name) = self.store.chat_name(&jid) {
            return name;
        }
        if chat.server == Server::Group {
            if let Some(n) = conv_name {
                return n.to_string();
            }
            if let Ok(client) = self.client() {
                if let Ok(meta) = client.groups().get_metadata(chat).await {
                    if !meta.subject.is_empty() {
                        return meta.subject;
                    }
                }
            }
            return format!("Group {}", chat.user);
        }
        self.store
            .contact_name(&jid)
            .or_else(|| conv_name.map(str::to_owned))
            .or_else(|| (!sender.is_empty()).then(|| sender.to_string()))
            .unwrap_or_else(|| chat.user.to_string())
    }

    /// Rewrites a chat row still holding the number placeholder, under the
    /// phone JID and the LID alike.
    fn refresh_name(&self, jid: &Jid) {
        let Some(name) = self.store.contact_name(&jid.to_string()) else { return };
        let mut rows = vec![jid.to_string()];
        if let Some(lid) = self.store.lid_for_pn(&jid.user) {
            rows.push(format!("{lid}@lid"));
        }
        for row in rows {
            if let Ok(true) = self.store.fill_chat_name(&row, &name) {
                info!("chat name for {row} is now {name}");
            }
        }
    }

    fn refresh_all_names(&self) {
        let Ok(rows) = self.store.unnamed_direct_chats() else { return };
        let total = rows.len();
        let mut named = 0;
        for row in rows {
            if let Some(name) = self.store.contact_name(&row) {
                if let Ok(true) = self.store.fill_chat_name(&row, &name) {
                    named += 1;
                }
            }
        }
        info!("chat name refresh: {named} of {total} chats updated");
    }

    // ---- send ---------------------------------------------------------------

    /// Text, or a file whose extension picks how it is sent. `voice` turns any
    /// decodable audio into a voice note. `reply_to` quotes a message of that chat.
    pub async fn send(&self, recipient: &str, text: &str, file: Option<media::Outgoing>, voice: bool, reply_to: Option<&str>) -> Result<String> {
        let client = self.client()?;
        if !client.is_connected() {
            bail!("Not connected to WhatsApp");
        }
        let to = parse_recipient(recipient)?;
        let chat = self.canonical(&to).await;
        let context = self.context_info(client, &chat, text, reply_to)?;
        let msg = match file {
            None if context.as_option().is_none() => wa::Message { conversation: Some(text.to_string()), ..Default::default() },
            None => wa::Message {
                extended_text_message: MessageField::some(wa::message::ExtendedTextMessage {
                    text: Some(text.to_string()),
                    context_info: context,
                    ..Default::default()
                }),
                ..Default::default()
            },
            Some(f) => self.media_message(client, text, f, voice, context).await?,
        };
        let sent = client.send_message(to.clone(), msg.clone()).await.map_err(|e| anyhow!("Error sending message: {e}"))?;
        // WhatsApp does not echo a sent message back, so it is recorded here or
        // never appears in the history.
        let own = client.pn().map(|j| j.user.to_string()).unwrap_or_default();
        let now = Utc::now();
        let name = self.chat_name(&chat, None, "").await;
        if let Err(e) = self.store.store_chat(&chat.to_string(), &name, now).and_then(|_| {
            self.store.store_message(&sent.message_id, &chat.to_string(), &own, text, now, true, &extract::media(&msg, now))
        }) {
            warn!("failed to store sent message: {e:#}");
        }
        Ok(format!("Message sent to {recipient}"))
    }

    /// Rewrites the text of a message you sent; WhatsApp accepts this only in
    /// the first 15 minutes or so.
    pub async fn edit(&self, recipient: &str, id: &str, text: &str) -> Result<String> {
        let (client, to, chat) = self.own_message(recipient, id, true).await?;
        let msg = wa::Message { conversation: Some(text.to_string()), ..Default::default() };
        client.edit_message(to, id, msg).await.map_err(|e| anyhow!("Error editing message: {e}"))?;
        self.store.edit_message(id, &chat, text)?;
        Ok(format!("Message {id} edited"))
    }

    /// Deletes a message you sent for everyone in the chat.
    pub async fn delete(&self, recipient: &str, id: &str) -> Result<String> {
        let (client, to, chat) = self.own_message(recipient, id, false).await?;
        client.revoke_message(to, id, whatsapp_rust::RevokeType::Sender).await.map_err(|e| anyhow!("Error deleting message: {e}"))?;
        self.store.revoke_message(id, &chat)?;
        Ok(format!("Message {id} deleted"))
    }

    /// The client, the JID to send to and the stored chat of a message of
    /// ours: WhatsApp would refuse the edit of anyone else's message anyway.
    async fn own_message(&self, recipient: &str, id: &str, text_only: bool) -> Result<(&Arc<Client>, Jid, String)> {
        let client = self.client()?;
        if !client.is_connected() {
            bail!("Not connected to WhatsApp");
        }
        let to = parse_recipient(recipient)?;
        let chat = self.canonical(&to).await.to_string();
        let (media, from_me, _) = self.store.media(id, &chat)?.ok_or_else(|| anyhow!("message {id} is not in this chat"))?;
        if !from_me {
            bail!("message {id} was not sent by you");
        }
        if text_only && !media.kind.is_empty() {
            bail!("only text messages can be edited");
        }
        Ok((client, to, chat))
    }

    /// Mentions (every "@<phone number>" in the text) and the quoted message of
    /// a reply. None when there is neither.
    fn context_info(&self, client: &Client, chat: &Jid, text: &str, reply_to: Option<&str>) -> Result<MessageField<wa::ContextInfo>> {
        let mut info = wa::ContextInfo { mentioned_jid: mentions(text), ..Default::default() };
        if let Some(id) = reply_to.filter(|id| !id.is_empty()) {
            let chat_jid = chat.to_string();
            let (from_me, sender, content) =
                self.store.quote(id, &chat_jid)?.ok_or_else(|| anyhow!("message {id} to reply to is not in this chat"))?;
            let author = if from_me { client.pn().map(|j| j.to_non_ad().user.to_string()).unwrap_or_default() } else { sender };
            info.stanza_id = Some(id.to_string());
            info.participant = Some(format!("{author}@s.whatsapp.net"));
            info.quoted_message = MessageField::some(wa::Message { conversation: Some(content), ..Default::default() });
        } else if info.mentioned_jid.is_empty() {
            return Ok(MessageField::none());
        }
        Ok(MessageField::some(info))
    }

    async fn media_message(
        &self,
        client: &Client,
        caption: &str,
        f: media::Outgoing,
        voice: bool,
        context_info: MessageField<wa::ContextInfo>,
    ) -> Result<wa::Message> {
        let ext = Path::new(&f.name).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let caption = (!caption.is_empty()).then(|| caption.to_string());
        if voice || ext == "ogg" || ext == "opus" {
            let data = f.data;
            let note = tokio::task::spawn_blocking(move || audio::voice_note(&data, Some(&ext)))
                .await?
                .context("Error converting file to opus ogg")?;
            let up = client.upload(note.ogg, MediaType::Audio, UploadOptions::default()).await?;
            return Ok(wa::Message {
                audio_message: MessageField::some(wa::message::AudioMessage {
                    mimetype: Some("audio/ogg; codecs=opus".into()),
                    url: Some(up.url),
                    direct_path: Some(up.direct_path),
                    media_key: Some(up.media_key.to_vec()),
                    file_enc_sha256: Some(up.file_enc_sha256.to_vec()),
                    file_sha256: Some(up.file_sha256.to_vec()),
                    file_length: Some(up.file_length),
                    seconds: Some(note.seconds),
                    ptt: Some(true),
                    waveform: Some(note.waveform),
                    context_info,
                    ..Default::default()
                }),
                ..Default::default()
            });
        }
        let mime = media::mime_of(Path::new(&f.name));
        let kind = match ext.as_str() {
            "jpg" | "jpeg" | "png" | "gif" | "webp" => MediaType::Image,
            "mp4" | "avi" | "mov" => MediaType::Video,
            _ => MediaType::Document,
        };
        let up = client.upload(f.data, kind, UploadOptions::default()).await?;
        let mut msg = wa::Message::default();
        match kind {
            MediaType::Image => {
                msg.image_message = MessageField::some(wa::message::ImageMessage {
                    caption,
                    mimetype: Some(mime),
                    url: Some(up.url),
                    direct_path: Some(up.direct_path),
                    media_key: Some(up.media_key.to_vec()),
                    file_enc_sha256: Some(up.file_enc_sha256.to_vec()),
                    file_sha256: Some(up.file_sha256.to_vec()),
                    file_length: Some(up.file_length),
                    context_info,
                    ..Default::default()
                })
            }
            MediaType::Video => {
                msg.video_message = MessageField::some(wa::message::VideoMessage {
                    caption,
                    mimetype: Some(mime),
                    url: Some(up.url),
                    direct_path: Some(up.direct_path),
                    media_key: Some(up.media_key.to_vec()),
                    file_enc_sha256: Some(up.file_enc_sha256.to_vec()),
                    file_sha256: Some(up.file_sha256.to_vec()),
                    file_length: Some(up.file_length),
                    context_info,
                    ..Default::default()
                })
            }
            _ => {
                msg.document_message = MessageField::some(wa::message::DocumentMessage {
                    title: Some(f.name.clone()),
                    file_name: Some(f.name),
                    caption,
                    mimetype: Some(mime),
                    url: Some(up.url),
                    direct_path: Some(up.direct_path),
                    media_key: Some(up.media_key.to_vec()),
                    file_enc_sha256: Some(up.file_enc_sha256.to_vec()),
                    file_sha256: Some(up.file_sha256.to_vec()),
                    file_length: Some(up.file_length),
                    context_info,
                    ..Default::default()
                })
            }
        }
        Ok(msg)
    }

    // ---- download -------------------------------------------------------------

    /// Saves a message's media under the store and returns (kind, path).
    pub async fn download(&self, message_id: &str, chat_jid: &str) -> Result<(String, PathBuf)> {
        let (mut m, from_me, sender) = self.store.media(message_id, chat_jid)?.ok_or_else(|| anyhow!("failed to find message"))?;
        if m.kind.is_empty() {
            bail!("not a media message");
        }
        let path = media::download_path(&self.store_dir, chat_jid, message_id, &m.filename);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return Ok((m.kind, path));
        }
        let data = match self.fetch(&m).await {
            Ok(data) => data,
            // The CDN drops media after a few weeks; the phone can upload it again.
            Err(e) => {
                warn!("download of {message_id} failed ({e:#}); asking the phone to re-upload it");
                m.direct_path = self.reupload(message_id, chat_jid, &m, from_me, &sender).await.context(format!("{e:#}"))?;
                self.store.set_direct_path(message_id, chat_jid, &m.direct_path)?;
                self.fetch(&m).await?
            }
        };
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        tokio::fs::write(&path, &data).await.context("failed to save media file")?;
        info!("downloaded {} media to {} ({} bytes)", m.kind, path.display(), data.len());
        Ok((m.kind, path))
    }

    async fn reupload(&self, message_id: &str, chat_jid: &str, m: &Media, from_me: bool, sender: &str) -> Result<String> {
        use whatsapp_rust::{MediaReuploadRequest, MediaRetryResult};
        if m.media_key.is_empty() {
            bail!("no media key to request a re-upload with");
        }
        let chat: Jid = chat_jid.parse().map_err(|e| anyhow!("bad chat JID {chat_jid}: {e:?}"))?;
        let participant = (chat.server == Server::Group && !from_me && !sender.is_empty()).then(|| Jid::new(sender, Server::Pn));
        // Chats are stored under the phone number, but the phone only answers for
        // the address the message was keyed under, which is the LID when one exists.
        let mut chats = vec![chat.clone()];
        if chat.is_pn() {
            if let Some(lid) = self.store.lid_for_pn(&chat.user) {
                chats.insert(0, Jid::new(&lid, Server::Lid));
            }
        }
        let mut last = anyhow!("no address to ask");
        for chat in &chats {
            let req = MediaReuploadRequest {
                msg_id: message_id,
                chat_jid: chat,
                media_key: &m.media_key,
                is_from_me: from_me,
                participant: participant.as_ref(),
            };
            match self.client()?.media_reupload().request(&req).await {
                Ok(MediaRetryResult::Success { direct_path }) => return Ok(direct_path),
                Ok(other) => bail!("media re-upload refused: {other:?}"),
                Err(e) => last = anyhow!("media re-upload via {chat} failed: {e}"),
            }
        }
        Err(last)
    }

    async fn fetch(&self, m: &Media) -> Result<Vec<u8>> {
        if m.media_key.is_empty() || m.file_sha256.is_empty() || m.file_enc_sha256.is_empty() || m.file_length == 0 {
            bail!("incomplete media information for download");
        }
        // Rows written before direct_path existed carry it inside the URL, query
        // string included: the CDN signature lives there.
        let direct_path = if m.direct_path.is_empty() {
            m.url.split_once(".net/").map(|(_, rest)| format!("/{rest}")).ok_or_else(|| anyhow!("no direct path"))?
        } else {
            m.direct_path.clone()
        };
        let kind = match m.kind.as_str() {
            "image" | "sticker" => MediaType::Image,
            "video" => MediaType::Video,
            "audio" => MediaType::Audio,
            "document" => MediaType::Document,
            other => bail!("unsupported media type: {other}"),
        };
        let params = DownloadParams::encrypted(direct_path, &m.media_key, &m.file_sha256, &m.file_enc_sha256, m.file_length, kind);
        self.client()?.download(&params).await.map_err(|e| anyhow!("failed to download media: {e}"))
    }
}

/// JIDs for every "@<phone number>" in the text.
fn parse_recipient(recipient: &str) -> Result<Jid> {
    if recipient.contains('@') {
        recipient.parse().map_err(|e| anyhow!("Error parsing JID: {e:?}"))
    } else {
        Ok(Jid::new(recipient, Server::Pn))
    }
}

fn mentions(text: &str) -> Vec<String> {
    text.split('@')
        .skip(1)
        .filter_map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            (8..=15).contains(&digits.len()).then(|| format!("{digits}@s.whatsapp.net"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn mentions_are_phone_numbers_after_an_at() {
        assert_eq!(
            super::mentions("oi @5511999999999, fala com @554391524065! e-mail a@b.com @123"),
            vec!["5511999999999@s.whatsapp.net".to_string(), "554391524065@s.whatsapp.net".to_string()]
        );
        assert!(super::mentions("sem menção").is_empty());
    }
}

fn print_qr(code: &str, ttl: std::time::Duration) {
    use qrcode::render::unicode::Dense1x2;
    match qrcode::QrCode::new(code.as_bytes()) {
        Ok(qr) => {
            let art = qr.render::<Dense1x2>().quiet_zone(true).build();
            println!("\nScan this QR code with WhatsApp (valid {}s):\n{art}\n{code}\n", ttl.as_secs());
        }
        Err(_) => println!("\nQR code (valid {}s): {code}\n", ttl.as_secs()),
    }
}
