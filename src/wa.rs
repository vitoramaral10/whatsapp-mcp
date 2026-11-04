//! The WhatsApp side: session, events into the history store, send, download.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{TimeZone, Utc};
use log::{debug, info, warn};
use whatsapp_rust::ErrorChainExt;
use whatsapp_rust::download::{DownloadParams, MediaType};
use whatsapp_rust::prelude::*;
use whatsapp_rust::upload::UploadOptions;

use crate::store::{ContactField, DELETED, EDITED, Media, Store, TRANSCRIBED};
use crate::{audio, extract, media};

pub struct Wa {
    pub store: Arc<Store>,
    pub store_dir: PathBuf,
    client: OnceLock<Arc<Client>>,
    /// Groups whose subject was already asked for this run.
    groups_tried: Mutex<HashSet<String>>,
}

impl Wa {
    pub fn new(store: Arc<Store>, store_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self { store, store_dir, client: OnceLock::new(), groups_tried: Mutex::new(HashSet::new()) })
    }

    fn client(&self) -> Result<&Arc<Client>> {
        self.client.get().ok_or_else(|| anyhow!("WhatsApp client is not started"))
    }

    /// Builds the client and starts it in the background. The handle resolves
    /// when the client stops; `shutdown()` on it flushes the session state.
    /// `pair_phone` asks for an 8-character pairing code for that number, to type
    /// on the phone instead of scanning a QR code from a second screen.
    pub async fn start(self: &Arc<Self>, session_db: &Path, pair_phone: Option<String>) -> Result<BotHandle> {
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
        Ok(bot.spawn())
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
                // Plain SQLite work over every chat: off the async workers.
                tokio::task::spawn_blocking(move || {
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
            // Reading on the phone: live receipts, and app-state marks (also
            // replayed in full after pairing).
            Event::Receipt(r) if r.r#type == whatsapp_rust::wacore::types::presence::ReceiptType::ReadSelf => {
                let chat = self.canonical(&r.source.chat).await;
                self.store_read(&chat, r.timestamp);
            }
            Event::MarkChatAsReadUpdate(u) if u.action.read == Some(true) => {
                let chat = self.canonical(&u.jid).await;
                self.store_read(&chat, u.timestamp);
            }
            Event::LoggedOut(l) => warn!("device logged out ({:?}); restart and scan the QR code again", l.reason),
            Event::PairSuccess(_) => info!("paired successfully"),
            _ => {}
        }
    }

    fn store_read(&self, chat: &Jid, at: chrono::DateTime<Utc>) {
        if let Err(e) = self.store.mark_read(&chat.to_string(), at) {
            warn!("failed to store read mark of {chat}: {e:#}");
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
        self.put_lid(&user(lid), &user(pn));
    }

    fn put_lid(&self, lid: &str, pn: &str) {
        if let Err(e) = self.store.put_lid(lid, pn) {
            warn!("failed to store LID mapping {lid} -> {pn}: {e:#}");
        }
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
        if let Ok(client) = self.client()
            && let Ok(Some(entry)) = client.get_lid_pn_entry(jid).await
        {
            self.put_lid(&entry.lid, &entry.phone_number);
            return Jid::new(&*entry.phone_number, Server::Pn);
        }
        jid.to_non_ad()
    }

    /// The JID a chat's rows are stored under (a LID becomes its phone number);
    /// what does not parse is returned as is.
    pub async fn canonical_jid(&self, chat_jid: &str) -> String {
        match chat_jid.parse::<Jid>() {
            Ok(jid) => self.canonical(&jid).await.to_string(),
            Err(_) => chat_jid.to_string(),
        }
    }

    /// Our own phone number, empty before pairing.
    fn own_user(&self) -> String {
        self.client().ok().and_then(|c| c.pn()).map(|j| j.user.to_string()).unwrap_or_default()
    }

    async fn on_message(&self, msg: &wa::Message, info: &MessageInfo) -> Result<()> {
        // sender_alt carries the other address (LID <-> phone) of the sender.
        if let Some(alt) = &info.source.sender_alt {
            let (lid, pn) = if alt.is_lid() { (alt, &info.source.sender) } else { (&info.source.sender, alt) };
            if lid.is_lid() && pn.is_pn() {
                self.put_lid(&lid.user, &pn.user);
            }
        }
        let chat = self.canonical(&info.source.chat).await;
        let chat_jid = chat.to_string();
        let sender = stored_sender(&self.canonical(&info.source.sender).await);
        if !info.source.is_from_me
            && !info.push_name.is_empty()
            && let Err(e) = self.store.upsert_contact(&sender_jid(&sender).to_string(), ContactField::PushName, &info.push_name)
        {
            warn!("failed to store push name of {sender}: {e:#}");
        }

        let change = extract::change(msg);
        let (content, media) = match change {
            Some(_) => Default::default(),
            None => (extract::text(msg), extract::media(msg, info.timestamp)),
        };
        if change.is_none() && content.is_empty() && media.kind.is_empty() {
            debug!("skipping message {} in {chat_jid}: no extractable content (fields: {})", info.id, extract::describe(msg));
            return Ok(());
        }
        let name = self.chat_name(&chat, None).await;
        self.store.store_chat(&chat_jid, &name, info.timestamp)?;
        if let Some(change) = change {
            return self.apply(change, &info.id, &chat_jid, &sender, info.timestamp, info.source.is_from_me);
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
    fn apply(
        &self,
        change: extract::Change,
        id: &str,
        chat_jid: &str,
        sender: &str,
        ts: chrono::DateTime<Utc>,
        from_me: bool,
    ) -> Result<()> {
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
            if let (Some(id), Some(name)) = (&p.id, &p.pushname)
                && let Ok(jid) = id.parse::<Jid>()
            {
                let jid = self.canonical(&jid).await;
                if let Err(e) = self.store.upsert_contact(&jid.to_string(), ContactField::PushName, name) {
                    warn!("history sync: failed to store push name of {jid}: {e:#}");
                }
            }
        }
        let own = self.own_user();
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
            // Looked up once, and only for a conversation with something to store.
            let mut name: Option<String> = None;
            for hm in &conv.messages {
                let Some(w) = hm.message.as_option() else { continue };
                let Some(m) = w.message.as_option() else { continue };
                let Some(ts) = w.message_timestamp.and_then(|t| Utc.timestamp_opt(t as i64, 0).single()) else { continue };
                let key = w.key.as_option();
                let Some(id) = key.and_then(|k| k.id.as_deref()).filter(|id| !id.is_empty()) else {
                    debug!("history sync: message without an id in {chat_jid} skipped");
                    continue;
                };
                let change = extract::change(m);
                let (content, media) = match change {
                    Some(_) => Default::default(),
                    None => (extract::text(m), extract::media(m, ts)),
                };
                if change.is_none() && content.is_empty() && media.kind.is_empty() {
                    continue;
                }
                let from_me = key.and_then(|k| k.from_me).unwrap_or(false);
                let participant = key.and_then(|k| k.participant.clone()).or_else(|| w.participant.clone()).unwrap_or_default();
                let sender = if from_me {
                    own.clone()
                } else if !participant.is_empty() {
                    match participant.parse::<Jid>() {
                        Ok(p) => stored_sender(&self.canonical(&p).await),
                        Err(_) => participant.split('@').next().unwrap_or_default().to_string(),
                    }
                } else {
                    stored_sender(&chat)
                };
                if name.is_none() {
                    name = Some(self.chat_name(&chat, conv_name).await);
                }
                if let Err(e) = self.store.store_chat(&chat_jid, name.as_deref().unwrap_or_default(), ts) {
                    warn!("history sync: failed to store chat {chat_jid}: {e:#}");
                    continue;
                }
                if let Some(change) = change {
                    if let Err(e) = self.apply(change, id, &chat_jid, &sender, ts, from_me) {
                        warn!("history sync: failed to apply change {id} in {chat_jid}: {e:#}");
                    }
                    continue;
                }
                match self.store.store_message(id, &chat_jid, &sender, &content, ts, from_me, &media) {
                    Ok(true) => stored += 1,
                    Ok(false) => {}
                    Err(e) => warn!("failed to store history message: {e:#}"),
                }
            }
        }
        info!("history sync: {} conversations, {stored} messages stored", sync.conversations.len());
    }

    fn groups_tried(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        self.groups_tried.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Stored real name, else group subject / address book / push name, else
    /// the number. A stored name equal to the number is a placeholder and is
    /// looked up again, otherwise the number would stick forever; a group's
    /// subject is asked for once per run.
    async fn chat_name(&self, chat: &Jid, conv_name: Option<&str>) -> String {
        let jid = chat.to_string();
        if let Some(name) = self.store.chat_name(&jid) {
            return name;
        }
        if chat.server == Server::Group {
            if let Some(n) = conv_name {
                return n.to_string();
            }
            let first_try = self.groups_tried().insert(jid.clone());
            if first_try && let Ok(client) = self.client() {
                match client.groups().get_metadata(chat).await {
                    Ok(meta) if !meta.subject.is_empty() => return meta.subject,
                    Ok(_) => {}
                    // Being offline says nothing about the group: ask again later.
                    Err(e) if e.is_transport_unavailable() || e.is_timeout() => {
                        self.groups_tried().remove(&jid);
                    }
                    Err(e) => debug!("group {jid} metadata: {e}"),
                }
            }
            return format!("Group {}", chat.user);
        }
        self.store.contact_name(&jid).or_else(|| conv_name.map(str::to_owned)).unwrap_or_else(|| chat.user.to_string())
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
        let rows = match self.store.unnamed_direct_chats() {
            Ok(rows) => rows,
            Err(e) => {
                warn!("chat name refresh failed: {e:#}");
                return;
            }
        };
        let total = rows.len();
        let mut named = 0;
        for row in rows {
            if let Some(name) = self.store.contact_name(&row)
                && let Ok(true) = self.store.fill_chat_name(&row, &name)
            {
                named += 1;
            }
        }
        info!("chat name refresh: {named} of {total} chats updated");
    }

    // ---- send ---------------------------------------------------------------

    /// Text, or a file whose extension picks how it is sent. `voice` turns any
    /// decodable audio into a voice note. `reply_to` quotes a message of that chat.
    pub async fn send(
        &self,
        recipient: &str,
        text: &str,
        file: Option<media::Outgoing>,
        voice: bool,
        reply_to: Option<&str>,
    ) -> Result<String> {
        let client = self.connected()?;
        let to = recipient_jid(recipient)?;
        let chat = self.canonical(&to).await;
        let context = self.context_info(&chat, text, reply_to)?;
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
        let own = self.own_user();
        let now = Utc::now();
        let name = self.chat_name(&chat, None).await;
        if let Err(e) = self
            .store
            .store_chat(&chat.to_string(), &name, now)
            .and_then(|_| self.store.store_message(&sent.message_id, &chat.to_string(), &own, text, now, true, &extract::media(&msg, now)))
        {
            warn!("failed to store sent message: {e:#}");
        }
        Ok(format!("Message sent to {recipient} (message ID: {})", sent.message_id))
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

    /// Reacts to a message of the chat; an empty emoji takes the reaction back.
    pub async fn react(&self, recipient: &str, id: &str, emoji: &str) -> Result<String> {
        let client = self.connected()?;
        let to = recipient_jid(recipient)?;
        let chat = self.canonical(&to).await.to_string();
        let (_, from_me, sender) = self.store.media(id, &chat)?.ok_or_else(|| anyhow!("message {id} is not in this chat"))?;
        let key = wa::MessageKey {
            remote_jid: Some(to.to_string()),
            from_me: Some(from_me),
            id: Some(id.to_string()),
            participant: ((to.server == Server::Group) && !from_me).then(|| sender_jid(&sender).to_string()),
        };
        let sent = client.send_reaction(to, key, emoji).await.map_err(|e| anyhow!("Error sending reaction: {e}"))?;
        if !emoji.is_empty() {
            let change = extract::Change::React { id: id.to_string(), emoji: emoji.to_string() };
            self.apply(change, &sent.message_id, &chat, &self.own_user(), Utc::now(), true)?;
        }
        Ok(format!("Reacted {emoji} to {id}"))
    }

    /// Marks every unread message of the chat as read: blue ticks for the
    /// senders, and the chat read on your other devices.
    pub async fn mark_read(&self, recipient: &str) -> Result<String> {
        let client = self.connected()?;
        let to = recipient_jid(recipient)?;
        let chat = self.canonical(&to).await.to_string();
        let unread = self.store.unread_messages(&chat)?;
        let mut by_sender: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
        for (id, sender) in &unread {
            by_sender.entry(if to.server == Server::Group { sender.as_str() } else { "" }).or_default().push(id);
        }
        for (sender, ids) in by_sender {
            let sender = (!sender.is_empty()).then(|| sender_jid(sender));
            client.mark_as_read(&to, sender.as_ref(), &ids).await.map_err(|e| anyhow!("Error sending read receipts: {e:#}"))?;
        }
        if let Err(e) = client.chat_actions().mark_chat_as_read(&to, true, None).await {
            warn!("chat {chat} read on the phone failed: {e:?}");
        }
        self.store.mark_read(&chat, Utc::now())?;
        Ok(format!("{} messages marked as read in {recipient}", unread.len()))
    }

    pub fn is_connected(&self) -> bool {
        self.client().is_ok_and(|c| c.is_connected())
    }

    fn connected(&self) -> Result<&Arc<Client>> {
        let client = self.client()?;
        if !client.is_connected() {
            bail!("Not connected to WhatsApp");
        }
        Ok(client)
    }

    /// The client, the JID to send to and the stored chat of a message of
    /// ours: WhatsApp would refuse the edit of anyone else's message anyway.
    async fn own_message(&self, recipient: &str, id: &str, text_only: bool) -> Result<(&Arc<Client>, Jid, String)> {
        let client = self.connected()?;
        let to = recipient_jid(recipient)?;
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
    fn context_info(&self, chat: &Jid, text: &str, reply_to: Option<&str>) -> Result<MessageField<wa::ContextInfo>> {
        let mut info = wa::ContextInfo { mentioned_jid: mentions(text), ..Default::default() };
        if let Some(id) = reply_to.filter(|id| !id.is_empty()) {
            let chat_jid = chat.to_string();
            let (from_me, sender, content) =
                self.store.quote(id, &chat_jid)?.ok_or_else(|| anyhow!("message {id} to reply to is not in this chat"))?;
            let author = if from_me { self.own_user() } else { sender };
            info.stanza_id = Some(id.to_string());
            info.participant = Some(sender_jid(&author).to_string());
            info.quoted_message =
                MessageField::some(wa::Message { conversation: Some(quotable(&content).to_string()), ..Default::default() });
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
        // The fields every uploaded media message carries, plus its own.
        macro_rules! uploaded {
            ($ty:ident, $up:ident, $mime:expr, { $($field:ident: $value:expr),* $(,)? }) => {
                MessageField::some(wa::message::$ty {
                    mimetype: Some($mime),
                    url: Some($up.url),
                    direct_path: Some($up.direct_path),
                    media_key: Some($up.media_key.to_vec()),
                    file_enc_sha256: Some($up.file_enc_sha256.to_vec()),
                    file_sha256: Some($up.file_sha256.to_vec()),
                    file_length: Some($up.file_length),
                    context_info,
                    $($field: $value,)*
                    ..Default::default()
                })
            };
        }
        let ext = Path::new(&f.name).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let caption = (!caption.is_empty()).then(|| caption.to_string());
        if voice || ext == "ogg" || ext == "opus" {
            let data = f.data;
            let note = tokio::task::spawn_blocking(move || audio::voice_note(&data, Some(&ext)))
                .await?
                .context("Error converting file to opus ogg")?;
            let up = client.upload(note.ogg, MediaType::Audio, UploadOptions::default()).await?;
            return Ok(wa::Message {
                audio_message: uploaded!(AudioMessage, up, "audio/ogg; codecs=opus".into(), {
                    seconds: Some(note.seconds),
                    ptt: Some(true),
                    waveform: Some(note.waveform),
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
            MediaType::Image => msg.image_message = uploaded!(ImageMessage, up, mime, { caption: caption }),
            MediaType::Video => msg.video_message = uploaded!(VideoMessage, up, mime, { caption: caption }),
            _ => {
                msg.document_message = uploaded!(DocumentMessage, up, mime, {
                    title: Some(f.name.clone()),
                    file_name: Some(f.name),
                    caption: caption,
                })
            }
        }
        Ok(msg)
    }

    // ---- download -------------------------------------------------------------

    /// Saves a message's media under the store and returns (kind, path).
    pub async fn download(&self, message_id: &str, chat_jid: &str) -> Result<(String, PathBuf)> {
        let chat: Jid = chat_jid.parse().with_context(|| format!("bad chat JID {chat_jid}"))?;
        let chat_jid = self.canonical(&chat).await.to_string();
        let (mut m, from_me, sender) = self.store.media(message_id, &chat_jid)?.ok_or_else(|| anyhow!("failed to find message"))?;
        if m.kind.is_empty() {
            bail!("not a media message");
        }
        let path = media::download_path(&self.store_dir, &chat_jid, message_id, &m.filename);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return Ok((m.kind, path));
        }
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        // Written aside and renamed when complete, so the final path is either
        // absent or whole; one name per call keeps concurrent callers apart.
        let part = PartFile(part_path(&path));
        let part = &part.0;
        let saved = async {
            let len = match self.fetch(&m, part).await {
                Ok(len) => len,
                // The CDN drops media after a few weeks; the phone can upload it again.
                Err(e) if needs_reupload(&e) => {
                    warn!("download of {message_id} failed ({e:#}); asking the phone to re-upload it");
                    m.direct_path = self.reupload(message_id, &chat_jid, &m, from_me, &sender).await.context(format!("{e:#}"))?;
                    self.store.set_direct_path(message_id, &chat_jid, &m.direct_path)?;
                    self.fetch(&m, part).await?
                }
                Err(e) => return Err(e),
            };
            tokio::fs::rename(&part, &path).await.context("failed to save media file")?;
            Ok(len)
        }
        .await;
        let len = saved?;
        info!("downloaded {} media to {} ({len} bytes)", m.kind, path.display());
        Ok((m.kind, path))
    }

    async fn reupload(&self, message_id: &str, chat_jid: &str, m: &Media, from_me: bool, sender: &str) -> Result<String> {
        use whatsapp_rust::wacore::media_retry::{build_media_retry_receipt, encrypt_media_retry_receipt, parse_media_retry_notification};
        use whatsapp_rust::{MediaRetryResult, NodeFilter};
        if m.media_key.is_empty() {
            bail!("no media key to request a re-upload with");
        }
        let chat: Jid = chat_jid.parse().with_context(|| format!("bad chat JID {chat_jid}"))?;
        let participant = (chat.server == Server::Group && !from_me && !sender.is_empty()).then(|| sender_jid(sender));
        // Chats are stored under the phone number, but the phone only answers for
        // the address the message was keyed under, which is the LID when one exists.
        let mut chats = vec![chat.clone()];
        if chat.is_pn()
            && let Some(lid) = self.store.lid_for_pn(&chat.user)
        {
            chats.insert(0, Jid::new(&lid, Server::Lid));
        }
        let client = self.client()?;
        // shortcut: whatsapp-rust 0.7.0 media_reupload() addresses the receipt to
        // our own device (user:N@...), so the phone never sees it; whatsmeow and
        // the crate's own history-sync receipt use the bare JID. Back to
        // media_reupload() once the crate fixes it.
        let own = client.pn().ok_or_else(|| anyhow!("not logged in"))?.to_non_ad();
        let (ciphertext, iv) = encrypt_media_retry_receipt(&m.media_key, message_id)?;
        let mut last = anyhow!("no address to ask");
        for chat in &chats {
            let waiter = client.wait_for_node(NodeFilter::tag("notification").attr("type", "mediaretry").attr("id", message_id));
            let receipt = build_media_retry_receipt(&own, message_id, chat, from_me, participant.as_ref(), &ciphertext, &iv);
            client.send_node(receipt).await.context("media re-upload request failed")?;
            let node = match tokio::time::timeout(std::time::Duration::from_secs(30), waiter).await {
                Ok(Ok(node)) => node,
                Ok(Err(_)) => bail!("media retry waiter cancelled"),
                Err(_) => {
                    last = anyhow!("media re-upload via {chat} failed: the phone did not answer in 30 s");
                    continue;
                }
            };
            match parse_media_retry_notification(node.get(), &m.media_key) {
                Ok(MediaRetryResult::Success { direct_path }) => return Ok(direct_path),
                Ok(other) => last = anyhow!("media re-upload via {chat} refused: {other:?}"),
                Err(e) => last = e.context(format!("media re-upload via {chat}: unreadable answer")),
            }
        }
        Err(last)
    }

    /// Streams the media into `dest` and returns its size.
    async fn fetch(&self, m: &Media, dest: &Path) -> Result<u64> {
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
        let client = self.client()?;
        let file = tokio::fs::File::create(dest).await.with_context(|| format!("create {}", dest.display()))?.into_std().await;
        let file = client.download_from_params_to_writer(&params, file).await.context("failed to download media")?;
        let file = tokio::fs::File::from_std(file);
        file.sync_all().await.context("failed to save media file")?;
        Ok(file.metadata().await?.len())
    }
}

/// Whether a failed fetch is the CDN no longer having the media, which a
/// re-upload from the phone fixes, rather than a network or local failure.
fn needs_reupload(e: &anyhow::Error) -> bool {
    let cause: &(dyn std::error::Error + 'static) = e.as_ref();
    matches!(cause.http_status(), Some(401 | 403 | 404 | 410))
}

/// A download's temporary file, removed unless it was renamed into place: also
/// when the call is cancelled midway (a client hanging up drops the future).
struct PartFile(PathBuf);

impl Drop for PartFile {
    fn drop(&mut self) {
        // After a successful rename the path is gone and this is a no-op.
        let _ = std::fs::remove_file(&self.0);
    }
}

fn part_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut p = path.as_os_str().to_owned();
    p.push(format!(".{}.part", NEXT.fetch_add(1, Ordering::Relaxed)));
    p.into()
}

/// A recipient typed by a person or a model: a phone number with country code
/// (punctuation allowed) or the JID of a person, a group or a LID.
pub fn recipient_jid(s: &str) -> Result<Jid> {
    let s = s.trim();
    if s.contains('@') {
        let jid: Jid = s.parse().with_context(|| format!("invalid JID {s}"))?;
        if jid.user.is_empty() || !matches!(jid.server, Server::Pn | Server::Group | Server::Lid) {
            bail!("unsupported JID {s}: expected a phone number, a group or a LID");
        }
        return Ok(jid);
    }
    let digits: String = s.chars().filter(|c| !matches!(c, '+' | ' ' | '-' | '(' | ')')).collect();
    if !(8..=15).contains(&digits.len()) || !digits.bytes().all(|b| b.is_ascii_digit()) {
        bail!("invalid phone number {s}: expected 8 to 15 digits with the country code");
    }
    Ok(Jid::new(&digits, Server::Pn))
}

/// A stored sender back as a JID: a full JID as is, a bare user is a phone number.
fn sender_jid(s: &str) -> Jid {
    if s.contains('@') {
        s.parse().unwrap_or_else(|_| Jid::new(s.split('@').next().unwrap_or_default(), Server::Pn))
    } else {
        Jid::new(s, Server::Pn)
    }
}

/// How a (canonical) sender is stored: the bare phone number, or the whole
/// LID JID when no number is known, so the two cannot be confused.
fn stored_sender(jid: &Jid) -> String {
    if jid.is_lid() { format!("{}@lid", jid.user) } else { jid.user.to_string() }
}

/// Stored content without the markers this server adds: what the message said.
fn quotable(content: &str) -> &str {
    let mut s = content;
    while let Some(rest) = s.strip_prefix(EDITED).or_else(|| s.strip_prefix(DELETED)) {
        s = rest;
    }
    match s.find(TRANSCRIBED) {
        Some(i) => s[..i].trim_end_matches('\n'),
        None => s,
    }
}

/// JIDs for every "@<phone number>" in the text.
fn mentions(text: &str) -> Vec<String> {
    text.split('@')
        .skip(1)
        .filter_map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            (8..=15).contains(&digits.len()).then(|| format!("{digits}@s.whatsapp.net"))
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mentions_are_phone_numbers_after_an_at() {
        assert_eq!(
            mentions("oi @5511900000001, fala com @5511900000002! e-mail a@b.com @123"),
            vec!["5511900000001@s.whatsapp.net".to_string(), "5511900000002@s.whatsapp.net".to_string()]
        );
        assert!(mentions("sem menção").is_empty());
    }

    #[test]
    fn recipients_are_numbers_or_supported_jids() {
        let ok = |s: &str| recipient_jid(s).unwrap().to_string();
        assert_eq!(ok("5511900000001"), "5511900000001@s.whatsapp.net");
        assert_eq!(ok("+55 (11) 90000-0001"), "5511900000001@s.whatsapp.net");
        assert_eq!(ok("5511900000001@s.whatsapp.net"), "5511900000001@s.whatsapp.net");
        assert_eq!(ok("120363000000000001@g.us"), "120363000000000001@g.us");
        assert_eq!(ok("100000000000001@lid"), "100000000000001@lid");
        for bad in ["", "1234567", "5511900000001234567", "55119abc0001", "status@broadcast", "1@newsletter", "@s.whatsapp.net"] {
            assert!(recipient_jid(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn stored_senders_round_trip_to_jids() {
        let lid = Jid::new("100000000000001", Server::Lid);
        assert_eq!(stored_sender(&lid), "100000000000001@lid");
        assert_eq!(sender_jid(&stored_sender(&lid)), lid);
        let pn = Jid::new("5511900000001", Server::Pn);
        assert_eq!(stored_sender(&pn), "5511900000001");
        assert_eq!(sender_jid(&stored_sender(&pn)), pn);
    }

    #[test]
    fn quotes_drop_the_stored_markers() {
        assert_eq!(quotable("oi"), "oi");
        assert_eq!(quotable(&format!("{EDITED}oi")), "oi");
        assert_eq!(quotable(&format!("{DELETED}{EDITED}oi")), "oi");
        assert_eq!(quotable(&format!("legenda\n{TRANSCRIBED}texto falado")), "legenda");
        assert_eq!(quotable(&format!("{TRANSCRIBED}texto falado")), "");
    }

    #[test]
    fn only_a_cdn_refusal_asks_for_a_reupload() {
        let status = |s| anyhow::Error::new(whatsapp_rust::http::HttpStatusError { status: s }).context("download").context("outer");
        for s in [401, 403, 404, 410] {
            assert!(needs_reupload(&status(s)), "{s}");
        }
        assert!(!needs_reupload(&status(500)));
        assert!(!needs_reupload(&status(429)));
        assert!(!needs_reupload(&anyhow!("connection refused")));
    }
}
