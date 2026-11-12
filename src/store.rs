//! messages.db: chat history, contacts and the LID -> phone map.
//!
//! The schema and the text timestamp format (`YYYY-MM-DD HH:MM:SS+00:00`) are the
//! ones the Go bridge wrote, so an existing database keeps working unchanged.
//! Contacts and LID mappings used to be read from whatsmeow's own store; they now
//! live here, so the history no longer depends on the WhatsApp library's tables.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

/// The media columns of a redelivered message move as one set, never mixed
/// field by field: the incoming set wins only when nothing is stored yet or
/// it is another upload (other encrypted hash). A replay of the same upload
/// keeps the stored set, whose path may have been refreshed since.
const FRESH_MEDIA: &str = "(length(excluded.media_key) > 0 AND (length(COALESCE(messages.media_key, x'')) = 0
     OR excluded.file_enc_sha256 IS NOT messages.file_enc_sha256))";

const TS_FORMAT: &str = "%Y-%m-%d %H:%M:%S+00:00";

/// A chat name that is empty or only digits is a placeholder (the number, a
/// LID, or the account's own number the Go bridge used to write), never a real
/// name; so is "Group <digits>", the fallback written when a group's subject
/// was unknown. `is_placeholder` is the same rule in Rust.
const PLACEHOLDER: &str = "(name IS NULL OR name = ''
     OR (name GLOB '[0-9]*' AND name NOT GLOB '*[^0-9]*')
     OR (name GLOB 'Group [0-9]*' AND substr(name, 7) NOT GLOB '*[^0-9-]*'))";

fn is_placeholder(name: &str) -> bool {
    match name.strip_prefix("Group ") {
        // Old-style group ids are <creator>-<timestamp>.
        Some(id) => !id.is_empty() && id.starts_with(|c: char| c.is_ascii_digit()) && id.bytes().all(|b| b.is_ascii_digit() || b == b'-'),
        None => name.is_empty() || name.bytes().all(|b| b.is_ascii_digit()),
    }
}

/// The best name a contact row has: address book, business, first name, push name.
const BEST_NAME: &str = "COALESCE(NULLIF(full_name, ''), NULLIF(business_name, ''), NULLIF(first_name, ''), NULLIF(push_name, ''))";

pub const EDITED: &str = "[edited] ";
pub const DELETED: &str = "[deleted] ";
pub const TRANSCRIBED: &str = "[transcription] ";
/// A window needs one line this long to get a vector: one of only short
/// replies ("ok", "kk") sits near any query and buries the real matches.
const MIN_SEMANTIC_CHARS: usize = 15;
/// A message is embedded with up to this many messages before it in its chat,
/// so a reply like "sim, fechado" carries what it answers.
const WINDOW: i64 = 4;
/// Earlier messages further back than this belong to another conversation.
const WINDOW_GAP_SECS: i64 = 2 * 3600;
/// Characters per window (about 500 tokens); the message itself comes first,
/// the context fills what is left, each earlier message cut at CONTEXT_LINE_CHARS.
const WINDOW_CHARS: usize = 1500;
const CONTEXT_LINE_CHARS: usize = 300;
/// Bumped when what a vector encodes changes, so older vectors are dropped.
/// 1: windows of messages instead of single messages.
const VECTORS_VERSION: i64 = 1;

/// SQL that is true when the message (`p` is "", "new." or "old.") has text
/// that windows are made of: reactions and attachments without a caption are not.
fn embeddable(p: &str) -> String {
    format!("(COALESCE({p}content, '') <> '' AND {p}content NOT LIKE '[reacted %')")
}

/// Whether a window has too little text for a vector (see MIN_SEMANTIC_CHARS).
pub fn too_short(window: &str) -> bool {
    window.lines().all(|l| l.chars().count() < MIN_SEMANTIC_CHARS)
}

/// SQL that is true when `col` holds a transcript or starts with an edit or
/// deletion marker. Parameters ?n, ?n+1 and ?n+2 must bind `MARKERS`.
fn marked(col: &str, n: usize) -> String {
    let (e, d) = (n + 1, n + 2);
    format!(
        "(instr(COALESCE({col}, ''), ?{n}) > 0 OR substr(COALESCE({col}, ''), 1, length(?{e})) = ?{e}
          OR substr(COALESCE({col}, ''), 1, length(?{d})) = ?{d})"
    )
}
const MARKERS: [&str; 3] = [TRANSCRIBED, EDITED, DELETED];

/// User text as a LIKE pattern that matches it anywhere, for `LIKE ? ESCAPE '\'`.
fn like_any(s: &str) -> String {
    let mut out = String::from("%");
    for c in s.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('%');
    out
}

/// The user part of a JID ("5511@s.whatsapp.net" -> "5511"); a bare user as is.
fn user_of(jid: &str) -> &str {
    jid.split('@').next().unwrap_or(jid)
}

pub fn format_ts(ts: DateTime<Utc>) -> String {
    ts.format(TS_FORMAT).to_string()
}

/// Parses what the Go bridge stored; tolerates a fractional second or a `T`.
pub fn parse_ts(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f%:z")
        .or_else(|_| DateTime::parse_from_rfc3339(raw))
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

/// Attachment fields. They travel together because the CDN authenticates
/// direct_path: losing it shows up as a 403 at download time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Media {
    pub kind: String,
    pub filename: String,
    pub url: String,
    pub direct_path: String,
    pub media_key: Vec<u8>,
    pub file_sha256: Vec<u8>,
    pub file_enc_sha256: Vec<u8>,
    pub file_length: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub timestamp: String,
    pub sender: String,
    pub content: String,
    pub is_from_me: bool,
    pub chat_jid: String,
    pub id: String,
    pub chat_name: Option<String>,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Chat {
    pub jid: String,
    pub name: Option<String>,
    pub last_message_time: Option<String>,
    pub last_message: Option<String>,
    pub last_sender: Option<String>,
    pub last_is_from_me: Option<bool>,
    /// Messages from others since you last read or wrote in the chat.
    pub unread_count: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Contact {
    pub phone_number: String,
    pub name: String,
    pub jid: String,
}

#[derive(Debug, Serialize)]
pub struct MessageContext {
    pub message: Message,
    pub before: Vec<Message>,
    pub after: Vec<Message>,
}

#[derive(Default)]
pub struct MessageFilter {
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub sender: Option<String>,
    pub chat_jid: Option<String>,
    pub query: Option<String>,
    pub limit: u32,
    pub page: u32,
}

pub struct Store {
    db: Mutex<Connection>,
    /// What the MCP tools read through. With an allowlist its temp views
    /// shadow chats, messages and contacts, so every query sees only those chats.
    reads: Mutex<Connection>,
    allow: std::sync::OnceLock<Vec<String>>,
}

const MSG_COLS: &str = "messages.timestamp, messages.sender, chats.name, messages.content, \
     messages.is_from_me, messages.chat_jid, messages.id, messages.media_type";
const MSG_FROM: &str = "FROM messages JOIN chats ON messages.chat_jid = chats.jid";

fn row_message(row: &rusqlite::Row) -> rusqlite::Result<Message> {
    let raw: String = row.get(0)?;
    Ok(Message {
        // Re-rendered so every timestamp reads the same, whatever wrote it.
        timestamp: parse_ts(&raw).map(format_ts).unwrap_or(raw),
        sender: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
        chat_name: row.get(2)?,
        content: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        is_from_me: row.get::<_, Option<bool>>(4)?.unwrap_or(false),
        chat_jid: row.get(5)?,
        id: row.get(6)?,
        media_type: row.get::<_, Option<String>>(7)?.filter(|m| !m.is_empty()),
    })
}

fn row_chat(row: &rusqlite::Row) -> rusqlite::Result<Chat> {
    Ok(Chat {
        jid: row.get(0)?,
        name: row.get(1)?,
        last_message_time: row.get::<_, Option<String>>(2)?.map(|raw| parse_ts(&raw).map(|t| t.to_rfc3339()).unwrap_or(raw)),
        last_message: row.get(3)?,
        last_sender: row.get(4)?,
        last_is_from_me: row.get(5)?,
        unread_count: row.get(6)?,
    })
}

/// Unread messages `u` of chat `c`: from others, after the latest of the read
/// the phone reported, your own last message and the cut-off (`*`) set when
/// unread tracking began. Reactions do not count. `= 0`/`= 1` rather than
/// NOT, so messages_chat_me_ts serves both lookups.
macro_rules! unread_where {
    () => {
        "u.chat_jid = c.jid AND u.is_from_me = 0 AND COALESCE(u.content, '') NOT LIKE '[reacted %'
         AND u.timestamp > MAX(COALESCE((SELECT read_at FROM chat_reads r WHERE r.jid = c.jid), ''),
                               COALESCE((SELECT read_at FROM chat_reads r WHERE r.jid = '*'), ''),
                               COALESCE((SELECT MAX(o.timestamp) FROM messages o WHERE o.chat_jid = c.jid AND o.is_from_me = 1), ''))"
    };
}
const UNREAD: &str = concat!("(SELECT COUNT(*) FROM messages u WHERE ", unread_where!(), ")");

/// Newest message actually on file, not the one chats.last_message_time names:
/// that can be a message the bridge dropped as unreadable.
const LAST_MESSAGE_JOIN: &str = "LEFT JOIN messages m ON m.rowid = (
        SELECT x.rowid FROM messages x WHERE x.chat_jid = c.jid
        ORDER BY x.timestamp DESC, x.rowid DESC LIMIT 1)";

fn connect(path: &Path) -> Result<Connection> {
    let db = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    db.busy_timeout(std::time::Duration::from_secs(5))?;
    // NORMAL is durable under WAL short of a power cut, and syncs far less than FULL.
    db.execute_batch("PRAGMA temp_store = MEMORY; PRAGMA synchronous = NORMAL;")?;
    Ok(db)
}

/// Words of a free-text query as an FTS5 expression: each word a quoted prefix,
/// so user input never parses as FTS syntax. None when no word is left.
fn fts_query(q: &str, joiner: &str) -> Option<String> {
    let words: Vec<String> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| format!("\"{w}\"*")).collect();
    (!words.is_empty()).then(|| words.join(joiner))
}

/// Embeddings are stored as little-endian f32, normalized, so a dot product is the cosine.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dot(blob: &[u8], q: &[f32]) -> f32 {
    blob.as_chunks::<4>().0.iter().zip(q).map(|(b, x)| f32::from_le_bytes(*b) * x).sum()
}

/// Up to `k` embeddable messages of `chat` right before (or after) the one at
/// (`ts`, `rowid`), nearest first, as (rowid, timestamp, content).
fn neighbours(db: &Connection, chat: &str, ts: &str, rowid: i64, after: bool, k: i64) -> Result<Vec<(i64, String, String)>> {
    let (op, ord) = if after { (">", "ASC") } else { ("<", "DESC") };
    let mut stmt = db.prepare_cached(&format!(
        "SELECT rowid, timestamp, content FROM messages
         WHERE chat_jid = ?1 AND (timestamp, rowid) {op} (?2, ?3) AND {}
         ORDER BY timestamp {ord}, rowid {ord} LIMIT ?4",
        embeddable("")
    ))?;
    let rows = stmt.query_map(params![chat, ts, rowid, k], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// Whether two stored timestamps are within one window's gap.
fn close_in_time(a: &str, b: &str) -> bool {
    match (parse_ts(a), parse_ts(b)) {
        (Some(a), Some(b)) => (a - b).num_seconds().abs() <= WINDOW_GAP_SECS,
        _ => false,
    }
}

/// The text a message's vector encodes: the earlier messages of the same
/// conversation, oldest first, then the message itself on the last line.
/// None when the message is gone or has no text to embed.
fn window_of(db: &Connection, rowid: i64) -> Result<Option<String>> {
    let target: Option<(String, String, String)> = db
        .query_row(&format!("SELECT chat_jid, timestamp, content FROM messages WHERE rowid = ?1 AND {}", embeddable("")), [rowid], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .optional()?;
    let Some((chat, ts, content)) = target else { return Ok(None) };
    let mut budget = WINDOW_CHARS.saturating_sub(content.chars().count());
    let mut lines = Vec::new();
    for (_, at, text) in neighbours(db, &chat, &ts, rowid, false, WINDOW)? {
        let line: String = text.chars().take(CONTEXT_LINE_CHARS).collect();
        let cost = line.chars().count() + 1;
        if !close_in_time(&ts, &at) || cost > budget {
            break;
        }
        budget -= cost;
        lines.push(line);
    }
    lines.reverse();
    lines.push(content);
    Ok(Some(lines.join("\n")))
}

/// The first `n` of `ranked` (best first) that are not within a window of a
/// better one: consecutive messages of one conversation share most of their
/// windows, and would otherwise fill the results with the same stretch of chat.
fn distinct_conversations(db: &Connection, ranked: impl Iterator<Item = i64>, n: usize) -> Result<Vec<i64>> {
    let mut covered = std::collections::HashSet::new();
    let mut kept = Vec::new();
    for rowid in ranked {
        if kept.len() == n {
            break;
        }
        if covered.contains(&rowid) {
            continue;
        }
        let at: Option<(String, String)> = db
            .query_row("SELECT chat_jid, timestamp FROM messages WHERE rowid = ?1", [rowid], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        let Some((chat, ts)) = at else { continue };
        for after in [false, true] {
            for (id, t, _) in neighbours(db, &chat, &ts, rowid, after, WINDOW)? {
                if !close_in_time(&ts, &t) {
                    break;
                }
                covered.insert(id);
            }
        }
        kept.push(rowid);
    }
    Ok(kept)
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let mut db = connect(path)?;
        db.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS chats (
                 jid TEXT PRIMARY KEY,
                 name TEXT,
                 last_message_time TIMESTAMP
             );
             CREATE TABLE IF NOT EXISTS messages (
                 id TEXT,
                 chat_jid TEXT,
                 sender TEXT,
                 content TEXT,
                 timestamp TIMESTAMP,
                 is_from_me BOOLEAN,
                 media_type TEXT,
                 filename TEXT,
                 url TEXT,
                 media_key BLOB,
                 file_sha256 BLOB,
                 file_enc_sha256 BLOB,
                 file_length INTEGER,
                 direct_path TEXT,
                 PRIMARY KEY (id, chat_jid),
                 FOREIGN KEY (chat_jid) REFERENCES chats(jid)
             );
             CREATE INDEX IF NOT EXISTS messages_chat_ts ON messages (chat_jid, timestamp);
             CREATE INDEX IF NOT EXISTS messages_ts ON messages (timestamp);
             CREATE TABLE IF NOT EXISTS contacts (
                 jid TEXT PRIMARY KEY,
                 full_name TEXT NOT NULL DEFAULT '',
                 first_name TEXT NOT NULL DEFAULT '',
                 business_name TEXT NOT NULL DEFAULT '',
                 push_name TEXT NOT NULL DEFAULT ''
             );
             CREATE TABLE IF NOT EXISTS lid_map (
                 lid TEXT PRIMARY KEY,
                 pn TEXT NOT NULL
             );",
        )?;
        // Databases from before direct_path existed lack the column.
        let has_direct_path: bool = db.prepare("SELECT 1 FROM pragma_table_info('messages') WHERE name = 'direct_path'")?.exists([])?;
        if !has_direct_path {
            db.execute("ALTER TABLE messages ADD COLUMN direct_path TEXT", [])?;
        }
        // Unread counts (and your last message per chat), per-sender lookups, LID by phone.
        db.execute_batch(
            "CREATE INDEX IF NOT EXISTS messages_chat_me_ts ON messages (chat_jid, is_from_me, timestamp);
             CREATE INDEX IF NOT EXISTS messages_sender_ts ON messages (sender, timestamp);
             CREATE INDEX IF NOT EXISTS lid_map_pn ON lid_map (pn);",
        )?;
        // Full-text index over the content, kept in step by triggers; vectors
        // for semantic search, dropped whenever the text they encode changes.
        // One transaction, so a crash never leaves the index without its triggers.
        let tx = db.transaction()?;
        let has_fts: bool = tx.prepare("SELECT 1 FROM sqlite_master WHERE name = 'messages_fts'")?.exists([])?;
        tx.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                 content, content = 'messages', content_rowid = 'rowid', tokenize = 'unicode61 remove_diacritics 2');
             CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
                 INSERT INTO messages_fts (rowid, content) VALUES (new.rowid, new.content);
             END;
             CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
                 INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                 DELETE FROM message_vec WHERE rowid = old.rowid;
             END;
             -- Recreated every time: older databases have it without the WHEN,
             -- so every redelivered message re-indexed and lost its vector.
             DROP TRIGGER IF EXISTS messages_fts_au;
             CREATE TRIGGER messages_fts_au AFTER UPDATE OF content ON messages
             WHEN old.content IS NOT new.content BEGIN
                 INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                 INSERT INTO messages_fts (rowid, content) VALUES (new.rowid, new.content);
                 DELETE FROM message_vec WHERE rowid = old.rowid;
             END;
             CREATE TABLE IF NOT EXISTS message_vec (
                 rowid INTEGER PRIMARY KEY,
                 embedding BLOB NOT NULL
             );",
        )?;
        // A vector encodes its message's window (window_of), so a message that
        // arrives, changes or leaves also drops the vectors of the WINDOW
        // messages after it, which get re-embedded with the new context.
        let next = |p: &str| {
            format!(
                "DELETE FROM message_vec WHERE rowid IN (SELECT rowid FROM messages
                     WHERE chat_jid = {p}chat_jid AND (timestamp, rowid) > ({p}timestamp, {p}rowid)
                       AND rowid <> {p}rowid AND {e} ORDER BY timestamp, rowid LIMIT {WINDOW});",
                e = embeddable("")
            )
        };
        tx.execute_batch(&format!(
            "DROP TRIGGER IF EXISTS message_vec_ai;
                 CREATE TRIGGER message_vec_ai AFTER INSERT ON messages WHEN {new_e} BEGIN {next_new} END;
                 DROP TRIGGER IF EXISTS message_vec_ad;
                 CREATE TRIGGER message_vec_ad AFTER DELETE ON messages WHEN {old_e} BEGIN {next_old} END;
                 DROP TRIGGER IF EXISTS message_vec_au;
                 CREATE TRIGGER message_vec_au AFTER UPDATE OF content, timestamp, chat_jid ON messages
                 WHEN old.content IS NOT new.content OR old.timestamp IS NOT new.timestamp OR old.chat_jid IS NOT new.chat_jid
                 BEGIN
                     DELETE FROM message_vec WHERE rowid = new.rowid;
                     {next_old}
                     {next_new}
                 END;",
            new_e = embeddable("new."),
            old_e = embeddable("old."),
            next_new = next("new."),
            next_old = next("old."),
        ))?;
        let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < VECTORS_VERSION {
            tx.execute("DELETE FROM message_vec", [])?;
            tx.execute_batch(&format!("PRAGMA user_version = {VECTORS_VERSION};"))?;
        }
        if !has_fts {
            tx.execute("INSERT INTO messages_fts (messages_fts) VALUES ('rebuild')", [])?;
        }
        tx.commit()?;
        // When you last read each chat. Row '*' is when tracking began: older
        // messages, history synced later included, count as read, or every old
        // group would show its whole history as unread.
        db.execute_batch("CREATE TABLE IF NOT EXISTS chat_reads (jid TEXT PRIMARY KEY, read_at TIMESTAMP NOT NULL);")?;
        db.execute("INSERT OR IGNORE INTO chat_reads VALUES ('*', ?1)", [format_ts(Utc::now())])?;
        let reads = connect(path)?;
        reads.execute_batch("PRAGMA query_only = ON;")?;
        Ok(Self { db: Mutex::new(db), reads: Mutex::new(reads), allow: Default::default() })
    }

    fn reads(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.reads.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Limits every MCP read to these chats (full JIDs). Phone-number chats
    /// also cover their LID twin, and contacts are only the people in them.
    pub fn restrict(&self, chats: &[String]) -> Result<()> {
        let mut jids: Vec<String> = chats.to_vec();
        for jid in chats {
            if let Some(user) = jid.strip_suffix("@s.whatsapp.net")
                && let Some(lid) = self.lid_for_pn(user)
            {
                jids.push(format!("{lid}@lid"));
            }
        }
        let _ = self.allow.set(jids.clone());
        let list = jids.iter().map(|j| format!("'{}'", j.replace('\'', "''"))).collect::<Vec<_>>().join(", ");
        let db = self.reads();
        db.execute_batch("PRAGMA query_only = OFF;")?;
        db.execute_batch(&format!(
            "CREATE TEMP VIEW chats AS SELECT * FROM main.chats WHERE jid IN ({list});
             CREATE TEMP VIEW messages AS SELECT rowid AS rowid, * FROM main.messages WHERE chat_jid IN ({list});
             CREATE TEMP VIEW contacts AS SELECT * FROM main.contacts
                 WHERE jid IN ({list}) OR jid IN (
                     SELECT DISTINCT CASE WHEN instr(sender, '@') > 0 THEN sender ELSE sender || '@s.whatsapp.net' END
                     FROM main.messages WHERE chat_jid IN ({list}));
             PRAGMA query_only = ON;"
        ))?;
        Ok(())
    }

    /// Whether the MCP tools may touch this chat (a full JID).
    pub fn allowed(&self, jid: &str) -> bool {
        self.allow.get().is_none_or(|list| list.iter().any(|j| j == jid))
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A panic while holding the lock leaves SQLite itself consistent.
        self.db.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn has_contacts(&self) -> bool {
        self.conn().query_row("SELECT EXISTS(SELECT 1 FROM contacts)", [], |r| r.get(0)).unwrap_or(false)
    }

    /// One-time copy of whatsmeow's contacts and LID map, so names survive the
    /// switch away from the Go bridge. Missing tables are not an error.
    pub fn import_whatsmeow(&self, whatsmeow_db: &Path) -> Result<usize> {
        if !whatsmeow_db.exists() {
            return Ok(0);
        }
        let mut db = self.conn();
        db.execute("ATTACH DATABASE ?1 AS wa", [whatsmeow_db.to_string_lossy()])?;
        let result = (|| -> Result<usize> {
            let tx = db.transaction()?;
            let has = |table: &str| -> rusqlite::Result<bool> {
                tx.prepare("SELECT 1 FROM wa.sqlite_master WHERE type = 'table' AND name = ?1")?.exists([table])
            };
            let mut n = 0;
            if has("whatsmeow_contacts")? {
                n += tx.execute(
                    "INSERT INTO contacts (jid, full_name, first_name, business_name, push_name)
                     SELECT their_jid, COALESCE(full_name, ''), COALESCE(first_name, ''),
                            COALESCE(business_name, ''), COALESCE(push_name, '')
                     FROM wa.whatsmeow_contacts WHERE true
                     ON CONFLICT(jid) DO NOTHING",
                    [],
                )?;
            }
            if has("whatsmeow_lid_map")? {
                n += tx.execute("INSERT OR IGNORE INTO lid_map (lid, pn) SELECT lid, pn FROM wa.whatsmeow_lid_map", [])?;
            }
            tx.commit()?;
            Ok(n)
        })();
        db.execute("DETACH DATABASE wa", [])?;
        result
    }

    // ---- writes -------------------------------------------------------------

    pub fn store_chat(&self, jid: &str, name: &str, last: DateTime<Utc>) -> Result<()> {
        // Never move last_message_time backwards: history sync replays old messages.
        self.conn().execute(
            "INSERT INTO chats (jid, name, last_message_time) VALUES (?1, ?2, ?3)
             ON CONFLICT(jid) DO UPDATE SET
               name = CASE WHEN excluded.name != '' THEN excluded.name ELSE chats.name END,
               last_message_time = MAX(COALESCE(chats.last_message_time, ''), excluded.last_message_time)",
            params![jid, name, format_ts(last)],
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_message(
        &self,
        id: &str,
        chat_jid: &str,
        sender: &str,
        content: &str,
        ts: DateTime<Utc>,
        is_from_me: bool,
        media: &Media,
    ) -> Result<bool> {
        if content.is_empty() && media.kind.is_empty() {
            return Ok(false);
        }
        let [tr, ed, de] = MARKERS;
        self.conn().execute(
            // An upsert, not REPLACE: the row keeps its rowid, which the
            // full-text index and the vectors are keyed by. A transcript, an
            // edit or a deletion is never undone by the original arriving again,
            // an empty text never blanks a stored one, and attachment fields
            // already on file stay (a redelivery may come without them).
            &format!(
                "INSERT INTO messages
                 (id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, url,
                  media_key, file_sha256, file_enc_sha256, file_length, direct_path)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                 ON CONFLICT(id, chat_jid) DO UPDATE SET
                   sender = excluded.sender,
                   content = CASE WHEN excluded.content = '' OR {} THEN messages.content ELSE excluded.content END,
                   timestamp = excluded.timestamp,
                   is_from_me = excluded.is_from_me,
                   media_type = CASE WHEN {fresh} THEN excluded.media_type ELSE messages.media_type END,
                   filename = CASE WHEN {fresh} THEN excluded.filename ELSE messages.filename END,
                   url = CASE WHEN {fresh} THEN excluded.url ELSE messages.url END,
                   media_key = CASE WHEN {fresh} THEN excluded.media_key ELSE messages.media_key END,
                   file_sha256 = CASE WHEN {fresh} THEN excluded.file_sha256 ELSE messages.file_sha256 END,
                   file_enc_sha256 = CASE WHEN {fresh} THEN excluded.file_enc_sha256 ELSE messages.file_enc_sha256 END,
                   file_length = CASE WHEN {fresh} THEN excluded.file_length ELSE messages.file_length END,
                   direct_path = CASE WHEN {fresh} THEN excluded.direct_path ELSE messages.direct_path END",
                marked("messages.content", 15),
                fresh = FRESH_MEDIA
            ),
            params![
                id,
                chat_jid,
                sender,
                content,
                format_ts(ts),
                is_from_me,
                media.kind,
                media.filename,
                media.url,
                media.media_key,
                media.file_sha256,
                media.file_enc_sha256,
                media.file_length as i64,
                media.direct_path,
                tr,
                ed,
                de
            ],
        )?;
        Ok(true)
    }

    pub fn upsert_contact(&self, jid: &str, field: ContactField, value: &str) -> Result<()> {
        if value.is_empty() {
            return Ok(());
        }
        let col = field.column();
        self.conn().execute(
            &format!(
                "INSERT INTO contacts (jid, {col}) VALUES (?1, ?2)
                 ON CONFLICT(jid) DO UPDATE SET {col} = excluded.{col}"
            ),
            params![jid, value],
        )?;
        Ok(())
    }

    pub fn put_lid(&self, lid_user: &str, pn_user: &str) -> Result<()> {
        if lid_user.is_empty() || pn_user.is_empty() {
            return Ok(());
        }
        self.conn().execute(
            "INSERT INTO lid_map (lid, pn) VALUES (?1, ?2) ON CONFLICT(lid) DO UPDATE SET pn = excluded.pn",
            params![lid_user, pn_user],
        )?;
        Ok(())
    }

    // ---- name / LID resolution ---------------------------------------------

    pub fn pn_for_lid(&self, lid_user: &str) -> Option<String> {
        self.conn().query_row("SELECT pn FROM lid_map WHERE lid = ?1", [lid_user], |r| r.get(0)).optional().ok().flatten()
    }

    pub fn lid_for_pn(&self, pn_user: &str) -> Option<String> {
        self.conn().query_row("SELECT lid FROM lid_map WHERE pn = ?1 LIMIT 1", [pn_user], |r| r.get(0)).optional().ok().flatten()
    }

    /// Address book name first, then business, first name, push name.
    pub fn contact_name(&self, jid: &str) -> Option<String> {
        self.conn()
            .query_row(&format!("SELECT {BEST_NAME} FROM contacts WHERE jid = ?1"), [jid], |r| r.get::<_, Option<String>>(0))
            .optional()
            .ok()
            .flatten()
            .flatten()
    }

    /// The stored chat name, unless it is the number placeholder.
    pub fn chat_name(&self, jid: &str) -> Option<String> {
        self.conn()
            .query_row("SELECT name FROM chats WHERE jid = ?1", [jid], |r| r.get::<_, Option<String>>(0))
            .optional()
            .ok()
            .flatten()
            .flatten()
            .filter(|n| !is_placeholder(n))
    }

    /// Replaces a placeholder name (empty or the number) on a chat row.
    pub fn fill_chat_name(&self, jid: &str, name: &str) -> Result<bool> {
        let n = self.conn().execute(&format!("UPDATE chats SET name = ?1 WHERE jid = ?2 AND {PLACEHOLDER}"), params![name, jid])?;
        Ok(n > 0)
    }

    /// Individual chats whose name is still a placeholder.
    pub fn unnamed_direct_chats(&self) -> Result<Vec<String>> {
        let db = self.conn();
        let mut stmt = db.prepare(&format!("SELECT jid FROM chats WHERE jid NOT LIKE '%@g.us' AND {PLACEHOLDER}"))?;
        let rows = stmt.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// Folds chats stored under a LID into the phone-number row, as the Go
    /// bridge did, for conversations split before addresses were normalized.
    pub fn merge_lid_chats(&self) -> Result<(usize, usize)> {
        let (pending, total) = {
            let db = self.conn();
            let pending: Vec<(String, Option<String>, Option<String>, String)> = db
                .prepare(
                    "SELECT c.jid, c.name, c.last_message_time, l.pn FROM chats c
                     JOIN lid_map l ON l.lid = substr(c.jid, 1, instr(c.jid, '@') - 1)
                     WHERE c.jid LIKE '%@lid'",
                )?
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                .collect::<rusqlite::Result<_>>()?;
            let total = db.query_row("SELECT COUNT(*) FROM chats WHERE jid LIKE '%@lid'", [], |r| r.get::<_, i64>(0))?;
            (pending, total as usize)
        };
        let [tr, ed, de] = MARKERS;
        let mut merged = 0;
        for (lid_jid, name, last, pn) in pending {
            let lid_user = user_of(&lid_jid).to_string();
            let target = format!("{pn}@s.whatsapp.net");
            // One chat per lock: incoming messages wait for a chat, not the whole merge.
            let mut db = self.conn();
            let tx = db.transaction()?;
            // An empty name never replaces the target's, so a placeholder can't either.
            let name = name.filter(|n| !is_placeholder(n)).unwrap_or_default();
            tx.execute(
                "INSERT INTO chats (jid, name, last_message_time) VALUES (?1, ?2, ?3)
                 ON CONFLICT(jid) DO UPDATE SET
                   name = CASE WHEN excluded.name != '' THEN excluded.name ELSE chats.name END,
                   last_message_time = NULLIF(MAX(COALESCE(chats.last_message_time, ''),
                                                  COALESCE(excluded.last_message_time, '')), '')",
                params![target, name, last],
            )?;
            tx.execute(
                "UPDATE messages SET sender = ?1 WHERE chat_jid = ?2 AND sender IN (?3, ?3 || '@lid')",
                params![pn, lid_jid, lid_user],
            )?;
            // A message in both chats keeps the phone-number copy; first it takes
            // the LID copy's transcript, edit or deletion if only that one has it.
            tx.execute(
                &format!(
                    "UPDATE messages SET content = (SELECT l.content FROM messages l WHERE l.chat_jid = ?2 AND l.id = messages.id)
                     WHERE chat_jid = ?1 AND NOT {}
                       AND EXISTS (SELECT 1 FROM messages l WHERE l.chat_jid = ?2 AND l.id = messages.id AND {})",
                    marked("messages.content", 3),
                    marked("l.content", 3)
                ),
                params![target, lid_jid, tr, ed, de],
            )?;
            tx.execute(
                "DELETE FROM messages WHERE chat_jid = ?1 AND id IN (SELECT id FROM messages WHERE chat_jid = ?2)",
                params![lid_jid, target],
            )?;
            tx.execute("UPDATE messages SET chat_jid = ?1 WHERE chat_jid = ?2", params![target, lid_jid])?;
            tx.execute(
                "INSERT INTO chat_reads (jid, read_at) SELECT ?1, read_at FROM chat_reads WHERE jid = ?2
                 ON CONFLICT(jid) DO UPDATE SET read_at = MAX(read_at, excluded.read_at)",
                params![target, lid_jid],
            )?;
            tx.execute("DELETE FROM chat_reads WHERE jid = ?1", [&lid_jid])?;
            tx.execute("DELETE FROM chats WHERE jid = ?1", [&lid_jid])?;
            tx.commit()?;
            merged += 1;
        }
        Ok((merged, total))
    }

    // ---- media --------------------------------------------------------------

    /// A message's attachment, plus who sent it (is_from_me, sender user):
    /// asking the phone to re-upload expired media needs both.
    pub fn media(&self, id: &str, chat_jid: &str) -> Result<Option<(Media, bool, String)>> {
        Ok(self
            .conn()
            .query_row(
                "SELECT media_type, filename, url, direct_path, media_key, file_sha256, file_enc_sha256, file_length,
                        is_from_me, sender
                 FROM messages WHERE id = ?1 AND chat_jid = ?2",
                [id, chat_jid],
                |r| {
                    let media = Media {
                        kind: r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                        filename: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        url: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        direct_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        media_key: r.get::<_, Option<Vec<u8>>>(4)?.unwrap_or_default(),
                        file_sha256: r.get::<_, Option<Vec<u8>>>(5)?.unwrap_or_default(),
                        file_enc_sha256: r.get::<_, Option<Vec<u8>>>(6)?.unwrap_or_default(),
                        file_length: r.get::<_, Option<i64>>(7)?.unwrap_or(0) as u64,
                    };
                    Ok((media, r.get::<_, Option<bool>>(8)?.unwrap_or(false), r.get::<_, Option<String>>(9)?.unwrap_or_default()))
                },
            )
            .optional()?)
    }

    /// An edit replaces the text, a deletion keeps it under a marker: the record
    /// of what was said outlives the "deleted for everyone".
    /// False when the original is not on file.
    pub fn edit_message(&self, id: &str, chat_jid: &str, text: &str) -> Result<bool> {
        let n = self.conn().execute(
            "UPDATE messages SET content = ?3 WHERE id = ?1 AND chat_jid = ?2",
            params![id, chat_jid, format!("{EDITED}{text}")],
        )?;
        Ok(n > 0)
    }

    pub fn revoke_message(&self, id: &str, chat_jid: &str) -> Result<bool> {
        let n = self.conn().execute(
            "UPDATE messages SET content = ?3 || COALESCE(content, '')
             WHERE id = ?1 AND chat_jid = ?2 AND COALESCE(content, '') NOT LIKE ?3 || '%'",
            params![id, chat_jid, DELETED],
        )?;
        Ok(n > 0)
    }

    /// The voice note's transcript goes in the text of the message, so the
    /// searches find it; a caption, if any, stays in front.
    pub fn set_transcription(&self, id: &str, chat_jid: &str, text: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE messages SET content = ltrim(COALESCE(content, '') || char(10) || ?3, char(10))
             WHERE id = ?1 AND chat_jid = ?2 AND instr(COALESCE(content, ''), ?4) = 0",
            params![id, chat_jid, format!("{TRANSCRIBED}{text}"), TRANSCRIBED],
        )?;
        Ok(())
    }

    /// (id, chat) of voice notes since `since` with no transcript yet, newest first.
    pub fn untranscribed(&self, since: DateTime<Utc>, n: u32) -> Result<Vec<(String, String)>> {
        let db = self.conn();
        let mut stmt = db.prepare(
            "SELECT id, chat_jid FROM messages WHERE media_type = 'audio' AND timestamp > ?1
               AND instr(COALESCE(content, ''), ?2) = 0 ORDER BY timestamp DESC, rowid DESC LIMIT ?3",
        )?;
        let rows =
            stmt.query_map(params![format_ts(since), TRANSCRIBED, n], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn transcription(&self, id: &str, chat_jid: &str) -> Option<String> {
        let content = self.message_content(id, chat_jid)?;
        content.find(TRANSCRIBED).map(|i| content[i + TRANSCRIBED.len()..].to_string())
    }

    /// Moves the read mark of a chat forward to `at`; never back.
    pub fn mark_read(&self, jid: &str, at: DateTime<Utc>) -> Result<()> {
        self.conn().execute(
            "INSERT INTO chat_reads (jid, read_at) VALUES (?1, ?2)
             ON CONFLICT(jid) DO UPDATE SET read_at = MAX(read_at, excluded.read_at)",
            params![jid, format_ts(at)],
        )?;
        Ok(())
    }

    /// (id, sender) of the unread messages of a chat, oldest first, for read receipts.
    pub fn unread_messages(&self, jid: &str) -> Result<Vec<(String, String)>> {
        let db = self.reads();
        let mut stmt = db.prepare(&format!(
            "SELECT u.id, u.sender FROM messages u, (SELECT ?1 AS jid) c WHERE {} ORDER BY u.timestamp, u.rowid",
            unread_where!()
        ))?;
        let rows = stmt
            .query_map([jid], |r| Ok((r.get(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default())))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// Chats with messages in the window, busiest first: (chat, messages in the window).
    pub fn active_chats(&self, after: DateTime<Utc>, before: Option<DateTime<Utc>>, limit: u32) -> Result<Vec<(Chat, i64)>> {
        let db = self.reads();
        let mut stmt = db.prepare(&format!(
            "SELECT c.jid, c.name, c.last_message_time, NULL, NULL, NULL, {UNREAD}, n FROM chats c
             JOIN (SELECT chat_jid, COUNT(*) AS n FROM messages WHERE timestamp > ?1 AND timestamp < ?2 AND chat_jid <> 'status@broadcast' GROUP BY chat_jid) a
               ON a.chat_jid = c.jid
             ORDER BY n DESC LIMIT ?3"
        ))?;
        // Not a bare number: the TIMESTAMP column would compare it as one.
        let before = before.map(format_ts).unwrap_or_else(|| "9999-12-31".into());
        let rows = stmt
            .query_map(params![format_ts(after), before, limit], |r| Ok((row_chat(r)?, r.get(7)?)))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// (is_from_me, sender, content) of a message, to quote it in a reply.
    pub fn quote(&self, id: &str, chat_jid: &str) -> Result<Option<(bool, String, String)>> {
        Ok(self
            .conn()
            .query_row("SELECT is_from_me, sender, content FROM messages WHERE id = ?1 AND chat_jid = ?2", [id, chat_jid], |r| {
                Ok((
                    r.get::<_, Option<bool>>(0)?.unwrap_or(false),
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                ))
            })
            .optional()?)
    }

    pub fn message_content(&self, id: &str, chat_jid: &str) -> Option<String> {
        self.conn()
            .query_row("SELECT content FROM messages WHERE id = ?1 AND chat_jid = ?2", [id, chat_jid], |r| r.get(0))
            .optional()
            .ok()
            .flatten()
    }

    /// The new CDN path after the phone re-uploaded expired media.
    pub fn set_direct_path(&self, id: &str, chat_jid: &str, direct_path: &str) -> Result<()> {
        self.conn().execute("UPDATE messages SET direct_path = ?3 WHERE id = ?1 AND chat_jid = ?2", params![id, chat_jid, direct_path])?;
        Ok(())
    }

    // ---- semantic search -----------------------------------------------------

    /// Newest messages with text and no vector yet, each with its window
    /// (window_of). Read through the MCP connection: with an allowlist only the
    /// chats search can return get embedded. Windows `too_short` for a vector
    /// still come back, for the caller to mark with an empty one.
    pub fn unembedded(&self, n: usize) -> Result<Vec<(i64, String)>> {
        let db = self.reads();
        let mut stmt = db.prepare(&format!(
            "SELECT m.rowid FROM messages m LEFT JOIN message_vec v ON v.rowid = m.rowid
             WHERE v.rowid IS NULL AND {} ORDER BY m.timestamp DESC, m.rowid DESC LIMIT ?1",
            embeddable("m.")
        ))?;
        let ids: Vec<i64> = stmt.query_map([n as i64], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(w) = window_of(&db, id)? {
                out.push((id, w));
            }
        }
        Ok(out)
    }

    /// Stores (rowid, window embedded, vector) rows whose window is still that
    /// text: a message edited or arriving in it while the batch was out would
    /// otherwise leave a stale vector. Returns how many were saved.
    pub fn put_vectors(&self, rows: &[(i64, &str, Vec<f32>)]) -> Result<usize> {
        let mut db = self.conn();
        let tx = db.transaction()?;
        let mut saved = 0;
        for (rowid, text, v) in rows {
            if window_of(&tx, *rowid)?.as_deref() == Some(*text) {
                saved +=
                    tx.execute("INSERT OR REPLACE INTO message_vec (rowid, embedding) VALUES (?1, ?2)", params![rowid, vec_to_blob(v)])?;
            }
        }
        tx.commit()?;
        Ok(saved)
    }

    /// Hybrid search: full-text (BM25) and, given a query vector, cosine over the
    /// stored ones, merged by reciprocal rank. Best match first; ties go to the
    /// newer message, so the order (and paging) is stable.
    pub fn search(&self, query: &str, qvec: Option<&[f32]>, f: &MessageFilter) -> Result<Vec<Message>> {
        const POOL: usize = 50;
        let (filter, args) = filter_sql(f);
        let db = self.reads();
        let mut scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
        let mut add = |ranked: Vec<i64>| {
            for (rank, rowid) in ranked.into_iter().enumerate() {
                *scores.entry(rowid).or_default() += 1.0 / (60.0 + rank as f64);
            }
        };
        if let Some(fts) = fts_query(query, " OR ") {
            let mut stmt = db.prepare(&format!(
                "SELECT messages.rowid FROM messages_fts JOIN messages ON messages.rowid = messages_fts.rowid
                 WHERE messages_fts MATCH ? AND {filter} ORDER BY bm25(messages_fts), messages.rowid DESC LIMIT {POOL}"
            ))?;
            let p = std::iter::once(fts).chain(args.iter().cloned());
            add(stmt.query_map(rusqlite::params_from_iter(p), |r| r.get(0))?.collect::<rusqlite::Result<_>>()?);
        }
        if let Some(q) = qvec {
            // Vectors of another size (another model, or the empty "unembeddable"
            // mark) can't compare.
            let mut stmt = db.prepare(&format!(
                "SELECT message_vec.rowid, message_vec.embedding FROM message_vec
                 JOIN messages ON messages.rowid = message_vec.rowid
                 WHERE length(message_vec.embedding) = {} AND {filter}",
                4 * q.len()
            ))?;
            let mut best: Vec<(f32, i64)> = stmt
                .query_map(rusqlite::params_from_iter(args.iter()), |r| Ok((dot(&r.get::<_, Vec<u8>>(1)?, q), r.get(0)?)))?
                .collect::<rusqlite::Result<_>>()?;
            let by_score = |a: &(f32, i64), b: &(f32, i64)| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1));
            // Neighbouring windows share most of their text and score alike;
            // room for the ones dropped below.
            let room = POOL * 4;
            if best.len() > room {
                best.select_nth_unstable_by(room, by_score);
                best.truncate(room);
            }
            best.sort_unstable_by(by_score);
            add(distinct_conversations(&db, best.into_iter().map(|(_, id)| id), POOL)?);
        }
        let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
        let mut stmt = db.prepare(&format!("SELECT {MSG_COLS} {MSG_FROM} WHERE messages.rowid = ?1"))?;
        let mut out = Vec::new();
        for (rowid, _) in ranked.into_iter().skip(f.page as usize * f.limit as usize).take(f.limit as usize) {
            if let Some(m) = stmt.query_row([rowid], row_message).optional()? {
                out.push(m);
            }
        }
        Ok(out)
    }

    // ---- reads for the MCP tools -------------------------------------------

    pub fn list_messages(&self, f: &MessageFilter) -> Result<Vec<Message>> {
        let (mut filter, mut args) = filter_sql(f);
        if let Some(q) = &f.query {
            match fts_query(q, " ") {
                Some(fts) => {
                    filter.push_str(" AND messages.rowid IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?)");
                    args.push(fts);
                }
                None => {
                    filter.push_str(" AND messages.content LIKE ? ESCAPE '\\'");
                    args.push(like_any(q));
                }
            }
        }
        let sql = format!(
            "SELECT {MSG_COLS} {MSG_FROM} WHERE {filter}
             ORDER BY messages.timestamp DESC, messages.rowid DESC LIMIT {} OFFSET {}",
            f.limit,
            f.page as u64 * f.limit as u64
        );
        let db = self.reads();
        let mut stmt = db.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), row_message)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// A message with its neighbours in the chat, all in chronological order.
    /// Without `chat_jid` the ID alone picks it (the newest, should two chats share it).
    pub fn message_context(&self, id: &str, chat_jid: Option<&str>, before: u32, after: u32) -> Result<Option<MessageContext>> {
        let db = self.reads();
        let Some((target, raw_ts, rowid)) = db
            .query_row(
                &format!(
                    "SELECT {MSG_COLS}, messages.rowid {MSG_FROM} WHERE messages.id = ?1 AND (?2 IS NULL OR messages.chat_jid = ?2)
                     ORDER BY messages.timestamp DESC, messages.rowid DESC LIMIT 1"
                ),
                params![id, chat_jid],
                |r| Ok((row_message(r)?, r.get::<_, String>(0)?, r.get::<_, i64>(8)?)),
            )
            .optional()?
        else {
            return Ok(None);
        };
        // (timestamp, rowid), so messages in the same second are neither lost nor repeated.
        let around = |op: &str, order: &str, n: u32| -> Result<Vec<Message>> {
            let mut stmt = db.prepare(&format!(
                "SELECT {MSG_COLS} {MSG_FROM}
                 WHERE messages.chat_jid = ?1 AND (messages.timestamp, messages.rowid) {op} (?2, ?3)
                 ORDER BY messages.timestamp {order}, messages.rowid {order} LIMIT ?4"
            ))?;
            let rows = stmt.query_map(params![target.chat_jid, raw_ts, rowid, n], row_message)?.collect::<rusqlite::Result<_>>()?;
            Ok(rows)
        };
        let mut before = around("<", "DESC", before)?;
        before.reverse();
        let after = around(">", "ASC", after)?;
        Ok(Some(MessageContext { message: target, before, after }))
    }

    pub fn list_chats(
        &self,
        query: Option<&str>,
        limit: u32,
        page: u32,
        last: bool,
        by_name: bool,
        unread_only: bool,
    ) -> Result<Vec<Chat>> {
        let (cols, join) = if last { ("m.content, m.sender, m.is_from_me", LAST_MESSAGE_JOIN) } else { ("NULL, NULL, NULL", "") };
        let order = if by_name { "c.name, c.jid" } else { "c.last_message_time DESC, c.jid" };
        let mut filter = String::from("1");
        let mut args: Vec<String> = Vec::new();
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            filter.push_str(" AND (c.name LIKE ?1 ESCAPE '\\' OR c.jid LIKE ?1 ESCAPE '\\')");
            args.push(like_any(q));
        }
        if unread_only {
            filter.push_str(&format!(" AND {UNREAD} > 0"));
        }
        // The page is picked first: the unread count and the last message are
        // worked out only for the chats shown, not for every chat on file.
        let sql = format!(
            "SELECT c.jid, c.name, c.last_message_time, {cols}, {UNREAD} FROM
               (SELECT * FROM chats c WHERE {filter} ORDER BY {order} LIMIT {limit} OFFSET {}) c
             {join} ORDER BY {order}",
            page as u64 * limit as u64
        );
        let db = self.reads();
        let mut stmt = db.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), row_chat)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn get_chat(&self, jid: &str, last: bool) -> Result<Option<Chat>> {
        let sql = if last {
            format!(
                "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me, {UNREAD} FROM chats c {LAST_MESSAGE_JOIN} WHERE c.jid = ?1"
            )
        } else {
            format!("SELECT c.jid, c.name, c.last_message_time, NULL, NULL, NULL, {UNREAD} FROM chats c WHERE c.jid = ?1")
        };
        Ok(self.reads().query_row(&sql, [jid], row_chat).optional()?)
    }

    /// The direct chat of this number: the exact JID if on file, else the most
    /// recently active individual chat whose JID contains it.
    pub fn direct_chat_by_phone(&self, phone: &str) -> Result<Option<Chat>> {
        Ok(self
            .reads()
            .query_row(
                &format!(
                    "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me, {UNREAD} FROM chats c
                     {LAST_MESSAGE_JOIN}
                     WHERE c.jid = ?1 || '@s.whatsapp.net' OR (c.jid LIKE ?2 ESCAPE '\\' AND c.jid NOT LIKE '%@g.us')
                     ORDER BY c.jid = ?1 || '@s.whatsapp.net' DESC, c.last_message_time DESC LIMIT 1"
                ),
                [phone.to_string(), like_any(phone)],
                row_chat,
            )
            .optional()?)
    }

    /// The contact's direct chat plus every chat they wrote in. Messages store
    /// the sender's user part (or, unmapped, a full LID JID), so both are tried.
    pub fn contact_chats(&self, jid: &str, limit: u32, page: u32) -> Result<Vec<Chat>> {
        let db = self.reads();
        let mut stmt = db.prepare(&format!(
            "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me, {UNREAD} FROM chats c
             {LAST_MESSAGE_JOIN}
             WHERE c.jid = ?1 OR c.jid IN (SELECT s.chat_jid FROM messages s WHERE s.sender IN (?1, ?2))
             ORDER BY c.last_message_time DESC, c.jid LIMIT ?3 OFFSET ?4"
        ))?;
        let rows =
            stmt.query_map(params![jid, user_of(jid), limit, page as i64 * limit as i64], row_chat)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn last_interaction(&self, jid: &str) -> Result<Option<Message>> {
        Ok(self
            .reads()
            .query_row(
                &format!(
                    "SELECT {MSG_COLS} {MSG_FROM}
                     WHERE messages.sender IN (?1, ?2) OR messages.chat_jid = ?1
                     ORDER BY messages.timestamp DESC, messages.rowid DESC LIMIT 1"
                ),
                [jid, user_of(jid)],
                row_message,
            )
            .optional()?)
    }

    /// Name shown for a sender (a bare user or a full JID): its direct chat's
    /// name unless that is a placeholder, then its contact name, then the raw value.
    pub fn sender_name(&self, sender: &str) -> String {
        let (pn, lid) = if sender.contains('@') {
            (sender.to_string(), sender.to_string())
        } else {
            (format!("{sender}@s.whatsapp.net"), format!("{sender}@lid"))
        };
        self.reads()
            .query_row(
                &format!(
                    "SELECT COALESCE(
                         (SELECT name FROM chats WHERE jid IN (?1, ?2) AND NOT {PLACEHOLDER} ORDER BY jid = ?2 LIMIT 1),
                         (SELECT {BEST_NAME} FROM contacts WHERE jid IN (?1, ?2) AND {BEST_NAME} IS NOT NULL
                          ORDER BY jid = ?2 LIMIT 1))"
                ),
                [&pn, &lid],
                |r| r.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
            .unwrap_or_else(|| sender.to_string())
    }

    /// Address book plus chat partners, one entry per phone number, named first.
    pub fn search_contacts(&self, query: &str) -> Result<Vec<Contact>> {
        let pattern = like_any(query);
        let db = self.reads();
        let mut rows: Vec<(String, Option<String>)> = db
            .prepare(&format!(
                "SELECT jid, {BEST_NAME} FROM contacts WHERE jid NOT LIKE '%@g.us'
                 AND (COALESCE({BEST_NAME}, '') LIKE ?1 ESCAPE '\\' OR jid LIKE ?1 ESCAPE '\\')"
            ))?
            .query_map([&pattern], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        rows.extend(
            db.prepare(
                "SELECT jid, name FROM chats WHERE jid NOT LIKE '%@g.us'
                 AND (name LIKE ?1 ESCAPE '\\' OR jid LIKE ?1 ESCAPE '\\')",
            )?
            .query_map([&pattern], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        let lid_map: std::collections::HashMap<String, String> =
            db.prepare("SELECT lid, pn FROM lid_map")?.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
        drop(db);
        Ok(merge_contacts(rows, &lid_map))
    }
}

fn merge_contacts(rows: Vec<(String, Option<String>)>, lid_map: &std::collections::HashMap<String, String>) -> Vec<Contact> {
    let mut merged: std::collections::HashMap<String, Contact> = std::collections::HashMap::new();
    for (jid, name) in rows {
        if jid.is_empty() {
            continue;
        }
        let user = jid.split('@').next().unwrap_or_default();
        let phone = lid_map.get(user).cloned().unwrap_or_else(|| user.to_string());
        let name = name.unwrap_or_default().trim().to_string();
        if let Some(existing) = merged.get(&phone)
            && !existing.name.is_empty()
            && existing.name != phone
        {
            continue;
        }
        let jid = if !phone.is_empty() && phone.bytes().all(|b| b.is_ascii_digit()) { format!("{phone}@s.whatsapp.net") } else { jid };
        let name = if name.is_empty() { phone.clone() } else { name };
        merged.insert(phone.clone(), Contact { phone_number: phone, name, jid });
    }
    let mut out: Vec<Contact> = merged.into_values().collect();
    out.sort_by_key(|a| (a.name == a.phone_number, a.name.to_lowercase()));
    out.truncate(50);
    out
}

// The names are the contact columns; renaming them would only obscure that.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy)]
pub enum ContactField {
    FullName,
    FirstName,
    PushName,
}

impl ContactField {
    fn column(self) -> &'static str {
        match self {
            Self::FullName => "full_name",
            Self::FirstName => "first_name",
            Self::PushName => "push_name",
        }
    }
}

/// WHERE terms (never empty) and their arguments for the filter's chat, sender
/// and time window, shared by `search` and `list_messages`.
fn filter_sql(f: &MessageFilter) -> (String, Vec<String>) {
    let mut sql = String::from("1");
    let mut args = Vec::new();
    if let Some(c) = &f.chat_jid {
        sql.push_str(" AND messages.chat_jid = ?");
        args.push(c.clone());
    }
    if let Some(s) = &f.sender {
        // A number, a JID, or an unmapped "user@lid" as stored.
        sql.push_str(" AND messages.sender IN (?, ?)");
        args.extend([user_of(s).to_string(), s.clone()]);
    }
    if let Some(t) = f.after {
        sql.push_str(" AND messages.timestamp > ?");
        args.push(format_ts(t));
    }
    if let Some(t) = f.before {
        sql.push_str(" AND messages.timestamp < ?");
        args.push(format_ts(t));
    }
    (sql, args)
}

/// One line per message, the transcript format the Python server produced.
pub fn format_message(store: &Store, m: &Message, show_chat: bool) -> String {
    let from = if m.is_from_me { "Me".to_string() } else { store.sender_name(&m.sender) };
    render(m, show_chat, &from)
}

fn render(m: &Message, show_chat: bool, from: &str) -> String {
    let ts = parse_ts(&m.timestamp).map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_else(|| m.timestamp.clone());
    let mut out = match (&m.chat_name, show_chat) {
        (Some(name), true) => format!("[{ts}] Chat: {name} "),
        _ => format!("[{ts}] "),
    };
    // The ID is what reply_to_message_id, edit_message and delete_message take.
    let prefix = match &m.media_type {
        Some(kind) => format!("[{kind} - Message ID: {} - Chat JID: {}] ", m.id, m.chat_jid),
        None => format!("[Message ID: {} - Chat JID: {}] ", m.id, m.chat_jid),
    };
    out.push_str(&format!("From: {from}: {prefix}{}\n", m.content));
    out
}

pub fn format_messages(store: &Store, messages: &[Message]) -> String {
    if messages.is_empty() {
        return "No messages to display.".into();
    }
    // One name lookup per sender, not per message.
    let mut names: std::collections::HashMap<&str, String> = std::collections::HashMap::new();
    messages
        .iter()
        .map(|m| {
            let from = if m.is_from_me { "Me" } else { names.entry(&m.sender).or_insert_with(|| store.sender_name(&m.sender)).as_str() };
            render(m, true, from)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn temp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("messages.db")).unwrap();
        (dir, store)
    }

    #[test]
    fn timestamps_match_the_go_format() {
        let t = Utc.with_ymd_and_hms(2026, 11, 3, 14, 2, 32).unwrap();
        assert_eq!(format_ts(t), "2026-11-03 14:02:32+00:00");
        assert_eq!(parse_ts("2026-11-03 14:02:32+00:00"), Some(t));
        assert_eq!(parse_ts("2026-11-03 11:02:32-03:00"), Some(t));
    }

    #[test]
    fn stores_and_reads_back_a_message_with_media() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        s.store_chat("55@s.whatsapp.net", "Ana", t).unwrap();
        let media =
            Media { kind: "image".into(), filename: "a.jpg".into(), direct_path: "/v/x?oh=1".into(), file_length: 9, ..Default::default() };
        assert!(s.store_message("ID1", "55@s.whatsapp.net", "55", "oi", t, false, &media).unwrap());
        assert!(!s.store_message("ID2", "55@s.whatsapp.net", "55", "", t, false, &Media::default()).unwrap());
        assert_eq!(s.media("ID1", "55@s.whatsapp.net").unwrap().unwrap(), (media.clone(), false, "55".to_string()));
        s.set_direct_path("ID1", "55@s.whatsapp.net", "/v/new").unwrap();
        assert_eq!(s.media("ID1", "55@s.whatsapp.net").unwrap().unwrap().0.direct_path, "/v/new");
        let msgs = s.list_messages(&MessageFilter { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            format_message(&s, &msgs[0], true),
            "[2026-01-02 03:04:05] Chat: Ana From: Ana: [image - Message ID: ID1 - Chat JID: 55@s.whatsapp.net] oi\n"
        );
        s.store_message("ID3", "55@s.whatsapp.net", "55", "tchau", t, false, &Media::default()).unwrap();
        let text = s.list_messages(&MessageFilter { query: Some("tchau".into()), limit: 1, ..Default::default() }).unwrap();
        assert!(format_message(&s, &text[0], false).contains("[Message ID: ID3 - Chat JID: 55@s.whatsapp.net] tchau"));
    }

    #[test]
    fn edits_replace_text_and_deletions_keep_it_once() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        s.store_chat("55@s.whatsapp.net", "Ana", t).unwrap();
        s.store_message("A", "55@s.whatsapp.net", "55", "oi", t, false, &Media::default()).unwrap();
        assert!(s.edit_message("A", "55@s.whatsapp.net", "olá").unwrap());
        assert_eq!(s.message_content("A", "55@s.whatsapp.net").as_deref(), Some("[edited] olá"));
        assert!(s.revoke_message("A", "55@s.whatsapp.net").unwrap());
        assert!(!s.revoke_message("A", "55@s.whatsapp.net").unwrap());
        assert_eq!(s.message_content("A", "55@s.whatsapp.net").as_deref(), Some("[deleted] [edited] olá"));
        assert!(!s.edit_message("missing", "55@s.whatsapp.net", "x").unwrap());
        s.store_message("A", "55@s.whatsapp.net", "55", "oi", t, false, &Media::default()).unwrap();
        assert_eq!(s.message_content("A", "55@s.whatsapp.net").as_deref(), Some("[deleted] [edited] olá"));
    }

    #[test]
    fn transcripts_are_searchable_text_that_a_redelivery_keeps() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let audio = Media { kind: "audio".into(), ..Default::default() };
        s.store_chat("55@s.whatsapp.net", "Ana", t).unwrap();
        s.store_message("V", "55@s.whatsapp.net", "55", "", t, false, &audio).unwrap();
        assert_eq!(s.transcription("V", "55@s.whatsapp.net"), None);
        s.set_transcription("V", "55@s.whatsapp.net", "chego às dez").unwrap();
        s.set_transcription("V", "55@s.whatsapp.net", "again").unwrap();
        s.store_message("V", "55@s.whatsapp.net", "55", "", t, false, &audio).unwrap();
        assert_eq!(s.message_content("V", "55@s.whatsapp.net").as_deref(), Some("[transcription] chego às dez"));
        assert_eq!(s.transcription("V", "55@s.whatsapp.net").as_deref(), Some("chego às dez"));
        let found = s.list_messages(&MessageFilter { query: Some("dez".into()), limit: 10, ..Default::default() }).unwrap();
        assert_eq!(found.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["V"]);
        s.store_message("C", "55@s.whatsapp.net", "55", "ouve", t, false, &audio).unwrap();
        s.set_transcription("C", "55@s.whatsapp.net", "oi").unwrap();
        assert_eq!(s.message_content("C", "55@s.whatsapp.net").as_deref(), Some("ouve\n[transcription] oi"));
    }

    fn chat_with(s: &Store, jid: &str, msgs: &[(&str, &str)]) {
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        s.store_chat(jid, "x", t).unwrap();
        for (i, (id, text)) in msgs.iter().enumerate() {
            s.store_message(id, jid, "55", text, t + chrono::Duration::seconds(i as i64), false, &Media::default()).unwrap();
        }
    }

    #[test]
    fn full_text_query_ignores_case_accents_and_syntax_and_follows_edits() {
        let (_d, s) = temp_store();
        chat_with(&s, "55@s.whatsapp.net", &[("A", "Vamos viajar para Floripa"), ("B", "Reunião amanhã às 9")]);
        let q = |q: &str| {
            s.list_messages(&MessageFilter { query: Some(q.into()), limit: 10, ..Default::default() })
                .unwrap()
                .into_iter()
                .map(|m| m.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(q("floripa"), ["A"]);
        assert_eq!(q("reuniao"), ["B"]);
        assert_eq!(q("viaj"), ["A"]);
        assert_eq!(q("\"NEAR( *"), Vec::<String>::new());
        // Re-storing keeps the rowid; an edit re-indexes the text.
        chat_with(&s, "55@s.whatsapp.net", &[("A", "Vamos viajar para Floripa")]);
        assert_eq!(q("floripa"), ["A"]);
        s.edit_message("A", "55@s.whatsapp.net", "mudou para Curitiba").unwrap();
        assert_eq!(q("curitiba"), ["A"]);
        assert!(q("floripa").is_empty());
    }

    #[test]
    fn hybrid_search_merges_words_and_vectors_and_edits_drop_stale_vectors() {
        let (_d, s) = temp_store();
        // One message per chat: three separate conversations.
        chat_with(&s, "1@s.whatsapp.net", &[("A", "passagem comprada")]);
        chat_with(&s, "2@s.whatsapp.net", &[("B", "bolo de cenoura")]);
        chat_with(&s, "3@s.whatsapp.net", &[("C", "viagem de férias")]);
        let rows = s.unembedded(10).unwrap();
        assert_eq!(rows.len(), 3);
        let axis = |i: usize| (0..3).map(|j| if i == j { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
        let by_text: std::collections::HashMap<String, i64> = rows.iter().map(|(r, t)| (t.clone(), *r)).collect();
        let put = |text: &'static str, v: Vec<f32>| (by_text[text], text, v);
        let saved = s.put_vectors(&[put("passagem comprada", axis(0)), put("bolo de cenoura", axis(1)), put("viagem de férias", axis(2))]);
        assert_eq!(saved.unwrap(), 3);
        assert!(s.unembedded(10).unwrap().is_empty());
        // The same text arriving again keeps its vector.
        chat_with(&s, "1@s.whatsapp.net", &[("A", "passagem comprada")]);
        assert!(s.unembedded(10).unwrap().is_empty());
        let f = MessageFilter { limit: 2, ..Default::default() };
        let ids = |v: Vec<Message>| v.into_iter().map(|m| m.id).collect::<Vec<_>>();
        // "ferias" hits C by words; the vector points at A: both come back, C (found twice) first.
        let q = [0.1, 0.0, 0.9_f32];
        assert_eq!(ids(s.search("ferias", Some(&q), &f).unwrap()), ["C", "A"]);
        assert_eq!(ids(s.search("cenoura", None, &f).unwrap()), ["B"]);
        // A vector of another size never compares.
        assert_eq!(ids(s.search("xyz", Some(&[1.0, 0.0]), &f).unwrap()), Vec::<String>::new());
        s.edit_message("B", "2@s.whatsapp.net", "torta de chocolate").unwrap();
        assert_eq!(s.unembedded(10).unwrap().len(), 1);
        // A vector for the text before the edit is not saved.
        assert_eq!(s.put_vectors(&[(by_text["bolo de cenoura"], "bolo de cenoura", axis(1))]).unwrap(), 0);
        assert_eq!(s.unembedded(10).unwrap().len(), 1);
    }

    #[test]
    fn a_vector_encodes_the_message_with_the_ones_before_it() {
        let (_d, s) = temp_store();
        let jid = "55@s.whatsapp.net";
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        s.store_chat(jid, "Ana", t).unwrap();
        let msg = |id: &str, min: i64, text: &str| {
            s.store_message(id, jid, "55", text, t + chrono::Duration::minutes(min), false, &Media::default()).unwrap()
        };
        msg("A", 0, "bom dia, tudo certo para amanhã?");
        msg("B", 300, "que dia eles vão viajar?");
        msg("C", 301, "[reacted 👍 to: que dia eles vão viajar?]");
        msg("D", 302, "dia 20");
        msg("E", 303, "sim, fechado");
        let windows = |s: &Store| {
            let mut w = s.unembedded(10).unwrap();
            w.sort();
            w.into_iter().map(|(_, t)| t).collect::<Vec<_>>()
        };
        // A is five hours before B: another conversation. The reaction is no context.
        assert_eq!(
            windows(&s),
            [
                "bom dia, tudo certo para amanhã?",
                "que dia eles vão viajar?",
                "que dia eles vão viajar?\ndia 20",
                "que dia eles vão viajar?\ndia 20\nsim, fechado",
            ]
        );
        assert!(!too_short("que dia eles vão viajar?\ndia 20"));
        assert!(too_short("ok\nkk\nsim"));
        let rows = s.unembedded(10).unwrap();
        let saved = s.put_vectors(&rows.iter().map(|(r, t)| (*r, t.as_str(), vec![1.0_f32])).collect::<Vec<_>>()).unwrap();
        assert_eq!(saved, 4);
        // A message synced late into the conversation changes the windows after it, and only those.
        msg("X", 301, "acho que de avião");
        // By rowid: D and E, whose windows now hold X, then X itself; B keeps its vector.
        assert_eq!(
            windows(&s),
            [
                "que dia eles vão viajar?\nacho que de avião\ndia 20",
                "que dia eles vão viajar?\nacho que de avião\ndia 20\nsim, fechado",
                "que dia eles vão viajar?\nacho que de avião",
            ]
        );
    }

    #[test]
    fn one_conversation_takes_one_semantic_slot() {
        let (_d, s) = temp_store();
        chat_with(
            &s,
            "1@s.whatsapp.net",
            &[("A", "vamos viajar em março"), ("B", "para onde exatamente?"), ("C", "talvez Floripa de carro")],
        );
        chat_with(&s, "2@s.whatsapp.net", &[("D", "a viagem foi cancelada")]);
        let rows = s.unembedded(10).unwrap();
        // The three windows of chat 1 score alike, chat 2 a bit lower.
        let v = |r: i64| if rows.iter().any(|(x, t)| *x == r && t.contains("cancelada")) { vec![0.8_f32, 0.6] } else { vec![1.0, 0.0] };
        s.put_vectors(&rows.iter().map(|(r, t)| (*r, t.as_str(), v(*r))).collect::<Vec<_>>()).unwrap();
        let f = MessageFilter { limit: 5, ..Default::default() };
        let ids = |v: Vec<Message>| v.into_iter().map(|m| m.id).collect::<Vec<_>>();
        // Ties go to the newer message: C stands for its conversation, then D.
        assert_eq!(ids(s.search("xyz", Some(&[1.0, 0.0]), &f).unwrap()), ["C", "D"]);
    }

    #[test]
    fn vectors_from_before_windows_are_dropped_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("messages.db");
        let s = Store::open(&path).unwrap();
        chat_with(&s, "1@s.whatsapp.net", &[("A", "passagem comprada")]);
        let rows = s.unembedded(10).unwrap();
        s.put_vectors(&[(rows[0].0, rows[0].1.as_str(), vec![1.0])]).unwrap();
        drop(s);
        assert!(Store::open(&path).unwrap().unembedded(10).unwrap().is_empty(), "current vectors stay");
        Connection::open(&path).unwrap().execute_batch("PRAGMA user_version = 0;").unwrap();
        assert_eq!(Store::open(&path).unwrap().unembedded(10).unwrap().len(), 1);
    }

    #[test]
    fn unread_counts_start_after_your_reply_or_the_phone_read_mark() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let at = |n: i64| t + chrono::Duration::seconds(n);
        let jid = "55@s.whatsapp.net";
        s.store_chat(jid, "Ana", t).unwrap();
        let msg = |id: &str, n: i64, me: bool, text: &str| s.store_message(id, jid, "55", text, at(n), me, &Media::default()).unwrap();
        msg("A", 0, false, "oi");
        msg("B", 1, true, "olá");
        msg("C", 2, false, "tudo bem?");
        msg("D", 3, false, "[reacted 👍 to: olá]");
        msg("E", 4, false, "e aí?");
        let unread = || s.get_chat(jid, false).unwrap().unwrap().unread_count;
        assert_eq!(unread(), 0, "history from before tracking began counts as read");
        s.conn().execute("UPDATE chat_reads SET read_at = ?1 WHERE jid = '*'", [format_ts(at(-1))]).unwrap();
        assert_eq!(unread(), 2);
        assert_eq!(s.unread_messages(jid).unwrap(), [("C".to_string(), "55".to_string()), ("E".to_string(), "55".to_string())]);
        assert_eq!(s.list_chats(None, 10, 0, false, false, true).unwrap().len(), 1);
        s.mark_read(jid, at(2)).unwrap();
        assert_eq!(unread(), 1);
        s.mark_read(jid, at(0)).unwrap();
        assert_eq!(unread(), 1, "a read mark never moves back");
        s.mark_read(jid, at(9)).unwrap();
        assert_eq!(unread(), 0);
        assert!(s.list_chats(None, 10, 0, false, false, true).unwrap().is_empty());
        let active = s.active_chats(at(1), None, 10).unwrap();
        assert_eq!(active.iter().map(|(c, n)| (c.jid.as_str(), *n)).collect::<Vec<_>>(), [(jid, 3)]);
    }

    #[test]
    fn allowlist_scopes_every_read() {
        let (_d, s) = temp_store();
        chat_with(&s, "1@s.whatsapp.net", &[("A", "segredo")]);
        chat_with(&s, "2@g.us", &[("B", "liberado")]);
        s.restrict(&["2@g.us".to_string()]).unwrap();
        assert!(s.allowed("2@g.us") && !s.allowed("1@s.whatsapp.net"));
        let all = s.list_messages(&MessageFilter { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(all.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["B"]);
        assert_eq!(s.list_chats(None, 10, 0, true, false, false).unwrap().len(), 1);
        assert!(s.get_chat("1@s.whatsapp.net", true).unwrap().is_none());
        assert!(s.message_context("A", None, 1, 1).unwrap().is_none());
        assert!(s.direct_chat_by_phone("1").unwrap().is_none());
        assert!(s.last_interaction("1@s.whatsapp.net").unwrap().is_none());
        assert!(s.search("segredo", None, &MessageFilter { limit: 5, ..Default::default() }).unwrap().is_empty());
        // Writes still land: history keeps recording every chat.
        chat_with(&s, "1@s.whatsapp.net", &[("C", "outro")]);
        assert_eq!(s.message_content("C", "1@s.whatsapp.net").as_deref(), Some("outro"));
    }

    #[test]
    fn numeric_names_are_placeholders_and_get_replaced() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.store_chat("55@s.whatsapp.net", "100000000000001", t).unwrap();
        assert_eq!(s.chat_name("55@s.whatsapp.net"), None);
        assert_eq!(s.unnamed_direct_chats().unwrap(), vec!["55@s.whatsapp.net".to_string()]);
        assert!(s.fill_chat_name("55@s.whatsapp.net", "Mevo").unwrap());
        assert!(!s.fill_chat_name("55@s.whatsapp.net", "Outro").unwrap());
        assert_eq!(s.chat_name("55@s.whatsapp.net").as_deref(), Some("Mevo"));
        s.store_chat("3@g.us", "Group 5511900000001-1500000000", t).unwrap();
        assert_eq!(s.chat_name("3@g.us"), None, "old-style group ids are placeholders too");
        assert!(super::is_placeholder("Group 5511900000001-1500000000"));
        assert!(!super::is_placeholder("Group -"));
        s.store_chat("1@g.us", "Group 1203630", t).unwrap();
        assert_eq!(s.chat_name("1@g.us"), None);
        assert!(s.fill_chat_name("1@g.us", "Família").unwrap(), "SQL and Rust agree on placeholders");
        s.store_chat("2@g.us", "Group ", t).unwrap();
        assert_eq!(s.chat_name("2@g.us").as_deref(), Some("Group "));
        assert!(!s.fill_chat_name("2@g.us", "x").unwrap());
    }

    #[test]
    fn last_message_time_never_moves_back() {
        let (_d, s) = temp_store();
        let new = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
        let old = Utc.with_ymd_and_hms(2025, 5, 1, 0, 0, 0).unwrap();
        s.store_chat("g@g.us", "G", new).unwrap();
        s.store_chat("g@g.us", "", old).unwrap();
        let c = s.get_chat("g@g.us", false).unwrap().unwrap();
        assert_eq!(c.name.as_deref(), Some("G"));
        assert_eq!(c.last_message_time.as_deref(), Some("2026-05-01T00:00:00+00:00"));
    }

    #[test]
    fn merges_lid_chat_into_phone_row() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.put_lid("999", "5543").unwrap();
        s.store_chat("999@lid", "Bia", t).unwrap();
        s.store_message("A", "999@lid", "999", "oi", t, false, &Media::default()).unwrap();
        assert_eq!(s.merge_lid_chats().unwrap(), (1, 1));
        let msgs =
            s.list_messages(&MessageFilter { limit: 5, chat_jid: Some("5543@s.whatsapp.net".into()), ..Default::default() }).unwrap();
        assert_eq!(msgs[0].sender, "5543");
        assert!(s.get_chat("999@lid", false).unwrap().is_none());
        assert_eq!(s.chat_name("5543@s.whatsapp.net").as_deref(), Some("Bia"));
    }

    #[test]
    fn lid_merge_keeps_real_names_markers_and_read_marks() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let (lid, pn) = ("999@lid", "5543@s.whatsapp.net");
        s.put_lid("999", "5543").unwrap();
        s.store_chat(pn, "Bia Souza", t).unwrap();
        s.store_chat(lid, "999", t + chrono::Duration::seconds(5)).unwrap();
        for chat in [lid, pn] {
            s.store_message("A", chat, "5543", "oi", t, false, &Media::default()).unwrap();
        }
        s.revoke_message("A", lid).unwrap();
        s.mark_read(lid, t + chrono::Duration::seconds(3)).unwrap();
        s.mark_read(pn, t + chrono::Duration::seconds(1)).unwrap();
        assert_eq!(s.merge_lid_chats().unwrap(), (1, 1));
        let c = s.get_chat(pn, false).unwrap().unwrap();
        assert_eq!(c.name.as_deref(), Some("Bia Souza"), "a placeholder never replaces a real name");
        assert_eq!(c.last_message_time.as_deref(), Some("2026-01-01T00:00:05+00:00"));
        assert_eq!(s.message_content("A", pn).as_deref(), Some("[deleted] oi"));
        assert_eq!(s.message_content("A", lid), None);
        let read: String = s.conn().query_row("SELECT read_at FROM chat_reads WHERE jid = ?1", [pn], |r| r.get(0)).unwrap();
        assert_eq!(read, "2026-01-01 00:00:03+00:00");
        assert!(!s.conn().prepare("SELECT 1 FROM chat_reads WHERE jid = ?1").unwrap().exists([lid]).unwrap());
    }

    #[test]
    fn contacts_collapse_by_phone_and_named_first() {
        let lid = [("777".to_string(), "5511".to_string())].into_iter().collect();
        let rows = vec![
            ("5511@s.whatsapp.net".into(), None),
            ("777@lid".into(), Some("Caio".into())),
            ("5522@s.whatsapp.net".into(), Some("5522".into())),
        ];
        let out = merge_contacts(rows, &lid);
        assert_eq!(out[0], Contact { phone_number: "5511".into(), name: "Caio".into(), jid: "5511@s.whatsapp.net".into() });
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn reopening_replaces_the_old_update_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("messages.db");
        let s = Store::open(&path).unwrap();
        s.conn()
            .execute_batch(
                "DROP TRIGGER messages_fts_au;
                 CREATE TRIGGER messages_fts_au AFTER UPDATE OF content ON messages BEGIN
                     DELETE FROM message_vec WHERE rowid = old.rowid;
                 END;",
            )
            .unwrap();
        drop(s);
        let s = Store::open(&path).unwrap();
        let sql: String = s.conn().query_row("SELECT sql FROM sqlite_master WHERE name = 'messages_fts_au'", [], |r| r.get(0)).unwrap();
        assert!(sql.contains("WHEN old.content IS NOT new.content"));
        assert_eq!(s.conn().query_row("PRAGMA synchronous", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    }

    #[test]
    fn a_redelivery_keeps_stored_media_and_text() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let jid = "55@s.whatsapp.net";
        s.store_chat(jid, "Ana", t).unwrap();
        let media = Media {
            kind: "image".into(),
            filename: "a.jpg".into(),
            url: "https://x".into(),
            direct_path: "/v/x".into(),
            media_key: vec![1],
            file_sha256: vec![2],
            file_enc_sha256: vec![3],
            file_length: 9,
        };
        s.store_message("A", jid, "55", "legenda", t, false, &media).unwrap();
        // History sync may bring the message back stripped of its attachment fields.
        s.store_message("A", jid, "55", "", t, false, &Media { kind: "image".into(), ..Default::default() }).unwrap();
        assert_eq!(s.media("A", jid).unwrap().unwrap().0, media);
        assert_eq!(s.message_content("A", jid).as_deref(), Some("legenda"));
        // A copy stored without keys takes the whole incoming set.
        s.store_message("B", jid, "55", "doc", t, false, &Media { kind: "document".into(), ..Default::default() }).unwrap();
        s.store_message("B", jid, "55", "doc", t, false, &media).unwrap();
        assert_eq!(s.media("B", jid).unwrap().unwrap().0, media);
        // A replay of the same upload keeps the path refreshed since.
        s.set_direct_path("A", jid, "/v/fresh").unwrap();
        s.store_message("A", jid, "55", "legenda", t, false, &media).unwrap();
        assert_eq!(s.media("A", jid).unwrap().unwrap().0.direct_path, "/v/fresh");
        // Another upload replaces the set as a whole, never field by field.
        let other =
            Media { url: "https://y".into(), direct_path: "/v/y".into(), media_key: vec![7], file_enc_sha256: vec![8], ..media.clone() };
        s.store_message("A", jid, "55", "legenda", t, false, &other).unwrap();
        assert_eq!(s.media("A", jid).unwrap().unwrap().0, other);
    }

    #[test]
    fn sender_names_come_from_the_direct_chat_or_the_contact_never_a_group() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.store_chat("120355@g.us", "Grupo", t).unwrap();
        assert_eq!(s.sender_name("55"), "55", "a group JID containing the digits is not the sender");
        s.store_chat("55@s.whatsapp.net", "55", t).unwrap();
        s.upsert_contact("55@s.whatsapp.net", ContactField::PushName, "Ana").unwrap();
        assert_eq!(s.sender_name("55"), "Ana", "a placeholder chat name falls back to the contact");
        s.store_chat("55@s.whatsapp.net", "Ana Lima", t).unwrap();
        assert_eq!(s.sender_name("55"), "Ana Lima");
        s.store_chat("999@lid", "Bia", t).unwrap();
        assert_eq!(s.sender_name("999"), "Bia");
        assert_eq!(s.sender_name("999@lid"), "Bia");
        chat_with(&s, "120355@g.us", &[("G1", "oi")]);
        let msgs = s.list_messages(&MessageFilter { limit: 5, ..Default::default() }).unwrap();
        assert!(format_messages(&s, &msgs).contains("From: Ana Lima: "));
    }

    #[test]
    fn contact_lookups_match_the_sender_user_part() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.store_chat("55@s.whatsapp.net", "Ana", t).unwrap();
        s.store_chat("1@g.us", "Grupo", t).unwrap();
        s.store_chat("2@g.us", "Outro", t).unwrap();
        s.store_message("D", "55@s.whatsapp.net", "55", "oi", t, false, &Media::default()).unwrap();
        s.store_message("G", "1@g.us", "55", "no grupo", t + chrono::Duration::seconds(1), false, &Media::default()).unwrap();
        s.store_message("H", "2@g.us", "66", "não é ela", t + chrono::Duration::seconds(2), false, &Media::default()).unwrap();
        let mut chats: Vec<String> = s.contact_chats("55@s.whatsapp.net", 10, 0).unwrap().into_iter().map(|c| c.jid).collect();
        chats.sort();
        assert_eq!(chats, ["1@g.us", "55@s.whatsapp.net"]);
        assert_eq!(s.last_interaction("55@s.whatsapp.net").unwrap().unwrap().id, "G");
        let from = |v: &str| s.list_messages(&MessageFilter { sender: Some(v.into()), limit: 10, ..Default::default() }).unwrap().len();
        assert_eq!((from("55"), from("55@s.whatsapp.net")), (2, 2));
    }

    #[test]
    fn context_keeps_same_second_neighbours_in_order() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let jid = "55@s.whatsapp.net";
        s.store_chat(jid, "Ana", t).unwrap();
        for (id, secs) in [("A", 0), ("B", 0), ("C", 0), ("D", 1)] {
            s.store_message(id, jid, "55", id, t + chrono::Duration::seconds(secs), false, &Media::default()).unwrap();
        }
        // The same ID in another chat is not this message.
        s.store_chat("66@s.whatsapp.net", "Bia", t).unwrap();
        s.store_message("C", "66@s.whatsapp.net", "66", "outro", t + chrono::Duration::seconds(9), false, &Media::default()).unwrap();
        let ctx = s.message_context("C", Some(jid), 5, 5).unwrap().unwrap();
        let ids = |v: &[Message]| v.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
        assert_eq!(
            (ids(&ctx.before), ctx.message.chat_jid.as_str(), ids(&ctx.after)),
            (vec!["A".into(), "B".into()], jid, vec!["D".to_string()])
        );
        let ctx = s.message_context("B", Some(jid), 1, 1).unwrap().unwrap();
        assert_eq!((ids(&ctx.before), ids(&ctx.after)), (vec!["A".to_string()], vec!["C".to_string()]));
        assert_eq!(s.message_context("C", None, 0, 0).unwrap().unwrap().message.chat_jid, "66@s.whatsapp.net");
    }

    #[test]
    fn like_searches_take_user_text_literally() {
        let (_d, s) = temp_store();
        chat_with(&s, "55@s.whatsapp.net", &[("A", "50% off"), ("B", "oi")]);
        let q = |q: &str| s.list_messages(&MessageFilter { query: Some(q.into()), limit: 10, ..Default::default() }).unwrap().len();
        assert_eq!((q("%"), q("_")), (1, 0));
        assert_eq!(s.list_chats(Some("_"), 10, 0, false, false, false).unwrap().len(), 0);
        assert_eq!(s.list_chats(Some("X"), 10, 0, false, false, false).unwrap().len(), 1, "still case-insensitive");
        assert!(s.search_contacts("%").unwrap().is_empty());
        let t = Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap();
        s.store_chat("155@s.whatsapp.net", "Outro", t).unwrap();
        assert_eq!(s.direct_chat_by_phone("55").unwrap().unwrap().jid, "55@s.whatsapp.net", "the exact JID wins");
        assert_eq!(s.direct_chat_by_phone("15").unwrap().unwrap().jid, "155@s.whatsapp.net");
    }

    #[test]
    fn chat_pages_are_stable_and_carry_unread_counts() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.conn().execute("UPDATE chat_reads SET read_at = ?1 WHERE jid = '*'", [format_ts(t)]).unwrap();
        for jid in ["1@g.us", "2@g.us", "3@g.us"] {
            s.store_chat(jid, "G", t).unwrap();
            s.store_message("A", jid, "55", "oi", t + chrono::Duration::seconds(1), false, &Media::default()).unwrap();
        }
        let page = |p: u32| s.list_chats(None, 2, p, true, true, false).unwrap();
        let jids: Vec<(String, i64)> = page(0).into_iter().chain(page(1)).map(|c| (c.jid, c.unread_count)).collect();
        assert_eq!(jids, [("1@g.us".to_string(), 1), ("2@g.us".to_string(), 1), ("3@g.us".to_string(), 1)]);
    }

    #[test]
    fn whatsmeow_import_skips_missing_tables() {
        let (dir, s) = temp_store();
        let wa = dir.path().join("whatsapp.db");
        Connection::open(&wa)
            .unwrap()
            .execute_batch(
                "CREATE TABLE whatsmeow_contacts (their_jid TEXT, full_name TEXT, first_name TEXT, business_name TEXT, push_name TEXT);
                 INSERT INTO whatsmeow_contacts VALUES ('55@s.whatsapp.net', 'Ana', NULL, NULL, NULL);",
            )
            .unwrap();
        assert_eq!(s.import_whatsmeow(&wa).unwrap(), 1);
        assert_eq!(s.contact_name("55@s.whatsapp.net").as_deref(), Some("Ana"));
    }
}
