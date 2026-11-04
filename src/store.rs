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
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

const TS_FORMAT: &str = "%Y-%m-%d %H:%M:%S+00:00";

/// A chat name that is only digits is a placeholder (the number, a LID, or the
/// account's own number the Go bridge used to write), never a real name.
const PLACEHOLDER: &str = "(name IS NULL OR name = '' OR (name GLOB '[0-9]*' AND name NOT GLOB '*[^0-9]*'))";

fn is_placeholder(name: &str) -> bool {
    // "Group <id>" is the fallback written when a group's subject was unknown.
    let digits = name.strip_prefix("Group ").unwrap_or(name);
    digits.is_empty() || digits.bytes().all(|b| b.is_ascii_digit())
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
}

const MSG_COLS: &str = "messages.timestamp, messages.sender, chats.name, messages.content, \
     messages.is_from_me, messages.chat_jid, messages.id, messages.media_type";

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
        last_message_time: row
            .get::<_, Option<String>>(2)?
            .map(|raw| parse_ts(&raw).map(|t| t.to_rfc3339()).unwrap_or(raw)),
        last_message: row.get(3)?,
        last_sender: row.get(4)?,
        last_is_from_me: row.get(5)?,
    })
}

/// Newest message actually on file, not the one chats.last_message_time names:
/// that can be a message the bridge dropped as unreadable.
const LAST_MESSAGE_JOIN: &str = "LEFT JOIN messages m ON m.rowid = (
        SELECT x.rowid FROM messages x WHERE x.chat_jid = c.jid
        ORDER BY x.timestamp DESC LIMIT 1)";

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA temp_store = MEMORY;
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
        let has_direct_path: bool = db
            .prepare("SELECT 1 FROM pragma_table_info('messages') WHERE name = 'direct_path'")?
            .exists([])?;
        if !has_direct_path {
            db.execute("ALTER TABLE messages ADD COLUMN direct_path TEXT", [])?;
        }
        Ok(Self { db: Mutex::new(db) })
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
        let db = self.conn();
        db.execute("ATTACH DATABASE ?1 AS wa", [whatsmeow_db.to_string_lossy()])?;
        let result = (|| -> Result<usize> {
            let mut n = db.execute(
                "INSERT INTO contacts (jid, full_name, first_name, business_name, push_name)
                 SELECT their_jid, COALESCE(full_name, ''), COALESCE(first_name, ''),
                        COALESCE(business_name, ''), COALESCE(push_name, '')
                 FROM wa.whatsmeow_contacts WHERE true
                 ON CONFLICT(jid) DO NOTHING",
                [],
            )?;
            n += db.execute(
                "INSERT OR IGNORE INTO lid_map (lid, pn) SELECT lid, pn FROM wa.whatsmeow_lid_map",
                [],
            )?;
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
        self.conn().execute(
            "INSERT OR REPLACE INTO messages
             (id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, url,
              media_key, file_sha256, file_enc_sha256, file_length, direct_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
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
                media.direct_path
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
        self.conn()
            .query_row("SELECT pn FROM lid_map WHERE lid = ?1", [lid_user], |r| r.get(0))
            .optional()
            .ok()
            .flatten()
    }

    pub fn lid_for_pn(&self, pn_user: &str) -> Option<String> {
        self.conn()
            .query_row("SELECT lid FROM lid_map WHERE pn = ?1 LIMIT 1", [pn_user], |r| r.get(0))
            .optional()
            .ok()
            .flatten()
    }

    /// Address book name first, then business, first name, push name.
    pub fn contact_name(&self, jid: &str) -> Option<String> {
        self.conn()
            .query_row(
                "SELECT COALESCE(NULLIF(full_name, ''), NULLIF(business_name, ''),
                                 NULLIF(first_name, ''), NULLIF(push_name, ''))
                 FROM contacts WHERE jid = ?1",
                [jid],
                |r| r.get::<_, Option<String>>(0),
            )
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
        let mut db = self.conn();
        let pending: Vec<(String, Option<String>, Option<String>, String)> = db
            .prepare(
                "SELECT c.jid, c.name, c.last_message_time, l.pn FROM chats c
                 JOIN lid_map l ON l.lid = substr(c.jid, 1, instr(c.jid, '@') - 1)
                 WHERE c.jid LIKE '%@lid'",
            )?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let total = db.query_row("SELECT COUNT(*) FROM chats WHERE jid LIKE '%@lid'", [], |r| {
            r.get::<_, i64>(0)
        })? as usize;
        let mut merged = 0;
        for (lid_jid, name, last, pn) in pending {
            let lid_user = lid_jid.split('@').next().unwrap_or_default().to_string();
            let target = format!("{pn}@s.whatsapp.net");
            let tx = db.transaction()?;
            tx.execute(
                "INSERT INTO chats (jid, name, last_message_time) VALUES (?1, ?2, ?3)
                 ON CONFLICT(jid) DO UPDATE SET
                   name = CASE WHEN excluded.name != '' THEN excluded.name ELSE chats.name END,
                   last_message_time = MAX(chats.last_message_time, excluded.last_message_time)",
                params![target, name.unwrap_or_default(), last],
            )?;
            tx.execute(
                "UPDATE messages SET sender = ?1 WHERE chat_jid = ?2 AND sender = ?3",
                params![pn, lid_jid, lid_user],
            )?;
            tx.execute(
                "DELETE FROM messages WHERE chat_jid = ?1 AND id IN (SELECT id FROM messages WHERE chat_jid = ?2)",
                params![lid_jid, target],
            )?;
            tx.execute("UPDATE messages SET chat_jid = ?1 WHERE chat_jid = ?2", params![target, lid_jid])?;
            tx.execute("DELETE FROM chats WHERE jid = ?1", [&lid_jid])?;
            tx.commit()?;
            merged += 1;
        }
        Ok((merged, total))
    }

    // ---- media --------------------------------------------------------------

    pub fn media(&self, id: &str, chat_jid: &str) -> Result<Option<Media>> {
        Ok(self
            .conn()
            .query_row(
                "SELECT media_type, filename, url, direct_path, media_key, file_sha256, file_enc_sha256, file_length
                 FROM messages WHERE id = ?1 AND chat_jid = ?2",
                [id, chat_jid],
                |r| {
                    Ok(Media {
                        kind: r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                        filename: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        url: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        direct_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        media_key: r.get::<_, Option<Vec<u8>>>(4)?.unwrap_or_default(),
                        file_sha256: r.get::<_, Option<Vec<u8>>>(5)?.unwrap_or_default(),
                        file_enc_sha256: r.get::<_, Option<Vec<u8>>>(6)?.unwrap_or_default(),
                        file_length: r.get::<_, Option<i64>>(7)?.unwrap_or(0) as u64,
                    })
                },
            )
            .optional()?)
    }

    // ---- reads for the MCP tools -------------------------------------------

    pub fn list_messages(&self, f: &MessageFilter) -> Result<Vec<Message>> {
        let mut sql = format!("SELECT {MSG_COLS} FROM messages JOIN chats ON messages.chat_jid = chats.jid");
        let mut clauses = Vec::new();
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(t) = f.after {
            clauses.push("messages.timestamp > ?");
            args.push(Box::new(format_ts(t)));
        }
        if let Some(t) = f.before {
            clauses.push("messages.timestamp < ?");
            args.push(Box::new(format_ts(t)));
        }
        if let Some(s) = &f.sender {
            clauses.push("messages.sender = ?");
            args.push(Box::new(s.clone()));
        }
        if let Some(c) = &f.chat_jid {
            clauses.push("messages.chat_jid = ?");
            args.push(Box::new(c.clone()));
        }
        if let Some(q) = &f.query {
            clauses.push("LOWER(messages.content) LIKE LOWER(?)");
            args.push(Box::new(format!("%{q}%")));
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY messages.timestamp DESC LIMIT ? OFFSET ?");
        args.push(Box::new(f.limit as i64));
        args.push(Box::new(f.page as i64 * f.limit as i64));
        let db = self.conn();
        let mut stmt = db.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())), row_message)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn message_context(&self, id: &str, before: u32, after: u32) -> Result<Option<MessageContext>> {
        let db = self.conn();
        let Some(target) = db
            .query_row(
                &format!("SELECT {MSG_COLS} FROM messages JOIN chats ON messages.chat_jid = chats.jid WHERE messages.id = ?1"),
                [id],
                row_message,
            )
            .optional()?
        else {
            return Ok(None);
        };
        let raw_ts: String = db.query_row(
            "SELECT timestamp FROM messages WHERE id = ?1 AND chat_jid = ?2",
            [&target.id, &target.chat_jid],
            |r| r.get(0),
        )?;
        let around = |op: &str, order: &str, n: u32| -> Result<Vec<Message>> {
            let mut stmt = db.prepare(&format!(
                "SELECT {MSG_COLS} FROM messages JOIN chats ON messages.chat_jid = chats.jid
                 WHERE messages.chat_jid = ?1 AND messages.timestamp {op} ?2
                 ORDER BY messages.timestamp {order} LIMIT ?3"
            ))?;
            let rows = stmt
                .query_map(params![target.chat_jid, raw_ts, n], row_message)?
                .collect::<rusqlite::Result<_>>()?;
            Ok(rows)
        };
        let before = around("<", "DESC", before)?;
        let after = around(">", "ASC", after)?;
        Ok(Some(MessageContext { message: target, before, after }))
    }

    pub fn list_chats(&self, query: Option<&str>, limit: u32, page: u32, last: bool, by_name: bool) -> Result<Vec<Chat>> {
        let cols = if last {
            "m.content, m.sender, m.is_from_me"
        } else {
            "NULL, NULL, NULL"
        };
        let mut sql = format!("SELECT c.jid, c.name, c.last_message_time, {cols} FROM chats c ");
        if last {
            sql.push_str(LAST_MESSAGE_JOIN);
        }
        let mut args: Vec<String> = Vec::new();
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            sql.push_str(" WHERE (LOWER(c.name) LIKE LOWER(?1) OR c.jid LIKE ?1)");
            args.push(format!("%{q}%"));
        }
        sql.push_str(if by_name { " ORDER BY c.name" } else { " ORDER BY c.last_message_time DESC" });
        sql.push_str(&format!(" LIMIT {} OFFSET {}", limit, page as u64 * limit as u64));
        let db = self.conn();
        let mut stmt = db.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args.iter()), row_chat)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn get_chat(&self, jid: &str, last: bool) -> Result<Option<Chat>> {
        let sql = if last {
            format!("SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me FROM chats c {LAST_MESSAGE_JOIN} WHERE c.jid = ?1")
        } else {
            "SELECT c.jid, c.name, c.last_message_time, NULL, NULL, NULL FROM chats c WHERE c.jid = ?1".into()
        };
        Ok(self.conn().query_row(&sql, [jid], row_chat).optional()?)
    }

    pub fn direct_chat_by_phone(&self, phone: &str) -> Result<Option<Chat>> {
        Ok(self
            .conn()
            .query_row(
                &format!(
                    "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me FROM chats c
                     {LAST_MESSAGE_JOIN} WHERE c.jid LIKE ?1 AND c.jid NOT LIKE '%@g.us' LIMIT 1"
                ),
                [format!("%{phone}%")],
                row_chat,
            )
            .optional()?)
    }

    pub fn contact_chats(&self, jid: &str, limit: u32, page: u32) -> Result<Vec<Chat>> {
        let db = self.conn();
        let mut stmt = db.prepare(&format!(
            "SELECT c.jid, c.name, c.last_message_time, m.content, m.sender, m.is_from_me FROM chats c
             {LAST_MESSAGE_JOIN}
             WHERE c.jid = ?1 OR EXISTS (SELECT 1 FROM messages s WHERE s.chat_jid = c.jid AND s.sender = ?1)
             ORDER BY c.last_message_time DESC LIMIT ?2 OFFSET ?3"
        ))?;
        let rows = stmt
            .query_map(params![jid, limit, page as i64 * limit as i64], row_chat)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn last_interaction(&self, jid: &str) -> Result<Option<Message>> {
        Ok(self
            .conn()
            .query_row(
                &format!(
                    "SELECT {MSG_COLS} FROM messages JOIN chats ON messages.chat_jid = chats.jid
                     WHERE messages.sender = ?1 OR chats.jid = ?1 ORDER BY messages.timestamp DESC LIMIT 1"
                ),
                [jid],
                row_message,
            )
            .optional()?)
    }

    /// Name shown for a sender: the chat row for that JID, then any chat whose
    /// JID contains the number, then the raw value.
    pub fn sender_name(&self, sender: &str) -> String {
        let db = self.conn();
        let exact = db
            .query_row("SELECT name FROM chats WHERE jid = ?1 LIMIT 1", [sender], |r| r.get::<_, Option<String>>(0))
            .optional()
            .ok()
            .flatten();
        let found = match exact {
            Some(name) => name,
            None => {
                let phone = sender.split('@').next().unwrap_or(sender);
                db.query_row("SELECT name FROM chats WHERE jid LIKE ?1 LIMIT 1", [format!("%{phone}%")], |r| {
                    r.get::<_, Option<String>>(0)
                })
                .optional()
                .ok()
                .flatten()
                .flatten()
            }
        };
        found.filter(|n| !n.is_empty()).unwrap_or_else(|| sender.to_string())
    }

    /// Address book plus chat partners, one entry per phone number, named first.
    pub fn search_contacts(&self, query: &str) -> Result<Vec<Contact>> {
        let pattern = format!("%{query}%");
        let db = self.conn();
        let best = "COALESCE(NULLIF(full_name, ''), NULLIF(business_name, ''), NULLIF(first_name, ''), NULLIF(push_name, ''))";
        let mut rows: Vec<(String, Option<String>)> = db
            .prepare(&format!(
                "SELECT jid, {best} FROM contacts WHERE jid NOT LIKE '%@g.us'
                 AND (LOWER(COALESCE({best}, '')) LIKE LOWER(?1) OR LOWER(jid) LIKE LOWER(?1))"
            ))?
            .query_map([&pattern], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        rows.extend(
            db.prepare(
                "SELECT jid, name FROM chats WHERE jid NOT LIKE '%@g.us'
                 AND (LOWER(name) LIKE LOWER(?1) OR LOWER(jid) LIKE LOWER(?1))",
            )?
            .query_map([&pattern], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        let lid_map: std::collections::HashMap<String, String> = db
            .prepare("SELECT lid, pn FROM lid_map")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
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
        if let Some(existing) = merged.get(&phone) {
            if !existing.name.is_empty() && existing.name != phone {
                continue;
            }
        }
        let jid = if !phone.is_empty() && phone.bytes().all(|b| b.is_ascii_digit()) {
            format!("{phone}@s.whatsapp.net")
        } else {
            jid
        };
        let name = if name.is_empty() { phone.clone() } else { name };
        merged.insert(phone.clone(), Contact { phone_number: phone, name, jid });
    }
    let mut out: Vec<Contact> = merged.into_values().collect();
    out.sort_by(|a, b| {
        (a.name == a.phone_number, a.name.to_lowercase()).cmp(&(b.name == b.phone_number, b.name.to_lowercase()))
    });
    out.truncate(50);
    out
}

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

/// One line per message, the transcript format the Python server produced.
pub fn format_message(store: &Store, m: &Message, show_chat: bool) -> String {
    let ts = parse_ts(&m.timestamp)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| m.timestamp.clone());
    let mut out = match (&m.chat_name, show_chat) {
        (Some(name), true) => format!("[{ts}] Chat: {name} "),
        _ => format!("[{ts}] "),
    };
    let prefix = match &m.media_type {
        Some(kind) => format!("[{kind} - Message ID: {} - Chat JID: {}] ", m.id, m.chat_jid),
        None => String::new(),
    };
    let from = if m.is_from_me { "Me".to_string() } else { store.sender_name(&m.sender) };
    out.push_str(&format!("From: {from}: {prefix}{}\n", m.content));
    out
}

pub fn format_messages(store: &Store, messages: &[Message]) -> String {
    if messages.is_empty() {
        return "No messages to display.".into();
    }
    messages.iter().map(|m| format_message(store, m, true)).collect()
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
        let media = Media { kind: "image".into(), filename: "a.jpg".into(), direct_path: "/v/x?oh=1".into(), file_length: 9, ..Default::default() };
        assert!(s.store_message("ID1", "55@s.whatsapp.net", "55", "oi", t, false, &media).unwrap());
        assert!(!s.store_message("ID2", "55@s.whatsapp.net", "55", "", t, false, &Media::default()).unwrap());
        assert_eq!(s.media("ID1", "55@s.whatsapp.net").unwrap().unwrap(), media);
        let msgs = s.list_messages(&MessageFilter { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(format_message(&s, &msgs[0], true), "[2026-01-02 03:04:05] Chat: Ana From: Ana: [image - Message ID: ID1 - Chat JID: 55@s.whatsapp.net] oi\n");
    }

    #[test]
    fn numeric_names_are_placeholders_and_get_replaced() {
        let (_d, s) = temp_store();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        s.store_chat("55@s.whatsapp.net", "144560596463731", t).unwrap();
        assert_eq!(s.chat_name("55@s.whatsapp.net"), None);
        assert_eq!(s.unnamed_direct_chats().unwrap(), vec!["55@s.whatsapp.net".to_string()]);
        assert!(s.fill_chat_name("55@s.whatsapp.net", "Mevo").unwrap());
        assert!(!s.fill_chat_name("55@s.whatsapp.net", "Outro").unwrap());
        assert_eq!(s.chat_name("55@s.whatsapp.net").as_deref(), Some("Mevo"));
        s.store_chat("1@g.us", "Group 1203630", t).unwrap();
        assert_eq!(s.chat_name("1@g.us"), None);
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
        let msgs = s.list_messages(&MessageFilter { limit: 5, chat_jid: Some("5543@s.whatsapp.net".into()), ..Default::default() }).unwrap();
        assert_eq!(msgs[0].sender, "5543");
        assert!(s.get_chat("999@lid", false).unwrap().is_none());
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
}
