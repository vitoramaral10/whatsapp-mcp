//! One-shot upload URLs: send_file hands the client a URL, the client PUTs the
//! file there, and the message goes out when the upload completes. A model
//! cannot type a PDF as base64 into a tool call, and a client on another
//! machine has no path on this one; a `curl -T` it can always run.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::{Json, Router, body::Bytes, routing::put};
use serde_json::{Value, json};

use crate::mcp::Ctx;
use crate::media;

/// How long an upload URL stays usable.
pub const TTL: Duration = Duration::from_secs(15 * 60);

/// What to send once the bytes arrive; everything but the file is fixed when
/// the URL is made.
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub recipient: String,
    pub caption: String,
    /// Basename with extension, as with content_base64.
    pub name: String,
    pub voice: bool,
    pub reply_to: Option<String>,
}

pub struct Uploads {
    /// Public base URL the client reaches the upload listener at.
    base_url: String,
    slots: Mutex<HashMap<String, (Slot, Instant)>>,
}

impl Uploads {
    pub fn new(base_url: String) -> Self {
        Self { base_url: base_url.trim_end_matches('/').to_string(), slots: Mutex::default() }
    }

    /// Registers a slot and returns its URL. The token is the only credential,
    /// so it is 128 random bits.
    pub fn create(&self, slot: Slot) -> Result<String> {
        let mut raw = [0u8; 16];
        std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom")?, &mut raw).context("read /dev/urandom")?;
        let token: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        let mut slots = self.slots.lock().unwrap();
        let now = Instant::now();
        slots.retain(|_, (_, expires)| *expires > now);
        slots.insert(token.clone(), (slot, now + TTL));
        Ok(format!("{}/upload/{token}", self.base_url))
    }

    /// Takes the slot for a token: each URL works once, and not after TTL.
    pub fn take(&self, token: &str) -> Option<Slot> {
        let (slot, expires) = self.slots.lock().unwrap().remove(token)?;
        (expires > Instant::now()).then_some(slot)
    }
}

/// The upload listener: only `PUT /upload/{token}`, nothing else of the API.
pub fn router(ctx: Arc<Ctx>) -> Router {
    Router::new().route("/upload/{token}", put(receive)).layer(DefaultBodyLimit::max(media::MAX_SEND_BYTES as usize)).with_state(ctx)
}

async fn receive(State(ctx): State<Arc<Ctx>>, Path(token): Path<String>, body: Bytes) -> (StatusCode, Json<Value>) {
    let fail = |status: StatusCode, m: String| (status, Json(json!({"success": false, "message": m})));
    // Before taking the slot, so a wrong curl does not burn the URL.
    if body.is_empty() {
        return fail(StatusCode::BAD_REQUEST, "Empty upload: PUT the file's bytes (curl -T <file> <url>)".into());
    }
    let Some(slot) = ctx.uploads.as_ref().and_then(|u| u.take(&token)) else {
        return fail(StatusCode::NOT_FOUND, "Unknown, used or expired upload URL; ask send_file for a new one".into());
    };
    // From, not to_vec: the buffer is ours alone, so it moves instead of being copied.
    let file = media::Outgoing { data: Vec::from(body), name: slot.name };
    match ctx.wa.send(&slot.recipient, &slot.caption, Some(file), slot.voice, slot.reply_to.as_deref()).await {
        Ok(m) => (StatusCode::OK, Json(json!({"success": true, "message": m}))),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> Slot {
        Slot { recipient: "5511900000001".into(), caption: "oi".into(), name: "a.pdf".into(), voice: false, reply_to: None }
    }

    #[test]
    fn a_url_works_once_and_only_for_its_token() {
        let u = Uploads::new("http://h:8090/".into());
        let url = u.create(slot()).unwrap();
        let token = url.strip_prefix("http://h:8090/upload/").expect(&url);
        assert_eq!(token.len(), 32);
        assert_ne!(u.create(slot()).unwrap(), url, "tokens are random");
        assert_eq!(u.take("nope"), None);
        assert_eq!(u.take(token), Some(slot()));
        assert_eq!(u.take(token), None, "single use");
    }

    #[test]
    fn expired_slots_are_refused_and_pruned() {
        let u = Uploads::new("http://h".into());
        let old = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        u.slots.lock().unwrap().insert("old".into(), (slot(), old));
        assert_eq!(u.take("old"), None);
        u.slots.lock().unwrap().insert("old".into(), (slot(), old));
        u.create(slot()).unwrap();
        assert!(!u.slots.lock().unwrap().contains_key("old"));
    }
}
