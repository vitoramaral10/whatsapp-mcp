//! Voice-note transcription through a whisper.cpp server.
//!
//! One run takes ~25 s on the homelab GPU, longer than the gateway's per-call
//! deadline, so it runs in the background and the text is saved as the text of
//! the message, where the searches find it. A call waits `wait` and otherwise
//! reports the job as pending; a later call reads the saved text.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use tokio::sync::{watch, Semaphore};

use crate::store::Store;

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Text(String),
    Pending,
    Error(String),
}

type Job = watch::Receiver<Option<Result<String, String>>>;

pub struct Transcriber {
    url: String,
    store: Arc<Store>,
    wait: Duration,
    http: reqwest::Client,
    jobs: Mutex<HashMap<PathBuf, Job>>,
    // whisper-server runs one request at a time anyway.
    slot: Arc<Semaphore>,
}

impl Transcriber {
    pub fn new(url: String, store: Arc<Store>, wait: Duration, timeout: Duration) -> Arc<Self> {
        let http = reqwest::Client::builder().timeout(timeout).build().expect("http client");
        Arc::new(Self { url, store, wait, http, jobs: Mutex::new(HashMap::new()), slot: Arc::new(Semaphore::new(1)) })
    }

    /// The transcript of voice note `id` of `chat_jid`, stored at `file`.
    pub async fn transcription(self: &Arc<Self>, id: &str, chat_jid: &str, file: &Path) -> Outcome {
        if let Some(text) = self.store.transcription(id, chat_jid) {
            return Outcome::Text(text);
        }
        // Earlier versions cached the text next to the file.
        let legacy = legacy_cache(file);
        if let Ok(text) = tokio::fs::read_to_string(&legacy).await {
            if self.store.set_transcription(id, chat_jid, &text).is_ok() {
                let _ = tokio::fs::remove_file(&legacy).await;
            }
            return Outcome::Text(text);
        }
        let mut rx = {
            let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            jobs.entry(file.to_path_buf()).or_insert_with(|| self.spawn(id.to_string(), chat_jid.to_string(), file.to_path_buf())).clone()
        };
        let done = tokio::time::timeout(self.wait, rx.wait_for(|r| r.is_some())).await;
        match done {
            Ok(Ok(result)) => match result.clone().expect("checked by wait_for") {
                Ok(text) => Outcome::Text(text),
                Err(e) => Outcome::Error(e),
            },
            Ok(Err(_)) => Outcome::Error("transcription task ended without a result".into()),
            Err(_) => Outcome::Pending,
        }
    }

    fn spawn(self: &Arc<Self>, id: String, chat_jid: String, file: PathBuf) -> Job {
        let (tx, rx) = watch::channel(None);
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let result = this.run(&file).await.and_then(|text| {
                this.store.set_transcription(&id, &chat_jid, &text)?;
                Ok(text)
            });
            let result = result.map_err(|e| format!("{e:#}"));
            // Errors are not cached, so the next call retries.
            this.jobs.lock().unwrap_or_else(|e| e.into_inner()).remove(&file);
            let _ = tx.send(Some(result));
        });
        rx
    }

    async fn run(&self, file: &Path) -> Result<String> {
        let _slot = self.slot.acquire().await?;
        let data = tokio::fs::read(file).await.with_context(|| format!("read {}", file.display()))?;
        let ext = file.extension().and_then(|e| e.to_str()).map(str::to_owned);
        let wav = tokio::task::spawn_blocking(move || crate::audio::whisper_wav(&data, ext.as_deref()))
            .await?
            .context("failed to convert audio to wav")?;
        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(wav).file_name("audio.wav").mime_str("audio/wav")?)
            .text("response_format", "json");
        let resp = self
            .http
            .post(format!("{}/inference", self.url.trim_end_matches('/')))
            .multipart(form)
            .send()
            .await
            .context("transcription failed")?
            .error_for_status()
            .context("transcription failed")?;
        let body: serde_json::Value = resp.json().await.context("transcription failed")?;
        Ok(body["text"].as_str().ok_or_else(|| anyhow!("transcription failed: no text in reply"))?.trim().to_string())
    }
}

fn legacy_cache(file: &Path) -> PathBuf {
    let mut p = file.as_os_str().to_owned();
    p.push(".txt");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Media;
    use chrono::Utc;

    fn setup(dir: &Path, wait: Duration) -> (Arc<Store>, Arc<Transcriber>, PathBuf) {
        let store = Arc::new(Store::open(&dir.join("messages.db")).unwrap());
        store.store_chat("55@s.whatsapp.net", "Ana", Utc::now()).unwrap();
        store.store_message("V", "55@s.whatsapp.net", "55", "", Utc::now(), false, &Media { kind: "audio".into(), ..Default::default() }).unwrap();
        let t = Transcriber::new("http://127.0.0.1:9".into(), Arc::clone(&store), wait, Duration::from_secs(1));
        (store, t, dir.join("a.ogg"))
    }

    #[tokio::test]
    async fn a_file_cache_of_an_earlier_version_moves_into_the_history() {
        let dir = tempfile::tempdir().unwrap();
        let (store, t, f) = setup(dir.path(), Duration::from_millis(50));
        std::fs::write(&f, b"x").unwrap();
        std::fs::write(dir.path().join("a.ogg.txt"), "olá").unwrap();
        assert_eq!(t.transcription("V", "55@s.whatsapp.net", &f).await, Outcome::Text("olá".into()));
        assert!(!dir.path().join("a.ogg.txt").exists());
        assert_eq!(store.transcription("V", "55@s.whatsapp.net").as_deref(), Some("olá"));
        assert_eq!(t.transcription("V", "55@s.whatsapp.net", &f).await, Outcome::Text("olá".into()));
    }

    #[tokio::test]
    async fn bad_audio_reports_an_error_and_is_retried() {
        let dir = tempfile::tempdir().unwrap();
        let (store, t, f) = setup(dir.path(), Duration::from_secs(5));
        std::fs::write(&f, b"not audio").unwrap();
        assert!(matches!(t.transcription("V", "55@s.whatsapp.net", &f).await, Outcome::Error(e) if e.contains("convert")));
        tokio::task::yield_now().await;
        assert!(t.jobs.lock().unwrap().is_empty());
        assert_eq!(store.transcription("V", "55@s.whatsapp.net"), None);
    }
}
