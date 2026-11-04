//! Voice-note transcription through a whisper.cpp server.
//!
//! One run takes ~25 s on the homelab GPU, longer than the gateway's per-call
//! deadline, so it runs in the background and the text is cached next to the
//! media file as `<file>.txt`. A call waits `wait` and otherwise reports the job
//! as pending; a later call reads the cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use tokio::sync::{watch, Semaphore};

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Text(String),
    Pending,
    Error(String),
}

type Job = watch::Receiver<Option<Result<String, String>>>;

pub struct Transcriber {
    url: String,
    wait: Duration,
    http: reqwest::Client,
    jobs: Mutex<HashMap<PathBuf, Job>>,
    // whisper-server runs one request at a time anyway.
    slot: Arc<Semaphore>,
}

impl Transcriber {
    pub fn new(url: String, wait: Duration, timeout: Duration) -> Arc<Self> {
        let http = reqwest::Client::builder().timeout(timeout).build().expect("http client");
        Arc::new(Self { url, wait, http, jobs: Mutex::new(HashMap::new()), slot: Arc::new(Semaphore::new(1)) })
    }

    pub async fn transcription(self: &Arc<Self>, file: &Path) -> Outcome {
        let cache = cache_path(file);
        if let Ok(text) = tokio::fs::read_to_string(&cache).await {
            return Outcome::Text(text);
        }
        let mut rx = {
            let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            jobs.entry(file.to_path_buf()).or_insert_with(|| self.spawn(file.to_path_buf())).clone()
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

    fn spawn(self: &Arc<Self>, file: PathBuf) -> Job {
        let (tx, rx) = watch::channel(None);
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let result = this.run(&file).await.map_err(|e| format!("{e:#}"));
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
        let text = body["text"].as_str().ok_or_else(|| anyhow!("transcription failed: no text in reply"))?.trim().to_string();
        let cache = cache_path(file);
        let tmp = cache.with_extension("txt.tmp");
        tokio::fs::write(&tmp, &text).await?;
        tokio::fs::rename(&tmp, &cache).await?;
        Ok(text)
    }
}

fn cache_path(file: &Path) -> PathBuf {
    let mut p = file.as_os_str().to_owned();
    p.push(".txt");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sits_next_to_the_file() {
        assert_eq!(cache_path(Path::new("/s/a.ogg")), PathBuf::from("/s/a.ogg.txt"));
    }

    #[tokio::test]
    async fn cached_text_is_returned_without_a_server() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.ogg");
        std::fs::write(&f, b"x").unwrap();
        std::fs::write(dir.path().join("a.ogg.txt"), "olá").unwrap();
        let t = Transcriber::new("http://127.0.0.1:9".into(), Duration::from_millis(50), Duration::from_secs(1));
        assert_eq!(t.transcription(&f).await, Outcome::Text("olá".into()));
    }

    #[tokio::test]
    async fn bad_audio_reports_an_error_and_is_retried() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.ogg");
        std::fs::write(&f, b"not audio").unwrap();
        let t = Transcriber::new("http://127.0.0.1:9".into(), Duration::from_secs(5), Duration::from_secs(1));
        assert!(matches!(t.transcription(&f).await, Outcome::Error(e) if e.contains("convert")));
        tokio::task::yield_now().await;
        assert!(t.jobs.lock().unwrap().is_empty());
    }
}
