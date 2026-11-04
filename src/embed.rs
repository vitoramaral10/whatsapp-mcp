//! Message embeddings for semantic search, from an OpenAI-compatible
//! /v1/embeddings server (llama.cpp's llama-server on the internal network).
//! A background loop gives every message with text a vector, newest first.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use log::{info, warn};
use serde_json::{json, Value};

use crate::store::Store;

const BATCH: usize = 32;
/// Characters per message sent to the model (about 500 tokens, under the
/// server's 1024-token batch); longer text is cut, not split.
const MAX_CHARS: usize = 1500;

pub struct Embedder {
    http: reqwest::Client,
    url: String,
    model: String,
}

impl Embedder {
    pub fn new(url: String, model: String) -> Arc<Self> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("http client");
        Arc::new(Self { http, url: url.trim_end_matches('/').to_string(), model })
    }

    /// Unit-length vectors, one per input, in order.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts.iter().map(|t| t.chars().take(MAX_CHARS).collect()).collect();
        let resp: Value = self
            .http
            .post(format!("{}/v1/embeddings", self.url))
            .json(&json!({ "model": self.model, "input": input }))
            .send()
            .await
            .context("embedding server unreachable")?
            .error_for_status()?
            .json()
            .await?;
        let mut data: Vec<(u64, Vec<f32>)> = resp["data"]
            .as_array()
            .context("embedding response has no data")?
            .iter()
            .map(|d| {
                let v: Vec<f32> = d["embedding"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                (d["index"].as_u64().unwrap_or(0), normalize(v))
            })
            .collect();
        if data.len() != texts.len() || data.iter().any(|(_, v)| v.is_empty()) {
            bail!("embedding server returned {} vectors for {} texts", data.len(), texts.len());
        }
        data.sort_by_key(|(i, _)| *i);
        Ok(data.into_iter().map(|(_, v)| v).collect())
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text.to_string()]).await?.remove(0))
    }

    /// Runs for the life of the process. Idle polls are cheap: one indexed query.
    pub async fn backfill(self: Arc<Self>, store: Arc<Store>) {
        let mut done = 0usize;
        loop {
            let rows = match store.unembedded(BATCH) {
                Ok(rows) => rows,
                Err(e) => {
                    warn!("embedding backfill: {e:#}");
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    continue;
                }
            };
            if rows.is_empty() {
                if done > 0 {
                    info!("embedding backfill: {done} messages embedded, index up to date");
                    done = 0;
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            }
            let texts: Vec<String> = rows.iter().map(|(_, t)| t.clone()).collect();
            match self.embed(&texts).await {
                Ok(vecs) => {
                    let pairs: Vec<(i64, Vec<f32>)> = rows.iter().map(|(id, _)| *id).zip(vecs).collect();
                    if let Err(e) = store.put_vectors(&pairs) {
                        warn!("embedding backfill: {e:#}");
                    }
                    done += pairs.len();
                    if done % 2048 < BATCH {
                        info!("embedding backfill: {done} messages so far");
                    }
                }
                // The server is up but refuses this batch: embed one by one and
                // mark the rows it rejects with an empty vector, so they are skipped.
                Err(e) if self.embed_one("ok").await.is_ok() => {
                    warn!("embedding batch rejected ({e:#}); retrying its {} messages one by one", rows.len());
                    let mut pairs = Vec::with_capacity(rows.len());
                    for (id, text) in &rows {
                        pairs.push((*id, self.embed_one(text).await.unwrap_or_default()));
                    }
                    if let Err(e) = store.put_vectors(&pairs) {
                        warn!("embedding backfill: {e:#}");
                    }
                }
                Err(e) => {
                    warn!("embedding backfill paused: {e:#}");
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
            }
        }
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}
