//! Message embeddings for semantic search, from an OpenAI-compatible
//! /v1/embeddings server, such as llama.cpp's llama-server.
//! A background loop gives every message with text a vector, newest first;
//! each vector encodes the message with the ones before it (see store::window_of).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use log::{info, warn};
use serde_json::{Value, json};

use crate::store::{Store, too_short};

const BATCH: usize = 32;
/// Characters per text sent to the model (about 500 tokens, under the
/// server's 1024-token batch); longer text is cut, not split.
const MAX_CHARS: usize = 1500;
/// Backfill batches can queue behind each other on a busy server.
const BATCH_TIMEOUT: Duration = Duration::from_secs(120);
/// A search waits on the query vector; past this it goes full-text only.
const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Embedder {
    http: reqwest::Client,
    url: String,
    model: String,
}

impl Embedder {
    pub fn new(url: String, model: String) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(3)).build().expect("http client");
        Arc::new(Self { http, url: url.trim_end_matches('/').to_string(), model })
    }

    /// Unit-length vectors, one per input, in order.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.request(texts, BATCH_TIMEOUT).await
    }

    async fn request(&self, texts: &[String], timeout: Duration) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts.iter().map(|t| t.chars().take(MAX_CHARS).collect()).collect();
        let resp: Value = self
            .http
            .post(format!("{}/v1/embeddings", self.url))
            .timeout(timeout)
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

    /// The query vector of a search, on the short timeout.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.request(&[text.to_string()], QUERY_TIMEOUT).await?.remove(0))
    }

    /// Runs for the life of the process. Each idle poll is one query that
    /// walks every message looking for one without a vector: a full scan,
    /// cheap at tens of thousands of messages, every 30 seconds.
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
            // Windows too short for a vector get the empty "unembeddable" mark.
            let texts: Vec<String> = rows.iter().filter(|(_, t)| !too_short(t)).map(|(_, t)| t.clone()).collect();
            let (vecs, mut pause) = if texts.is_empty() {
                (Vec::new(), false)
            } else {
                match self.embed(&texts).await {
                    Ok(vecs) => (vecs.into_iter().map(Some).collect(), false),
                    Err(e) if self.embed_one("ok").await.is_ok() => {
                        warn!("embedding batch rejected ({e:#}); retrying its {} messages one by one", texts.len());
                        self.one_by_one(&texts).await
                    }
                    Err(e) => {
                        warn!("embedding backfill paused: {e:#}");
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        continue;
                    }
                }
            };
            let mut vecs = vecs.into_iter();
            let triples: Vec<(i64, &str, Vec<f32>)> = rows
                .iter()
                .filter_map(|(id, t)| {
                    let v = if too_short(t) { Some(Vec::new()) } else { vecs.next().flatten() };
                    Some((*id, t.as_str(), v?))
                })
                .collect();
            match store.put_vectors(&triples) {
                Ok(saved) => {
                    let before = done;
                    done += saved;
                    if done / 2048 > before / 2048 {
                        info!("embedding backfill: {done} messages so far");
                    }
                }
                Err(e) => {
                    warn!("embedding backfill: saving vectors failed: {e:#}");
                    pause = true;
                }
            }
            if pause {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        }
    }
}

impl Embedder {
    /// The server is up but refuses a batch: embeds its texts one by one. A
    /// row it rejects (4xx) gets an empty vector, so it is skipped from now on;
    /// so does a row it fails on (5xx) when a later row goes through, which
    /// shows the server is fine and the row is the problem. None is a row left
    /// for later; the bool asks the caller to pause, after a transport error
    /// or when failures were not followed by a success.
    async fn one_by_one(&self, texts: &[String]) -> (Vec<Option<Vec<f32>>>, bool) {
        let mut vecs: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut failed = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            match self.embed(std::slice::from_ref(text)).await {
                Ok(mut v) => {
                    vecs[i] = Some(v.remove(0));
                    failed.drain(..).for_each(|j: usize| vecs[j] = Some(Vec::new()));
                }
                Err(e) => match status(&e) {
                    Some(s) if s.is_client_error() => vecs[i] = Some(Vec::new()),
                    Some(_) => failed.push(i),
                    None => {
                        warn!("embedding backfill paused: {e:#}");
                        return (vecs, true);
                    }
                },
            }
        }
        if !failed.is_empty() {
            warn!("embedding backfill paused: the server failed on the last {} messages of the batch", failed.len());
        }
        (vecs, !failed.is_empty())
    }
}

/// The HTTP status of a reply the server did send.
fn status(e: &anyhow::Error) -> Option<reqwest::StatusCode> {
    e.downcast_ref::<reqwest::Error>().and_then(reqwest::Error::status)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, http::StatusCode, routing::post};

    /// An embedding server that fails (500) on any text with "bad" and
    /// refuses (400) any with "huge".
    async fn server() -> String {
        async fn embed(Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
            let input: Vec<String> = serde_json::from_value(body["input"].clone()).unwrap();
            if input.iter().any(|t| t.contains("bad")) {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({})));
            }
            if input.iter().any(|t| t.contains("huge")) {
                return (StatusCode::BAD_REQUEST, Json(json!({})));
            }
            let data: Vec<Value> = (0..input.len()).map(|i| json!({ "index": i, "embedding": [1.0, 0.0] })).collect();
            (StatusCode::OK, Json(json!({ "data": data })))
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, Router::new().route("/v1/embeddings", post(embed))).await });
        url
    }

    #[tokio::test]
    async fn a_row_the_server_fails_on_is_skipped_once_a_later_row_works() {
        let e = Embedder::new(server().await, "m".into());
        let texts = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let (v, pause) = e.one_by_one(&texts(&["bad", "huge", "ok"])).await;
        assert_eq!(v, vec![Some(vec![]), Some(vec![]), Some(vec![1.0, 0.0])]);
        assert!(!pause);
        // Nothing after the failure worked: kept for later, and the loop pauses.
        let (v, pause) = e.one_by_one(&texts(&["ok", "bad", "huge"])).await;
        assert_eq!(v, vec![Some(vec![1.0, 0.0]), None, Some(vec![])]);
        assert!(pause);
    }
}
