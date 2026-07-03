//! Zero-work OpenAI-compatible mock upstream.
//!
//! Purpose: isolate **gateway overhead** in perf runs. A load generator pointed
//! directly at this binary establishes the floor; the same load pointed at a
//! Routeplane gateway whose `self_hosted` provider targets this binary measures
//! floor + gateway. The delta is the gateway's added cost — see
//! `benchmarks/perf/README.md` for the pinned methodology.
//!
//! Per request the handler does the minimum a faithful upstream must do:
//! consume the request body (so the gateway's write is not back-pressured) and
//! return a static, plausible chat-completion JSON with a `usage` object.
//! No parsing, no allocation beyond axum's framing — deliberately ~zero work,
//! so the mock never becomes the bottleneck being measured.

use axum::body::Bytes;
use axum::http::header::CONTENT_TYPE;
use axum::routing::{get, post};
use axum::Router;

/// Static, plausible OpenAI-shaped completion (same shape the adapter test
/// fixtures use in `crates/routeplane/tests/common/mod.rs`).
const CHAT_BODY: &str = r#"{
  "id": "chatcmpl-mock-upstream-000",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "mock-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Acknowledged. This is a static mock completion used for gateway overhead benchmarking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 128, "completion_tokens": 17, "total_tokens": 145 }
}"#;

async fn chat_completions(_body: Bytes) -> impl axum::response::IntoResponse {
    ([(CONTENT_TYPE, "application/json")], CHAT_BODY)
}

async fn healthz() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("MOCK_UPSTREAM_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(9100);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/healthz", get(healthz));

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("mock-upstream: cannot bind {addr}: {e}"));
    println!("mock-upstream listening on http://{addr} (POST /v1/chat/completions)");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("mock-upstream server error");
}
