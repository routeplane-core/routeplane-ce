//! Black-box acceptance suite — ADR-049 / PRD-037 (tracked: routeplane-core/docs#120,
//! impl routeplane#136).
//!
//! This suite runs against a **deployed** Routeplane gateway over real HTTP — it
//! is NOT the in-process / wiremock integration suite. It is the `routeplane`-owned
//! half of the staging acceptance gate (ADR-049 §8c): the same suite runs report-only
//! against the live **dev** env now (its raw ACA FQDN) and becomes the enforcing
//! staging→prod gate once staging go-live arms it.
//!
//! ## Gating (so `cargo test --all` / the commit-stage CI stay green)
//! Every test no-ops with a loud `SKIP` unless `ACCEPTANCE_BASE_URL` is set:
//! - `ACCEPTANCE_BASE_URL` — the deployed gateway, e.g. `https://<dev>.azurecontainerapps.io`.
//! - `ACCEPTANCE_API_KEY` — a scoped `rp_` test key (the `x-routeplane-api-key`).
//! - `ACCEPTANCE_COMPLETIONS=1` — enable the provider-dependent tiers (real 200 + SSE); needs a working provider key on the target, so it is OFF for a local run with no provider configured.
//! - `ACCEPTANCE_EDGE=1` — enable the Cloudflare edge/ingress tier (Cloudflare-fronted envs only; never dev, which answers on its raw ACA FQDN).
//! - `ACCEPTANCE_MODEL` — model id for the completion tiers (default `gpt-4o-mini`).
//!
//! Run via `just acceptance` (see the justfile).

use std::time::Duration;

const BASE_ENV: &str = "ACCEPTANCE_BASE_URL";

/// The deployed gateway base URL (trailing slash trimmed), or `None` when unset.
fn base_url() -> Option<String> {
    std::env::var(BASE_ENV)
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
}

fn api_key() -> String {
    std::env::var("ACCEPTANCE_API_KEY").unwrap_or_default()
}

fn model() -> String {
    std::env::var("ACCEPTANCE_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string())
}

/// A `1`/`true` env flag.
fn flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn skip(test: &str, why: &str) {
    eprintln!("SKIP {test}: {why}");
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build reqwest client")
}

/// Resolve the base URL or no-op the test (loud SKIP). Keeps the suite inert in
/// the commit stage where there is no deployed target.
macro_rules! base_or_skip {
    ($name:literal) => {
        match base_url() {
            Some(b) => b,
            None => {
                skip(
                    $name,
                    concat!(
                        "set ",
                        "ACCEPTANCE_BASE_URL",
                        " to run against a deployed gateway"
                    ),
                );
                return;
            }
        }
    };
}

/// A minimal OpenAI-shaped chat request.
fn chat_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": model(),
        "messages": [{ "role": "user", "content": "Reply with the single word: pong" }],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 — Smoke / health + fail-closed auth. No provider required → this tier
// runs green against any live gateway, including a local instance with no
// provider key configured.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn smoke_healthz_ok() {
    let base = base_or_skip!("smoke_healthz_ok");
    let resp = client()
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("GET /healthz");
    assert!(
        resp.status().is_success(),
        "/healthz status {}",
        resp.status()
    );
    let body = resp.text().await.expect("read /healthz body");
    assert_eq!(body.trim(), "OK", "/healthz body");
}

#[tokio::test]
async fn smoke_metrics_ok() {
    let base = base_or_skip!("smoke_metrics_ok");
    let resp = client()
        .get(format!("{base}/metrics"))
        .send()
        .await
        .expect("GET /metrics");
    assert!(
        resp.status().is_success(),
        "/metrics status {}",
        resp.status()
    );
    // /metrics is Prometheus text exposition (ADR-025), not JSON. Assert the
    // load-shed counter is exposed — `rp_shed_total`, with a `shed_total`
    // back-compat alias (so `contains("shed_total")` matches either).
    let body = resp.text().await.expect("read /metrics body");
    assert!(
        body.contains("shed_total"),
        "/metrics must expose the shed_total counter (Prometheus text); got: {}",
        body.chars().take(200).collect::<String>()
    );
}

#[tokio::test]
async fn smoke_root_banner() {
    let base = base_or_skip!("smoke_root_banner");
    let resp = client()
        .get(format!("{base}/"))
        .send()
        .await
        .expect("GET /");
    assert!(resp.status().is_success(), "/ status {}", resp.status());
}

#[tokio::test]
async fn auth_rejects_missing_key() {
    let base = base_or_skip!("auth_rejects_missing_key");
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST without key");
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::UNAUTHORIZED,
        "missing key must be 401 (fail-closed), got {}",
        resp.status()
    );
}

#[tokio::test]
async fn auth_rejects_bad_key() {
    let base = base_or_skip!("auth_rejects_bad_key");
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-routeplane-api-key", "rp_definitely_not_a_valid_key")
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST with bad key");
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::UNAUTHORIZED,
        "bad key must be 401 (fail-closed), got {}",
        resp.status()
    );
}

/// A valid `rp_` key must clear auth — proven without a working provider by
/// asserting the response is anything *other than* 401 (a provider error is a
/// later concern; auth is what this asserts).
#[tokio::test]
async fn auth_accepts_valid_key() {
    let base = base_or_skip!("auth_accepts_valid_key");
    let key = api_key();
    if key.is_empty() {
        skip(
            "auth_accepts_valid_key",
            "set ACCEPTANCE_API_KEY to assert a valid key clears auth",
        );
        return;
    }
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-routeplane-api-key", &key)
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST with valid key");
    assert_ne!(
        resp.status(),
        reqwest::StatusCode::UNAUTHORIZED,
        "a valid rp_ key must clear auth (got 401)"
    );
}

/// FR-13 (PRD-011): `/v1/responses` is a deliberate decline. A valid key must
/// get a **501 `endpoint_not_supported`** pointer-bearing decline (not the bare
/// unknown-route 404). No provider required — the 501 returns before any upstream
/// call.
#[tokio::test]
async fn responses_endpoint_declines_501() {
    let base = base_or_skip!("responses_endpoint_declines_501");
    let key = api_key();
    if key.is_empty() {
        skip(
            "responses_endpoint_declines_501",
            "set ACCEPTANCE_API_KEY to assert the /v1/responses decline past auth",
        );
        return;
    }
    let resp = client()
        .post(format!("{base}/v1/responses"))
        .header("x-routeplane-api-key", &key)
        .json(&serde_json::json!({ "model": model(), "input": "hello" }))
        .send()
        .await
        .expect("POST /v1/responses");
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_IMPLEMENTED,
        "/v1/responses must be a 501 decline, got {}",
        resp.status()
    );
    let v: serde_json::Value = resp.json().await.expect("/v1/responses JSON");
    assert_eq!(
        v.pointer("/error/code").and_then(serde_json::Value::as_str),
        Some("endpoint_not_supported"),
        "error.code must be endpoint_not_supported, got {v}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 2 — OpenAI-compat contract + E2E happy-path. Provider-dependent: gated on
// ACCEPTANCE_COMPLETIONS=1 (a working provider on the target). Asserts the wire
// contract an OpenAI SDK hard-depends on.
// ─────────────────────────────────────────────────────────────────────────────

/// Common precondition for the completion tiers: base + key + the COMPLETIONS
/// flag. Returns `(base, key)` or signals skip.
fn completion_ctx(test: &str) -> Option<(String, String)> {
    let base = base_url()?;
    if !flag("ACCEPTANCE_COMPLETIONS") {
        skip(test, "set ACCEPTANCE_COMPLETIONS=1 (+ a working provider on the target) to run the provider tiers");
        return None;
    }
    let key = api_key();
    if key.is_empty() {
        skip(test, "set ACCEPTANCE_API_KEY for the completion tiers");
        return None;
    }
    Some((base, key))
}

#[tokio::test]
async fn contract_chat_completion_shape() {
    let Some((base, key)) = completion_ctx("contract_chat_completion_shape") else {
        return;
    };
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-routeplane-api-key", &key)
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST chat completion");
    assert!(
        resp.status().is_success(),
        "completion status {}",
        resp.status()
    );

    // Branding contract: the gateway echoes which provider served the response.
    assert!(
        resp.headers().get("x-routeplane-provider").is_some(),
        "response must carry the x-routeplane-provider header"
    );

    let v: serde_json::Value = resp.json().await.expect("completion JSON");
    // The fields an OpenAI SDK hard-depends on (mirrors routeplane_types::ChatCompletionResponse).
    assert_eq!(v["object"], "chat.completion", "object field");
    assert!(v["id"].is_string(), "id must be a string");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant", "choices[0].message.role");
    assert!(
        msg["content"].is_string(),
        "choices[0].message.content must be a string"
    );

    let usage = &v["usage"];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens");
    let completion = usage["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens");
    let total = usage["total_tokens"].as_u64().expect("usage.total_tokens");
    assert!(total > 0, "usage.total_tokens must be > 0");
    assert_eq!(
        total,
        prompt + completion,
        "usage totals must be consistent"
    );
}

/// `Authorization: Bearer rp_…` is the OpenAI-SDK-compat fallback for the branded
/// `x-routeplane-api-key` (ADR-041). An SDK pointed at the gateway must work.
#[tokio::test]
async fn contract_bearer_fallback() {
    let Some((base, key)) = completion_ctx("contract_bearer_fallback") else {
        return;
    };
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST with Bearer auth");
    assert!(
        resp.status().is_success(),
        "Bearer rp_ fallback must be accepted, got {}",
        resp.status()
    );
}

/// Streaming wire contract: `text/event-stream`, OpenAI-shaped `chat.completion.chunk`
/// objects on `data:` lines, terminated by the literal `data: [DONE]`.
#[tokio::test]
async fn contract_streaming_sse() {
    let Some((base, key)) = completion_ctx("contract_streaming_sse") else {
        return;
    };
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-routeplane-api-key", &key)
        .json(&chat_body(true))
        .send()
        .await
        .expect("POST streaming completion");
    assert!(
        resp.status().is_success(),
        "stream status {}",
        resp.status()
    );

    // Branding contract: the streamed response echoes the serving provider too
    // (parity with contract_chat_completion_shape).
    assert!(
        resp.headers().get("x-routeplane-provider").is_some(),
        "streamed response must carry the x-routeplane-provider header"
    );

    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ctype.starts_with("text/event-stream"),
        "streaming Content-Type must be text/event-stream, got {ctype:?}"
    );

    let body = resp.text().await.expect("read SSE body");
    let mut saw_chunk = false;
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            continue;
        }
        let chunk: serde_json::Value =
            serde_json::from_str(payload).expect("each data: line must be a JSON chunk");
        assert_eq!(chunk["object"], "chat.completion.chunk", "SSE chunk object");
        saw_chunk = true;
    }
    assert!(
        saw_chunk,
        "expected at least one chat.completion.chunk before [DONE]"
    );
    assert!(
        body.contains("data: [DONE]"),
        "SSE stream must terminate with data: [DONE]"
    );
}

/// E2E happy-path: a real completion returns a non-empty assistant message — the
/// user-visible value, end to end through the gateway and a real provider.
#[tokio::test]
async fn e2e_completion_nonempty() {
    let Some((base, key)) = completion_ctx("e2e_completion_nonempty") else {
        return;
    };
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-routeplane-api-key", &key)
        .json(&chat_body(false))
        .send()
        .await
        .expect("POST e2e completion");
    assert!(resp.status().is_success(), "e2e status {}", resp.status());
    let v: serde_json::Value = resp.json().await.expect("e2e JSON");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        !content.trim().is_empty(),
        "assistant message content must be non-empty"
    );
}

/// Optional load signal: a small concurrent burst must all succeed — surfaces
/// gross regressions and feeds (not replaces) the prod soak decision.
#[tokio::test]
async fn probe_concurrent_completions() {
    let Some((base, key)) = completion_ctx("probe_concurrent_completions") else {
        return;
    };
    let mut handles = Vec::new();
    for _ in 0..5 {
        let (base, key) = (base.clone(), key.clone());
        handles.push(tokio::spawn(async move {
            client()
                .post(format!("{base}/v1/chat/completions"))
                .header("x-routeplane-api-key", &key)
                .json(&chat_body(false))
                .send()
                .await
                .map(|r| r.status())
        }));
    }
    for h in handles {
        let status = h.await.expect("join task").expect("concurrent request");
        assert!(
            status.is_success(),
            "concurrent completion failed: {status}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge / ingress tier — env-scoped (ACCEPTANCE_EDGE=1), only where Cloudflare
// fronts the gateway. NOT run on dev (raw ACA FQDN, no proxied gateway host yet);
// first runs at staging once a proxied hostname + ingress lockdown exist.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn edge_cloudflare_fronting() {
    let base = base_or_skip!("edge_cloudflare_fronting");
    if !flag("ACCEPTANCE_EDGE") {
        skip(
            "edge_cloudflare_fronting",
            "set ACCEPTANCE_EDGE=1 on a Cloudflare-fronted env (never dev)",
        );
        return;
    }
    assert!(
        base.starts_with("https://"),
        "edge tier expects an https base URL, got {base}"
    );
    let resp = client()
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("GET /healthz via edge");
    assert!(
        resp.status().is_success(),
        "edge /healthz status {}",
        resp.status()
    );
    assert!(
        resp.headers().get("cf-ray").is_some(),
        "a Cloudflare-fronted response must carry the cf-ray header"
    );
}
