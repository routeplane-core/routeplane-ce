//! Hermetic integration tests: the runtime custom-provider registry resolves on
//! EVERY inference endpoint, not just `/v1/chat/completions` (findings #2/#3).
//!
//! Same composition as `custom_providers_integration.rs`: handlers are invoked
//! directly with the extensions the auth middleware would inject; the upstream
//! is a real wiremock OpenAI-compatible server whose matchers ENFORCE the
//! registered `Authorization: Bearer` key, so each test proves the full loop —
//! lock-free ArcSwap resolution → `SelfHostedProvider` egress with the
//! registered key → the usage ring records the custom provider's name.

mod common;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use common::build_stub_state;
use routeplane::api_error::OpenAiJson;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::custom_providers::CustomProviderConfig;
use routeplane::proxy::AppState;
use routeplane_entitlements::{CapabilitySet, Tier};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CUSTOM_KEY: &str = "sk-custom-endpoints-7777";

fn vk() -> VirtualKey {
    // Deliberately NO "myvllm" entry: the custom provider's REGISTERED key must
    // be the one reaching the upstream (the wiremock matcher enforces it).
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "openai": "test-api-key" }
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    TenantContext {
        tenant_id: "t_test".into(),
        tier: Tier::Standard,
        capabilities: CapabilitySet::resolve(
            Tier::Standard,
            &std::collections::BTreeSet::new(),
            &std::collections::BTreeSet::new(),
        ),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

/// Register a custom provider straight on the (ephemeral) store — the same
/// snapshot swap the `/v1/providers` handler performs after validation.
async fn register(state: &Arc<AppState>, name: &str, base_url: &str, models: &[&str]) {
    state
        .custom_providers
        .upsert(CustomProviderConfig {
            name: name.into(),
            base_url: base_url.into(),
            api_key: CUSTOM_KEY.into(),
            models: models.iter().map(|m| m.to_string()).collect(),
            stream_include_usage: None,
            created_at: None,
        })
        .await
        .expect("provider registers");
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Poll the observability ring (its writer drains asynchronously) until an
/// event for `provider` appears, bounded.
async fn await_ring_provider(state: &Arc<AppState>, provider: &str) {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if state
            .observability_engine
            .get_recent_events()
            .iter()
            .any(|e| e.provider == provider)
        {
            return;
        }
        if Instant::now() > deadline {
            panic!("no usage event recorded for provider {provider}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// --- /v1/embeddings -------------------------------------------------------------

async fn drive_embeddings(state: &Arc<AppState>, headers: HeaderMap, model: &str) -> Response {
    let payload: routeplane_types::EmbeddingRequest = serde_json::from_value(json!({
        "model": model,
        "input": "hello"
    }))
    .expect("embedding request");
    routeplane::embeddings::embeddings(
        State(state.clone()),
        axum::Extension(vk()),
        axum::Extension(ctx()),
        headers,
        OpenAiJson(payload),
    )
    .await
    .into_response()
}

#[tokio::test]
async fn embeddings_route_by_model_id_hits_custom_upstream_with_bearer_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "custom-embed",
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        })))
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "myvllm", &server.uri(), &["custom-embed"]).await;

    // Header-less: the model id resolves through the runtime registry.
    let resp = drive_embeddings(&state, HeaderMap::new(), "custom-embed").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["model"], "custom-embed");
    assert_eq!(v["data"][0]["embedding"][0], 0.1);

    // UsageEvent.provider = the custom name, exactly like chat.
    await_ring_provider(&state, "myvllm").await;
}

#[tokio::test]
async fn embeddings_route_by_explicit_provider_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "model": "any-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "myvllm", &server.uri(), &["any-embed"]).await;

    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", "myvllm".parse().expect("header"));
    let resp = drive_embeddings(&state, headers, "some-other-model").await;
    assert_eq!(resp.status(), StatusCode::OK);
    await_ring_provider(&state, "myvllm").await;
}

#[tokio::test]
async fn embeddings_upstream_error_surfaces_cleanly_not_a_panic() {
    // Faithful passthrough: a chat-only custom model hitting /v1/embeddings gets
    // the UPSTREAM's error, surfaced as the gateway's clean all-failed envelope
    // (no fabricated capability tracking).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(400).set_body_string("model does not embed"))
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "myvllm", &server.uri(), &["chat-only-model"]).await;

    let resp = drive_embeddings(&state, HeaderMap::new(), "chat-only-model").await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "upstream_error");
}

#[tokio::test]
async fn embeddings_default_path_is_unchanged_without_a_registration() {
    // Empty registry: a non-catalog model still defaults to openai (the stub,
    // which has no embeddings override) → the typed 422, byte-identical.
    let state = build_stub_state();
    let resp = drive_embeddings(&state, HeaderMap::new(), "custom-embed").await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "embeddings_not_supported");
}

// --- /v1/messages (Anthropic-native inbound) -------------------------------------

async fn drive_messages(state: &Arc<AppState>, headers: HeaderMap, model: &str) -> Response {
    let payload: routeplane::messages_api::AnthropicMessagesRequest =
        serde_json::from_value(json!({
            "model": model,
            "max_tokens": 64,
            "messages": [ { "role": "user", "content": "hello" } ]
        }))
        .expect("anthropic request");
    routeplane::messages_api::messages(
        State(state.clone()),
        axum::Extension(vk()),
        axum::Extension(ctx()),
        axum::Extension(TenantGuardrails(None)),
        headers,
        OpenAiJson(payload),
    )
    .await
    .into_response()
}

#[tokio::test]
async fn messages_routes_custom_model_through_openai_compatible_translation() {
    // The Anthropic-native inbound is translated to the canonical shape, egresses
    // through SelfHostedProvider to the custom provider's OpenAI-compatible
    // /v1/chat/completions (Bearer = the REGISTERED key), and the OpenAI-shaped
    // reply is translated back OUTBOUND to the Anthropic Messages shape.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-custom-msg",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "local-claude",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from custom upstream"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
        })))
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "myvllm", &server.uri(), &["local-claude"]).await;

    // Header-less: the custom model must NOT be defaulted to `anthropic` — the
    // core's model-index routing resolves it to the custom provider.
    let resp = drive_messages(&state, HeaderMap::new(), "local-claude").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["type"], "message", "response must be Anthropic-shaped");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "hello from custom upstream");
    assert_eq!(v["usage"]["input_tokens"], 7);
    assert_eq!(v["usage"]["output_tokens"], 4);

    await_ring_provider(&state, "myvllm").await;
}

#[tokio::test]
async fn messages_without_custom_mapping_still_defaults_to_anthropic() {
    // Empty registry: the synthetic `x-routeplane-provider: anthropic` default is
    // byte-identical. The stub state registers only "openai", so the attempt
    // fails with the all-failed envelope — proving the target was anthropic,
    // not a custom provider and not the openai default.
    let state = build_stub_state();
    let resp = drive_messages(&state, HeaderMap::new(), "claude-3-5-sonnet").await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "upstream_error");
}

// --- /v1/rerank -------------------------------------------------------------------

#[tokio::test]
async fn rerank_routes_custom_model_with_bearer_key_and_records_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "custom-reranker",
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1}
            ],
            "usage": {"search_units": 1, "total_tokens": 1}
        })))
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "myreranker", &server.uri(), &["custom-reranker"]).await;

    let payload: routeplane_types::RerankRequest = serde_json::from_value(json!({
        "model": "custom-reranker",
        "query": "which doc",
        "documents": ["doc a", "doc b"]
    }))
    .expect("rerank request");
    // Header-less: the model id resolves through the runtime registry (default
    // would otherwise be cohere, which the stub state does not register).
    let resp = routeplane::rerank_api::rerank(
        State(state.clone()),
        axum::Extension(vk()),
        axum::Extension(ctx()),
        HeaderMap::new(),
        OpenAiJson(payload),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["results"][0]["index"], 1);

    await_ring_provider(&state, "myreranker").await;
}

// --- /v1/audio/speech ---------------------------------------------------------------

#[tokio::test]
async fn speech_routes_custom_model_and_passes_binary_audio_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(b"FAKE-MP3-BYTES".to_vec())
                .insert_header("content-type", "audio/mpeg"),
        )
        .mount(&server)
        .await;

    let state = build_stub_state();
    register(&state, "mytts", &server.uri(), &["kokoro"]).await;

    let payload: routeplane_types::SpeechRequest = serde_json::from_value(json!({
        "model": "kokoro",
        "input": "hello there",
        "voice": "alloy"
    }))
    .expect("speech request");
    // Header-less: the model id resolves through the runtime registry.
    let resp = routeplane::audio_api::speech(
        State(state.clone()),
        axum::Extension(vk()),
        axum::Extension(ctx()),
        HeaderMap::new(),
        OpenAiJson(payload),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("audio/mpeg")
    );
    assert_eq!(
        resp.headers()
            .get("x-routeplane-provider")
            .and_then(|v| v.to_str().ok()),
        Some("mytts")
    );
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    assert_eq!(&bytes[..], b"FAKE-MP3-BYTES");

    await_ring_provider(&state, "mytts").await;
}
