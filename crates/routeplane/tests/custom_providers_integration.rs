//! Hermetic integration tests for the runtime custom-provider registry
//! (`/v1/providers` + hot-path routing + discovery + observability).
//!
//! Handlers are invoked directly with the extensions the auth middleware would
//! inject (the same composition the logs/finops/prompts suites use). The
//! upstream is a real wiremock OpenAI-compatible server, so the tests prove the
//! full loop: register over the API (persist-free ephemeral store) → route a
//! chat request to the custom base_url with the registered key as
//! `Authorization: Bearer` → the usage ring records the custom provider name —
//! all with NO restart, and DELETE takes effect immediately (hot swap).

mod common;

use axum::body::to_bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use common::build_stub_state;
use routeplane::api_error::OpenAiJson;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::custom_providers::CustomProviderConfig;
use routeplane::models_api::{list_models, retrieve_model};
use routeplane::providers_api::{delete_provider, list_providers, upsert_provider};
use routeplane::proxy::{chat_completions, AppState};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_types::{ChatCompletionRequest, Message};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CUSTOM_KEY: &str = "sk-custom-test-9999";

fn vk() -> VirtualKey {
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

fn provider_cfg(name: &str, base_url: &str, models: &[&str]) -> CustomProviderConfig {
    CustomProviderConfig {
        name: name.into(),
        base_url: base_url.into(),
        api_key: CUSTOM_KEY.into(),
        models: models.iter().map(|m| m.to_string()).collect(),
        created_at: None,
    }
}

fn chat_payload(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.into(),
        messages: vec![Message {
            role: "user".into(),
            content: "hello".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: Some(false),
        ..Default::default()
    }
}

async fn drive_chat(state: &Arc<AppState>, headers: HeaderMap, model: &str) -> Response {
    chat_completions(
        State(state.clone()),
        axum::Extension(vk()),
        axum::Extension(ctx()),
        axum::Extension(TenantGuardrails(None)),
        headers,
        OpenAiJson(chat_payload(model)),
    )
    .await
    .into_response()
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Mount an OpenAI-compatible chat completion answering ONLY when the custom
/// provider's Bearer key is presented — proving the registered key reaches the
/// upstream.
async fn mount_custom_upstream(server: &MockServer, model: &str) {
    let resp = json!({
        "id": "chatcmpl-custom-1",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello from custom upstream"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header(
            "authorization",
            format!("Bearer {CUSTOM_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(resp))
        .mount(server)
        .await;
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

// --- registry API lifecycle ---------------------------------------------------

#[tokio::test]
async fn upsert_list_delete_lifecycle_masks_the_key_and_404s_unknown() {
    let state = build_stub_state();

    // Create → 201 with the MASKED view.
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg(
            "myvllm",
            "http://vllm.internal:8000/",
            &["custom-llama"],
        )),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CREATED);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "provider");
    assert_eq!(v["name"], "myvllm");
    assert_eq!(v["base_url"], "http://vllm.internal:8000"); // trailing / stripped
    assert_eq!(v["api_key"], "…9999", "response must carry the masked key");
    assert!(v["created_at"].as_str().is_some());

    // Upsert again → 200 (updated), created_at preserved.
    let created_at = v["created_at"].clone();
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg(
            "myvllm",
            "http://vllm.internal:8000",
            &["custom-llama", "custom-qwen"],
        )),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["created_at"], created_at);

    // List → masked; the raw key must never appear anywhere in the body.
    let resp = list_providers(State(state.clone())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    let raw = String::from_utf8_lossy(&bytes).to_string();
    assert!(
        !raw.contains(CUSTOM_KEY),
        "GET /v1/providers must never leak the raw api_key"
    );
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["name"], "myvllm");
    assert_eq!(v["data"][0]["api_key"], "…9999");

    // Delete → 200 deleted:true; second delete → 404 provider_not_found.
    let resp = delete_provider(State(state.clone()), Path("myvllm".to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["deleted"], true);
    let resp = delete_provider(State(state.clone()), Path("myvllm".to_string())).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "provider_not_found");
}

#[tokio::test]
async fn built_in_names_and_invalid_configs_are_rejected() {
    let state = build_stub_state(); // registers built-in "openai"

    // A built-in name can never be shadowed.
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg(
            "openai",
            "http://evil.internal:1",
            &["gpt-4o"],
        )),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "provider_name_reserved");

    // Bad scheme → invalid_provider_config with the offending param.
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg("myvllm", "ftp://host:21", &["m"])),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_provider_config");
    assert_eq!(v["error"]["param"], "base_url");

    // Empty model list → 400.
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg("myvllm", "http://host:8000", &[])),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "models");
}

// --- hot-path routing ----------------------------------------------------------

#[tokio::test]
async fn chat_routes_by_model_id_with_bearer_key_and_records_usage_then_delete_hot_swaps() {
    // wiremock runs on loopback — opt into private endpoints for this test
    // (the SSRF guard blocks loopback by default). Metadata stays blocked.
    std::env::set_var("RP_CUSTOM_PROVIDER_ALLOW_PRIVATE", "on");
    let server = MockServer::start().await;
    mount_custom_upstream(&server, "custom-llama").await;

    let state = build_stub_state();
    // Register with NO restart: the very next request can use it.
    let resp = upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg("myvllm", &server.uri(), &["custom-llama"])),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Header-less request whose model matches the custom provider's list →
    // routed to the custom base_url with `Authorization: Bearer <registered key>`
    // (the wiremock matcher enforces the header).
    let resp = drive_chat(&state, HeaderMap::new(), "custom-llama").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "hello from custom upstream"
    );

    // Observability: the SAME proxy path recorded the usage event under the
    // custom provider's name — usage/logs/analytics/metrics see it for free.
    await_ring_provider(&state, "myvllm").await;

    // DELETE hot-swaps immediately: the same model id now falls back to the
    // default (openai stub) with no restart.
    let resp = delete_provider(State(state.clone()), Path("myvllm".to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = drive_chat(&state, HeaderMap::new(), "custom-llama").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(
        v["choices"][0]["message"]["content"], "hello back",
        "after delete the request must fall back to the default provider"
    );
}

#[tokio::test]
async fn chat_routes_by_explicit_provider_header() {
    // wiremock runs on loopback — opt into private endpoints for this test
    // (the SSRF guard blocks loopback by default). Metadata stays blocked.
    std::env::set_var("RP_CUSTOM_PROVIDER_ALLOW_PRIVATE", "on");
    let server = MockServer::start().await;
    mount_custom_upstream(&server, "any-model").await;

    let state = build_stub_state();
    upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg("myvllm", &server.uri(), &["any-model"])),
    )
    .await;

    // `x-routeplane-provider: myvllm` addresses the custom provider explicitly,
    // regardless of the model id.
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", "myvllm".parse().expect("header"));
    let resp = drive_chat(&state, headers, "some-model-the-upstream-accepts").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "hello from custom upstream"
    );
}

#[tokio::test]
async fn custom_model_mapping_never_shadows_a_builtin_model_id() {
    // wiremock runs on loopback — opt into private endpoints for this test
    // (the SSRF guard blocks loopback by default). Metadata stays blocked.
    std::env::set_var("RP_CUSTOM_PROVIDER_ALLOW_PRIVATE", "on");
    // A wiremock that would answer if (wrongly) called — the assertion below is
    // on the RESPONSE + ring, so a stub-openai answer proves no shadowing.
    let server = MockServer::start().await;
    mount_custom_upstream(&server, "gpt-4o").await;

    let state = build_stub_state();
    upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg("myvllm", &server.uri(), &["gpt-4o"])),
    )
    .await;

    // gpt-4o is a BUILT-IN catalog id: the header-less default must remain the
    // legacy openai route (the in-process stub), NOT the custom provider.
    let resp = drive_chat(&state, HeaderMap::new(), "gpt-4o").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["choices"][0]["message"]["content"], "hello back");
    await_ring_provider(&state, "openai").await;
}

// --- discovery ------------------------------------------------------------------

#[tokio::test]
async fn models_surface_lists_custom_models_owned_by_the_provider() {
    let state = build_stub_state();
    upsert_provider(
        State(state.clone()),
        OpenAiJson(provider_cfg(
            "myvllm",
            "http://vllm.internal:8000",
            &["custom-llama"],
        )),
    )
    .await;

    let resp = list_models(State(state.clone())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let entry = v["data"]
        .as_array()
        .expect("data array")
        .iter()
        .find(|m| m["id"] == "custom-llama")
        .expect("custom model listed in /v1/models")
        .clone();
    assert_eq!(entry["owned_by"], "myvllm");
    assert_eq!(entry["object"], "model");

    let resp = retrieve_model(State(state.clone()), Path("custom-llama".to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["id"], "custom-llama");
    assert_eq!(v["owned_by"], "myvllm");
}
