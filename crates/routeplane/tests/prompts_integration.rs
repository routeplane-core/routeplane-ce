//! Hermetic integration tests for the prompt render surface (PRD-010 / G3.5).
//! Handlers are invoked directly (no HTTP server, no auth round-trip — the
//! extensions the auth middleware would inject are passed explicitly, exactly
//! like the embeddings/limits suites). The prompt registry rides an
//! `Extension<SharedPromptRegistry>`. The only "network" is a localhost wiremock
//! standing in for OpenAI on the completions path.
//!
//! Covers: entitlement gate (Free → 403 feature_not_entitled; held-back
//! Enterprise → 403 feature_not_released; cleared → resolves, AC-7); GET stored
//! version (FR-7); pure render with no upstream (FR-8); render-and-run happy path
//! (FR-9); render-and-run inherits residency (PII + IN → 422, FR-9/§8.2); tenant
//! isolation 404 (FR-15).

use axum::extract::{Json, Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::prompts_api::{
    get_prompt, prompt_completions, render_prompt, CompletionsBody, RenderBody,
};
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Feature, Tier};
use routeplane_prompts::{Bounds, PromptRegistry, SharedPromptRegistry};
use routeplane_router::HealthTracker;
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const PROMPTS: &str = r#"{"prompts":[
    {"tenant_id":"t_test","id":"prompt_greeting","name":"Greeting","latest_version":2,
     "labels":{"prod":1,"staging":2},
     "experiments":{"tone":{"variants":[
        {"version":1,"weight":50,"label":"concise"},
        {"version":2,"weight":50,"label":"friendly"}
     ]}},
     "versions":[
       {"version":1,"template":"Hi {{name}}.","variables":[{"name":"name"}],"default_model":"gpt-4o"},
       {"version":2,"template":"Hello {{name}}!","variables":[{"name":"name"}],"default_model":"gpt-4o","default_params":{"temperature":0.2}}
     ]},
    {"tenant_id":"t_test","id":"prompt_pii","name":"PII","latest_version":1,
     "versions":[{"version":1,"template":"Contact {{email}} for details","variables":[{"name":"email"}],"default_model":"gpt-4o"}]}
]}"#;

fn build_state(providers: ProviderRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        ..AppState::for_tests(providers)
    })
}

fn openai_registry(base_url: &str) -> ProviderRegistry {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(base_url)) as Arc<dyn Provider>,
    );
    providers
}

fn shared_prompts() -> SharedPromptRegistry {
    let reg = PromptRegistry::load_from_json(PROMPTS, "test", &Bounds::default())
        .expect("valid prompt registry");
    routeplane_prompts::new_shared_registry(reg)
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "openai": "test-api-key" }
    }))
    .expect("virtual key deserializes")
}

fn ctx(tier: Tier, tenant: &str) -> TenantContext {
    TenantContext {
        tenant_id: tenant.into(),
        tier,
        capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn ctx_held(tier: Tier, tenant: &str) -> TenantContext {
    let holdbacks = BTreeSet::from([Feature::PromptRegistry]);
    TenantContext {
        tenant_id: tenant.into(),
        tier,
        capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &holdbacks),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn render_body(v: serde_json::Value) -> RenderBody {
    serde_json::from_value(v).expect("render body")
}

fn completions_body(v: serde_json::Value) -> CompletionsBody {
    serde_json::from_value(v).expect("completions body")
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn mount_chat_ok(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })))
        .mount(server)
        .await;
}

// --- AC-7: entitlement gate ---------------------------------------------------

#[tokio::test]
async fn free_tenant_gets_403_feature_not_entitled() {
    let prompts = shared_prompts();
    let resp = get_prompt(
        Extension(prompts),
        Extension(ctx(Tier::Free, "t_test")),
        Path("prompt_greeting".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "feature_not_entitled");
}

#[tokio::test]
async fn held_back_entitled_tenant_gets_403_feature_not_released_then_resolves() {
    let state = build_state(openai_registry("http://127.0.0.1:9"));

    // Held back → entitled-by-baseline (Enterprise) but inactive → feature_not_released.
    let held = render_prompt(
        State(state.clone()),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx_held(Tier::Enterprise, "t_test")),
        Path("prompt_greeting".to_string()),
        HeaderMap::new(),
        Json(render_body(json!({ "variables": { "name": "Ada" } }))),
    )
    .await;
    assert_eq!(held.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        body_json(held).await["error"]["code"],
        "feature_not_released"
    );

    // Holdback cleared → resolves (200).
    let ok = render_prompt(
        State(state),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx(Tier::Enterprise, "t_test")),
        Path("prompt_greeting".to_string()),
        HeaderMap::new(),
        Json(render_body(json!({ "variables": { "name": "Ada" } }))),
    )
    .await;
    assert_eq!(ok.status(), StatusCode::OK);
}

// --- FR-7: GET stored version -------------------------------------------------

#[tokio::test]
async fn get_returns_stored_version_with_concrete_resolved_version() {
    let resp = get_prompt(
        Extension(shared_prompts()),
        Extension(ctx(Tier::Standard, "t_test")),
        Path("prompt_greeting@v1".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["version"], 1);
    assert_eq!(v["template"], "Hi {{name}}.");
    assert_eq!(v["default_model"], "gpt-4o");
}

// --- FR-8: pure render, no upstream ------------------------------------------

#[tokio::test]
async fn render_returns_messages_with_no_upstream_call() {
    let state = build_state(openai_registry("http://127.0.0.1:9")); // never dialed
    let resp = render_prompt(
        State(state),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx(Tier::Standard, "t_test")),
        Path("prompt_greeting".to_string()), // latest == v2
        HeaderMap::new(),
        Json(render_body(json!({ "variables": { "name": "Ada" } }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["version"], 2);
    assert_eq!(v["messages"][0]["content"], "Hello Ada!");
}

// --- FR-9: render-and-run happy path -----------------------------------------

#[tokio::test]
async fn completions_renders_then_runs_the_chat_pipeline() {
    let server = MockServer::start().await;
    mount_chat_ok(&server).await;
    let state = build_state(openai_registry(&server.uri()));

    let resp = prompt_completions(
        State(state),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx(Tier::Standard, "t_test")),
        Extension(TenantGuardrails(None)),
        Path("prompt_greeting".to_string()),
        HeaderMap::new(),
        Json(completions_body(json!({ "variables": { "name": "Ada" } }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "Hi there");
    assert_eq!(v["usage"]["total_tokens"], 7);
}

// --- FR-9 / §8.2: render-and-run inherits residency --------------------------

#[tokio::test]
async fn completions_with_pii_and_required_region_is_422() {
    // The rendered text carries an email (PII); IN residency is requested and
    // openai is NOT IN-resident → the chat pipeline sovereign-blocks (422). The
    // mock server is never dialed.
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-residency", HeaderValue::from_static("IN"));

    let resp = prompt_completions(
        State(state),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx(Tier::Standard, "t_test")),
        Extension(TenantGuardrails(None)),
        Path("prompt_pii".to_string()),
        headers,
        Json(completions_body(json!({
            "variables": { "email": "user@example.com" }
        }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// --- FR-15: tenant isolation --------------------------------------------------

#[tokio::test]
async fn tenant_cannot_resolve_another_tenants_prompt_404() {
    // Identically-named prompt, different tenant → 404 prompt_not_found (no leak).
    let resp = get_prompt(
        Extension(shared_prompts()),
        Extension(ctx(Tier::Standard, "t_other")),
        Path("prompt_greeting".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "prompt_not_found");
}

// --- FR-4 / FR-16: reference-grammar 404 codes at the HTTP layer -------------

#[tokio::test]
async fn unknown_label_is_404_prompt_version_not_found() {
    let resp = get_prompt(
        Extension(shared_prompts()),
        Extension(ctx(Tier::Standard, "t_test")),
        Path("prompt_greeting@nope".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        body_json(resp).await["error"]["code"],
        "prompt_version_not_found"
    );
}

// --- PRD-010 (A/B testing): experiment resolution at the HTTP layer -----------

async fn render_with_cohort(state: &Arc<AppState>, cohort: Option<&str>) -> serde_json::Value {
    let mut headers = HeaderMap::new();
    if let Some(c) = cohort {
        headers.insert(
            "x-routeplane-cohort",
            HeaderValue::from_str(c).expect("ascii cohort"),
        );
    }
    let resp = render_prompt(
        State(state.clone()),
        Extension(shared_prompts()),
        Extension(vk()),
        Extension(ctx(Tier::Standard, "t_test")),
        Path("prompt_greeting@tone".to_string()),
        headers,
        Json(render_body(json!({ "variables": { "name": "Ada" } }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp).await
}

#[tokio::test]
async fn experiment_cohort_header_is_sticky_and_resolves_an_arm() {
    let state = build_state(openai_registry("http://127.0.0.1:9")); // never dialed
                                                                    // Same cohort header ⇒ the SAME arm every call (sticky), and the rendered
                                                                    // text matches the assigned version's template.
    let first = render_with_cohort(&state, Some("cohort-7")).await;
    let v = first["version"].as_u64().unwrap();
    assert!(v == 1 || v == 2);
    let expected = if v == 1 { "Hi Ada." } else { "Hello Ada!" };
    assert_eq!(first["messages"][0]["content"], expected);
    for _ in 0..5 {
        let again = render_with_cohort(&state, Some("cohort-7")).await;
        assert_eq!(again["version"], first["version"]);
    }
}

#[tokio::test]
async fn experiment_no_cohort_serves_control_arm() {
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    // No cohort header ⇒ the control (first declared) arm = version 1 ("concise").
    let v = render_with_cohort(&state, None).await;
    assert_eq!(v["version"], 1);
    assert_eq!(v["messages"][0]["content"], "Hi Ada.");
}

#[tokio::test]
async fn experiment_render_annotates_usage_event_with_served_variant() {
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    let _ = render_with_cohort(&state, Some("cohort-xyz")).await;

    // The prompt.render join event is recorded on the async ring; poll briefly.
    let mut found = None;
    for _ in 0..50 {
        let events = state.observability_engine.get_recent_events();
        if let Some(ev) = events.into_iter().find(|e| e.provider == "(prompt_render)") {
            found = Some(ev);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let ev = found.expect("prompt.render event recorded");
    assert_eq!(ev.prompt_id.as_deref(), Some("prompt_greeting"));
    assert_eq!(ev.prompt_experiment.as_deref(), Some("tone"));
    // The served variant label is one of the two arms, and matches the version.
    let variant = ev.prompt_variant.as_deref().expect("served variant");
    match ev.prompt_version {
        Some(1) => assert_eq!(variant, "concise"),
        Some(2) => assert_eq!(variant, "friendly"),
        other => panic!("unexpected experiment version {other:?}"),
    }
    // An experiment-resolved render carries no static label.
    assert!(ev.prompt_label.is_none());
}
