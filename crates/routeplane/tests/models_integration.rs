//! Hermetic integration tests for the OpenAI-compatible model-discovery surface
//! (`GET /v1/models`, `GET /v1/models/{id}`).
//!
//! The handlers are stateless (a read of the static catalog), so they are
//! invoked directly. The auth path is exercised separately against a minimal
//! router wired with the REAL `auth_middleware` + `shared_auth_state` layer (the
//! same composition `main.rs` uses for the authed `/v1/*` routes), so an
//! unauthenticated caller is rejected with the standard 401 invalid_api_key
//! envelope before the handler runs.
//!
//! Covers: the `{"object":"list", data:[…]}` shape + a known model present
//! (FR-1); `GET /v1/models/{id}` 200 for a known id and a 404 OpenAI error
//! envelope for an unknown id; and the authed-route auth gate (no key ⇒ 401).

mod common;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use common::build_stub_state;
use routeplane::auth::{auth_middleware, shared_auth_state, AuthState, SharedAuthState};
use routeplane::models_api::{list_models, retrieve_model};
use routeplane::proxy::AppState;
use routeplane_policy::load_registry_from_file;
use std::sync::Arc;
use tower::ServiceExt;

/// A stub state whose routing-policy registry carries one operator-defined combo
/// (`fast-cheap` → `gpt-4o`), for the ADR-086 `/v1/models` surface tests.
fn state_with_combo() -> Arc<AppState> {
    // Unique per CALL, not just per process: tests in one binary run on parallel
    // threads, and a PID-only name let one test's post-load remove_file delete the
    // path out from under another mid-load (observed as a NotFound flake).
    static FIXTURE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let state = build_stub_state();
    let path =
        std::env::temp_dir().join(format!("rp_combo_it_{}_{}.json", std::process::id(), seq));
    std::fs::write(
        &path,
        r#"{"configs":[{"id":"cfg_fast","combo":"fast-cheap","routing":{"strategy":"cost","targets":[{"provider":"openai","params":{"override":{"model":"gpt-4o"}}}]}}]}"#,
    )
    .unwrap();
    let reg = load_registry_from_file(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    state.policies.store(Arc::new(reg));
    state
}

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

// --- list -------------------------------------------------------------------

#[tokio::test]
async fn list_models_returns_openai_list_shape_with_known_model() {
    let resp = list_models(State(build_stub_state())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().expect("data is an array");
    assert!(!data.is_empty(), "catalog must not be empty");

    // Every entry has the OpenAI model shape.
    for m in data {
        assert_eq!(m["object"], "model");
        assert!(m["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(m["owned_by"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(m["created"].as_u64().is_some());
    }

    // A well-known model is present and tagged with its provider.
    let gpt4o = data
        .iter()
        .find(|m| m["id"] == "gpt-4o")
        .expect("gpt-4o present in catalog");
    assert_eq!(gpt4o["owned_by"], "openai");

    // Audio transcription (STT) models are catalogued and tagged with their
    // provider — Groq's Whisper is the flagship fast/cheap STT entry.
    let whisper = data
        .iter()
        .find(|m| m["id"] == "whisper-large-v3")
        .expect("whisper-large-v3 present in catalog");
    assert_eq!(whisper["owned_by"], "groq");
}

// --- retrieve ---------------------------------------------------------------

#[tokio::test]
async fn retrieve_known_model_returns_the_object() {
    let resp = retrieve_model(
        State(build_stub_state()),
        Path("claude-3-5-sonnet-latest".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["id"], "claude-3-5-sonnet-latest");
    assert_eq!(v["object"], "model");
    assert_eq!(v["owned_by"], "anthropic");
    assert!(v["created"].as_u64().is_some());
}

#[tokio::test]
async fn retrieve_unknown_model_returns_404_openai_envelope() {
    let resp = retrieve_model(
        State(build_stub_state()),
        Path("no-such-model-xyz".to_string()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v = body_json(resp).await;
    // OpenAI error envelope shape.
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "model_not_found");
    assert_eq!(v["error"]["param"], "model");
    assert!(v["error"]["message"]
        .as_str()
        .is_some_and(|m| m.contains("no-such-model-xyz")));
}

// --- auth gate --------------------------------------------------------------

/// A minimal router mirroring `main.rs`'s authed wiring: the `/v1/models` routes
/// behind `auth_middleware` + the `SharedAuthState` extension. Used to assert
/// the auth gate; the handlers themselves are stateless.
fn authed_router() -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(retrieve_model))
        .with_state(build_stub_state())
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
}

#[tokio::test]
async fn unauthenticated_list_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn authenticated_list_passes_the_gate() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("x-routeplane-api-key", "rp_acme")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "list");
}

// --- ADR-086: combos surface in /v1/models ----------------------------------

#[tokio::test]
async fn list_models_includes_operator_combos_additively() {
    let resp = list_models(State(state_with_combo())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let data = v["data"].as_array().expect("data is an array");
    // The combo is discoverable as a model id, owned by the gateway.
    let combo = data
        .iter()
        .find(|m| m["id"] == "fast-cheap")
        .expect("combo surfaced in /v1/models");
    assert_eq!(combo["object"], "model");
    assert_eq!(combo["owned_by"], "routeplane");
    // Additive: the base provider catalog is still present.
    assert!(data
        .iter()
        .any(|m| m["id"] == "gpt-4o" && m["owned_by"] == "openai"));
}

#[tokio::test]
async fn retrieve_combo_by_id_returns_the_object() {
    let resp = retrieve_model(State(state_with_combo()), Path("fast-cheap".to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["id"], "fast-cheap");
    assert_eq!(v["object"], "model");
    assert_eq!(v["owned_by"], "routeplane");
}

#[tokio::test]
async fn no_combos_configured_yields_no_gateway_owned_entries() {
    // Empty registry (the default) ⇒ no combo entries ⇒ base catalog byte-identical.
    let resp = list_models(State(build_stub_state())).await;
    let v = body_json(resp).await;
    let data = v["data"].as_array().expect("data is an array");
    assert!(
        data.iter().all(|m| m["owned_by"] != "routeplane"),
        "no combos configured must yield no routeplane-owned model entries"
    );
}
