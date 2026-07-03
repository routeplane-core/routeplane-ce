//! Hermetic integration tests for the recent-request-logs surface
//! (`GET /v1/logs`). The handler is invoked directly (no HTTP server / auth
//! round-trip — the extensions the auth middleware would inject are passed
//! explicitly, exactly like the finops/prompts suites). The only state needed is
//! an `AppState` (for the observability ring) and a `SharedAuthState` registry
//! (the key→tenant ownership map the handler resolves the scope from).
//!
//! Covers: any authed tenant gets 200 (no entitlement gate beyond auth/ownership);
//! the `{ events: [...] }` envelope; and tenant isolation — a seeded event on the
//! tenant's own key appears, and the response never reflects another tenant's keys.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::logs_api::list_logs;
use routeplane::observability::UsageEvent;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme_a","routeplane_key":"rp_acme_a","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"},
    {"name":"k_acme_b","routeplane_key":"rp_acme_b","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"},
    {"name":"k_other","routeplane_key":"rp_other","provider_keys":{"openai":"x"},"tenant_id":"t_other","tier":"free"}
]}"#;

fn build_state() -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        ..AppState::for_tests(ProviderRegistry::new())
    })
}

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
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

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Record an event and wait (bounded) until the async drain task has pushed it
/// into the ring — the same real-time-budget poll the finops/ab_parity harnesses
/// use, so the assertion never races the background writer.
async fn record_and_settle(state: &AppState, event: UsageEvent) {
    let before = state.observability_engine.get_recent_events().len();
    state.observability_engine.record_usage(event);
    let deadline = Instant::now() + Duration::from_secs(2);
    while state.observability_engine.get_recent_events().len() <= before {
        if Instant::now() > deadline {
            panic!("usage event never reached the ring");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn free_tenant_with_no_traffic_gets_200_empty_list() {
    // No entitlement gate beyond auth: even a Free tenant can read its own logs.
    let resp = list_logs(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert!(v["events"].is_array());
    assert_eq!(v["events"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn logs_are_tenant_isolated_and_exclude_other_tenants() {
    let state = build_state();

    // One success on the tenant's OWN key…
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_a".into(),
            "openai".into(),
            "gpt-4o".into(),
            10,
            5,
            15,
            None,
            false,
        )
        .with_latency(33),
    )
    .await;
    // …and one on ANOTHER tenant's key (k_other) that must never appear.
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_other".into(),
            "openai".into(),
            "gpt-4o".into(),
            99,
            99,
            198,
            None,
            true,
        ),
    )
    .await;

    let resp = list_logs(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    let events = v["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1, "only the tenant's own attempt is returned");
    assert_eq!(events[0]["virtual_key_name"], "k_acme_a");
    assert_eq!(events[0]["model"], "gpt-4o");
    assert_eq!(events[0]["provider"], "openai");
    assert_eq!(events[0]["outcome"], "success");
    assert_eq!(events[0]["latency_ms"], 33);
    assert!(events[0]["id"].as_str().unwrap().starts_with("log_"));
    // The other tenant's key never appears anywhere in the body.
    assert!(!v.to_string().contains("k_other"));
}
