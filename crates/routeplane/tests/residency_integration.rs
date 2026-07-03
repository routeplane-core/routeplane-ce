//! Hermetic integration tests for the residency-observability surface
//! (`GET /v1/residency/summary` + `GET /v1/residency/ledger`). The handlers are
//! invoked directly (no HTTP server / auth round-trip — the extensions the auth
//! middleware would inject are passed explicitly, exactly like the logs/finops
//! suites). The only state needed is an `AppState` (for the observability ring)
//! and a `SharedAuthState` registry (the key→tenant ownership map the handler
//! resolves the scope from).
//!
//! Covers: any authed tenant gets 200 (no entitlement gate beyond auth/ownership);
//! the summary + ledger envelopes; honest-absent `series`/`framework`; and tenant
//! isolation — a seeded event on the tenant's own key appears, and the response
//! never reflects another tenant's keys.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane::residency_api::{residency_ledger, residency_summary};
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
/// into the ring — the same real-time-budget poll the logs/finops harnesses use,
/// so the assertion never races the background writer.
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
async fn free_tenant_with_no_traffic_gets_200_empty_report() {
    // No entitlement gate beyond auth: even a Free tenant reads its own residency.
    let resp = residency_summary(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["total"], 0);
    assert_eq!(v["regulated_pct"], 0.0);
    // HONEST-ABSENT: the summary carries no dated daily series.
    assert!(v.get("series").is_none());

    let resp = residency_ledger(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert!(v["entries"].is_array());
    assert_eq!(v["entries"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn residency_report_is_tenant_isolated_and_classifies() {
    let state = build_state();

    // A sovereign-routed success on the tenant's OWN key (regulated, IN).
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_a".into(),
            "gemini".into(),
            "gemini-pro".into(),
            20,
            10,
            30,
            Some("IN".into()),
            true,
        ),
    )
    .await;
    // A sovereign BLOCK on the tenant's OWN key (residency_blocked, 422).
    record_and_settle(
        &state,
        UsageEvent::sovereign_block("k_acme_b".into(), "gpt-4o".into(), Some("IN".into())),
    )
    .await;
    // Another tenant's sovereign-routed success — must never appear.
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_other".into(),
            "gemini".into(),
            "gemini-pro".into(),
            99,
            99,
            198,
            Some("EU".into()),
            true,
        ),
    )
    .await;

    // --- summary ---
    let resp = residency_summary(
        State(state.clone()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    // Only the two owned attempts contribute.
    assert_eq!(v["total"], 2);
    assert_eq!(v["regulated_count"], 2);
    assert_eq!(v["sovereign_routed_count"], 1);
    assert_eq!(v["blocked_count"], 1);
    assert_eq!(v["by_region"]["IN"], 2);
    assert_eq!(v["by_outcome"]["sovereign_routed"], 1);
    assert_eq!(v["by_outcome"]["residency_blocked"], 1);
    // The other tenant's region/key never leaks anywhere in the body.
    assert!(!v.to_string().contains("k_other"));
    assert!(v["by_region"].get("EU").is_none());

    // --- ledger ---
    let resp = residency_ledger(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let entries = v["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 2, "only the tenant's own decisions");
    // Newest-first: the block was recorded last among owned events.
    assert_eq!(entries[0]["virtual_key_name"], "k_acme_b");
    assert_eq!(entries[0]["outcome"], "residency_blocked");
    assert_eq!(entries[0]["required_region"], "IN");
    // A block routed nowhere → routed_region omitted (honest-absent).
    assert!(entries[0].get("routed_region").is_none());
    // The successful sovereign route DID route to IN.
    assert_eq!(entries[1]["outcome"], "sovereign_routed");
    assert_eq!(entries[1]["routed_region"], "IN");
    // HONEST-ABSENT: no per-request compliance framework on the ring.
    assert!(entries[0].get("framework").is_none());
    assert!(entries
        .iter()
        .all(|e| e["id"].as_str().unwrap().starts_with("log_")));
    // The other tenant's key never appears anywhere in the body.
    assert!(!v.to_string().contains("k_other"));
}
