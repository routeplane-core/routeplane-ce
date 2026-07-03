//! Hermetic integration tests for the recent-usage surface (`GET /analytics`).
//! The handler is invoked directly (the extensions the auth middleware would inject
//! are passed explicitly, exactly like the logs/finops suites). Covers: any authed
//! tenant gets 200; and tenant isolation — a seeded event on the tenant's own key
//! appears, while another tenant's event (and its raw provider `error` body) never
//! does. Regression for the cross-tenant `/analytics` disclosure.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::analytics_api::analytics_events;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme_a","routeplane_key":"rp_acme_a","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"},
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
async fn analytics_empty_for_a_tenant_with_no_traffic() {
    let resp = analytics_events(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await
    .into_response();
    let v = body_json(resp).await;
    assert!(v.is_array());
    assert_eq!(v.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn analytics_is_tenant_isolated_and_hides_other_tenant_error_bodies() {
    let state = build_state();

    // The tenant's OWN success event…
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
        ),
    )
    .await;
    // …and ANOTHER tenant's failure whose raw upstream body sits in `error`.
    record_and_settle(
        &state,
        UsageEvent::failure(
            "k_other".into(),
            "openai".into(),
            "gpt-4o".into(),
            None,
            false,
            "openai API error (400): {\"prompt\":\"tenant OTHER secret\"}".into(),
        ),
    )
    .await;

    let resp = analytics_events(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await
    .into_response();
    let v = body_json(resp).await;

    let events = v.as_array().expect("array body");
    assert_eq!(events.len(), 1, "only the caller's own event is returned");
    assert_eq!(events[0]["virtual_key_name"], "k_acme_a");
    // Neither the other tenant's key name nor its raw provider error body leaks.
    assert!(!v.to_string().contains("k_other"));
    assert!(!v.to_string().contains("tenant OTHER secret"));
}
