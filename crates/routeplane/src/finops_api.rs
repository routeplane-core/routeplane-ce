//! The FinOps chargeback/showback export surface (PRD-008 FR-24, the FinOps moat).
//!
//! Three routes, all gated FIRST on `Feature::FinOpsExport` (Business tier or above):
//!   * `GET /v1/finops/usage` — a per-tenant usage + cost rollup over the recent
//!     observability window, broken down by model and by virtual key.
//!   * `GET /v1/finops/timeseries` — a per-tenant RECENT-WINDOW usage time-series
//!     over the same ring, bucketed by `timestamp` (powers the Console trend charts).
//!     HONEST: the ring holds only the last ~1000 events, so the series is the recent
//!     in-memory window, not durable history (a true multi-day series needs the
//!     telemetry store, ADR-024) — the response carries a `note` saying so.
//!   * `GET /v1/finops/cache-savings` — a per-tenant RECENT-WINDOW cache-savings
//!     rollup over the same ring (served cache hits → avoided spend + tokens not
//!     re-sent), powering the Console Cache page's "Cost saved" + "Tokens saved"
//!     StatCards. HONEST: the cost is the sum of the per-hit RECORDED ESTIMATE
//!     (`estimated_saved_cost_micro_usd`), and the window is the recent ring, not
//!     durable history — the response `note` says both.
//!
//! This is a NEW module (the chat orchestrator in `proxy.rs` is UNTOUCHED), wired
//! exactly like `prompts_api`: it extracts the resolved `TenantContext` and the
//! `SharedAuthState` registry as Axum `Extension`s. Tenant isolation is by KEY
//! OWNERSHIP — the report only aggregates events whose `virtual_key_name` belongs
//! to a key the requesting tenant owns in the current registry snapshot, resolved
//! server-side. No client-supplied identifier selects the scope (the ADR-023
//! bypass rule: scope is keyed off the authenticated context only). The endpoint
//! is read-only and emits no usage event of its own.

use crate::auth::{SharedAuthState, TenantContext};
use crate::proxy::AppState;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use routeplane_entitlements::{tier_baseline, Feature};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Default recent window for `GET /v1/finops/timeseries` (minutes) when the caller
/// omits `window_mins`.
const DEFAULT_WINDOW_MINS: i64 = 60;
/// Hard cap on the recent window (24h). The ring only holds ~1000 events anyway, so
/// a wider window simply yields more empty leading buckets — honest, never fabricated.
const MAX_WINDOW_MINS: i64 = 1440;
/// Default bucket count when the caller omits `buckets`.
const DEFAULT_BUCKETS: usize = 30;
/// Hard cap on the bucket count (bounds the response size + per-bucket work).
const MAX_BUCKETS: usize = 200;

/// Query params for the recent-window time-series. Both optional + clamped to sane
/// bounds server-side (never trusted raw): `window_mins` ∈ [1, 1440], `buckets`
/// ∈ [1, 200]. The honest `note` in the response states this is the recent
/// in-memory window, not durable history.
#[derive(Debug, Deserialize)]
pub struct TimeseriesQuery {
    pub window_mins: Option<i64>,
    pub buckets: Option<usize>,
}

/// `GET /v1/finops/usage` — the tenant's chargeback/showback rollup (FR-24).
pub async fn usage_export(
    State(state): State<Arc<AppState>>,
    Extension(auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx, "/v1/finops/usage") {
        return resp;
    }

    // Resolve the virtual-key NAMES this tenant owns from the current registry
    // snapshot. Tenant isolation is structural (by key ownership), never by a
    // client-supplied id — a tenant can only ever see spend attributed to its own
    // keys' names in the recent-event window.
    let snapshot = auth_state.load();
    let key_names: BTreeSet<String> = snapshot
        .keys
        .values()
        .filter(|vk| vk.resolved_tenant_id() == tenant_ctx.tenant_id)
        .map(|vk| vk.name.clone())
        .collect();

    let report = state.observability_engine.chargeback(&key_names);

    // Attach the tenant id to the report envelope so an exported artifact is
    // self-describing. `to_value` on a plain Serialize report cannot fail.
    let mut body = serde_json::to_value(&report).unwrap_or_else(|_| json!({}));
    if let serde_json::Value::Object(map) = &mut body {
        map.insert("tenant_id".to_string(), json!(tenant_ctx.tenant_id));
    }
    (StatusCode::OK, Json(body)).into_response()
}

/// `GET /v1/finops/timeseries` — the tenant's RECENT-WINDOW usage time-series over
/// the in-memory observability ring (powers the Console trend charts).
///
/// Identical scoping + gating to `usage_export`: gated on `Feature::FinOpsExport`
/// (held-back → 403 `feature_not_released`, not-entitled → 403 `feature_not_entitled`),
/// tenant-isolated by KEY OWNERSHIP resolved server-side (never a client-supplied id).
/// Read-only over the existing ring; emits no usage event.
///
/// HONESTY: the ring holds only the last ~1000 events, so the series is the recent
/// in-memory window, NOT durable history — the response `note` says so explicitly.
/// An empty ring yields all-zero buckets (200), never fabricated traffic.
pub async fn usage_timeseries(
    State(state): State<Arc<AppState>>,
    Extension(auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Query(params): Query<TimeseriesQuery>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx, "/v1/finops/timeseries") {
        return resp;
    }

    // Clamp the caller's params server-side — never trust the raw query values.
    let window_mins = params
        .window_mins
        .unwrap_or(DEFAULT_WINDOW_MINS)
        .clamp(1, MAX_WINDOW_MINS);
    let bucket_count = params
        .buckets
        .unwrap_or(DEFAULT_BUCKETS)
        .clamp(1, MAX_BUCKETS);

    // Same key-ownership scoping as `usage_export`: tenant isolation is structural
    // (by key ownership), never a client-supplied id.
    let snapshot = auth_state.load();
    let key_names: BTreeSet<String> = snapshot
        .keys
        .values()
        .filter(|vk| vk.resolved_tenant_id() == tenant_ctx.tenant_id)
        .map(|vk| vk.name.clone())
        .collect();

    let series = state.observability_engine.usage_timeseries(
        &key_names,
        chrono::Duration::minutes(window_mins),
        bucket_count,
    );

    // Self-describing envelope. The honest `note` states this is the recent
    // in-memory window (last ~1000 events), not durable history (ADR-024 telemetry
    // store would be needed for a true multi-day series). `to_value` on a plain
    // Serialize report cannot fail.
    let mut body = serde_json::to_value(&series).unwrap_or_else(|_| json!({}));
    if let serde_json::Value::Object(map) = &mut body {
        map.insert("window_mins".to_string(), json!(window_mins));
        map.insert("tenant_id".to_string(), json!(tenant_ctx.tenant_id));
        map.insert(
            "note".to_string(),
            json!(
                "Recent activity from the in-memory observability ring (last ~1000 events) — \
                 a recent window, not durable history. Buckets outside the captured window are \
                 honestly empty; a multi-day series requires the telemetry store (ADR-024)."
            ),
        );
    }
    (StatusCode::OK, Json(body)).into_response()
}

/// Query params for the cache-savings rollup. Optional + clamped server-side
/// (never trusted raw): `window_mins` ∈ [1, 1440].
#[derive(Debug, Deserialize)]
pub struct CacheSavingsQuery {
    pub window_mins: Option<i64>,
}

/// `GET /v1/finops/cache-savings` — the tenant's RECENT-WINDOW cache-savings rollup
/// over the in-memory observability ring (powers the Console Cache page's
/// "Cost saved" + "Tokens saved" StatCards).
///
/// Identical scoping + gating to `usage_timeseries`: gated on `Feature::FinOpsExport`
/// (held-back → 403 `feature_not_released`, not-entitled → 403 `feature_not_entitled`),
/// tenant-isolated by KEY OWNERSHIP resolved server-side (never a client-supplied id).
/// Read-only over the existing ring; emits no usage event.
///
/// HONESTY: `saved_cost_micro_usd` sums the ring's per-hit
/// `estimated_saved_cost_micro_usd` — a real recorded ESTIMATE (the source field name
/// says so), not a fabricated number. The ring holds only the last ~1000 events, so
/// this is the recent in-memory window, NOT durable history — the response `note`
/// says so. No cache hits in the window → honest zeroes (200), never fabricated.
pub async fn cache_savings(
    State(state): State<Arc<AppState>>,
    Extension(auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Query(params): Query<CacheSavingsQuery>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx, "/v1/finops/cache-savings") {
        return resp;
    }

    // Clamp the caller's window server-side — never trust the raw query value.
    let window_mins = params
        .window_mins
        .unwrap_or(DEFAULT_WINDOW_MINS)
        .clamp(1, MAX_WINDOW_MINS);

    // Same key-ownership scoping as `usage_timeseries`: tenant isolation is
    // structural (by key ownership), never a client-supplied id.
    let snapshot = auth_state.load();
    let key_names: BTreeSet<String> = snapshot
        .keys
        .values()
        .filter(|vk| vk.resolved_tenant_id() == tenant_ctx.tenant_id)
        .map(|vk| vk.name.clone())
        .collect();

    let report = state
        .observability_engine
        .cache_savings(&key_names, chrono::Duration::minutes(window_mins));

    // Shape the response with the Console's expected field names. The honest `note`
    // states this is the recent in-memory window AND that the cost is an estimate.
    let body = json!({
        "tenant_id": tenant_ctx.tenant_id,
        "window_mins": window_mins,
        "cache_hits": report.cache_hits,
        "cacheable_lookups": report.cacheable_lookups,
        "saved_cost_micro_usd": report.saved_cost_micro_usd,
        "saved_tokens": report.saved_tokens,
        "note":
            "Recent cache savings from the in-memory observability ring (last ~1000 \
             events) — a recent window, not durable history. `saved_cost_micro_usd` \
             sums the per-hit ESTIMATED avoided upstream spend (a recorded estimate, \
             not a billed figure); `saved_tokens` is the served responses' tokens not \
             re-sent to providers. A durable, multi-day savings series requires the \
             telemetry store (ADR-024).",
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// The entitlement gate, mirroring `prompts_api::entitlement_gate`. An
/// entitled-but-held-back tenant → 403 `feature_not_released` (an operator
/// rollout holdback — truthful on both builds). A NOT-ENTITLED tenant:
///
/// * **Enterprise build** — 403 `feature_not_entitled` (a commercial-tier
///   decision inside a licensed deployment).
/// * **Community Edition** — the uniform 402 `enterprise_only` upsell
///   (`api_error::enterprise_only`), naming the endpoint called: on CE the
///   honest message is "switch to Enterprise", not "you lack a tier". Keys an
///   operator DID grant `FinOpsExport` (e.g. `tier: business` in `keys.json` —
///   the bundled Console's Usage/Cache pages ride those) pass the gate exactly
///   as before; only the refusal envelope changes.
#[cfg_attr(feature = "enterprise", allow(unused_variables))]
fn entitlement_gate(ctx: &TenantContext, endpoint: &str) -> Option<Response> {
    if ctx.capabilities.active(Feature::FinOpsExport) {
        return None;
    }
    let entitled_by_baseline = tier_baseline(ctx.tier).contains(&Feature::FinOpsExport);
    Some(if entitled_by_baseline {
        openai_error(
            StatusCode::FORBIDDEN,
            "feature_not_released",
            "FinOps usage export is entitled for this tenant but not yet released (rollout holdback).",
        )
    } else {
        #[cfg(not(feature = "enterprise"))]
        {
            crate::api_error::enterprise_only(endpoint)
        }
        #[cfg(feature = "enterprise")]
        {
            openai_error(
                StatusCode::FORBIDDEN,
                "feature_not_entitled",
                "FinOps usage export requires the Business tier or above.",
            )
        }
    })
}

/// The OpenAI-shaped error envelope `{ error: { message, type, code } }`, matching
/// the codes the prompt surface uses so clients parse one error shape everywhere.
fn openai_error(status: StatusCode, code: &str, message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use routeplane_entitlements::{CapabilitySet, Tier};

    fn ctx(tier: Tier) -> TenantContext {
        TenantContext {
            tenant_id: "t_test".into(),
            tier,
            capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &BTreeSet::new()),
            compliance_frameworks: Vec::new(),
            compliance_mode: crate::auth::ComplianceMode::Strict,
        }
    }

    #[tokio::test]
    async fn entitled_tenant_passes_the_gate_unchanged() {
        // A Business-tier key (FinOpsExport in its baseline) still passes — the
        // CE change only reshapes the NOT-ENTITLED refusal, so the bundled
        // Console's Usage/Cache pages keep working for granted keys.
        assert!(entitlement_gate(&ctx(Tier::Business), "/v1/finops/usage").is_none());
    }

    #[cfg(not(feature = "enterprise"))]
    #[tokio::test]
    async fn ce_not_entitled_is_the_uniform_enterprise_only_402() {
        let resp = entitlement_gate(&ctx(Tier::Free), "/v1/finops/usage")
            .expect("Free tier is not entitled");
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
        assert_eq!(
            resp.headers()
                .get("x-routeplane-upgrade")
                .and_then(|v| v.to_str().ok()),
            Some("https://routeplane.ai")
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(v["error"]["code"], "enterprise_only");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .expect("message")
            .starts_with("/v1/finops/usage is a Routeplane Enterprise feature"));
    }
}
