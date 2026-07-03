//! The read-only recent-request-logs surface (`GET /v1/logs`).
//!
//! One route that returns the authenticated tenant's most-recent request events
//! from the in-memory observability ring (the last ~1000 `UsageEvent`s). It is the
//! read-only twin of `finops_api` and follows the SAME tenant-isolation model
//! exactly: the scope is the set of virtual-key NAMES the requesting tenant owns,
//! resolved SERVER-SIDE from the `SharedAuthState` snapshot (key ownership). No
//! client-supplied identifier ever selects the scope (the ADR-023 bypass rule), so
//! a tenant can only ever see logs for its own keys.
//!
//! Entitlement choice (documented deliberately): UNLIKE `finops_api` (gated on
//! `Feature::FinOpsExport`, Business+), `/v1/logs` is gated on **auth + key
//! ownership only** — no extra feature gate. Observability of one's OWN recent
//! requests is a reasonable baseline for any authenticated tenant (it is the
//! request-log equivalent of `/analytics`, which is authed-only), and it carries no
//! cross-tenant data and no raw content. Richer log analytics (durable retention,
//! trace spans) would live behind the telemetry store + an entitlement when ADR-024
//! lands.
//!
//! This is a NEW module (the chat orchestrator in `proxy.rs` is UNTOUCHED). It is
//! read-only over the EXISTING ring (no new durable store, same posture as
//! `/v1/finops/usage` and `/metrics`) and emits NO usage event of its own.

use crate::auth::{SharedAuthState, TenantContext};
use crate::proxy::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Max rows returned in one `GET /v1/logs` read. Bounded so the response (and the
/// per-event projection work) is capped regardless of ring size; the ring itself
/// holds ≤1000 events.
const LOGS_LIMIT: usize = 200;

/// `GET /v1/logs` — the tenant's recent request-log rows (newest-first).
///
/// Read-only, tenant-isolated by key ownership, no entitlement gate beyond auth.
/// Returns `{ "events": [ LogRow, ... ] }` over the existing observability ring.
pub async fn list_logs(
    State(state): State<Arc<AppState>>,
    Extension(auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> Response {
    // Resolve the virtual-key NAMES this tenant owns from the current registry
    // snapshot — identical to finops_api. Tenant isolation is structural (by key
    // ownership), never by a client-supplied id.
    let snapshot = auth_state.load();
    let key_names: BTreeSet<String> = snapshot
        .keys
        .values()
        .filter(|vk| vk.resolved_tenant_id() == tenant_ctx.tenant_id)
        .map(|vk| vk.name.clone())
        .collect();

    let events = state
        .observability_engine
        .recent_events(&key_names, LOGS_LIMIT);

    (StatusCode::OK, Json(json!({ "events": events }))).into_response()
}
