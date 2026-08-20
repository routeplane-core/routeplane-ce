//! The read-only recent-request-logs surface (`GET /v1/logs`).
//!
//! One route that returns the authenticated tenant's most-recent request events
//! from the in-memory observability ring (the last ~1000 `UsageEvent`s). It is the
//! read-only twin of `finops_api` and follows the SAME tenant-isolation model
//! exactly: the scope is the typed tenant authority resolved at authentication.
//! No client-supplied identifier or collidable display-key name selects the scope
//! (the ADR-023 bypass rule), so a tenant can only ever see its own records.
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
use std::sync::Arc;

/// Max rows returned in one `GET /v1/logs` read. Bounded so the response (and the
/// per-event projection work) is capped regardless of ring size; the ring itself
/// holds ≤1000 events.
const LOGS_LIMIT: usize = 200;

/// `GET /v1/logs` — the tenant's recent request-log rows (newest-first).
///
/// Read-only, tenant-isolated by typed auth authority, no extra entitlement gate.
/// Returns `{ "events": [ LogRow, ... ] }` over the existing observability ring.
pub async fn list_logs(
    State(state): State<Arc<AppState>>,
    Extension(_auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> Response {
    let events = state
        .observability_engine
        .recent_events(tenant_ctx.resource_tenant_id.as_ref(), LOGS_LIMIT);

    (StatusCode::OK, Json(json!({ "events": events }))).into_response()
}
