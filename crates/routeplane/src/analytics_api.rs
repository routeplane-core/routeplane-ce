//! The read-only recent-usage surface (`GET /analytics`).
//!
//! Returns the authenticated tenant's most-recent `UsageEvent`s from the in-memory
//! observability ring. It follows the SAME tenant-isolation model as `/v1/logs`,
//! `/v1/finops/*`, and `/v1/residency/*`: the scope is the validated typed tenant
//! authority resolved at authentication (ADR-023). No client-supplied identifier or
//! collidable display-key name ever selects the scope, so a tenant sees only its OWN
//! `virtual_key_name`, cost, `use_case` labels, or provider `error` strings (which
//! can echo prompt text).
//!
//! Unlike `/v1/logs` (which projects to the sanitized `LogRow`), `/analytics`
//! returns the full `UsageEvent` shape — the richer per-request view — but now scoped
//! to the caller. `/analytics/latency` returns only per-PROVIDER latency aggregates
//! (no per-tenant rows), so it needs no ownership scope and is wired separately.
//!
//! Entitlement: authenticated tenant authority only (no extra feature gate) — observability of
//! one's own recent requests is a baseline for any authenticated tenant.

use crate::auth::{SharedAuthState, TenantContext};
use crate::proxy::AppState;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{Extension, Json};
use std::sync::Arc;

/// `GET /analytics` — the tenant's own recent `UsageEvent`s, tenant-isolated by
/// typed auth authority. Read-only over the existing observability ring; no entitlement gate
/// beyond auth.
pub async fn analytics_events(
    State(state): State<Arc<AppState>>,
    Extension(_auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> impl IntoResponse {
    Json(
        state
            .observability_engine
            .recent_events_owned(tenant_ctx.resource_tenant_id.as_ref()),
    )
}
