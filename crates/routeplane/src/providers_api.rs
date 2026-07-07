//! `POST|GET /v1/providers` + `DELETE /v1/providers/{name}` — the runtime
//! custom-provider registry API (Community Edition operator surface).
//!
//! Rides the AUTHED router (the same `auth_middleware` layer as
//! `/v1/chat/completions`): an unauthenticated caller is 401'd before any
//! handler runs. NOT entitlement-gated — this is a CE feature.
//!
//! Contract:
//!   * `POST /v1/providers`          → upsert `{name, base_url, api_key, models}`.
//!     201 (created) / 200 (updated) with the MASKED provider view.
//!   * `GET /v1/providers`           → `{"object":"list","data":[<view>, …]}`,
//!     api_key always masked (`…last4`) — the raw key is WRITE-ONLY.
//!   * `DELETE /v1/providers/{name}` → `{"object":"provider","name",…,"deleted":true}`,
//!     or the OpenAI-shaped 404 for an unknown name.
//!
//! Secret handling: the raw `api_key` is never returned, never logged (the
//! tracing lines below carry name/base_url/model-count only), and persists only
//! to the 0600 `configs/providers.json` (see `custom_providers.rs`).
//!
//! Concurrency: reads are one lock-free `ArcSwap::load`; mutations run inside a
//! `tokio::spawn`ed task so a client disconnect mid-request cannot cancel the
//! future between the durable file write and the in-memory swap (the
//! persist-then-swap pair always completes together).

use crate::api_error::{error_response, OpenAiJson};
use crate::custom_providers::{validate_and_normalize, CustomProviderConfig};
use crate::proxy::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

/// `POST /v1/providers` — validate, persist, hot-swap. 201 created / 200 updated.
pub async fn upsert_provider(
    State(state): State<Arc<AppState>>,
    OpenAiJson(mut cfg): OpenAiJson<CustomProviderConfig>,
) -> Response {
    if let Err((param, msg)) = validate_and_normalize(&mut cfg) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_provider_config",
            msg,
            "invalid_request_error",
            Some(&param),
        );
    }
    // A custom provider may NEVER shadow a built-in provider name (that would
    // silently reroute existing traffic); `routeplane` is reserved for combos.
    // Checked BEFORE the SSRF guard: a pure string check, no DNS for a doomed name.
    if state.providers.contains_key(cfg.name.as_str()) || cfg.name == "routeplane" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "provider_name_reserved",
            format!(
                "'{}' is a reserved built-in provider name; choose a different name.",
                cfg.name
            ),
            "invalid_request_error",
            Some("name"),
        );
    }
    // SSRF guard (fail-closed): resolve base_url and refuse link-local/metadata
    // always, loopback/private unless RP_CUSTOM_PROVIDER_ALLOW_PRIVATE=on. DNS
    // resolution is blocking, so it runs off the async worker.
    {
        let allow_private = crate::custom_providers::custom_provider_allow_private();
        let ssrf_url = cfg.base_url.clone();
        match tokio::task::spawn_blocking(move || {
            crate::custom_providers::ssrf_check(&ssrf_url, allow_private)
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err((param, msg))) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "provider_ssrf_blocked",
                    msg,
                    "invalid_request_error",
                    Some(&param),
                );
            }
            Err(e) => {
                tracing::error!("custom provider SSRF check task failed: {e}");
                return crate::api_error::internal_error();
            }
        }
    }
    let name = cfg.name.clone();
    let base_url = cfg.base_url.clone();
    let model_count = cfg.models.len();
    let store = state.custom_providers.clone();
    // Spawn so the persist→swap pair completes even if the caller disconnects
    // (a dropped handler future must not leave disk and memory out of step).
    match tokio::spawn(async move { store.upsert(cfg).await }).await {
        Ok(Ok((view, created))) => {
            // Register a circuit breaker + latency EWMA + in-flight gauge for the
            // provider so it is fast-failed and latency-ordered like a built-in
            // (ADR-113). Idempotent: an update never resets an existing breaker.
            state.health.register(&name);
            tracing::info!(
                "custom provider {}: name={name} base_url={base_url} models={model_count}",
                if created { "created" } else { "updated" },
            );
            let status = if created {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            };
            (status, Json(view)).into_response()
        }
        Ok(Err(e)) => {
            // `e` is a store/persist error string — never contains the api_key.
            tracing::error!("custom provider upsert failed (name={name}): {e}");
            crate::api_error::internal_error()
        }
        Err(e) => {
            tracing::error!("custom provider upsert task failed (name={name}): {e}");
            crate::api_error::internal_error()
        }
    }
}

/// `GET /v1/providers` — every custom provider, MASKED, sorted by name.
pub async fn list_providers(State(state): State<Arc<AppState>>) -> Response {
    let data = state.custom_providers.list();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "object": "list", "data": data })),
    )
        .into_response()
}

/// `DELETE /v1/providers/{name}` — remove, persist, hot-swap; 404 if absent.
pub async fn delete_provider(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Response {
    let store = state.custom_providers.clone();
    let removed_name = name.clone();
    // Same cancellation-safety posture as the upsert: the persist→swap pair
    // runs in a spawned task the client cannot cancel mid-way.
    match tokio::spawn(async move { store.remove(&removed_name).await }).await {
        Ok(Ok(true)) => {
            tracing::info!("custom provider deleted: name={name}");
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "object": "provider",
                    "name": name,
                    "deleted": true,
                })),
            )
                .into_response()
        }
        Ok(Ok(false)) => error_response(
            StatusCode::NOT_FOUND,
            "provider_not_found",
            format!("No custom provider named '{name}' is registered."),
            "invalid_request_error",
            Some("name"),
        ),
        Ok(Err(e)) => {
            tracing::error!("custom provider delete failed (name={name}): {e}");
            crate::api_error::internal_error()
        }
        Err(e) => {
            tracing::error!("custom provider delete task failed (name={name}): {e}");
            crate::api_error::internal_error()
        }
    }
}
