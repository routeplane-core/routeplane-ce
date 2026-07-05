//! The Data-Plane prompt render surface (PRD-010 / G3.5, FR-7..FR-17).
//!
//! Three routes, all gated FIRST on `Feature::PromptRegistry`:
//!   * `GET  /v1/prompts/{ref}`             — the stored version object (FR-7).
//!   * `POST /v1/prompts/{ref}/render`      — a pure render, no upstream (FR-8).
//!   * `POST /v1/prompts/{ref}/completions` — render-and-run through the unchanged
//!     chat pipeline (FR-9), by invoking `crate::proxy::chat_completions` directly.
//!
//! This is a NEW module (the chat orchestrator in `proxy.rs` is UNTOUCHED). The
//! prompt registry is carried as an Axum `Extension<SharedPromptRegistry>` —
//! exactly the pattern `main.rs` uses for `auth_state` — rather than as a field on
//! `AppState` (which lives in `proxy.rs`). This keeps the 2074-line orchestrator
//! byte-for-byte intact; the completions path forwards to `chat_completions`,
//! which does not need the registry. See the PR body for the deviation argument.
//!
//! Deliberate deviations (documented, argued in the PR body):
//!   * FR-13/FR-14 are distinguished via `tier_baseline(tier)` (the only
//!     entitlement signal on `TenantContext`): a not-entitled tenant → 403
//!     `feature_not_entitled`; an entitled-but-held-back tenant → 403
//!     `feature_not_released`. The blind spot (a Free tenant with an override that
//!     is then held back) reports `feature_not_entitled` — overrides aren't visible
//!     at this layer.
//!   * FR-17 join: rather than threading `prompt_id/version/label` into the chat
//!     success event (which would touch `proxy.rs`), the render + completions paths
//!     emit a lightweight parallel `prompt.render` `UsageEvent` carrying those
//!     fields. The chat pipeline records its own usage/cost event unchanged.

use crate::auth::{TenantContext, TenantGuardrails, VirtualKey};
use crate::observability::UsageEvent;
use crate::proxy::{chat_completions, AppState};
use axum::extract::{Json, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane_entitlements::{tier_baseline, Feature};
use routeplane_prompts::{MissingPolicy, PromptError, SharedPromptRegistry};
use routeplane_types::ChatCompletionRequest;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Body for `POST /v1/prompts/{ref}/render` (FR-8).
#[derive(Debug, Deserialize)]
pub struct RenderBody {
    #[serde(default)]
    pub variables: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub missing: MissingPolicy,
}

/// Body for `POST /v1/prompts/{ref}/completions` (FR-9). `variables`/`missing`
/// drive the render; every other field is captured as a request override (FR-10:
/// body overrides win over the version's `default_model`/`default_params`).
#[derive(Debug, Deserialize)]
pub struct CompletionsBody {
    #[serde(default)]
    pub variables: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub missing: MissingPolicy,
    #[serde(flatten)]
    pub overrides: serde_json::Map<String, serde_json::Value>,
}

/// `GET /v1/prompts/{ref}` — the stored version object, NO render, NO upstream
/// (FR-7). Reports the concrete resolved `version` (even for a label ref).
pub async fn get_prompt(
    Extension(prompts): Extension<SharedPromptRegistry>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Path(reference): Path<String>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx) {
        return resp;
    }
    let reg = prompts.load();
    match reg.resolve(&tenant_ctx.tenant_id, &reference) {
        Ok(r) => {
            let body = json!({
                "id": r.prompt.id,
                "name": r.prompt.name,
                "description": r.prompt.description,
                "latest_version": r.prompt.latest_version,
                "version": r.version_number,
                "label": r.label,
                "template": r.version.template,
                "variables": r.version.variables,
                "default_model": r.version.default_model,
                "default_params": r.version.default_params,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => prompt_error_response(&e),
    }
}

/// `POST /v1/prompts/{ref}/render` — a pure render, NO upstream (FR-8).
pub async fn render_prompt(
    State(state): State<Arc<AppState>>,
    Extension(prompts): Extension<SharedPromptRegistry>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Path(reference): Path<String>,
    headers: HeaderMap,
    Json(body): Json<RenderBody>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx) {
        return resp;
    }
    // A/B testing (PRD-010): the cohort key for sticky weighted assignment. The
    // pure-render surface has no chat body, so the cohort comes from the explicit
    // header only (no `user` field to fall back to). None ⇒ control arm.
    let cohort = cohort_from_header(&headers);
    let rendered = {
        let reg = prompts.load();
        match reg.render_with_cohort(
            &tenant_ctx.tenant_id,
            &reference,
            &body.variables,
            body.missing,
            cohort.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => return prompt_error_response(&e),
        }
    };

    // FR-17: a render-only call emits a lightweight prompt.render event (no
    // tokens, no cost, no upstream). A/B testing: annotate the served variant.
    state.observability_engine.record_usage(
        UsageEvent::prompt_render(
            virtual_key.name.clone(),
            rendered.model.clone().unwrap_or_default(),
            rendered.prompt_id.clone(),
            rendered.version,
            rendered.label.clone(),
            None,
            false,
        )
        .with_experiment(rendered.experiment.clone()),
    );

    let body = json!({
        "prompt_id": rendered.prompt_id,
        "version": rendered.version,
        "label": rendered.label,
        "model": rendered.model,
        "params": rendered.params,
        "messages": [{ "role": "user", "content": rendered.text }],
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// `POST /v1/prompts/{ref}/completions` — render then run the UNCHANGED chat
/// pipeline (FR-9): residency classification on the rendered text, guardrails,
/// eligibility/ordering, cache, budgets/rate-limits, streaming.
#[allow(clippy::too_many_arguments)]
pub async fn prompt_completions(
    State(state): State<Arc<AppState>>,
    Extension(prompts): Extension<SharedPromptRegistry>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Extension(tenant_guardrails): Extension<TenantGuardrails>,
    Path(reference): Path<String>,
    headers: HeaderMap,
    Json(body): Json<CompletionsBody>,
) -> Response {
    if let Some(resp) = entitlement_gate(&tenant_ctx) {
        return resp;
    }

    // A/B testing (PRD-010): the cohort key for sticky weighted assignment.
    // Precedence: the explicit `x-routeplane-cohort` header wins; otherwise fall
    // back to the OpenAI `user` field in the body overrides. None ⇒ control arm.
    let cohort = cohort_from_header(&headers).or_else(|| cohort_from_overrides(&body.overrides));
    let rendered = {
        let reg = prompts.load();
        match reg.render_with_cohort(
            &tenant_ctx.tenant_id,
            &reference,
            &body.variables,
            body.missing,
            cohort.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => return prompt_error_response(&e),
        }
    };

    // FR-10 precedence: version default_params (base) ◁ default_model ◁ body
    // overrides (win). Sovereign residency remains the hard override over all of
    // it — enforced downstream by chat_completions, not weakened here.
    let mut merged = serde_json::Map::new();
    if let Some(serde_json::Value::Object(params)) = &rendered.params {
        for (k, v) in params {
            merged.insert(k.clone(), v.clone());
        }
    }
    if let Some(model) = &rendered.model {
        merged.insert("model".to_string(), json!(model));
    }
    for (k, v) in body.overrides {
        merged.insert(k, v); // body wins
    }
    if !merged.contains_key("model") {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "prompt_render_failed",
            "No model resolved: the prompt version declares no default_model and the request supplied no model override.",
            Some("model"),
        );
    }
    // FR-9: the rendered template is the message source; the body cannot override
    // messages (v1 renders the template as a single user message — documented).
    merged.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": rendered.text }]),
    );

    let chat_request: ChatCompletionRequest =
        match serde_json::from_value(serde_json::Value::Object(merged)) {
            Ok(r) => r,
            Err(e) => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "prompt_render_failed",
                    format!("Failed to bind the prompt to a chat request: {e}"),
                    None,
                )
            }
        };

    // FR-17 (deviation): the prompt.render join event carrying the resolved
    // prompt_id + integer version + label. Region/sovereign are recorded by the
    // chat pipeline's own event; this is purely the prompt-attribution join row.
    state.observability_engine.record_usage(
        UsageEvent::prompt_render(
            virtual_key.name.clone(),
            chat_request.model.clone(),
            rendered.prompt_id.clone(),
            rendered.version,
            rendered.label.clone(),
            None,
            false,
        )
        .with_experiment(rendered.experiment.clone()),
    );

    // chat_completions IS a plain `pub async fn` — invoke it directly with the
    // extractor wrappers reconstructed from what auth injected.
    chat_completions(
        State(state),
        Extension(virtual_key),
        Extension(tenant_ctx),
        Extension(tenant_guardrails),
        headers,
        crate::api_error::OpenAiJson(chat_request),
    )
    .await
}

/// The FR-13/FR-14 entitlement gate. Returns `Some` when the prompt registry
/// is not active for this tenant, distinguishing not-entitled from held-back.
///
/// An entitled-but-held-back tenant → 403 `feature_not_released` (an operator
/// rollout holdback — truthful on both builds). A NOT-ENTITLED tenant on the
/// **enterprise build** → 403 `feature_not_entitled` (FR-13); on the
/// **Community Edition** → the uniform 402 `enterprise_only` upsell
/// (`api_error::enterprise_only`, the same envelope as /v1/finops/* and the
/// /v1/mcp/* stubs). Keys an operator DID grant `PromptRegistry` (Standard+
/// tier in `keys.json`) pass the gate exactly as before.
fn entitlement_gate(ctx: &TenantContext) -> Option<Response> {
    if ctx.capabilities.active(Feature::PromptRegistry) {
        return None;
    }
    // The only entitlement signal on TenantContext is `tier` + the resolved
    // (post-holdback) CapabilitySet. If the tier baseline grants the feature yet
    // it is inactive, the only cause is a rollout holdback (released = false) →
    // FR-14. Otherwise the tenant is simply not entitled → FR-13.
    let entitled_by_baseline = tier_baseline(ctx.tier).contains(&Feature::PromptRegistry);
    Some(if entitled_by_baseline {
        openai_error(
            StatusCode::FORBIDDEN,
            "invalid_request_error",
            "feature_not_released",
            "Prompt registry is entitled for this tenant but not yet released (rollout holdback).",
            None,
        )
    } else {
        #[cfg(not(feature = "enterprise"))]
        {
            crate::api_error::enterprise_only("/v1/prompts")
        }
        #[cfg(feature = "enterprise")]
        {
            openai_error(
                StatusCode::FORBIDDEN,
                "invalid_request_error",
                "feature_not_entitled",
                "Prompt registry requires the Standard tier or above.",
                None,
            )
        }
    })
}

/// A/B testing (PRD-010): the cohort key from the explicit `x-routeplane-cohort`
/// header. A trimmed, non-empty ASCII/UTF-8 header value; absent/blank ⇒ `None`
/// (the experiment then serves its control arm). The header is the PREFERRED
/// cohort source — an explicit, stable caller-chosen identity for sticky
/// assignment, independent of the OpenAI `user` field.
fn cohort_from_header(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-routeplane-cohort")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// A/B testing fallback: the OpenAI `user` field from the completions body
/// overrides. Used only when no `x-routeplane-cohort` header is present, so
/// callers that already tag requests with a stable `user` get sticky assignment
/// for free. A non-string or blank `user` ⇒ `None` (control arm).
fn cohort_from_overrides(overrides: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    overrides
        .get("user")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Map a registry/render error to the OpenAI envelope with the FR-16 code +
/// status + optional `param`.
fn prompt_error_response(e: &PromptError) -> Response {
    openai_error(
        StatusCode::from_u16(e.status()).unwrap_or(StatusCode::BAD_REQUEST),
        "invalid_request_error",
        e.code(),
        e.message(),
        e.param(),
    )
}

/// The OpenAI-shaped error envelope `{ error: { message, type, code, param? } }`
/// (FR-16).
fn openai_error(
    status: StatusCode,
    err_type: &str,
    code: &str,
    message: impl Into<String>,
    param: Option<&str>,
) -> Response {
    let body = json!({
        "error": {
            "message": message.into(),
            "type": err_type,
            "code": code,
            "param": param,
        }
    });
    (status, Json(body)).into_response()
}
