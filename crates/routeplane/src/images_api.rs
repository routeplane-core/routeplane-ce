//! `POST /v1/images/generations` — the OpenAI-compatible image-generation route
//! (PARITY: OpenAI exposes `/v1/images/generations`, LiteLLM/Portkey proxy image
//! generation). Closes a parity gap: Routeplane's OpenAI-compatible surface
//! (chat, embeddings, rerank, moderations) had no image endpoint.
//!
//! This is a NEW module modelled on `rerank_api.rs` / `embeddings.rs` (the chat
//! orchestrator in `proxy.rs` is untouched). It runs the same lean slice of the
//! pipeline: auth (extensions injected by `auth_middleware`), residency
//! classification on the INPUT prompt, PII masking of the prompt (so image-gen
//! cannot become a PII-egress bypass — the SAME always-on masking the
//! chat/embeddings/rerank path uses), provider eligibility (sovereign filter when
//! a region is required), router ordering, budget/rate-limit admission, and a
//! per-attempt-timeout fallback loop. `resolve_api_key` /
//! `limit_rejection_response` / `apply_advisory_headers` are reused from the
//! sibling `embeddings` module so the 429/402/advisory envelopes stay
//! byte-identical to chat.
//!
//! ## PII masking posture — SAME as rerank (NOT moderations)
//! The `prompt` is user TEXT bound for an external image API. Like
//! `/v1/chat`, `/v1/embeddings` and `/v1/rerank` (and UNLIKE `/v1/moderations`,
//! which must see raw content to classify it), the prompt is classified for
//! residency on the RAW text, THEN masked BEFORE egress. Classify-then-mask: the
//! classifier sees the raw text so masking cannot hide the PII it looks for.
//!
//! ## Provider selection
//! `x-routeplane-provider` (default `openai` — OpenAI is the canonical image
//! backend, `gpt-image-1` / `dall-e-3`). A provider with no first-party image
//! endpoint returns a typed 422 `image_generation_not_supported`.
//!
//! ## Usage accounting
//! Image generation does not bill in tokens. gpt-image-1 MAY return a top-level
//! `usage` block; we tolerate it but do not depend on it. We record the number
//! of generated images as the usage total so FinOps has a real number, and cost
//! via the pricing crate (which returns 0 for non-token image models — recorded
//! gracefully, never fabricated).
//!
//! ADR note: a new OpenAI-compatible endpoint with **no new standing cost, no DB,
//! no new dependency** (reuses reqwest/serde + the existing OpenAI adapter) is
//! incremental — rerank/moderations/embeddings shipped the same way without an
//! ADR. No architectural shift, so no ADR is written.

use crate::auth::{TenantContext, VirtualKey};
use crate::embeddings::{apply_advisory_headers, apply_warning_header, resolve_api_key};
use crate::guardrails::GuardrailConfig;
use crate::ledger_sink;
use crate::ledger_sink::{Outcome, UsageTotals};
use crate::observability::UsageEvent;
use crate::provenance::stamp_provenance;
use crate::proxy::AppState;
use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_limits::{estimate_cost_micro_usd, now_unix_ms, Admission};
use routeplane_router::RoutingStrategy;
use routeplane_types::{ImageGenerationRequest, Region};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub async fn image_generation(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    crate::api_error::OpenAiJson(mut payload): crate::api_error::OpenAiJson<ImageGenerationRequest>,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let request_deadline = state.deadline_config.request_deadline;
    let per_attempt = state.deadline_config.per_attempt_timeout;
    let registry = &state.providers;

    // 0. Validate the request shape at the route edge: an empty prompt is a
    //    malformed request independent of the provider — reject it with a clean
    //    422 invalid_request envelope BEFORE any residency/masking/network work
    //    (never a panic, never a generic 500).
    if payload.prompt.trim().is_empty() {
        return crate::api_error::error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            "`prompt` must be a non-empty string.",
            "invalid_request_error",
            Some("prompt"),
        );
    }

    // A stable model label for usage/ledger when the caller omits `model` (the
    // adapter fills the provider default at egress; we mirror it here for FinOps).
    let model_label = payload
        .model
        .clone()
        .unwrap_or_else(|| "gpt-image-1".to_string());

    let guards = state
        .limits
        .resolve(&virtual_key.routeplane_key, &tenant_ctx.tenant_id);

    // 1. Residency: classify the prompt BEFORE masking — masking would hide the
    //    very PII the classifier looks for (same invariant as chat/rerank).
    let classification = state.residency_engine.classify(&payload.prompt);
    let header_region = headers
        .get("x-routeplane-residency")
        .and_then(|h| h.to_str().ok());
    let requested_region: Option<Region> = virtual_key
        .effective_requested_region(header_region)
        .map(Region::new);
    let required_region = state
        .residency_engine
        .required_region(requested_region.as_ref(), &classification);
    let sovereign = required_region.is_some();
    let client_provider_requested = headers.get("x-routeplane-provider").is_some();

    // 2. Pre-guardrails: mask PII in the prompt by default, so
    //    /v1/images/generations does not become a PII-egress bypass (the same
    //    always-on masking the chat/embeddings/rerank path uses). Classify-then-
    //    mask: classification (step 1) already ran on the raw text.
    let guard_config = GuardrailConfig::masking();
    payload.prompt = state
        .guardrail_engine
        .process_text(&payload.prompt, &guard_config);

    // 3. Eligibility: a required region is a HARD filter over the registry's
    //    resident providers (overrides the client's chain); otherwise the
    //    client's x-routeplane-provider chain (default `openai` — the canonical
    //    image-generation backend).
    let eligible: Vec<String> = if let Some(region) = &required_region {
        let mut names: Vec<String> = registry
            .iter()
            .filter(|(_, p)| p.is_resident_in(region.as_str()))
            .map(|(name, _)| name.to_string())
            .collect();
        names.sort();
        if names.is_empty() {
            tracing::warn!(
                "Sovereign block (images): personal data requires {}-residency but no resident provider is eligible (entities={:?})",
                region.as_str(),
                classification.entities
            );
            state
                .observability_engine
                .record_usage(UsageEvent::sovereign_block(
                    virtual_key.name.clone(),
                    model_label.clone(),
                    Some(region.0.clone()),
                ));
            ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
                ledger_sink::decision_draft(
                    &tenant_ctx.tenant_id,
                    &request_id,
                    &model_label,
                    None,
                    None,
                    &classification,
                    Some(region.as_str()),
                    true,
                    client_provider_requested,
                    Outcome::ResidencyBlocked,
                    UsageTotals::default(),
                )
            });
            return crate::api_error::sovereign_block(region.as_str());
        }
        tracing::info!(
            "Sovereign routing enforced (images): region={} eligible={:?}",
            region.as_str(),
            names
        );
        names
    } else {
        let requested = headers
            .get("x-routeplane-provider")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("openai");
        let chain: Vec<String> = requested
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if chain.is_empty() {
            vec!["openai".to_string()]
        } else {
            chain
        }
    };

    // 4. Ordering: strategy + health (drops circuit-OPEN providers).
    let strategy = headers
        .get("x-routeplane-strategy")
        .and_then(|h| h.to_str().ok())
        .map(RoutingStrategy::parse)
        .unwrap_or_default();
    let ordered = state
        .router
        .order_candidates(&eligible, strategy, &state.health);

    // 5. Budgets & rate limits admission — check-before, fail-stop.
    let admit_now = now_unix_ms();
    if let Admission::Denied(breach) = guards.admit(admit_now) {
        tracing::info!(
            "limit rejection (images): tenant={} kind={} scope={} policy={}",
            tenant_ctx.tenant_id,
            breach.kind_header(),
            breach.scope_header(),
            breach.policy_id()
        );
        state.observability_engine.record_usage(UsageEvent::failure(
            virtual_key.name.clone(),
            format!("({})", breach.kind_header()),
            model_label.clone(),
            required_region.as_ref().map(|r| r.0.clone()),
            sovereign,
            if breach.is_budget() {
                "budget_exceeded".to_string()
            } else {
                "rate_limit_exceeded".to_string()
            },
        ));
        return crate::embeddings::limit_rejection_response(&breach);
    }

    // 6. Attempt loop — no streaming, no cache. First success wins.
    let mut last_error = "No providers available".to_string();
    let mut last_not_supported = false;

    for provider_name in &ordered {
        let provider: &dyn Provider = match registry.get(provider_name.as_str()) {
            Some(p) => p.as_ref(),
            None => {
                last_error = format!("Unsupported provider: {provider_name}");
                continue;
            }
        };
        if !state.health.is_available(provider_name) {
            tracing::warn!("Skipping {} — circuit breaker is OPEN", provider_name);
            last_error = format!("circuit breaker open for {provider_name}");
            continue;
        }
        let api_key = match resolve_api_key(&virtual_key, provider_name) {
            Some(k) => k,
            None => {
                last_error = format!("API key for {provider_name} not configured");
                continue;
            }
        };

        let remaining = request_deadline.saturating_sub(started_at.elapsed());
        if remaining.is_zero() {
            last_error = "request deadline exceeded before all providers were tried".to_string();
            break;
        }
        let attempt_timeout = remaining.min(per_attempt);

        tracing::info!(
            "Attempting image generation via {} (sovereign={} timeout={}ms)",
            provider_name,
            sovereign,
            attempt_timeout.as_millis()
        );

        let started = Instant::now();
        let result = match tokio::time::timeout(
            attempt_timeout,
            provider.image_generation(payload.clone(), api_key),
        )
        .await
        {
            Ok(r) => r,
            Err(_elapsed) => Err(ProviderError::timeout(
                provider_name.clone(),
                format!("timed out after {}ms", attempt_timeout.as_millis()),
            )),
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;

        match result {
            Ok(response) => {
                state.health.record_latency(provider_name, elapsed_ms);
                state.health.record_success(provider_name);

                // Image generation has no token billing; record the number of
                // generated images as the usage total so FinOps has a real number
                // (completion = 0). Cost via the pricing crate returns 0 for
                // non-token image models — recorded gracefully, never fabricated.
                let units = response.data.len() as u32;
                let route_region = provider.resident_regions().into_iter().next();
                ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
                    ledger_sink::decision_draft(
                        &tenant_ctx.tenant_id,
                        &request_id,
                        &model_label,
                        Some(provider_name.as_str()),
                        route_region.as_deref(),
                        &classification,
                        required_region.as_ref().map(|r| r.as_str()),
                        sovereign,
                        client_provider_requested,
                        Outcome::Ok,
                        UsageTotals {
                            prompt_tokens: units,
                            completion_tokens: 0,
                            total_tokens: units,
                        },
                    )
                });

                state.observability_engine.record_usage(UsageEvent::success(
                    virtual_key.name.clone(),
                    provider_name.clone(),
                    model_label.clone(),
                    units,
                    0,
                    units,
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                ));

                let settle_now = now_unix_ms();
                let cost = estimate_cost_micro_usd(&model_label, units, 0);
                for a in &guards.settle(settle_now, units as u64, cost) {
                    state.export_spend_alert(&tenant_ctx.tenant_id, a);
                }

                let mut ok = (StatusCode::OK, Json(response)).into_response();
                if !guards.is_unlimited() {
                    let adv = guards.advisory(settle_now);
                    if !adv.is_empty() {
                        apply_advisory_headers(ok.headers_mut(), &adv);
                    }
                    if let Some(w) = guards.warning(settle_now) {
                        apply_warning_header(ok.headers_mut(), &w);
                    }
                }
                // Provenance trio (provider + trace/request correlation ids).
                stamp_provenance(ok.headers_mut(), provider_name, &request_id);
                return ok;
            }
            Err(e) => {
                // A provider that lacks first-party image generation returns a
                // typed 422 `image_generation_not_supported`. That is a CAPABILITY
                // GAP, not a health fault — do NOT trip its circuit breaker
                // (shared with chat) and do NOT pollute its latency EWMA.
                let this_not_supported = matches!(
                    &e,
                    ProviderError::BadRequest { status: 422, body, .. }
                        if body.starts_with("image_generation_not_supported")
                );
                if !this_not_supported {
                    state.health.record_latency(provider_name, elapsed_ms);
                    if crate::proxy::counts_as_health_failure(&e) {
                        state.health.record_failure(provider_name);
                    }
                }
                last_not_supported = this_not_supported;
                last_error = e.to_string();
                state.observability_engine.record_usage(UsageEvent::failure(
                    virtual_key.name.clone(),
                    provider_name.clone(),
                    model_label.clone(),
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                    last_error.clone(),
                ));
                tracing::warn!(
                    "Image generation via {} failed: {}. Trying fallback...",
                    provider_name,
                    last_error
                );
                continue;
            }
        }
    }

    // 7. Exhausted. A pure unsupported-image outcome is an explicit 422 envelope
    //    (never a generic 500); anything else is the all-failed 500.
    if last_not_supported {
        return image_generation_not_supported_response();
    }
    ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
        ledger_sink::decision_draft(
            &tenant_ctx.tenant_id,
            &request_id,
            &model_label,
            None,
            None,
            &classification,
            required_region.as_ref().map(|r| r.as_str()),
            sovereign,
            client_provider_requested,
            Outcome::AllFailed,
            UsageTotals::default(),
        )
    });
    tracing::warn!(
        "images: all providers failed (request_id={}): {}",
        request_id,
        last_error
    );
    crate::api_error::upstream_all_failed()
}

/// The explicit 422 `image_generation_not_supported` envelope — an OpenAI-shaped
/// error, not a generic failure, so a client routing image generation to a
/// provider without a first-party image endpoint gets an actionable message.
fn image_generation_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party image-generation endpoint. Route /v1/images/generations to an image-capable provider (openai) via x-routeplane-provider.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "image_generation_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}
