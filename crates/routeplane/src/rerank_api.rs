//! `POST /v1/rerank` — the Cohere/LiteLLM-compatible reranking route (parity
//! gap: LiteLLM exposes `/rerank` and Cohere/Jina offer reranking, core to RAG
//! pipelines). Reranking takes a `query` plus a candidate `documents` set and
//! returns the documents ordered by relevance.
//!
//! This is a NEW module modelled on `embeddings.rs` (the chat orchestrator in
//! `proxy.rs` is untouched). It runs the same lean slice of the pipeline as
//! embeddings: auth (extensions injected by `auth_middleware`), residency
//! classification on the INPUT text, PII masking of the query + every document
//! (so rerank cannot become a PII-egress bypass), provider eligibility
//! (sovereign filter when a region is required), router ordering, budget/rate-
//! limit admission, and a per-attempt-timeout fallback loop.
//!
//! Default provider is `cohere` (not `openai`): OpenAI has no rerank endpoint,
//! Cohere does (and is the canonical rerank backend, like Jina). The
//! `x-routeplane-provider` header still overrides.
//!
//! Deliberate scope cuts vs chat (same as embeddings): NO response cache, NO
//! streaming (rerank is non-streaming). `resolve_api_key` /
//! `limit_rejection_response` / `apply_advisory_headers` are reused from the
//! sibling `embeddings` module so the 429/402/advisory envelopes stay
//! byte-identical to chat.
//!
//! ADR note: a new OpenAI-compatible read endpoint with **no new standing cost,
//! no DB, no new dependency** (reuses reqwest/serde + the existing Cohere
//! adapter) is incremental — embeddings shipped the same way without an ADR.
//! No architectural shift, so no ADR is written.

use crate::auth::{TenantContext, VirtualKey};
use crate::embeddings::{apply_advisory_headers, apply_warning_header, resolve_api_key};
use crate::guardrails::GuardrailConfig;
use crate::ledger_sink;
use crate::ledger_sink::{Outcome, UsageTotals};
use crate::observability::UsageEvent;
use crate::proxy::AppState;
use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use routeplane_adapters::ProviderError;
use routeplane_limits::{estimate_cost_micro_usd, now_unix_ms, Admission};
use routeplane_router::RoutingStrategy;
use routeplane_types::{Region, RerankRequest};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    crate::api_error::OpenAiJson(mut payload): crate::api_error::OpenAiJson<RerankRequest>,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let request_deadline = state.deadline_config.request_deadline;
    let per_attempt = state.deadline_config.per_attempt_timeout;
    let registry = &state.providers;

    // 0. Validate the request shape at the route edge: an empty document set is
    //    a malformed request independent of the provider — reject it with a
    //    clean 422 invalid_request envelope BEFORE any residency/masking/network
    //    work (never a panic, never a generic 500). The Cohere adapter guards the
    //    same case as defense-in-depth.
    if payload.documents.is_empty() {
        return crate::api_error::error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            "`documents` must contain at least one string.",
            "invalid_request_error",
            Some("documents"),
        );
    }

    let guards = state
        .limits
        .resolve(&virtual_key.routeplane_key, &tenant_ctx.tenant_id);

    // 1. Residency: classify the joined query + documents BEFORE masking —
    //    masking would hide the very PII the classifier looks for (same
    //    invariant as chat/embeddings).
    let mut original_text = String::with_capacity(payload.query.len());
    original_text.push_str(&payload.query);
    for d in &payload.documents {
        original_text.push('\n');
        original_text.push_str(d);
    }
    let classification = state.residency_engine.classify(&original_text);
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

    // 2. Pre-guardrails: mask PII in the query AND every document by default, so
    //    /v1/rerank does not become a PII-egress bypass (the same always-on
    //    masking the chat/embeddings path uses). Mask-then-rerank: a masked
    //    document changes the relevance score vs the raw text — accepted, the
    //    same documented tradeoff as mask-then-embed.
    let guard_config = GuardrailConfig::masking();
    payload.query = state
        .guardrail_engine
        .process_text(&payload.query, &guard_config);
    payload.documents = payload
        .documents
        .iter()
        .map(|d| state.guardrail_engine.process_text(d, &guard_config))
        .collect();

    // 3. Eligibility: a required region is a HARD filter over the registry's
    //    resident providers (overrides the client's chain); otherwise the
    //    client's x-routeplane-provider chain (default `cohere` — OpenAI has no
    //    rerank endpoint, Cohere is the canonical rerank backend).
    let eligible: Vec<String> = if let Some(region) = &required_region {
        let mut names: Vec<String> = registry
            .iter()
            .filter(|(_, p)| p.is_resident_in(region.as_str()))
            .map(|(name, _)| name.to_string())
            .collect();
        names.sort();
        if names.is_empty() {
            tracing::warn!(
                "Sovereign block (rerank): personal data requires {}-residency but no resident provider is eligible (entities={:?})",
                region.as_str(),
                classification.entities
            );
            state
                .observability_engine
                .record_usage(UsageEvent::sovereign_block(
                    virtual_key.name.clone(),
                    payload.model.clone(),
                    Some(region.0.clone()),
                ));
            ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
                ledger_sink::decision_draft(
                    &tenant_ctx.tenant_id,
                    &request_id,
                    &payload.model,
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
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "Sovereign routing: request contains personal data but no {}-resident provider is configured",
                    region.as_str()
                ),
            )
                .into_response();
        }
        tracing::info!(
            "Sovereign routing enforced (rerank): region={} eligible={:?}",
            region.as_str(),
            names
        );
        names
    } else {
        match headers
            .get("x-routeplane-provider")
            .and_then(|h| h.to_str().ok())
        {
            // Explicit addressing (comma chain) — unchanged, and it may name a
            // runtime custom provider directly (resolved in the loop below).
            Some(requested) => {
                let chain: Vec<String> = requested
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if chain.is_empty() {
                    vec!["cohere".to_string()]
                } else {
                    chain
                }
            }
            // No header: runtime custom-provider MODEL routing — the SAME
            // precedence as chat (a custom provider never shadows a built-in
            // catalog id). One lock-free `ArcSwap::load` + `HashMap` probe;
            // an empty registry ⇒ instant miss ⇒ byte-identical legacy default.
            None => match state
                .custom_providers
                .provider_for_model(&payload.model)
                .filter(|_| !crate::models_api::is_builtin_model(&payload.model))
            {
                Some(custom) => vec![custom],
                None => vec!["cohere".to_string()],
            },
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
            "limit rejection (rerank): tenant={} kind={} scope={} policy={}",
            tenant_ctx.tenant_id,
            breach.kind_header(),
            breach.scope_header(),
            breach.policy_id()
        );
        state.observability_engine.record_usage(UsageEvent::failure(
            virtual_key.name.clone(),
            format!("({})", breach.kind_header()),
            payload.model.clone(),
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
        // Built-in registry FIRST, then the runtime custom registry — the same
        // `resolve_provider` resolution chat uses (one lock-free ArcSwap load +
        // HashMap probe; the owned Arc clone is one refcount bump per attempt).
        let Some(provider) = state.resolve_provider(provider_name.as_str()) else {
            last_error = format!("Unsupported provider: {provider_name}");
            continue;
        };
        if !state.health.is_available(provider_name) {
            tracing::warn!("Skipping {} — circuit breaker is OPEN", provider_name);
            last_error = format!("circuit breaker open for {provider_name}");
            continue;
        }
        // Key precedence: the virtual key's authored `provider_keys` entry (if
        // one exists for this name), else a runtime custom provider's
        // registered upstream key — identical to the chat path's fallback.
        let api_key = match resolve_api_key(&virtual_key, provider_name)
            .or_else(|| state.custom_providers.api_key(provider_name))
        {
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
            "Attempting rerank via {} (sovereign={} timeout={}ms)",
            provider_name,
            sovereign,
            attempt_timeout.as_millis()
        );

        let started = Instant::now();
        let result =
            match tokio::time::timeout(attempt_timeout, provider.rerank(payload.clone(), api_key))
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

                // Rerank bills in search-units (Cohere), not tokens; record the
                // search-unit count as total_tokens for usage/cost so the FinOps
                // path has a real number (prompt/completion = 0).
                let units = response.usage.total_tokens;
                let route_region = provider.resident_regions().into_iter().next();
                ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
                    ledger_sink::decision_draft(
                        &tenant_ctx.tenant_id,
                        &request_id,
                        &response.model,
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
                    response.model.clone(),
                    units,
                    0,
                    units,
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                ));

                let settle_now = now_unix_ms();
                let cost = estimate_cost_micro_usd(&response.model, units, 0);
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
                return ok;
            }
            Err(e) => {
                // A provider that lacks first-party rerank returns a typed 422
                // `rerank_not_supported`. That is a CAPABILITY GAP, not a health
                // fault — do NOT trip its circuit breaker (shared with chat) and
                // do NOT pollute its latency EWMA.
                let this_not_supported = matches!(
                    &e,
                    ProviderError::BadRequest { status: 422, body, .. }
                        if body.starts_with("rerank_not_supported")
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
                    payload.model.clone(),
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                    last_error.clone(),
                ));
                tracing::warn!(
                    "Rerank via {} failed: {}. Trying fallback...",
                    provider_name,
                    last_error
                );
                continue;
            }
        }
    }

    // 7. Exhausted. A pure unsupported-rerank outcome is an explicit 422
    //    envelope (never a generic 500); anything else is the all-failed 500.
    if last_not_supported {
        return rerank_not_supported_response();
    }
    ledger_sink::record_decision(&state.ledger, &tenant_ctx.capabilities, || {
        ledger_sink::decision_draft(
            &tenant_ctx.tenant_id,
            &request_id,
            &payload.model,
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
        "rerank: all providers failed (request_id={}): {}",
        request_id,
        last_error
    );
    crate::api_error::upstream_all_failed()
}

/// The explicit 422 `rerank_not_supported` envelope — an OpenAI-shaped error,
/// not a generic failure, so a client routing rerank to a provider without a
/// first-party rerank endpoint gets an actionable message.
fn rerank_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party rerank endpoint. Route /v1/rerank to a rerank-capable provider (cohere) via x-routeplane-provider.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "rerank_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}
