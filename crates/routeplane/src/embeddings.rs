//! `POST /v1/embeddings` — the OpenAI-compatible embeddings route (G2.7,
//! PRD-011 §5 FR-1..7).
//!
//! This is a NEW module (the chat orchestrator in `proxy.rs` is untouched). It
//! runs a deliberately lean slice of the same pipeline as chat: auth (extensions
//! injected by `auth_middleware`), residency classification on the INPUT text,
//! PII masking of the inputs (mask-then-embed, FR-5), provider eligibility
//! (sovereign filter when a region is required), simple router ordering,
//! budget/rate-limit admission, and a per-attempt-timeout fallback loop.
//!
//! Deliberate scope cuts vs chat (documented deviations, see the PR body):
//!   * NO response cache (FR-6) and NO streaming (embeddings are non-streaming;
//!     `EmbeddingRequest` carries no `stream` field).
//!   * NO routing-policy config / Guardrails v2 in v1 — PII masking and usage
//!     events cover FR-5/FR-7; the rest are additive fast-follows. (Audit-ledger
//!     records now mirror chat — F5/ADR-021 — gated on the AuditLedger
//!     capability, so a non-entitled tenant is byte-identical.)
//!   * `resolve_api_key` / `limit_rejection_response` / `apply_advisory_headers`
//!     are duplicated from `proxy.rs` (which must not be rewritten) so the
//!     429/402/advisory envelopes stay byte-identical.

use crate::auth::{TenantContext, VirtualKey};
use crate::guardrails::GuardrailConfig;
use crate::ledger_sink;
use crate::ledger_sink::{Outcome, UsageTotals};
use crate::observability::UsageEvent;
use crate::proxy::AppState;
use axum::{
    extract::{Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use routeplane_adapters::ProviderError;
use routeplane_limits::{
    estimate_cost_micro_usd, now_unix_ms, Admission, Advisory, Breach, BudgetWarning, LimitKind,
};
use routeplane_router::RoutingStrategy;
use routeplane_types::{EmbeddingInput, EmbeddingRequest, Region};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    crate::api_error::OpenAiJson(mut payload): crate::api_error::OpenAiJson<EmbeddingRequest>,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let request_deadline = state.deadline_config.request_deadline;
    let per_attempt = state.deadline_config.per_attempt_timeout;
    let registry = &state.providers;

    let guards = state
        .limits
        .resolve(&virtual_key.routeplane_key, &tenant_ctx.tenant_id);

    // 1. Residency: classify the joined input text(s) BEFORE masking — masking
    //    would hide the very PII the classifier looks for (the same invariant as
    //    chat). FR-4/FR-5.
    let inputs = payload.input.to_vec();
    let original_text = inputs.join("\n");
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

    // 2. Pre-guardrails: mask PII in every input by default (FR-5, mask-then-
    //    embed). DOCUMENTED FIDELITY TRADEOFF: masking alters the embedded text,
    //    so a masked input yields a different vector than the raw text. v1 is
    //    safe-by-default; a "faithful in-region embedding without masking" mode
    //    is the deferred per-config option (PRD-011 decision 011-G).
    let guard_config = GuardrailConfig::masking();
    let masked_input = match &payload.input {
        EmbeddingInput::Single(s) => {
            EmbeddingInput::Single(state.guardrail_engine.process_text(s, &guard_config))
        }
        EmbeddingInput::Batch(v) => EmbeddingInput::Batch(
            v.iter()
                .map(|t| state.guardrail_engine.process_text(t, &guard_config))
                .collect(),
        ),
    };
    payload.input = masked_input;

    // 3. Eligibility: a required region is a HARD filter over the registry's
    //    resident providers (overrides the client's chain); otherwise the
    //    client's x-routeplane-provider chain (default openai). FR-4.
    let eligible: Vec<String> = if let Some(region) = &required_region {
        let mut names: Vec<String> = registry
            .iter()
            .filter(|(_, p)| p.is_resident_in(region.as_str()))
            .map(|(name, _)| name.to_string())
            .collect();
        names.sort();
        if names.is_empty() {
            tracing::warn!(
                "Sovereign block (embeddings): personal data requires {}-residency but no resident provider is eligible (entities={:?})",
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
            // F5: mirror chat's sovereign-block ledger record (Outcome::ResidencyBlocked).
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
            "Sovereign routing enforced (embeddings): region={} eligible={:?}",
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
                    vec!["openai".to_string()]
                } else {
                    chain
                }
            }
            // No header: runtime custom-provider MODEL routing — the SAME
            // precedence as chat (a custom provider never shadows a built-in
            // catalog id; reach it explicitly via `x-routeplane-provider`).
            // One lock-free `ArcSwap::load` + `HashMap` probe; an empty
            // registry is an instant miss ⇒ byte-identical legacy default.
            None => match state
                .custom_providers
                .provider_for_model(&payload.model)
                .filter(|_| !crate::models_api::is_builtin_model(&payload.model))
            {
                Some(custom) => vec![custom],
                None => vec!["openai".to_string()],
            },
        }
    };

    // 4. Ordering: strategy + health (drops circuit-OPEN providers). Embeddings
    //    use the same priority/cost/latency/weighted ordering as chat.
    let strategy = headers
        .get("x-routeplane-strategy")
        .and_then(|h| h.to_str().ok())
        .map(RoutingStrategy::parse)
        .unwrap_or_default();
    let ordered = state
        .router
        .order_candidates(&eligible, strategy, &state.health);

    // 5. Budgets & rate limits admission — check-before, fail-stop (FR-4).
    let admit_now = now_unix_ms();
    if let Admission::Denied(breach) = guards.admit(admit_now) {
        tracing::info!(
            "limit rejection (embeddings): tenant={} kind={} scope={} policy={}",
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
        return limit_rejection_response(&breach);
    }

    // 6. Attempt loop — no streaming, no cache (FR-6). First success wins.
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
            "Attempting embeddings via {} (sovereign={} timeout={}ms)",
            provider_name,
            sovereign,
            attempt_timeout.as_millis()
        );

        let started = Instant::now();
        let result = match tokio::time::timeout(
            attempt_timeout,
            provider.embeddings(payload.clone(), api_key),
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

                // F5: mirror chat's success ledger record (Outcome::Ok), with the
                // provider's first resident region and embeddings token totals
                // (completion_tokens = 0).
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
                            prompt_tokens: response.usage.prompt_tokens,
                            completion_tokens: 0,
                            total_tokens: response.usage.total_tokens,
                        },
                    )
                });

                // FR-7 observability: completion_tokens = 0 for embeddings.
                state.observability_engine.record_usage(UsageEvent::success(
                    virtual_key.name.clone(),
                    provider_name.clone(),
                    response.model.clone(),
                    response.usage.prompt_tokens,
                    0,
                    response.usage.total_tokens,
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                ));

                let settle_now = now_unix_ms();
                let cost =
                    estimate_cost_micro_usd(&response.model, response.usage.prompt_tokens, 0);
                // Soft-budget: fan edge-triggered crossings out to the existing
                // off-path export seam, once per window per scope.
                for a in &guards.settle(settle_now, response.usage.total_tokens as u64, cost) {
                    state.export_spend_alert(&tenant_ctx.tenant_id, a);
                }

                // FR-6: embeddings responses carry NO cache header.
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
                if let Ok(v) = HeaderValue::from_str(&request_id) {
                    ok.headers_mut().insert("x-routeplane-request-id", v);
                }
                return ok;
            }
            Err(e) => {
                // A provider that lacks first-party embeddings (Anthropic) returns
                // a typed 422 `embeddings_not_supported` (FR-3). That is a
                // CAPABILITY GAP, not a health fault — do NOT trip its circuit
                // breaker (shared with chat) and do NOT pollute its latency EWMA.
                let this_not_supported = matches!(
                    &e,
                    ProviderError::BadRequest { status: 422, body, .. }
                        if body.starts_with("embeddings_not_supported")
                );
                if !this_not_supported {
                    state.health.record_latency(provider_name, elapsed_ms);
                    // F12 (ADR-021 A1): a 429 is the caller's key/quota throttle,
                    // not provider health — record latency but never trip the
                    // breaker on it.
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
                    "Embeddings via {} failed: {}. Trying fallback...",
                    provider_name,
                    last_error
                );
                continue;
            }
        }
    }

    // 7. Exhausted. A pure unsupported-embeddings outcome is an explicit 422
    //    envelope (FR-3, never a generic 500); anything else is the all-failed
    //    500 (matching the chat path's last-error 500).
    if last_not_supported {
        return embeddings_not_supported_response();
    }
    // F5: mirror chat's exhausted-fallback ledger record (Outcome::AllFailed).
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
        "embeddings: all providers failed (request_id={}): {}",
        request_id,
        last_error
    );
    crate::api_error::upstream_all_failed()
}

/// The explicit 422 `embeddings_not_supported` envelope (PRD-011 FR-3) — an
/// OpenAI-shaped error, not a generic failure, so a client routing embeddings to
/// a chat-only provider (Anthropic) gets an actionable message.
fn embeddings_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party embeddings endpoint. Route /v1/embeddings to an embeddings-capable provider (openai, azure_openai, gemini) via x-routeplane-provider.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "embeddings_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}

/// Resolve a provider's API key from the virtual key, expanding `env:`
/// indirection. Shared by embeddings / images / moderations / rerank / audio.
///
/// ADR-087 multi-account: a comma-pool value resolves to its **first resolvable
/// pool element** (the same behavior as `proxy::resolve_api_key`). These endpoints
/// do not yet do intra-pool failover — that is the chat attempt loop's job — but
/// they MUST parse a pool so a pooled provider keeps working here rather than
/// hard-failing (the pre-fix bug: an `env:` pool looked up a variable literally
/// named `A,env:B` → None; a literal pool was sent whole as the bearer key → 401).
pub(crate) fn resolve_api_key(virtual_key: &VirtualKey, provider_name: &str) -> Option<String> {
    let value = virtual_key.provider_keys.get(provider_name)?;
    if crate::proxy::is_key_pool(value) {
        return crate::proxy::resolve_pool(value)
            .into_iter()
            .next()
            .map(|(_, key)| key);
    }
    let mut api_key = value.clone();
    if let Some(env_var) = api_key.strip_prefix("env:") {
        api_key = std::env::var(env_var).unwrap_or_default();
    }
    if api_key.is_empty() {
        None
    } else {
        Some(api_key)
    }
}

/// The OpenAI-shaped 429/402 limit envelope + truthful headers. Duplicated
/// verbatim from `proxy.rs` so the embeddings route's rejection is byte-identical
/// to chat's (PRD-008 FR-17/18/19).
pub(crate) fn limit_rejection_response(breach: &Breach) -> Response {
    let (status, err_type, code) = if breach.is_budget() {
        (
            StatusCode::PAYMENT_REQUIRED,
            "insufficient_quota",
            "routeplane_budget_exceeded",
        )
    } else {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "routeplane_rate_limit_exceeded",
        )
    };
    let body = json!({
        "error": {
            "message": breach.message(),
            "type": err_type,
            "param": serde_json::Value::Null,
            "code": code,
        }
    });
    let mut resp = (status, Json(body)).into_response();
    let h = resp.headers_mut();
    h.insert(
        "x-routeplane-limit-type",
        HeaderValue::from_static(breach.kind_header()),
    );
    h.insert(
        "x-routeplane-limit-scope",
        HeaderValue::from_static(breach.scope_header()),
    );
    if let Ok(v) = HeaderValue::from_str(breach.policy_id()) {
        h.insert("x-routeplane-limit-policy", v);
    }
    if let Some(secs) = breach.retry_after_secs() {
        if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
            h.insert("retry-after", v);
        }
    }
    match breach.kind() {
        LimitKind::RateRequests => {
            if let Ok(v) = HeaderValue::from_str(&breach.limit().to_string()) {
                h.insert("x-ratelimit-limit-requests", v);
            }
            h.insert(
                "x-ratelimit-remaining-requests",
                HeaderValue::from_static("0"),
            );
            if let Some(secs) = breach.reset_secs() {
                if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                    h.insert("x-ratelimit-reset-requests", v);
                }
            }
        }
        LimitKind::RateTokens => {
            if let Ok(v) = HeaderValue::from_str(&breach.limit().to_string()) {
                h.insert("x-ratelimit-limit-tokens", v);
            }
            h.insert(
                "x-ratelimit-remaining-tokens",
                HeaderValue::from_static("0"),
            );
            if let Some(secs) = breach.reset_secs() {
                if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                    h.insert("x-ratelimit-reset-tokens", v);
                }
            }
        }
        LimitKind::BudgetCost | LimitKind::BudgetTokens => {}
    }
    resp
}

/// Advisory `x-ratelimit-*` / budget headers on a successful response.
/// Duplicated verbatim from `proxy.rs`.
pub(crate) fn apply_advisory_headers(headers: &mut HeaderMap, a: &Advisory) {
    fn put(headers: &mut HeaderMap, name: &'static str, val: u64) {
        if let Ok(v) = HeaderValue::from_str(&val.to_string()) {
            headers.insert(name, v);
        }
    }
    if let Some(v) = a.limit_requests {
        put(headers, "x-ratelimit-limit-requests", v);
    }
    if let Some(v) = a.remaining_requests {
        put(headers, "x-ratelimit-remaining-requests", v);
    }
    if let Some(v) = a.reset_requests_secs {
        put(headers, "x-ratelimit-reset-requests", v);
    }
    if let Some(v) = a.limit_tokens {
        put(headers, "x-ratelimit-limit-tokens", v);
    }
    if let Some(v) = a.remaining_tokens {
        put(headers, "x-ratelimit-remaining-tokens", v);
    }
    if let Some(v) = a.budget_remaining_micro_usd {
        put(headers, "x-routeplane-budget-remaining", v);
    }
}

/// Soft-budget warning header `x-routeplane-budget-warning` for the embeddings
/// path. Duplicated verbatim from `proxy.rs` (same convention as
/// `apply_advisory_headers` above). Present only in the warning zone ⇒ additive.
pub(crate) fn apply_warning_header(headers: &mut HeaderMap, w: &BudgetWarning) {
    let v = format!(
        "{}; scope={}; period={}; threshold={}",
        w.consumed_permille,
        w.scope.header(),
        w.period.code(),
        w.threshold_permille
    );
    if let Ok(hv) = HeaderValue::from_str(&v) {
        headers.insert("x-routeplane-budget-warning", hv);
    }
}
