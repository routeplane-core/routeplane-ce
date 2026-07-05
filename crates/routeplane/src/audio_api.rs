//! `POST /v1/audio/transcriptions` — the OpenAI-compatible speech-to-text route
//! (PARITY: OpenAI exposes `/v1/audio/transcriptions`, LiteLLM/Portkey proxy
//! audio). Closes the audio parity gap: Routeplane had no audio surface. Backed
//! by OpenAI AND Groq — Groq's Whisper (`whisper-large-v3` /
//! `whisper-large-v3-turbo`) is the flagship fast/cheap STT use case.
//!
//! This is a NEW module modelled on `rerank_api.rs` / `moderations_api.rs` (the
//! chat orchestrator in `proxy.rs` is untouched). It runs the same lean slice of
//! the pipeline: auth (extensions injected by `auth_middleware`), provider
//! eligibility (sovereign filter when a region is required), router ordering,
//! budget/rate-limit admission, and a per-attempt-timeout fallback loop.
//! `resolve_api_key` / `limit_rejection_response` / `apply_advisory_headers` are
//! reused from the sibling `embeddings` module so the 429/402/advisory envelopes
//! stay byte-identical to chat.
//!
//! ## INBOUND CONTRACT IS MULTIPART (not JSON)
//! Unlike every other vertical (chat/embeddings/rerank/moderations/images), the
//! inbound body is `multipart/form-data` with a binary `file` part + text fields
//! (`model` required; optional `language`, `prompt`, `response_format`,
//! `temperature`). The handler uses axum's `Multipart` extractor, buffers the
//! file under a route-specific body cap (`RP_AUDIO_MAX_BODY_BYTES`, ~26 MiB,
//! applied in `main.rs`), and threads the fields to the adapter via
//! `TranscriptionInput`. A missing `file` or `model` is a clean 400 (never a
//! panic). The audio bytes are NEVER logged.
//!
//! ## SOVEREIGNTY / PII CAVEAT (IMPORTANT — deliberate deviation)
//! Audio is BINARY. It cannot be text-PII-masked or residency-classified the way
//! chat/rerank/embeddings *text* inputs are — there is no text to scan inbound.
//! Therefore:
//!   * There is NO text masking step (there is nothing to mask).
//!   * There is NO content-derived residency classification (the classifier
//!     reads text; binary audio yields nothing). We use an EMPTY classification.
//!   * Residency-region eligibility STILL applies: a region required by the
//!     virtual key or the `x-routeplane-residency` header is a HARD filter to
//!     resident providers. If no resident provider supports transcription, we
//!     fail with the standard region error (same as the siblings).
//!
//! Audio CONTENT is not guardrailed in this pass; a follow-on could add
//! audio-aware scanning (transcribe-then-classify, or an audio moderation
//! detector). This is documented, not silent.
//!
//! ## Provider selection
//! `x-routeplane-provider` (default `openai`; `groq` allowed, comma chain for
//! fallback). A provider with no first-party STT endpoint returns a typed 422
//! `transcription_not_supported`.
//!
//! ## Audio TRANSLATIONS (`POST /v1/audio/translations`) — the near-twin
//! `/v1/audio/translations` (speech-in-any-language → ENGLISH text) is the
//! near-twin of transcriptions: identical multipart inbound contract EXCEPT there
//! is no `language` field (the output is always English). It REUSES this module's
//! whole slice — the same multipart parse, residency posture, audio-not-text-
//! maskable caveat, provider selection (default `openai`; `groq`), and attempt
//! loop — via the shared [`run_audio_text`] core. The only differences are the
//! per-attempt call (`provider.translate` vs `provider.transcribe`) and the typed
//! 422 envelope (`translation_not_supported`). Whisper models (`whisper-1`,
//! `whisper-large-v3`) support translations; the gpt-4o-transcribe models do not.
//!
//! ADR note: a new OpenAI-compatible endpoint with **no new standing cost, no DB,
//! and no new crate** (multipart is a *feature* on the existing reqwest/axum
//! deps; reuses the OpenAI + Groq adapters) is incremental — rerank/moderations/
//! images shipped the same way without an ADR. No architectural shift, so no ADR
//! is written.

use crate::auth::{TenantContext, VirtualKey};
use crate::embeddings::{apply_advisory_headers, apply_warning_header, resolve_api_key};
use crate::guardrails::GuardrailConfig;
use crate::ledger_sink;
use crate::ledger_sink::{Outcome, UsageTotals};
use crate::observability::UsageEvent;
use crate::proxy::AppState;
use axum::{
    extract::{Json, Multipart, State},
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_limits::{now_unix_ms, Admission};
use routeplane_residency::Classification;
use routeplane_router::RoutingStrategy;
use routeplane_types::{Region, SpeechRequest, TranscriptionInput, TranscriptionParams};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

/// The two audio-to-text operations served by this module. They share the entire
/// pipeline slice (multipart parse, residency posture, provider selection,
/// attempt loop) and differ only in the per-attempt provider call and the typed
/// 422 capability-gap envelope. OpenAI/Groq expose them as near-twin endpoints:
/// `/v1/audio/transcriptions` (speech → text in the SAME language) and
/// `/v1/audio/translations` (speech-in-any-language → ENGLISH text, no `language`
/// field).
#[derive(Clone, Copy)]
enum AudioOp {
    Transcribe,
    Translate,
}

impl AudioOp {
    /// The human label used in tracing/log lines.
    fn label(self) -> &'static str {
        match self {
            AudioOp::Transcribe => "transcription",
            AudioOp::Translate => "translation",
        }
    }
    /// The `*_not_supported` body prefix the typed 422 carries, so the loop can
    /// recognise a capability gap without parsing.
    fn not_supported_prefix(self) -> &'static str {
        match self {
            AudioOp::Transcribe => "transcription_not_supported",
            AudioOp::Translate => "translation_not_supported",
        }
    }
    /// The explicit 422 capability-gap response for this op.
    fn not_supported_response(self) -> Response {
        match self {
            AudioOp::Transcribe => transcription_not_supported_response(),
            AudioOp::Translate => translation_not_supported_response(),
        }
    }
}

/// `POST /v1/audio/transcriptions` — OpenAI-compatible speech-to-text (same
/// language in, text out). A thin wrapper over the shared [`run_audio_text`] core.
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    run_audio_text(
        AudioOp::Transcribe,
        state,
        virtual_key,
        tenant_ctx,
        headers,
        multipart,
    )
    .await
}

/// `POST /v1/audio/translations` — OpenAI-compatible audio translation
/// (speech-in-any-language → ENGLISH text). The near-twin of [`transcriptions`]:
/// it REUSES the identical pipeline slice via the shared [`run_audio_text`] core,
/// differing only in the per-attempt provider call (`translate`) and the typed
/// 422 envelope (`translation_not_supported`). There is no `language` field (the
/// output is always English) — even if a caller sends one in the multipart, the
/// adapter omits it on egress. Same audio-not-text-maskable / residency posture
/// as transcriptions; rides the SAME dedicated audio router (larger body limit).
pub async fn translations(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    run_audio_text(
        AudioOp::Translate,
        state,
        virtual_key,
        tenant_ctx,
        headers,
        multipart,
    )
    .await
}

/// The shared core for BOTH audio-to-text endpoints (transcriptions +
/// translations). Runs the identical lean pipeline slice — multipart parse,
/// EMPTY residency classification (binary audio is not text-maskable), region
/// eligibility, router ordering, budget/rate-limit admission, and a
/// per-attempt-timeout fallback loop. The `op` parameter selects the per-attempt
/// provider call (`transcribe` vs `translate`) and the typed 422 capability-gap
/// envelope; everything else is byte-identical between the two endpoints.
async fn run_audio_text(
    op: AudioOp,
    state: Arc<AppState>,
    virtual_key: VirtualKey,
    tenant_ctx: TenantContext,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let request_deadline = state.deadline_config.request_deadline;
    let per_attempt = state.deadline_config.per_attempt_timeout;
    let registry = &state.providers;

    // 0. Parse the multipart body: extract the file bytes + form fields. A
    //    missing `file` or `model` is a malformed request → clean 400 (never a
    //    panic, never a generic 500). The audio bytes are NEVER logged.
    let parsed = match parse_multipart(multipart).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };
    let ParsedTranscription {
        file_bytes,
        filename,
        params,
    } = parsed;
    let model_label = params.model.clone();

    let guards = state
        .limits
        .resolve(&virtual_key.routeplane_key, &tenant_ctx.tenant_id);

    // 1. Residency: audio is BINARY — there is no text to classify and no text to
    //    mask (the deliberate deviation, see module docs). We use an EMPTY
    //    classification (no personal data detectable from binary). Region
    //    ELIGIBILITY still applies from the key/header.
    let classification = Classification {
        contains_personal_data: false,
        entities: Vec::new(),
    };
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

    // 2. Eligibility: a required region is a HARD filter over the registry's
    //    resident providers (overrides the client's chain); otherwise the
    //    client's x-routeplane-provider chain (default `openai`; `groq` allowed).
    let eligible: Vec<String> = if let Some(region) = &required_region {
        let mut names: Vec<String> = registry
            .iter()
            .filter(|(_, p)| p.is_resident_in(region.as_str()))
            .map(|(name, _)| name.to_string())
            .collect();
        names.sort();
        if names.is_empty() {
            tracing::warn!(
                "Sovereign block ({}s): {}-residency required but no resident provider is eligible",
                op.label(),
                region.as_str()
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
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "Sovereign routing: a {}-resident provider is required but none is configured for {}",
                    region.as_str(),
                    op.label()
                ),
            )
                .into_response();
        }
        tracing::info!(
            "Sovereign routing enforced ({}s): region={} eligible={:?}",
            op.label(),
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

    // 3. Ordering: strategy + health (drops circuit-OPEN providers).
    let strategy = headers
        .get("x-routeplane-strategy")
        .and_then(|h| h.to_str().ok())
        .map(RoutingStrategy::parse)
        .unwrap_or_default();
    let ordered = state
        .router
        .order_candidates(&eligible, strategy, &state.health);

    // 4. Budgets & rate limits admission — check-before, fail-stop.
    let admit_now = now_unix_ms();
    if let Admission::Denied(breach) = guards.admit(admit_now) {
        tracing::info!(
            "limit rejection ({}s): tenant={} kind={} scope={} policy={}",
            op.label(),
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

    // 5. Attempt loop — no streaming, no cache. First success wins. The audio
    //    bytes + filename are cloned per attempt so fallback can retry the next
    //    provider with the same file (reqwest::multipart consumes the body).
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
            "Attempting {} via {} (sovereign={} timeout={}ms)",
            op.label(),
            provider_name,
            sovereign,
            attempt_timeout.as_millis()
        );

        // Clone the bytes per attempt (the form consumes them). Audio is capped
        // at ~26 MiB so this is bounded; the bytes are never logged.
        let audio = TranscriptionInput {
            file_bytes: file_bytes.clone(),
            filename: filename.clone(),
            params: params.clone(),
        };

        // Dispatch to the op-specific provider call. transcribe/translate share
        // the TranscriptionInput/TranscriptionResponse types; translate omits the
        // `language` field on egress (handled in the adapter).
        let call = async {
            match op {
                AudioOp::Transcribe => provider.transcribe(audio, api_key).await,
                AudioOp::Translate => provider.translate(audio, api_key).await,
            }
        };
        let started = Instant::now();
        let result = match tokio::time::timeout(attempt_timeout, call).await {
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

                // Transcription bills by audio DURATION/seconds upstream, not
                // tokens — the gateway does not measure audio length here, so we
                // record 0 tokens (the usage event is still emitted so FinOps has
                // a row; cost is settled at 0 gracefully). A follow-on could parse
                // `duration` from verbose_json.
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
                        UsageTotals::default(),
                    )
                });

                state.observability_engine.record_usage(UsageEvent::success(
                    virtual_key.name.clone(),
                    provider_name.clone(),
                    model_label.clone(),
                    0,
                    0,
                    0,
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                ));

                let settle_now = now_unix_ms();
                // Audio has no token-based cost; settle 0 cost gracefully so
                // budgets/advisories still update (request count, not spend).
                for a in &guards.settle(settle_now, 0, 0) {
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
                // A provider that lacks the first-party endpoint returns a typed
                // 422 (`transcription_not_supported` / `translation_not_supported`).
                // That is a CAPABILITY GAP, not a health fault — do NOT trip its
                // circuit breaker (shared with chat) and do NOT pollute its
                // latency EWMA.
                let prefix = op.not_supported_prefix();
                let this_not_supported = matches!(
                    &e,
                    ProviderError::BadRequest { status: 422, body, .. }
                        if body.starts_with(prefix)
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
                    "{} via {} failed: {}. Trying fallback...",
                    op.label(),
                    provider_name,
                    last_error
                );
                continue;
            }
        }
    }

    // 6. Exhausted. A pure unsupported-capability outcome is an explicit 422
    //    envelope (never a generic 500); anything else is the all-failed 500.
    if last_not_supported {
        return op.not_supported_response();
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
        "{}s: all providers failed (request_id={}): {}",
        op.label(),
        request_id,
        last_error
    );
    crate::api_error::upstream_all_failed()
}

/// `POST /v1/audio/speech` — the OpenAI-compatible text-to-speech (TTS) route.
/// Completes the audio pair with [`transcriptions`] (STT). Closes a parity gap:
/// OpenAI exposes `/v1/audio/speech`; LiteLLM/Portkey proxy it.
///
/// ## Inbound JSON, BINARY response (unlike transcription)
/// The request is JSON ([`SpeechRequest`] = `{model?, input, voice,
/// response_format?, speed?}`), so it rides the NORMAL authed router under the
/// standard ~2 MiB body cap (the `input` is text — small). The RESPONSE is raw
/// binary audio bytes (NOT JSON), returned with a per-format `Content-Type`
/// (mp3→audio/mpeg, …) plus the standard `x-routeplane-*` headers.
///
/// ## PII masking posture — SAME as images/rerank (classify-then-mask)
/// The `input` is user TEXT bound for an external API → classified for residency
/// on the RAW text, THEN masked BEFORE egress (so /v1/audio/speech cannot become
/// a PII-egress bypass). This is the SAME always-on masking the
/// chat/embeddings/rerank/images path uses — and the deliberate DIFFERENCE from
/// transcription, where the inbound body is binary audio (nothing to mask/classify).
///
/// ## Provider selection
/// `x-routeplane-provider` (default `openai` — the canonical TTS backend,
/// `gpt-4o-mini-tts` / `tts-1` / `tts-1-hd`). A provider with no first-party TTS
/// endpoint returns a typed 422 `speech_not_supported`.
///
/// ## Usage accounting
/// TTS bills by INPUT characters upstream, not tokens — the gateway records 0
/// tokens/cost gracefully (like STT) but still emits a UsageEvent + ledger row so
/// FinOps has a row.
pub async fn speech(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    headers: HeaderMap,
    crate::api_error::OpenAiJson(mut payload): crate::api_error::OpenAiJson<SpeechRequest>,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let request_deadline = state.deadline_config.request_deadline;
    let per_attempt = state.deadline_config.per_attempt_timeout;
    let registry = &state.providers;

    // 0. Validate the request shape at the route edge: an empty input or voice is
    //    a malformed request independent of the provider — reject with a clean 422
    //    invalid_request envelope BEFORE any residency/masking/network work (never
    //    a panic, never a generic 500).
    if payload.input.trim().is_empty() {
        return crate::api_error::error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            "`input` must be a non-empty string.",
            "invalid_request_error",
            Some("input"),
        );
    }
    if payload.voice.trim().is_empty() {
        return crate::api_error::error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            "`voice` must be a non-empty string.",
            "invalid_request_error",
            Some("voice"),
        );
    }

    // A stable model label for usage/ledger when the caller omits `model` (the
    // adapter fills the provider default at egress; we mirror it here for FinOps).
    let model_label = payload
        .model
        .clone()
        .unwrap_or_else(|| "gpt-4o-mini-tts".to_string());

    let guards = state
        .limits
        .resolve(&virtual_key.routeplane_key, &tenant_ctx.tenant_id);

    // 1. Residency: classify the `input` BEFORE masking — masking would hide the
    //    very PII the classifier looks for (same invariant as chat/images/rerank).
    let classification = state.residency_engine.classify(&payload.input);
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

    // 2. Pre-guardrails: mask PII in the `input` by default, so /v1/audio/speech
    //    does not become a PII-egress bypass (the same always-on masking the
    //    chat/embeddings/rerank/images path uses). Classify-then-mask:
    //    classification (step 1) already ran on the raw text.
    let guard_config = GuardrailConfig::masking();
    payload.input = state
        .guardrail_engine
        .process_text(&payload.input, &guard_config);

    // 3. Eligibility: a required region is a HARD filter over the registry's
    //    resident providers (overrides the client's chain); otherwise the
    //    client's x-routeplane-provider chain (default `openai` — the canonical
    //    TTS backend).
    let eligible: Vec<String> = if let Some(region) = &required_region {
        let mut names: Vec<String> = registry
            .iter()
            .filter(|(_, p)| p.is_resident_in(region.as_str()))
            .map(|(name, _)| name.to_string())
            .collect();
        names.sort();
        if names.is_empty() {
            tracing::warn!(
                "Sovereign block (speech): personal data requires {}-residency but no resident provider is eligible (entities={:?})",
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
            "Sovereign routing enforced (speech): region={} eligible={:?}",
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
            // catalog id). `model` is optional on TTS: absent ⇒ the legacy
            // default. One lock-free `ArcSwap::load` + `HashMap` probe; an
            // empty registry ⇒ instant miss ⇒ byte-identical legacy default.
            None => match payload.model.as_deref().and_then(|m| {
                state
                    .custom_providers
                    .provider_for_model(m)
                    .filter(|_| !crate::models_api::is_builtin_model(m))
            }) {
                Some(custom) => vec![custom],
                None => vec!["openai".to_string()],
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
            "limit rejection (speech): tenant={} kind={} scope={} policy={}",
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

    // 6. Attempt loop — no streaming, no cache. First success wins. The request is
    //    cloned per attempt so fallback can retry the next provider.
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
            "Attempting speech synthesis via {} (sovereign={} timeout={}ms)",
            provider_name,
            sovereign,
            attempt_timeout.as_millis()
        );

        let started = Instant::now();
        let result =
            match tokio::time::timeout(attempt_timeout, provider.speech(payload.clone(), api_key))
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
            Ok(audio) => {
                state.health.record_latency(provider_name, elapsed_ms);
                state.health.record_success(provider_name);

                // TTS bills by input CHARACTERS upstream, not tokens — the gateway
                // does not price audio here, so we record 0 tokens (the usage event
                // is still emitted so FinOps has a row; cost is settled at 0
                // gracefully). Same posture as STT.
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
                        UsageTotals::default(),
                    )
                });

                state.observability_engine.record_usage(UsageEvent::success(
                    virtual_key.name.clone(),
                    provider_name.clone(),
                    model_label.clone(),
                    0,
                    0,
                    0,
                    required_region.as_ref().map(|r| r.0.clone()),
                    sovereign,
                ));

                let settle_now = now_unix_ms();
                // Audio has no token-based cost; settle 0 cost gracefully so
                // budgets/advisories still update (request count, not spend).
                for a in &guards.settle(settle_now, 0, 0) {
                    state.export_spend_alert(&tenant_ctx.tenant_id, a);
                }

                // BINARY response: the raw audio bytes with the per-format
                // Content-Type (from the adapter), NOT JSON. The bytes are never
                // logged. Branding-load-bearing x-routeplane-* headers are echoed.
                let mut ok = (StatusCode::OK, audio.bytes).into_response();
                // Override the default octet-stream content-type with the real one.
                if let Ok(ct) = HeaderValue::from_str(&audio.content_type) {
                    ok.headers_mut().insert(CONTENT_TYPE, ct);
                }
                if let Ok(p) = HeaderValue::from_str(provider_name) {
                    ok.headers_mut().insert("x-routeplane-provider", p);
                }
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
                // A provider that lacks first-party TTS returns a typed 422
                // `speech_not_supported`. That is a CAPABILITY GAP, not a health
                // fault — do NOT trip its circuit breaker (shared with chat) and
                // do NOT pollute its latency EWMA.
                let this_not_supported = matches!(
                    &e,
                    ProviderError::BadRequest { status: 422, body, .. }
                        if body.starts_with("speech_not_supported")
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
                    "Speech synthesis via {} failed: {}. Trying fallback...",
                    provider_name,
                    last_error
                );
                continue;
            }
        }
    }

    // 7. Exhausted. A pure unsupported-speech outcome is an explicit 422 envelope
    //    (never a generic 500); anything else is the all-failed 500.
    if last_not_supported {
        return speech_not_supported_response();
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
        "speech: all providers failed (request_id={}): {}",
        request_id,
        last_error
    );
    crate::api_error::upstream_all_failed()
}

/// The explicit 422 `speech_not_supported` envelope — an OpenAI-shaped error, not
/// a generic failure, so a client routing TTS to a provider without a first-party
/// speech endpoint gets an actionable message.
fn speech_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party text-to-speech endpoint. Route /v1/audio/speech to a speech-capable provider (openai) via x-routeplane-provider.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "speech_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}

/// The buffered file + threaded params parsed from the multipart body.
struct ParsedTranscription {
    file_bytes: Vec<u8>,
    filename: String,
    params: TranscriptionParams,
}

/// Parse the `multipart/form-data` body into the file bytes + form fields.
/// Returns a clean 400 `Response` (never a panic) when the body is malformed,
/// the `file` part is missing, or `model` is missing. The audio bytes are never
/// logged. The route-level `RequestBodyLimitLayer` (~26 MiB) already 413s an
/// oversized body before we buffer it here.
async fn parse_multipart(mut multipart: Multipart) -> Result<ParsedTranscription, Response> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut params = TranscriptionParams::default();

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                // Malformed multipart, OR a body that exceeded the route body cap
                // mid-read (axum surfaces the RequestBodyLimit error here as a
                // MultipartError whose `.status()` is 413). Honour that status so
                // an oversized upload is a clean 413, a malformed body a 400.
                return Err(multipart_error_response(&e));
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                // Capture the filename BEFORE consuming the bytes (the field is
                // borrowed by `bytes()`).
                let fname = field.file_name().map(|s| s.to_string());
                let data = match field.bytes().await {
                    Ok(b) => b,
                    // A 413 here means the audio exceeded the route body cap;
                    // honour the multipart error's status (413 vs 400).
                    Err(e) => return Err(multipart_error_response(&e)),
                };
                filename = Some(fname.unwrap_or_else(|| "audio.wav".to_string()));
                file_bytes = Some(data.to_vec());
            }
            "model" => params.model = read_text_field(field).await?,
            "language" => params.language = Some(read_text_field(field).await?),
            "prompt" => params.prompt = Some(read_text_field(field).await?),
            "response_format" => params.response_format = Some(read_text_field(field).await?),
            "temperature" => {
                let raw = read_text_field(field).await?;
                match raw.trim().parse::<f32>() {
                    Ok(t) => params.temperature = Some(t),
                    Err(_) => {
                        return Err(bad_request(
                            "`temperature` must be a number.",
                            Some("temperature"),
                        ))
                    }
                }
            }
            // Unknown fields are ignored (forward-compatible) — never a panic.
            _ => {
                // Drain the field so the stream advances cleanly.
                let _ = field.bytes().await;
            }
        }
    }

    let file_bytes = match file_bytes {
        Some(b) if !b.is_empty() => b,
        Some(_) => return Err(bad_request("the `file` part is empty.", Some("file"))),
        None => {
            return Err(bad_request(
                "a `file` part (the audio file) is required.",
                Some("file"),
            ))
        }
    };
    if params.model.trim().is_empty() {
        return Err(bad_request("a `model` field is required.", Some("model")));
    }
    let filename = filename.unwrap_or_else(|| "audio.wav".to_string());

    Ok(ParsedTranscription {
        file_bytes,
        filename,
        params,
    })
}

/// Read a multipart text field into a `String`, mapping a read failure to a clean
/// 400 (never a panic). Text fields are tiny (model/language/prompt), so reading
/// them fully into memory is safe under the route body cap.
async fn read_text_field(field: axum::extract::multipart::Field<'_>) -> Result<String, Response> {
    field.text().await.map_err(|e| multipart_error_response(&e))
}

/// Map an axum `MultipartError` to a clean OpenAI-shaped error response, honouring
/// its own status: a body that exceeded the route body cap surfaces here as a 413
/// (Payload Too Large); a genuinely malformed body is a 400. Never a panic, never
/// a 500. The error string is short and carries no body content (it is a parse/
/// limit error, not the audio bytes).
fn multipart_error_response(e: &axum::extract::multipart::MultipartError) -> Response {
    let status = e.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        crate::api_error::error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            "The audio upload exceeds the maximum allowed size.",
            "invalid_request_error",
            Some("file"),
        )
    } else {
        bad_request(&format!("invalid multipart/form-data body: {e}"), None)
    }
}

/// A clean OpenAI-shaped 400 invalid_request envelope (never a panic).
fn bad_request(message: &str, param: Option<&str>) -> Response {
    crate::api_error::error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        "invalid_request_error",
        param,
    )
}

/// The explicit 422 `transcription_not_supported` envelope — an OpenAI-shaped
/// error, not a generic failure, so a client routing transcription to a provider
/// without a first-party STT endpoint gets an actionable message.
fn transcription_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party audio-transcription endpoint. Route /v1/audio/transcriptions to a transcription-capable provider (openai or groq) via x-routeplane-provider.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "transcription_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}

/// The explicit 422 `translation_not_supported` envelope — an OpenAI-shaped error
/// (mirrors [`transcription_not_supported_response`]), so a client routing audio
/// translation to a provider without a first-party `/v1/audio/translations`
/// endpoint gets an actionable message. Translations is a Whisper feature
/// (`whisper-1`, `whisper-large-v3`).
fn translation_not_supported_response() -> Response {
    let body = json!({
        "error": {
            "message": "The selected provider does not offer a first-party audio-translation endpoint. Route /v1/audio/translations to a translation-capable provider (openai or groq) via x-routeplane-provider, using a Whisper model (whisper-1 or whisper-large-v3).",
            "type": "invalid_request_error",
            "param": "model",
            "code": "translation_not_supported"
        }
    });
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}
