use chrono::{DateTime, Utc};
use routeplane_guardrails::CheckOutcome;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use tokio::sync::mpsc;

/// A single recorded request outcome. Now records FAILURES and sovereign-blocks
/// in addition to successes (Task #5), so `/analytics` reflects real traffic
/// (a flapping or refused provider is no longer invisible). Guardrails v2
/// (G2.6) attaches its check results here — `observe` verdicts are *recorded*,
/// never blocking; `deny` outcomes appear on dedicated block events. The cache
/// fields (G2.5 / PRD-007 FR-16) are `Option` + skip-when-None, so a request
/// with no cache config emits a byte-identical event (AC-7). The prompt fields
/// (PRD-010 FR-17) are likewise `Option` + skip-when-None: a non-prompt request
/// (every chat/embeddings call) emits a byte-identical event, and only the
/// lightweight `prompt.render` join event sets them.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UsageEvent {
    pub timestamp: DateTime<Utc>,
    pub virtual_key_name: String,
    pub provider: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Provider-native prompt-cache READ tokens (the cached SUBSET of
    /// `prompt_tokens`): Anthropic `cache_read_input_tokens` /
    /// OpenAI `prompt_tokens_details.cached_tokens`. `Some` only when the provider
    /// reported a cache read; absent (and omitted from the wire) otherwise, so a
    /// non-cached request emits a byte-identical event (A/B parity). Lets
    /// analytics/FinOps see cache savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Business-process / use-case label the caller tagged this request with via
    /// `x-routeplane-use-case` (FinOps cost attribution down to the business
    /// process). `Some` only when the header was present; absent (and omitted from
    /// the wire) otherwise, so an untagged request emits a byte-identical event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_case: Option<String>,
    /// Residency jurisdiction this request was locked to, when sovereign routing
    /// was enforced (e.g. "IN"). None when no residency constraint applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// True when the request carried personal data and was routed to a
    /// region-resident provider to satisfy a data-residency requirement.
    pub sovereign_routed: bool,
    /// Whether the attempt succeeded. False for provider failures/timeouts and
    /// for sovereign blocks (which never reach a provider).
    pub success: bool,
    /// Failure detail (provider error / timeout / "sovereign_block" /
    /// "guardrails_denied"), when this event records a non-success outcome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Guardrails v2 check results for this request (G2.6) — pass/fail/error
    /// verdicts for every evaluated check, both hooks. Absent (and omitted from
    /// the wire) when no checks ran — a tenant without AdvancedGuardrails emits
    /// byte-identical events to today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrails: Option<Vec<CheckOutcome>>,
    /// G2.5 (PRD-007 FR-16): whether this request was served from the exact
    /// cache. FEEDS PRD-009's reserved field of the same name. `Some` only for
    /// cache-participating requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    /// Five-value verdict, snake_case (`hit|miss|refreshed|semantic_hit|bypass`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_status: Option<String>,
    /// The within-tenant cache namespace this request participated in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_namespace: Option<String>,
    /// Upstream cost a hit avoided, in micro-USD (priced via the limits crate's
    /// estimator until the FinOps pricing book lands — FR-16 allows null/None).
    /// Hits attribute UPSTREAM cost $0; this field carries the savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_saved_cost_micro_usd: Option<u64>,
    /// PRD-010 FR-17: the prompt this event is attributed to (the `prompt_<slug>`
    /// id). Set only on the lightweight `prompt.render` join event; None (and
    /// omitted) for every other event, so chat/embeddings events stay
    /// byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<String>,
    /// PRD-010 FR-17: the CONCRETE resolved integer version (recorded even when
    /// the request referenced a label — AC-10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<u32>,
    /// PRD-010 FR-17: the label the prompt was referenced by, if any (else None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_label: Option<String>,
    /// PRD-010 (A/B testing): the experiment NAME the prompt was referenced by,
    /// when resolution went through a weighted experiment. Set only on the
    /// `prompt.render` join event; None (and omitted) otherwise, so a
    /// non-experiment prompt render is byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_experiment: Option<String>,
    /// PRD-010 (A/B testing): the SERVED variant label for this experiment
    /// assignment (alongside the concrete `prompt_version`). Enables per-variant
    /// analytics. None (and omitted) when no experiment participated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_variant: Option<String>,
    /// F14 (G2.2 / ADR-021): coarse routing-config SOURCE label (`"inline"` /
    /// `"saved"`), present only when a routing config participated — otherwise
    /// omitted (byte-identical legacy event).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_ref: Option<String>,
    /// F14: the matched rule label within that config (the branch that fired).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_match: Option<String>,
    /// P1.c (FinOps): per-request cost attribution — canonical micro-USD plus a
    /// region-derived local-currency view. `Some` only on successful provider
    /// calls; absent (and omitted from the wire) otherwise, so non-priced events
    /// stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<routeplane_limits::pricing::CostBreakdown>,
    /// Observed upstream latency for this attempt, in milliseconds. For a
    /// buffered call this is the full provider round-trip; for a stream it is
    /// time-to-first-chunk (the same value fed to the router EWMA). `Some` only
    /// on real provider attempts that were timed; absent (and omitted from the
    /// wire) otherwise, so non-timed events stay byte-identical (A/B parity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// ADR-057 (tail-latency hedging): true when this successful response came
    /// from a speculative HEDGE attempt rather than the primary target. Defaults
    /// false and is OMITTED from the wire when false, so a non-hedged (the
    /// default, no-config) request emits a byte-identical event (A/B parity).
    #[serde(default, skip_serializing_if = "is_false")]
    pub hedged: bool,
    /// ADR-031 / PRD-036 (egress DLP): the opt-in OUTPUT-masking mode the caller
    /// explicitly requested via `x-routeplane-output-mask` (e.g. `"pii"`), when
    /// that masking pass was affirmed on this response. This is an AUDITABLE
    /// signal that egress PII/secret masking was explicitly engaged for the
    /// reply (parity with Lakera/Bedrock egress-DLP surfaces) — the deterministic
    /// `redact` baseline runs regardless, so the body is unchanged by the flag;
    /// this annotation records that the control was deliberately on. `None` (and
    /// OMITTED from the wire) when the header was absent or when reversible
    /// tokenize round-trip took precedence, so a request without the opt-in emits
    /// a byte-identical event (A/B parity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_masked: Option<&'static str>,
    /// Feedback API (`POST /v1/feedback`, Portkey/Helicone parity): the
    /// client-supplied `trace_id` this feedback record references. Set ONLY on
    /// the synthetic `(feedback)` join event; `None` (and OMITTED from the wire)
    /// for every other event, so chat/embeddings/etc. stay byte-identical (A/B
    /// parity). Bounded at the route edge (length-capped, non-empty), so the
    /// in-memory ring can never grow on an unbounded caller-controlled string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_trace_id: Option<String>,
    /// Feedback API: the weighted quality score in `-10..=10`. Set only on the
    /// `(feedback)` event; `None` (and omitted) otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_value: Option<i8>,
    /// Feedback API: the relative weight in `0.0..=1.0` (defaults to `1.0` when
    /// the caller omits it). Set only on the `(feedback)` event; `None` (and
    /// omitted) otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_weight: Option<f32>,
    /// Feedback API: the number of bounded, label-cleaned metadata key/value
    /// pairs the caller attached (a COUNT, never the values — raw caller metadata
    /// is not persisted into this audit-adjacent surface). Set only on the
    /// `(feedback)` event; `None` (and omitted) otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_metadata_keys: Option<u32>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl UsageEvent {
    /// Mark this event as a hedged win (ADR-057). No-op-by-default builder so the
    /// non-hedged path stays byte-identical.
    pub fn with_hedged(mut self, hedged: bool) -> Self {
        self.hedged = hedged;
        self
    }

    /// Annotate that opt-in egress OUTPUT masking (`x-routeplane-output-mask`)
    /// was explicitly engaged for this response (ADR-031 / PRD-036). `None`
    /// (the default) is a no-op and stays omitted from the wire, so a request
    /// without the opt-in emits a byte-identical event (A/B parity). The label
    /// is a `&'static str` (e.g. `"pii"`) — no caller-controlled bytes, no
    /// reflection of response content.
    #[must_use]
    pub fn with_output_masked(mut self, mode: Option<&'static str>) -> Self {
        self.output_masked = mode;
        self
    }

    /// Streaming truth: a stream that truncated mid-flight keeps its observed
    /// (real) token spend but must not read as a success. `None` (the clean-end
    /// path) is a no-op — byte-identical event.
    #[must_use]
    pub fn with_stream_error(mut self, error: Option<String>) -> Self {
        if let Some(e) = error {
            self.success = false;
            self.error = Some(format!("stream truncated: {e}"));
        }
        self
    }

    /// A successful provider call.
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        virtual_key_name: String,
        provider: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        region: Option<String>,
        sovereign_routed: bool,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed,
            success: true,
            error: None,
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// A failed provider attempt (error or timeout). Token counts unknown → 0.
    pub fn failure(
        virtual_key_name: String,
        provider: String,
        model: String,
        region: Option<String>,
        sovereign_routed: bool,
        error: String,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider,
            model,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed,
            success: false,
            error: Some(error),
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// A request refused for data-residency reasons (HTTP 422). No provider was
    /// called; `provider` is the sentinel "(sovereign_block)".
    pub fn sovereign_block(
        virtual_key_name: String,
        model: String,
        region: Option<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider: "(sovereign_block)".to_string(),
            model,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed: true,
            success: false,
            error: Some("sovereign_block".to_string()),
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// A request denied by a `before_request` guardrail (HTTP 446, G2.6). No
    /// provider was called; `provider` is the sentinel "(guardrails_denied)" —
    /// same pattern as the sovereign block. Token counts are zero (nothing was
    /// spent upstream). MOAT (ADR-088): only the advanced deny paths construct
    /// this — enterprise-only.
    #[cfg(feature = "enterprise")]
    pub fn guardrails_block(
        virtual_key_name: String,
        model: String,
        region: Option<String>,
        sovereign_routed: bool,
        outcomes: Vec<CheckOutcome>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider: "(guardrails_denied)".to_string(),
            model,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed,
            success: false,
            error: Some("guardrails_denied".to_string()),
            guardrails: Some(outcomes),
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// A response denied by an `after_request` guardrail (HTTP 446, G2.6). The
    /// provider call SUCCEEDED and tokens were consumed — FinOps must see the
    /// real counts even though the client received a denial. MOAT (ADR-088):
    /// enterprise-only (the advanced output-deny/leak/tool-policy paths).
    #[cfg(feature = "enterprise")]
    #[allow(clippy::too_many_arguments)]
    pub fn guardrails_output_denied(
        virtual_key_name: String,
        provider: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        region: Option<String>,
        sovereign_routed: bool,
        outcomes: Vec<CheckOutcome>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed,
            success: false,
            error: Some("guardrails_denied".to_string()),
            guardrails: Some(outcomes),
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// PRD-010 FR-17: a lightweight `prompt.render` join event. No tokens, no
    /// cost, no provider call — `provider` is the sentinel "(prompt_render)". It
    /// carries the resolved `prompt_id` + integer `prompt_version` + `prompt_label`
    /// so a downstream analytics consumer can attribute traffic to a prompt
    /// version even though the chat pipeline's own usage/cost event (recorded
    /// separately by `proxy.rs`, unchanged) carries no prompt fields.
    pub fn prompt_render(
        virtual_key_name: String,
        model: String,
        prompt_id: String,
        prompt_version: u32,
        prompt_label: Option<String>,
        region: Option<String>,
        sovereign_routed: bool,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider: "(prompt_render)".to_string(),
            model,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            use_case: None,
            region,
            sovereign_routed,
            success: true,
            error: None,
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: Some(prompt_id),
            prompt_version: Some(prompt_version),
            prompt_label,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// Feedback API (`POST /v1/feedback`, Portkey/Helicone parity): a
    /// lightweight `(feedback)` join event. No tokens, no cost, no provider call
    /// — `provider` is the sentinel "(feedback)" (so it is excluded from
    /// traffic/success/latency/chargeback aggregates exactly like the other
    /// synthetic sentinels, which all start with `'('`). It carries the
    /// referenced `trace_id` + the weighted `value`/`weight` + a COUNT of bounded
    /// metadata keys, so a downstream analytics consumer can correlate a quality
    /// signal back to a request trace (and to a served prompt variant). All
    /// inputs are validated + bounded at the route edge; raw caller metadata is
    /// never persisted here.
    pub fn feedback(
        virtual_key_name: String,
        trace_id: String,
        value: i8,
        weight: f32,
        metadata_keys: u32,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            virtual_key_name,
            provider: "(feedback)".to_string(),
            model: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            use_case: None,
            region: None,
            sovereign_routed: false,
            success: true,
            error: None,
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: Some(trace_id),
            feedback_value: Some(value),
            feedback_weight: Some(weight),
            feedback_metadata_keys: Some(metadata_keys),
        }
    }

    /// Attach guardrail check results to this event (G2.6). An empty outcome
    /// list attaches nothing, so a guardrails-free request emits an event
    /// byte-identical to today's (the `guardrails` field is skipped when None).
    pub fn with_guardrails(mut self, outcomes: Vec<CheckOutcome>) -> Self {
        if !outcomes.is_empty() {
            self.guardrails = Some(outcomes);
        }
        self
    }

    /// G2.5 (FR-16): mark this event as an exact-cache HIT. The base event must
    /// carry the STORED response's token counts; upstream cost is $0 by
    /// construction (no provider was called) and `saved_cost_micro_usd` carries
    /// the estimated avoided spend.
    pub fn with_cache_hit(mut self, namespace: Option<String>, saved_cost_micro_usd: u64) -> Self {
        self.cache_hit = Some(true);
        self.cache_status = Some("hit".to_string());
        self.cache_namespace = namespace;
        self.estimated_saved_cost_micro_usd = Some(saved_cost_micro_usd);
        self
    }

    /// G2.5 (FR-16): annotate a cache-PARTICIPATING non-hit outcome
    /// (`miss` / `refreshed` / `bypass`, snake_case). `None` status attaches
    /// nothing — a request with no cache config stays byte-identical (FR-2).
    pub fn with_cache_status(
        mut self,
        status: Option<&'static str>,
        namespace: Option<String>,
    ) -> Self {
        if let Some(s) = status {
            self.cache_hit = Some(false);
            self.cache_status = Some(s.to_string());
            self.cache_namespace = namespace;
        }
        self
    }

    /// F14 (G2.2 / ADR-021): annotate this event with the routing-config source
    /// label + matched rule. `None` `config_ref` attaches nothing (and ignores
    /// `config_match`), so a legacy no-config request stays byte-identical
    /// (skip-when-None on both fields).
    pub fn with_config(
        mut self,
        config_ref: Option<&'static str>,
        config_match: Option<String>,
    ) -> Self {
        if let Some(r) = config_ref {
            self.config_ref = Some(r.to_string());
            self.config_match = config_match;
        }
        self
    }

    /// Attach per-request cost attribution (P1.c FinOps). Additive — events
    /// without it serialize byte-identically.
    pub fn with_cost(mut self, cost: routeplane_limits::pricing::CostBreakdown) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Attach the observed upstream latency (ms) for this attempt. Additive —
    /// events without it serialize byte-identically (skip-when-None).
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    /// PRD-010 (A/B testing): annotate this `prompt.render` event with the
    /// experiment name + served variant label. `None` attaches nothing, so a
    /// non-experiment prompt render (or any other event) stays byte-identical
    /// (skip-when-None on both fields). The concrete served `prompt_version` is
    /// already carried by the base `prompt_render` event.
    pub fn with_experiment(mut self, experiment: Option<(String, String)>) -> Self {
        if let Some((name, variant)) = experiment {
            self.prompt_experiment = Some(name);
            self.prompt_variant = Some(variant);
        }
        self
    }

    /// Attach the provider-reported prompt-cache READ token count (prompt-caching
    /// surfacing). `None` attaches nothing, so a non-cached response stays
    /// byte-identical (skip-when-None); a `Some(0)` is recorded as a real zero.
    pub fn with_cached_tokens(mut self, cached_tokens: Option<u32>) -> Self {
        if let Some(c) = cached_tokens {
            self.cached_tokens = Some(c);
        }
        self
    }

    /// Attach the caller's business-process / use-case label (FinOps attribution).
    /// `None` attaches nothing, so an untagged request stays byte-identical
    /// (skip-when-None). The caller has already trimmed/length-capped the value.
    pub fn with_use_case(mut self, use_case: Option<String>) -> Self {
        if let Some(uc) = use_case {
            self.use_case = Some(uc);
        }
        self
    }
}

/// Bound on the in-flight ingest channel. Bounded (not unbounded) so a stalled
/// consumer or a usage spike applies backpressure / sheds the oldest semantics
/// via `try_send` rather than growing memory without limit (Task #5 / the
/// backpressure-not-buffering principle).
const INGEST_CHANNEL_CAPACITY: usize = 4096;

/// Max retained events in the in-memory ring (unchanged: last 1000, frugal — no
/// DB during Alpha; a DB migration is ADR-gated).
const MAX_RETAINED_EVENTS: usize = 1000;

/// Max retained MCP enforcement events in the in-memory ring (last 200). Same
/// frugal, in-memory, free-tier observability posture as the usage ring — NOT the
/// durable telemetry store (ADR-024), which remains gated.
const MAX_MCP_SECURITY_EVENTS: usize = 200;

/// One MCP-leg enforcement event (a tool-call authorization / egress / quota /
/// result-size / anomaly denial), retained in a bounded in-memory ring so the
/// Console can show a LIVE feed of the agentic-security moat's denials. Label-only
/// and secret-free by construction: `category`/`outcome` are closed-vocab and
/// `detail` is a closed-vocab code (never matched content). `tenant_id` scopes the
/// read so a tenant only ever sees its own events.
#[derive(Clone, Debug, serde::Serialize)]
pub struct McpSecurityEvent {
    /// RFC3339 timestamp.
    pub ts: String,
    /// Closed-vocab category label (e.g. `mcp_authorize_deny`, `mcp_egress_deny`).
    pub category: String,
    /// Closed-vocab outcome code (e.g. `deny`).
    pub outcome: String,
    /// Closed-vocab detail code (e.g. `authorize`, `anomaly_quarantine`) — never content.
    pub detail: Option<String>,
    /// The tenant the event belongs to (used to scope the read).
    pub tenant_id: String,
}

/// In-memory observability sink.
///
/// Recording is now **non-blocking on the hot path** (Task #5): `record_usage`
/// only does a single lock-free `try_send` into a bounded channel and returns
/// immediately — it no longer takes the events mutex, and it no longer logs
/// inside a critical section. A dedicated background task owns the `VecDeque`,
/// drains the channel, emits the `tracing::info!` USAGE line, and pushes into
/// the ring. The mutex is therefore only ever contended between that single
/// writer task and the (rare) `/analytics` reader — never on the request path.
pub struct ObservabilityEngine {
    tx: mpsc::Sender<UsageEvent>,
    // The retained ring, owned by the background drain task and read by
    // `/analytics`. The Mutex critical section is a single push/pop or a clone
    // for the reader — it is never held across an `.await`.
    recent_events: std::sync::Arc<Mutex<VecDeque<UsageEvent>>>,
    // Bounded ring of MCP enforcement denials (off the chat hot path; the MCP leg
    // is low-volume). A short push/clone critical section, never held across await.
    // CE (PRD-047): the only writers/readers live behind the `enterprise` gate
    // (record_mcp_security_outcome / mcp_api), so the ring sits idle there.
    #[cfg_attr(not(feature = "enterprise"), allow(dead_code))]
    mcp_security: std::sync::Arc<Mutex<VecDeque<McpSecurityEvent>>>,
}

impl Default for ObservabilityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservabilityEngine {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<UsageEvent>(INGEST_CHANNEL_CAPACITY);
        let recent_events =
            std::sync::Arc::new(Mutex::new(VecDeque::with_capacity(MAX_RETAINED_EVENTS)));

        // Background drain task: the SINGLE writer to the ring. Owns logging so
        // the hot path never logs under the lock. Ends when all senders drop.
        let ring = recent_events.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                // Log OUTSIDE the lock (Task #5): formatting + the tracing
                // subscriber's own work must not extend the mutex hold.
                if event.success {
                    tracing::info!(
                        "USAGE: Key='{}' Provider='{}' Model='{}' Tokens={}",
                        event.virtual_key_name,
                        event.provider,
                        event.model,
                        event.total_tokens
                    );
                } else {
                    tracing::warn!(
                        "USAGE(failed): Key='{}' Provider='{}' Model='{}' Error='{}'",
                        event.virtual_key_name,
                        event.provider,
                        event.model,
                        event.error.as_deref().unwrap_or("unknown"),
                    );
                }

                // Minimal critical section: push (and bound) only.
                if let Ok(mut events) = ring.lock() {
                    if events.len() >= MAX_RETAINED_EVENTS {
                        events.pop_front();
                    }
                    events.push_back(event);
                }
            }
        });

        Self {
            tx,
            recent_events,
            mcp_security: std::sync::Arc::new(Mutex::new(VecDeque::with_capacity(
                MAX_MCP_SECURITY_EVENTS,
            ))),
        }
    }

    /// Record an MCP enforcement event into the bounded ring. Off the chat hot
    /// path (the MCP leg is low-volume); a single bounded push under a short-lived
    /// lock, never held across an `.await`. Best-effort: a poisoned lock is ignored
    /// rather than propagated to the caller (observability must never fail a deny).
    /// CE (PRD-047): callers live behind the `enterprise` gate.
    #[cfg_attr(not(feature = "enterprise"), allow(dead_code))]
    pub fn record_mcp_security_event(&self, event: McpSecurityEvent) {
        if let Ok(mut ring) = self.mcp_security.lock() {
            if ring.len() >= MAX_MCP_SECURITY_EVENTS {
                ring.pop_front();
            }
            ring.push_back(event);
        }
    }

    /// Recent MCP enforcement events for ONE tenant, newest-first, capped at
    /// `limit`. Tenant-scoped so a caller only ever sees its own events.
    /// CE (PRD-047): the only caller is the gated `mcp_api::security_events`.
    #[cfg_attr(not(feature = "enterprise"), allow(dead_code))]
    pub fn recent_mcp_security(&self, tenant_id: &str, limit: usize) -> Vec<McpSecurityEvent> {
        self.mcp_security
            .lock()
            .map(|ring| {
                ring.iter()
                    .rev()
                    .filter(|e| e.tenant_id == tenant_id)
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Record a usage event. NON-BLOCKING and lock-free on the hot path: a
    /// single bounded `try_send`. If the channel is full (consumer stalled or a
    /// burst beyond capacity) the event is dropped with a rate-limited warning
    /// rather than blocking the request — observability must never add latency to
    /// or stall a caller's request.
    pub fn record_usage(&self, event: UsageEvent) {
        // OTLP export skeleton (PRD-009): best-effort, fire-and-forget. The
        // exporter is gated on OTEL_EXPORT_ENABLED; when disabled the function
        // returns immediately. Failure in the exporter must never affect the
        // primary observability path.
        crate::otel::export_event(&event);
        if let Err(e) = self.tx.try_send(event) {
            // `tracing` itself is non-blocking; this is the only work on the hot
            // path when the buffer is saturated.
            tracing::warn!("observability ingest channel full, dropping usage event: {e}");
        }
    }

    /// Returns the ENTIRE ring, unscoped — every tenant's events. This is an
    /// inspection/test accessor only (used by the crate's unit + integration test
    /// harnesses via the library target); it must NOT back any authenticated data-
    /// plane endpoint, since a cross-tenant dump is a data leak. Production reads go
    /// through the key-ownership-scoped `recent_events_owned` / `recent_events`
    /// (ADR-023). `allow(dead_code)` because the binary target has no caller — only
    /// the library's test consumers do.
    #[allow(dead_code)]
    pub fn get_recent_events(&self) -> Vec<UsageEvent> {
        self.recent_events
            .lock()
            .map(|events| events.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Tenant-scoped variant of `get_recent_events`: returns only the events whose
    /// `virtual_key_name` is one of `key_names` — the virtual-key names the caller's
    /// tenant owns. Isolation is structural, by key ownership (ADR-023), exactly like
    /// `recent_events`/`chargeback`; the scope is NEVER a client-supplied id. Insertion
    /// order is preserved (unlike the newest-first `recent_events` log projection), so
    /// the `/analytics` response keeps its prior ordering — only cross-tenant rows are
    /// removed. Same short mutex, never held across an `.await`.
    pub fn recent_events_owned(
        &self,
        key_names: &std::collections::BTreeSet<String>,
    ) -> Vec<UsageEvent> {
        self.recent_events
            .lock()
            .map(|events| {
                events
                    .iter()
                    .filter(|ev| key_names.contains(&ev.virtual_key_name))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Aggregate the retained ring into a non-sensitive operational summary for
    /// the read-only `GET /status` surface. Folds the last-1000-event ring ONCE
    /// under the same short mutex `get_recent_events` already takes (never held
    /// across an `.await`, never on the request hot path). Carries no keys,
    /// bodies, or PII — only counts. `window` is the sample size (≤1000), so the
    /// rates are read as "over the recent window", not all-time.
    pub fn usage_summary(&self) -> UsageSummary {
        let Ok(events) = self.recent_events.lock() else {
            return UsageSummary::default();
        };
        let mut s = UsageSummary {
            window: events.len(),
            ..Default::default()
        };
        for ev in events.iter() {
            // Blocks are classified by the `error` sentinel so both the
            // before-request (synthetic-provider) and after-request (real-
            // provider) denials are counted.
            match ev.error.as_deref() {
                Some("guardrails_denied") => s.guardrail_block_count += 1,
                Some("sovereign_block") => s.sovereign_block_count += 1,
                _ => {}
            }
            // Synthetic events (provider sentinels like "(sovereign_block)" /
            // "(prompt_render)") are not real provider attempts — exclude them
            // from traffic/success counts so the golden signals reflect calls.
            if ev.provider.starts_with('(') {
                continue;
            }
            *s.by_provider.entry(ev.provider.clone()).or_insert(0) += 1;
            if ev.success {
                s.success_count += 1;
            } else {
                s.failure_count += 1;
            }
            if let Some(hit) = ev.cache_hit {
                s.cache_participating_count += 1;
                if hit {
                    s.cache_hit_count += 1;
                }
            }
        }
        let attempts = s.success_count + s.failure_count;
        s.success_rate = if attempts > 0 {
            s.success_count as f64 / attempts as f64
        } else {
            0.0
        };
        s.cache_hit_rate = if s.cache_participating_count > 0 {
            s.cache_hit_count as f64 / s.cache_participating_count as f64
        } else {
            0.0
        };
        s
    }

    /// Latency percentiles (p50/p95/p99) over the retained ring — overall and
    /// per provider. Computed by exact nearest-rank over the latencies recorded
    /// on real provider attempts (synthetic sentinel events like
    /// "(sovereign_block)" / "(prompt_render)" carry no latency and are skipped).
    /// Folds the ring ONCE under the same short mutex `get_recent_events` takes;
    /// never held across an `.await`, never on the request hot path. The ring is
    /// ≤1000 events, so the per-provider sort is trivially cheap.
    pub fn latency_stats(&self) -> LatencyReport {
        let Ok(events) = self.recent_events.lock() else {
            return LatencyReport::default();
        };
        let mut overall: Vec<u64> = Vec::new();
        let mut per_provider: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for ev in events.iter() {
            // Only real provider attempts that were timed contribute.
            if ev.provider.starts_with('(') {
                continue;
            }
            if let Some(ms) = ev.latency_ms {
                overall.push(ms);
                per_provider
                    .entry(ev.provider.clone())
                    .or_default()
                    .push(ms);
            }
        }
        LatencyReport {
            overall: LatencyPercentiles::from_samples(&mut overall),
            by_provider: per_provider
                .into_iter()
                .map(|(p, mut v)| (p, LatencyPercentiles::from_samples(&mut v)))
                .collect(),
        }
    }

    /// FinOps chargeback/showback (PRD-008 FR-24, `Feature::FinOpsExport`) over the
    /// retained ring, scoped to ONE tenant. The caller passes the set of virtual-key
    /// NAMES it owns (resolved from the auth registry — tenant isolation is by key
    /// ownership, never a client-supplied id); only events whose `virtual_key_name`
    /// is in that set contribute, so a tenant can never see another tenant's spend.
    ///
    /// Synthetic sentinel events (providers like "(sovereign_block)" /
    /// "(prompt_render)") carry no real spend and are skipped — `window` still
    /// reports the full ring size so the caller knows the sample horizon. Folds the
    /// ring ONCE under the same short mutex `get_recent_events` takes; never held
    /// across an `.await`, never on the request hot path. Costs sum the canonical
    /// integer micro-USD (`cost_micro_usd`) plus per-currency minor-unit totals
    /// (`cost_by_currency`) — money stays integer end to end (never a float).
    pub fn chargeback(&self, key_names: &std::collections::BTreeSet<String>) -> ChargebackReport {
        let Ok(events) = self.recent_events.lock() else {
            return ChargebackReport::default();
        };
        let mut report = ChargebackReport {
            window: events.len(),
            ..Default::default()
        };
        // Tenant-scoped latency samples (ms) over timed provider attempts, for the
        // report's real p50/p95/p99 — collected during the same single fold.
        let mut latencies: Vec<u64> = Vec::new();
        for ev in events.iter() {
            // Tenant isolation: only this tenant's keys, and only real provider
            // attempts (synthetic sentinels carry no chargeable spend).
            if ev.provider.starts_with('(') || !key_names.contains(&ev.virtual_key_name) {
                continue;
            }
            report.events_matched += 1;
            report.totals.accumulate(ev);
            report
                .by_model
                .entry(ev.model.clone())
                .or_default()
                .accumulate(ev);
            report
                .by_key
                .entry(ev.virtual_key_name.clone())
                .or_default()
                .accumulate(ev);
            // Cost attribution by business-process — only tagged events contribute.
            if let Some(uc) = ev.use_case.as_ref() {
                report
                    .by_use_case
                    .entry(uc.clone())
                    .or_default()
                    .accumulate(ev);
            }
            if let Some(ms) = ev.latency_ms {
                latencies.push(ms);
            }
        }
        report.latency = LatencyPercentiles::from_samples(&mut latencies);
        report
    }

    /// Recent-window usage TIME-SERIES for the read-only `GET /v1/finops/timeseries`
    /// surface, scoped to ONE tenant. Mirrors `chargeback`'s isolation model EXACTLY:
    /// the caller passes the set of virtual-key NAMES it owns (resolved server-side
    /// from the auth registry — tenant isolation is by key ownership, never a
    /// client-supplied id), and only events whose `virtual_key_name` is in that set
    /// contribute.
    ///
    /// Buckets the tenant's real provider attempts by `timestamp` into `bucket_count`
    /// fixed-width buckets spanning `[now - window, now)`. Bucket width is
    /// `window / bucket_count` (the caller clamps both inputs). Events older than the
    /// window are EXCLUDED — honest: the ≤1000-event ring may not even span the
    /// window, so the series is the RECENT in-memory window, not durable history.
    ///
    /// Synthetic sentinel events (providers like "(sovereign_block)" /
    /// "(prompt_render)" / "(feedback)", all starting with `'('`) carry no real
    /// provider attempt and are skipped — the same `'('` rule `chargeback` uses.
    /// Costs sum the canonical integer `cost_micro_usd` (money stays integer); a
    /// missing cost/latency contributes 0. Latency is summed per bucket and divided
    /// by the bucket's timed-sample count to yield `avg_latency_ms` (0 when no sample
    /// was timed in that bucket).
    ///
    /// Read-only and lock-short: folds the ring ONCE under the same brief mutex
    /// `get_recent_events` takes (never held across an `.await`, never on the request
    /// hot path). No `unwrap`/panic — a poisoned lock yields all-zero buckets.
    pub fn usage_timeseries(
        &self,
        key_names: &std::collections::BTreeSet<String>,
        window: chrono::Duration,
        bucket_count: usize,
    ) -> UsageTimeseries {
        // Anchor the window at a single `now` so every bucket boundary is consistent
        // within this snapshot. The window is `[start, now)`; bucket `i` covers
        // `[start + i*width, start + (i+1)*width)`.
        let now = Utc::now();
        let bucket_count = bucket_count.max(1);
        let window_secs = window.num_seconds().max(1);
        // Integer bucket width in whole seconds (≥1). Derive the effective window
        // from width*count so no bucket boundary straddles a rounding gap.
        let bucket_secs = (window_secs / bucket_count as i64).max(1);
        let effective_window = chrono::Duration::seconds(bucket_secs * bucket_count as i64);
        let start = now - effective_window;

        let mut buckets: Vec<TimeseriesBucket> = (0..bucket_count)
            .map(|i| TimeseriesBucket {
                ts: start + chrono::Duration::seconds(bucket_secs * i as i64),
                requests: 0,
                errors: 0,
                cost_micro_usd: 0,
                tokens: 0,
                latency_sum_ms: 0,
                latency_samples: 0,
                avg_latency_ms: 0,
            })
            .collect();

        let mut total_events_in_window: u64 = 0;

        if let Ok(events) = self.recent_events.lock() {
            for ev in events.iter() {
                // Tenant isolation + only real provider attempts (skip the synthetic
                // sentinels — the same `'('` rule chargeback uses).
                if ev.provider.starts_with('(') || !key_names.contains(&ev.virtual_key_name) {
                    continue;
                }
                // Exclude events outside the recent window — honest: the ring may not
                // span the window, so an out-of-window event simply does not appear.
                if ev.timestamp < start || ev.timestamp >= now {
                    continue;
                }
                let offset = (ev.timestamp - start).num_seconds();
                // Map to a bucket index; clamp defensively (offset is in [0, window)).
                let idx = (offset / bucket_secs).clamp(0, bucket_count as i64 - 1) as usize;
                let b = &mut buckets[idx];
                b.requests = b.requests.saturating_add(1);
                if !ev.success {
                    b.errors = b.errors.saturating_add(1);
                }
                b.tokens = b.tokens.saturating_add(ev.total_tokens as u64);
                if let Some(cost) = &ev.cost {
                    b.cost_micro_usd = b.cost_micro_usd.saturating_add(cost.micro_usd);
                }
                if let Some(ms) = ev.latency_ms {
                    b.latency_sum_ms = b.latency_sum_ms.saturating_add(ms);
                    b.latency_samples = b.latency_samples.saturating_add(1);
                }
                total_events_in_window = total_events_in_window.saturating_add(1);
            }
        }

        for b in &mut buckets {
            // Mean over the bucket's timed samples (0 when none were timed) — never
            // a divide-by-zero panic.
            b.avg_latency_ms = b.latency_sum_ms.checked_div(b.latency_samples).unwrap_or(0);
        }

        UsageTimeseries {
            window_secs: effective_window.num_seconds(),
            bucket_secs,
            total_events_in_window,
            buckets,
        }
    }

    /// Guardrail detection outcomes for the read-only `GET /v1/guardrails/outcomes`
    /// surface, scoped to ONE tenant. Mirrors `usage_timeseries`'s isolation +
    /// windowing model EXACTLY: the caller passes the set of virtual-key NAMES it
    /// owns (resolved server-side from the auth registry — tenant isolation is by key
    /// ownership, never a client-supplied id), and only events whose
    /// `virtual_key_name` is in that set, with a `timestamp` inside `[now-window, now)`,
    /// contribute.
    ///
    /// The ring's `UsageEvent.guardrails` carries COARSE detector labels only —
    /// `CheckOutcome.check_type` is a string like `detect_pii` / `detect_secrets` /
    /// `prompt_injection` / `banned_keywords` / `invisible_unicode`. There are NO
    /// fine-grained PII subtypes in the event (the inline redactor surfaces category
    /// labels only, never reflected into the ring), so this method reports the REAL
    /// recorded `check_type` strings and never fabricates a finer breakdown.
    ///
    /// A "detection" is a guardrail outcome whose `verdict != Pass` (Fail OR Error —
    /// the same fail-closed semantics `CheckOutcome::is_blocking` uses for deny,
    /// extended to include `observe` fails, which ARE detections even though they did
    /// not block). `affected_requests` is the count of in-window owned events with at
    /// least one such non-pass outcome; `by_type` is the per-`check_type` detection
    /// count. A recent per-bucket `series` of total detections is produced with the
    /// SAME bucketing approach `usage_timeseries` uses.
    ///
    /// Unlike the traffic aggregates, this method DOES consider the synthetic block
    /// sentinels (`(guardrails_denied)`): a before-request deny is recorded on a
    /// `(guardrails_denied)` event that carries the outcomes, and that IS a real
    /// detection a tenant should see. The other join sentinels (`(prompt_render)` /
    /// `(feedback)` / `(sovereign_block)`) never carry guardrail outcomes, so they
    /// naturally contribute nothing.
    ///
    /// Read-only and lock-short: folds the ring ONCE under the same brief mutex
    /// `get_recent_events` takes (never held across an `.await`, never on the request
    /// hot path). No `unwrap`/panic — a poisoned lock yields all-zero results.
    /// MOAT (ADR-088): the sole caller is the enterprise-only /v1/guardrails/outcomes
    /// report handler, so this rides `enterprise`.
    #[cfg(feature = "enterprise")]
    pub fn guardrail_outcomes(
        &self,
        key_names: &std::collections::BTreeSet<String>,
        window: chrono::Duration,
        bucket_count: usize,
    ) -> GuardrailOutcomesReport {
        let now = Utc::now();
        let bucket_count = bucket_count.max(1);
        let window_secs = window.num_seconds().max(1);
        let bucket_secs = (window_secs / bucket_count as i64).max(1);
        let effective_window = chrono::Duration::seconds(bucket_secs * bucket_count as i64);
        let start = now - effective_window;

        let mut series: Vec<GuardrailDetectionBucket> = (0..bucket_count)
            .map(|i| GuardrailDetectionBucket {
                ts: start + chrono::Duration::seconds(bucket_secs * i as i64),
                detections: 0,
            })
            .collect();

        let mut report = GuardrailOutcomesReport {
            window_secs: effective_window.num_seconds(),
            bucket_secs,
            total_requests: 0,
            affected_requests: 0,
            by_type: BTreeMap::new(),
            series: Vec::new(),
        };

        if let Ok(events) = self.recent_events.lock() {
            for ev in events.iter() {
                // Tenant isolation + recent-window bound. We do NOT apply the `'('`
                // synthetic-provider skip here: a `(guardrails_denied)` block IS a
                // real detection a tenant should see, and the other sentinels carry
                // no guardrail outcomes anyway.
                if !key_names.contains(&ev.virtual_key_name) {
                    continue;
                }
                if ev.timestamp < start || ev.timestamp >= now {
                    continue;
                }
                report.total_requests = report.total_requests.saturating_add(1);

                let Some(outcomes) = ev.guardrails.as_ref() else {
                    continue;
                };
                // A detection = any outcome that did not pass (Fail OR Error),
                // covering both deny (blocking) and observe (non-blocking) fails.
                let mut event_had_detection = false;
                let offset = (ev.timestamp - start).num_seconds();
                let idx = (offset / bucket_secs).clamp(0, bucket_count as i64 - 1) as usize;
                for o in outcomes {
                    if o.verdict == routeplane_guardrails::Verdict::Pass {
                        continue;
                    }
                    event_had_detection = true;
                    *report.by_type.entry(o.check_type.clone()).or_insert(0) += 1;
                    let b = &mut series[idx];
                    b.detections = b.detections.saturating_add(1);
                }
                if event_had_detection {
                    report.affected_requests = report.affected_requests.saturating_add(1);
                }
            }
        }

        report.series = series;
        report
    }

    /// Cache-SAVINGS rollup for the read-only `GET /v1/finops/cache-savings`
    /// surface, scoped to ONE tenant. Mirrors `usage_timeseries`'s isolation +
    /// windowing model EXACTLY: the caller passes the set of virtual-key NAMES it
    /// owns (resolved server-side from the auth registry — tenant isolation is by
    /// key ownership, never a client-supplied id), and only events whose
    /// `virtual_key_name` is in that set, with a `timestamp` inside
    /// `[now-window, now)`, contribute.
    ///
    /// A served cache hit IS the event that carries the savings: the proxy records
    /// it as a `UsageEvent::success` on the `(cache)` / `(semantic-cache)` sentinel
    /// provider via `with_cache_hit`, which sets `cache_hit = Some(true)` and
    /// `estimated_saved_cost_micro_usd = Some(..)` while preserving the STORED
    /// response's token counts (`total_tokens`). The authoritative hit signal is
    /// therefore `cache_hit == Some(true)` (the sentinel provider name is accepted
    /// as a defensive OR, but is never the sole signal). For each such hit we:
    ///   * sum `estimated_saved_cost_micro_usd` — the avoided upstream spend. HONEST:
    ///     the field name says ESTIMATE (priced by the limits estimator until the
    ///     FinOps pricing book lands), so this is a real recorded estimate, never a
    ///     fabricated number; a missing value contributes 0.
    ///   * sum the served response's `total_tokens` (the completion that did NOT
    ///     have to be regenerated upstream — "not re-sent to providers"), PLUS any
    ///     provider-native prompt-cache `cached_tokens` recorded on the event. A
    ///     missing field contributes 0.
    ///   * count the hit.
    ///
    /// `cacheable_lookups` is the count of in-window owned events that PARTICIPATED
    /// in caching at all (`cache_hit.is_some()` — hit OR miss/refresh/bypass), so the
    /// Console can show hits against the lookups that were eligible. An empty ring /
    /// no cache activity yields honest zeroes.
    ///
    /// Read-only and lock-short: folds the ring ONCE under the same brief mutex
    /// `get_recent_events` takes (never held across an `.await`, never on the request
    /// hot path). No `unwrap`/panic — a poisoned lock yields an all-zero report.
    pub fn cache_savings(
        &self,
        key_names: &std::collections::BTreeSet<String>,
        window: chrono::Duration,
    ) -> CacheSavingsReport {
        let now = Utc::now();
        let window_secs = window.num_seconds().max(1);
        let start = now - chrono::Duration::seconds(window_secs);

        let mut report = CacheSavingsReport {
            window_secs,
            cache_hits: 0,
            cacheable_lookups: 0,
            saved_cost_micro_usd: 0,
            saved_tokens: 0,
        };

        if let Ok(events) = self.recent_events.lock() {
            for ev in events.iter() {
                // Tenant isolation + recent-window bound. We do NOT apply the `'('`
                // synthetic-provider skip here: a served cache hit is recorded on the
                // `(cache)` / `(semantic-cache)` sentinel and IS the event that carries
                // the savings a tenant should see.
                if !key_names.contains(&ev.virtual_key_name) {
                    continue;
                }
                if ev.timestamp < start || ev.timestamp >= now {
                    continue;
                }
                // A cache-PARTICIPATING event is any event that recorded a cache
                // verdict (`cache_hit` is `Some` for hit AND for miss/refresh/bypass).
                if ev.cache_hit.is_some() {
                    report.cacheable_lookups = report.cacheable_lookups.saturating_add(1);
                }
                // A served HIT: the authoritative signal is `cache_hit == Some(true)`
                // (set by `with_cache_hit`); the `(cache)`/`(semantic-cache)` sentinel
                // is accepted as a defensive OR but is never the sole signal.
                let is_hit = ev.cache_hit == Some(true)
                    || matches!(ev.provider.as_str(), "(cache)" | "(semantic-cache)");
                if !is_hit {
                    continue;
                }
                report.cache_hits = report.cache_hits.saturating_add(1);
                // Avoided upstream spend — a real recorded ESTIMATE (the field name
                // says so); a missing value contributes 0.
                report.saved_cost_micro_usd = report
                    .saved_cost_micro_usd
                    .saturating_add(ev.estimated_saved_cost_micro_usd.unwrap_or(0));
                // Tokens not re-sent to providers: the served response's total tokens
                // (the completion that did not have to be regenerated) plus any
                // provider-native prompt-cache read subset. Missing → 0.
                report.saved_tokens = report
                    .saved_tokens
                    .saturating_add(ev.total_tokens as u64)
                    .saturating_add(ev.cached_tokens.unwrap_or(0) as u64);
            }
        }

        report
    }

    /// Recent request-log rows for the read-only `GET /v1/logs` surface, scoped to
    /// ONE tenant. Mirrors `chargeback`'s isolation model exactly: the caller passes
    /// the set of virtual-key NAMES it owns (resolved server-side from the auth
    /// registry — tenant isolation is by key ownership, never a client-supplied id),
    /// and only events whose `virtual_key_name` is in that set are returned. Rows are
    /// newest-first and capped at `limit`.
    ///
    /// Synthetic sentinel events (providers like "(sovereign_block)" /
    /// "(prompt_render)" / "(feedback)", all of which start with `'('`) are EXCLUDED
    /// — the log shows real chat/embeddings/etc. request attempts (success, provider
    /// failure, and the synthetic-provider block events ARE included since a block is
    /// a real request outcome a tenant should see; only the non-attempt join events
    /// like prompt_render/feedback are dropped). Each row carries a synthesized
    /// stable-ish id (the ring stores no id) derived from the event fields so the
    /// detail lookup is deterministic within a snapshot. Guardrail outcomes are
    /// reduced to LABELS only (id + verdict), preserving the no-reflection posture
    /// (no detail strings that could echo masked content).
    ///
    /// Read-only and lock-short: folds the ring ONCE under the same brief mutex
    /// `get_recent_events` takes (never held across an `.await`, never on the request
    /// hot path). No `unwrap`/panic — a poisoned lock yields an empty list.
    pub fn recent_events(
        &self,
        key_names: &std::collections::BTreeSet<String>,
        limit: usize,
    ) -> Vec<LogRow> {
        let Ok(events) = self.recent_events.lock() else {
            return Vec::new();
        };
        // Iterate newest-first (the ring pushes to the back), filtering to the
        // tenant's own keys and to real request attempts, capped at `limit`.
        events
            .iter()
            .rev()
            .filter(|ev| key_names.contains(&ev.virtual_key_name) && is_logged_attempt(ev))
            .take(limit)
            .map(LogRow::from_event)
            .collect()
    }

    /// Residency observability for the read-only `GET /v1/residency/summary` +
    /// `GET /v1/residency/ledger` surfaces, scoped to ONE tenant. Mirrors
    /// `recent_events`'s isolation model EXACTLY: the caller passes the set of
    /// virtual-key NAMES it owns (resolved server-side from the auth registry —
    /// tenant isolation is by key ownership, never a client-supplied id), and only
    /// events whose `virtual_key_name` is in that set contribute. The ledger rows
    /// are newest-first and capped at `limit`.
    ///
    /// Folds the ring ONCE (a single short mutex hold, the same brief lock
    /// `recent_events`/`chargeback` take — never across an `.await`, never on the
    /// request hot path), producing BOTH the summary and the ledger from one pass.
    /// No `unwrap`/panic — a poisoned lock yields an empty summary + ledger.
    ///
    /// Read-only over the EXISTING ring (last ≤1000 events) — no new durable store
    /// (same posture as `/v1/logs`, `/v1/finops/usage`, `/metrics`; no ADR needed).
    /// The summary carries NO daily `series` (the ring is not dated history) and the
    /// ledger carries NO compliance `framework` (the ring records no per-request
    /// framework) — both are HONEST-ABSENT, never fabricated.
    ///
    /// Only real request attempts contribute (synthetic JOIN sentinels like
    /// `(prompt_render)` / `(feedback)` are excluded via `is_logged_attempt`, the
    /// same rule as `/v1/logs`); the block sentinel `(sovereign_block)` IS counted
    /// (a residency refusal is a real residency decision a tenant should see).
    pub fn residency_report(
        &self,
        key_names: &std::collections::BTreeSet<String>,
        limit: usize,
    ) -> (ResidencySummaryView, Vec<ResidencyLedgerRow>) {
        let Ok(events) = self.recent_events.lock() else {
            return (ResidencySummaryView::default(), Vec::new());
        };

        let mut summary = ResidencySummaryView {
            window: events.len(),
            ..Default::default()
        };
        let mut rows: Vec<ResidencyLedgerRow> = Vec::new();

        // Iterate newest-first (the ring pushes to the back) so the ledger is
        // newest-first and the `limit` cap keeps the most-recent decisions.
        for ev in events.iter().rev() {
            if !key_names.contains(&ev.virtual_key_name) || !is_logged_attempt(ev) {
                continue;
            }

            // An event "carried regulated data" when residency applied to it: a
            // locked region OR a sovereign-routed flag (a block carries both).
            let regulated = ev.region.is_some() || ev.sovereign_routed;
            let outcome = residency_outcome(ev);

            summary.total += 1;
            if regulated {
                summary.regulated_count += 1;
            }
            // Count by the derived OUTCOME (not the raw `sovereign_routed` flag,
            // which a block/failure also carries): a request is "sovereign-routed"
            // only when it SUCCEEDED in-region. This keeps `sovereign_routed_count`
            // consistent with `by_outcome["sovereign_routed"]` and the ledger's
            // `routed_region`.
            match outcome {
                "sovereign_routed" => summary.sovereign_routed_count += 1,
                "residency_blocked" => summary.blocked_count += 1,
                "all_failed" => summary.all_failed_count += 1,
                _ => {}
            }
            // by_region: None → the sentinel "none" (unconstrained / no residency).
            let region_key = ev.region.clone().unwrap_or_else(|| "none".to_string());
            *summary.by_region.entry(region_key).or_insert(0) += 1;
            *summary.by_outcome.entry(outcome.to_string()).or_insert(0) += 1;

            if rows.len() < limit {
                rows.push(ResidencyLedgerRow {
                    id: synthesize_id(ev),
                    timestamp: ev.timestamp,
                    classification: if regulated { "regulated" } else { "none" },
                    // The required region is the locked jurisdiction; the routed
                    // region is that same jurisdiction ONLY when the request was
                    // actually routed there (a block routed nowhere → None).
                    required_region: ev.region.clone(),
                    routed_region: if ev.sovereign_routed && ev.success {
                        ev.region.clone()
                    } else {
                        None
                    },
                    outcome,
                    model: ev.model.clone(),
                    virtual_key_name: ev.virtual_key_name.clone(),
                });
            }
        }

        (summary.finalize(), rows)
    }
}

/// Classify one ring event into a residency OUTCOME label. Snake_case strings the
/// Console's `ResidencyOutcome` union mirrors directly (the dashboard maps these
/// 1:1, no re-derivation). Derived purely from `success` + the `sovereign_block`
/// error sentinel + `sovereign_routed` + `region`:
/// - `residency_blocked` — a sovereign block (HTTP 422; never reached a provider)
/// - `all_failed` — a provider failure/timeout on a residency-constrained or
///   sovereign-routed request (no in-region success)
/// - `sovereign_routed` — a SUCCESS that was sovereign-routed to an in-region
///   provider to satisfy a residency requirement
/// - `not_regulated` — any other (a passthrough request with no residency
///   constraint and no sovereign routing)
fn residency_outcome(ev: &UsageEvent) -> &'static str {
    if ev.error.as_deref() == Some("sovereign_block") {
        return "residency_blocked";
    }
    if !ev.success {
        // A failure on a residency-constrained / sovereign-routed request is a
        // residency-relevant "all failed"; an ordinary (unconstrained) failure is
        // not a residency outcome → not_regulated.
        if ev.region.is_some() || ev.sovereign_routed {
            return "all_failed";
        }
        return "not_regulated";
    }
    if ev.sovereign_routed {
        return "sovereign_routed";
    }
    "not_regulated"
}

/// Whether an event is a real request attempt that belongs in the request-log
/// surface. Excludes the synthetic JOIN events that carry no request outcome
/// (`(prompt_render)`, `(feedback)`) while KEEPING the block sentinels
/// (`(sovereign_block)`, `(guardrails_denied)`) — a refusal is a real outcome a
/// tenant should see in its logs. Real provider attempts (no `'('` prefix) and the
/// `(cache)` hit pseudo-provider are kept.
fn is_logged_attempt(ev: &UsageEvent) -> bool {
    !matches!(ev.provider.as_str(), "(prompt_render)" | "(feedback)")
}

/// A lightweight, tenant-safe log-line view of a `UsageEvent` for `GET /v1/logs`.
/// Read-only projection — carries only the fields a log line needs, never raw
/// request/response content. Guardrail outcomes are reduced to labels only
/// (`GuardrailLabel`), preserving the no-reflection posture. The `id` is
/// synthesized (the ring stores no id) so a detail lookup is stable within a
/// snapshot.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LogRow {
    /// Synthesized stable-ish id (`log_<hex>`): a hash of timestamp + key + model +
    /// provider, so the same event maps to the same id across a snapshot read.
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub virtual_key_name: String,
    pub provider: String,
    pub model: String,
    /// Coarse outcome label: `success` | `error` | `blocked`.
    pub outcome: &'static str,
    /// The failure/block detail sentinel, when not a success
    /// (`sovereign_block` / `guardrails_denied` / a provider error string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// Canonical cost in micro-USD, when this attempt was priced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_micro_usd: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    pub sovereign_routed: bool,
    /// Five-value cache verdict (`hit|miss|refreshed|semantic_hit|bypass`), when the
    /// request participated in caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_status: Option<String>,
    /// Guardrail check outcomes reduced to LABELS only (id + verdict) — no detail
    /// strings (no-reflection). Absent (and omitted) when no checks ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrails: Option<Vec<GuardrailLabel>>,
    /// Business-process / use-case label (`x-routeplane-use-case`), when the
    /// request was tagged. Absent (and omitted) otherwise — A/B-parity-safe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_case: Option<String>,
}

/// A no-reflection label for one guardrail check on a log row: the stable check id
/// and its verdict only — never the `detail` string (which can echo masked
/// content). Verdict is the lowercased `Verdict` Display.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GuardrailLabel {
    pub id: String,
    pub verdict: String,
}

impl LogRow {
    /// Project a ring `UsageEvent` into a tenant-safe log row. The outcome is
    /// classified from the success flag + the error sentinel: a `sovereign_block` /
    /// `guardrails_denied` becomes `blocked`; any other non-success is `error`.
    fn from_event(ev: &UsageEvent) -> Self {
        let outcome = if ev.success {
            "success"
        } else {
            match ev.error.as_deref() {
                Some("sovereign_block") | Some("guardrails_denied") => "blocked",
                _ => "error",
            }
        };
        LogRow {
            id: synthesize_id(ev),
            timestamp: ev.timestamp,
            virtual_key_name: ev.virtual_key_name.clone(),
            provider: ev.provider.clone(),
            model: ev.model.clone(),
            outcome,
            error: ev.error.clone(),
            prompt_tokens: ev.prompt_tokens,
            completion_tokens: ev.completion_tokens,
            total_tokens: ev.total_tokens,
            latency_ms: ev.latency_ms,
            cost_micro_usd: ev.cost.as_ref().map(|c| c.micro_usd),
            region: ev.region.clone(),
            sovereign_routed: ev.sovereign_routed,
            cache_status: ev.cache_status.clone(),
            guardrails: ev.guardrails.as_ref().map(|outs| {
                outs.iter()
                    .map(|o| GuardrailLabel {
                        id: o.id.clone(),
                        verdict: format!("{:?}", o.verdict).to_lowercase(),
                    })
                    .collect()
            }),
            use_case: ev.use_case.clone(),
        }
    }
}

/// Synthesize a stable-ish id for a ring event (the ring stores none). A FNV-1a
/// hash of timestamp(ns) + key + provider + model — deterministic for a given
/// event, so the `/v1/logs` list and a follow-up detail lookup agree within a
/// snapshot. Collisions across distinct events are astronomically unlikely for a
/// ≤1000-event window and are non-load-bearing (this is a display id, not a
/// security or storage key).
fn synthesize_id(ev: &UsageEvent) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(&ev
        .timestamp
        .timestamp_nanos_opt()
        .unwrap_or(0)
        .to_le_bytes());
    mix(ev.virtual_key_name.as_bytes());
    mix(ev.provider.as_bytes());
    mix(ev.model.as_bytes());
    format!("log_{h:016x}")
}

/// One tenant's RECENT-WINDOW usage time-series over the in-memory observability
/// ring for `GET /v1/finops/timeseries`. This is the HONEST recent window — the ring
/// holds only the last ~1000 events (`MAX_RETAINED_EVENTS`), so the series spans the
/// recent activity captured there, NOT durable history (a true multi-day series
/// needs the telemetry store, ADR-024). `window_secs` is the effective span
/// (`bucket_secs * buckets`); `total_events_in_window` is how many of the tenant's
/// real attempts fell inside it. Buckets are oldest-first.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct UsageTimeseries {
    /// Effective window span in seconds (`bucket_secs * buckets.len()`).
    pub window_secs: i64,
    /// Width of each bucket in seconds.
    pub bucket_secs: i64,
    /// The tenant's real attempts that fell inside the window (sum over buckets).
    pub total_events_in_window: u64,
    /// Fixed-width buckets, oldest-first; every bucket is emitted (zeros included)
    /// so the series never has gaps — but an empty ring → all-zero buckets, never
    /// fabricated traffic.
    pub buckets: Vec<TimeseriesBucket>,
}

/// One fixed-width bucket of the recent-window usage series. `ts` is the bucket
/// START (ISO8601 / RFC3339, the `DateTime<Utc>` Serialize). Counts are summed over
/// the tenant's real provider attempts whose `timestamp` fell in `[ts, ts + width)`;
/// a missing cost/latency contributes 0. `avg_latency_ms` is the mean of the timed
/// samples in the bucket (0 when none were timed). The latency accumulators are
/// internal (not serialized).
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct TimeseriesBucket {
    /// Bucket start, ISO8601 (RFC3339 via `DateTime<Utc>`).
    pub ts: DateTime<Utc>,
    pub requests: u64,
    pub errors: u64,
    pub cost_micro_usd: u64,
    pub tokens: u64,
    /// Mean latency (ms) over the bucket's timed samples; 0 when none timed.
    pub avg_latency_ms: u64,
    /// Internal accumulator: summed latency of timed samples (not on the wire).
    #[serde(skip)]
    pub latency_sum_ms: u64,
    /// Internal accumulator: count of timed samples (not on the wire).
    #[serde(skip)]
    pub latency_samples: u64,
}

/// One tenant's guardrail DETECTION telemetry over the recent-event window for
/// `GET /v1/guardrails/outcomes`. Built ONLY from the COARSE detector labels the
/// ring actually records (`CheckOutcome.check_type` — e.g. `detect_pii`,
/// `detect_secrets`, `prompt_injection`, `banned_keywords`, `invisible_unicode`):
/// there are NO fine-grained PII subtypes in the event, so the breakdown is honest
/// raw `check_type` strings, never a fabricated finer split.
///
/// A "detection" is an outcome whose `verdict != Pass` (Fail OR Error). `by_type`
/// is a `BTreeMap` for deterministic JSON ordering; `affected_requests` is the count
/// of in-window owned events with ≥1 detection. The `series` is the recent in-memory
/// window (the ring is ≤1000 events), NOT durable history — the handler's `note`
/// says so (a multi-day series needs the telemetry store, ADR-024).
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct GuardrailOutcomesReport {
    /// Effective window span in seconds (`bucket_secs * series.len()`).
    pub window_secs: i64,
    /// Width of each `series` bucket in seconds.
    pub bucket_secs: i64,
    /// This tenant's real in-window events considered (the detection denominator).
    pub total_requests: u64,
    /// In-window owned events with at least one non-pass guardrail outcome.
    pub affected_requests: u64,
    /// Per-`check_type` detection count (verdict != Pass). RAW coarse gateway
    /// labels — `detect_pii` / `detect_secrets` / `prompt_injection` / etc.
    pub by_type: BTreeMap<String, u64>,
    /// Fixed-width buckets of total detections, oldest-first; every bucket emitted
    /// (zeros included) so the series never has gaps. Empty ring → all-zero buckets,
    /// never fabricated activity.
    pub series: Vec<GuardrailDetectionBucket>,
}

/// One fixed-width bucket of the recent-window detection series. `ts` is the bucket
/// START (ISO8601 / RFC3339 via `DateTime<Utc>`); `detections` is the count of
/// non-pass guardrail outcomes whose owning event fell in `[ts, ts + width)`.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct GuardrailDetectionBucket {
    pub ts: DateTime<Utc>,
    pub detections: u64,
}

/// One tenant's CACHE-SAVINGS rollup over the recent-event window for
/// `GET /v1/finops/cache-savings` (powers the Console Cache page's "Cost saved" +
/// "Tokens saved" StatCards). Built ONLY from served cache-hit events the ring
/// actually recorded (`cache_hit == Some(true)` on the `(cache)` / `(semantic-cache)`
/// sentinel). HONEST: `saved_cost_micro_usd` is the sum of the per-hit
/// `estimated_saved_cost_micro_usd` — a real recorded ESTIMATE (the source field name
/// says so), never a fabricated figure; an empty ring / no hits yields all zeroes.
/// The window is the recent in-memory ring (≤1000 events), NOT durable history (a
/// multi-day series needs the telemetry store, ADR-024) — the handler's `note` says
/// so.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct CacheSavingsReport {
    /// Effective recent window span in seconds.
    pub window_secs: i64,
    /// Served cache hits in the window (the events that carried the savings).
    pub cache_hits: u64,
    /// In-window owned events that PARTICIPATED in caching at all (hit OR
    /// miss/refresh/bypass) — the eligible-lookup denominator for a hit rate.
    pub cacheable_lookups: u64,
    /// Summed avoided upstream spend in micro-USD — a real recorded ESTIMATE.
    pub saved_cost_micro_usd: u64,
    /// Summed tokens not re-sent to providers (the served responses' total tokens
    /// plus any provider-native prompt-cache read subset).
    pub saved_tokens: u64,
}

/// One tenant's FinOps chargeback view over the recent-event window. `window` is
/// the full ring size (the sample horizon); `events_matched` is how many of those
/// belonged to the tenant's keys. Aggregates are broken down `by_model` and
/// `by_key` (both `BTreeMap`, so the JSON ordering is deterministic).
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct ChargebackReport {
    pub window: usize,
    pub events_matched: usize,
    pub totals: ChargebackTotals,
    pub by_model: BTreeMap<String, ChargebackTotals>,
    pub by_key: BTreeMap<String, ChargebackTotals>,
    /// Spend/usage attributed to the caller's business-process / use-case label
    /// (`x-routeplane-use-case`). Only events that carried a label contribute, so
    /// an all-untagged window yields an empty map — and `skip_serializing_if`
    /// keeps the wire byte-identical for tenants that never tag (A/B parity).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_use_case: BTreeMap<String, ChargebackTotals>,
    /// Real p50/p95/p99 upstream latency over THIS tenant's timed provider
    /// attempts in the window (nearest-rank). Percentile fields are `None` when
    /// no attempt was timed — never a misleading zero. Tenant-scoped (only the
    /// caller's key-owned events contribute), unlike the global `latency_stats`.
    pub latency: LatencyPercentiles,
}

/// Summed usage + spend for a chargeback grouping. Token counts widen to `u64`
/// and use saturating adds so a long window never overflows. `cost_micro_usd` is
/// the canonical USD sum budgets reconcile against; `cost_by_currency` carries the
/// per-region local-currency minor-unit totals (paise/cents/fils) — both integer.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct ChargebackTotals {
    pub requests: u64,
    pub successful_requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost_micro_usd: u64,
    pub cost_by_currency: BTreeMap<String, u64>,
}

impl ChargebackTotals {
    /// Fold one event into this grouping. Saturating throughout (money/tokens are
    /// integers and must never panic or wrap on a long window).
    fn accumulate(&mut self, ev: &UsageEvent) {
        self.requests = self.requests.saturating_add(1);
        if ev.success {
            self.successful_requests = self.successful_requests.saturating_add(1);
        }
        self.prompt_tokens = self.prompt_tokens.saturating_add(ev.prompt_tokens as u64);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(ev.completion_tokens as u64);
        self.total_tokens = self.total_tokens.saturating_add(ev.total_tokens as u64);
        if let Some(cost) = &ev.cost {
            self.cost_micro_usd = self.cost_micro_usd.saturating_add(cost.micro_usd);
            let entry = self
                .cost_by_currency
                .entry(cost.currency.clone())
                .or_insert(0);
            *entry = entry.saturating_add(cost.minor_units);
        }
    }
}

/// Latency percentiles in milliseconds over a sample window. `count` is the
/// number of timed samples; percentile fields are `None` when `count == 0`
/// (never a misleading zero). `BTreeMap` ordering in the report is deterministic.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct LatencyPercentiles {
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p95_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ms: Option<u64>,
}

impl LatencyPercentiles {
    /// Exact nearest-rank percentiles. Sorts the sample slice in place. Empty
    /// input → all-None with `count = 0`.
    pub fn from_samples(samples: &mut [u64]) -> Self {
        let count = samples.len();
        if count == 0 {
            return Self::default();
        }
        samples.sort_unstable();
        // Nearest-rank: rank = ceil(p/100 * N), 1-based, clamped to [1, N].
        let nearest = |p: u64| -> u64 {
            let rank = ((p as usize * count).div_ceil(100)).clamp(1, count);
            samples[rank - 1]
        };
        Self {
            count,
            p50_ms: Some(nearest(50)),
            p95_ms: Some(nearest(95)),
            p99_ms: Some(nearest(99)),
            max_ms: Some(samples[count - 1]),
        }
    }
}

/// Overall + per-provider latency percentiles for the `GET /analytics/latency`
/// surface. `by_provider` is a `BTreeMap` for deterministic JSON ordering.
#[derive(Debug, Default, Serialize)]
pub struct LatencyReport {
    pub overall: LatencyPercentiles,
    pub by_provider: BTreeMap<String, LatencyPercentiles>,
}

/// Non-sensitive aggregate of the recent usage ring for `GET /status`. Counts
/// only — no keys, bodies, models, or PII. Rates are `0.0` when their
/// denominator is 0 (never NaN). `by_provider` is a `BTreeMap` for deterministic
/// JSON ordering.
#[derive(Debug, Default, Serialize)]
pub struct UsageSummary {
    /// Number of events in the sampled ring (≤ 1000).
    pub window: usize,
    pub success_count: u64,
    pub failure_count: u64,
    pub success_rate: f64,
    pub by_provider: BTreeMap<String, u64>,
    pub guardrail_block_count: u64,
    pub sovereign_block_count: u64,
    pub cache_hit_count: u64,
    pub cache_participating_count: u64,
    pub cache_hit_rate: f64,
}

/// One tenant's residency observability SUMMARY over the recent-event window for
/// `GET /v1/residency/summary`. `window` is the full ring size sampled (the sample
/// horizon, ≤1000); `total` is how many of those were this tenant's real request
/// attempts. The `*_pct` fields are convenience ratios in `0.0..=1.0` (never NaN —
/// `0.0` when their denominator is 0), so the Console can render them directly.
/// `by_region`/`by_outcome` are `BTreeMap`s for deterministic JSON ordering.
///
/// HONEST-ABSENT by design: there is NO daily `series` (the in-memory ring is not
/// dated history — a durable daily series needs the telemetry store, ADR-024) and
/// NO compliance-framework breakdown (the ring records no per-request framework).
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct ResidencySummaryView {
    /// Full ring size sampled (the sample horizon, ≤1000).
    pub window: usize,
    /// This tenant's real request attempts in the window (the summary denominator).
    pub total: u64,
    /// Requests that carried regulated data (a locked region OR sovereign-routed).
    pub regulated_count: u64,
    /// Requests sovereign-routed to an in-region provider.
    pub sovereign_routed_count: u64,
    /// Requests refused for residency reasons (the `sovereign_block`, HTTP 422).
    pub blocked_count: u64,
    /// Residency-constrained requests where no in-region provider succeeded.
    pub all_failed_count: u64,
    /// Convenience ratios in `0.0..=1.0` over `total` (0.0 when `total == 0`).
    pub regulated_pct: f64,
    pub sovereign_routed_pct: f64,
    pub blocked_pct: f64,
    pub all_failed_pct: f64,
    /// Count by locked jurisdiction; `None` region → the sentinel key "none".
    pub by_region: BTreeMap<String, u64>,
    /// Count by residency outcome label (the `residency_outcome` snake_case set).
    pub by_outcome: BTreeMap<String, u64>,
}

impl ResidencySummaryView {
    /// Finalise the convenience percentages from the accumulated counts. Called
    /// once after the fold (denominator is `total`; `0.0` when there is no traffic,
    /// never NaN). Kept separate so the hot fold stays a pure count accumulation.
    pub fn finalize(mut self) -> Self {
        let total = self.total as f64;
        let pct = |n: u64| {
            if self.total > 0 {
                n as f64 / total
            } else {
                0.0
            }
        };
        self.regulated_pct = pct(self.regulated_count);
        self.sovereign_routed_pct = pct(self.sovereign_routed_count);
        self.blocked_pct = pct(self.blocked_count);
        self.all_failed_pct = pct(self.all_failed_count);
        self
    }
}

/// One row of the residency-decision LEDGER for `GET /v1/residency/ledger` — the
/// read-only, PII-free projection of a ring `UsageEvent`'s residency decision.
/// Carries ONLY decision labels (region, outcome, model, key), never raw request
/// or response content (the no-reflection posture). The `id` is the SAME
/// synthesized stable id `/v1/logs` uses (a hash of the event fields), so a row
/// is stable within a snapshot read.
///
/// The compliance `framework` is deliberately OMITTED — the in-memory ring records
/// no per-request framework, so honest-absent rather than fabricated.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResidencyLedgerRow {
    /// Synthesized stable id (`log_<hex>`), shared with the `/v1/logs` projection.
    pub id: String,
    pub timestamp: DateTime<Utc>,
    /// `regulated` when residency applied to the request, else `none`.
    pub classification: &'static str,
    /// The locked residency jurisdiction this request required, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_region: Option<String>,
    /// The jurisdiction the request was actually routed to (a SUCCESSFUL sovereign
    /// route); `None` for a block (routed nowhere) or a non-routed request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_region: Option<String>,
    /// The residency outcome label (the `residency_outcome` snake_case set).
    pub outcome: &'static str,
    pub model: String,
    pub virtual_key_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use routeplane_guardrails::{CheckAction, Hook, Verdict};
    use routeplane_limits::pricing::CostBreakdown;

    fn outcome() -> CheckOutcome {
        CheckOutcome {
            id: "c1".into(),
            check_type: "regex".into(),
            hook: Hook::BeforeRequest,
            action: CheckAction::Observe,
            verdict: Verdict::Fail,
            detail: Some("pattern matched".into()),
        }
    }

    #[test]
    fn guardrails_field_is_omitted_when_absent_byte_identical() {
        // Ship-dark guarantee at the event layer: an event with no guardrail
        // outcomes serializes WITHOUT the `guardrails` key — byte-identical to
        // the pre-G2.6 wire shape.
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        );
        let v = serde_json::to_value(&e).unwrap();
        assert!(v.get("guardrails").is_none());

        // with_guardrails(empty) must also attach nothing.
        let e2 = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_guardrails(Vec::new());
        assert!(serde_json::to_value(&e2)
            .unwrap()
            .get("guardrails")
            .is_none());
    }

    #[test]
    fn with_guardrails_attaches_outcomes() {
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_guardrails(vec![outcome()]);
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["guardrails"][0]["id"], "c1");
        assert_eq!(v["guardrails"][0]["verdict"], "fail");
        assert_eq!(v["guardrails"][0]["action"], "observe");
    }

    #[cfg(feature = "enterprise")]
    #[test]
    fn guardrails_block_event_shape() {
        let e =
            UsageEvent::guardrails_block("k".into(), "gpt-4o".into(), None, false, vec![outcome()]);
        assert_eq!(e.provider, "(guardrails_denied)");
        assert!(!e.success);
        assert_eq!(e.error.as_deref(), Some("guardrails_denied"));
        assert_eq!(e.total_tokens, 0);
        assert!(e.guardrails.is_some());
    }

    #[cfg(feature = "enterprise")]
    #[tokio::test]
    async fn usage_summary_aggregates_recent_ring() {
        // Push directly into the ring (the test is the writer) so the aggregation
        // is exercised deterministically, without racing the async drain task.
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            ring.push_back(UsageEvent::success(
                "k".into(),
                "openai".into(),
                "gpt-4o".into(),
                1,
                2,
                3,
                None,
                false,
            ));
            let mut cache_hit = UsageEvent::success(
                "k".into(),
                "openai".into(),
                "gpt-4o".into(),
                1,
                2,
                3,
                None,
                false,
            );
            cache_hit.cache_hit = Some(true);
            ring.push_back(cache_hit);
            ring.push_back(UsageEvent::failure(
                "k".into(),
                "anthropic".into(),
                "claude".into(),
                None,
                false,
                "boom".into(),
            ));
            ring.push_back(UsageEvent::sovereign_block(
                "k".into(),
                "m".into(),
                Some("IN".into()),
            ));
            ring.push_back(UsageEvent::guardrails_block(
                "k".into(),
                "m".into(),
                None,
                false,
                vec![outcome()],
            ));
        }

        let s = engine.usage_summary();
        assert_eq!(s.window, 5);
        // Real provider attempts only (sentinels excluded from traffic counts).
        assert_eq!(s.success_count, 2);
        assert_eq!(s.failure_count, 1);
        assert_eq!(s.by_provider.get("openai"), Some(&2));
        assert_eq!(s.by_provider.get("anthropic"), Some(&1));
        assert!(!s.by_provider.contains_key("(sovereign_block)"));
        // Blocks counted by the error sentinel.
        assert_eq!(s.guardrail_block_count, 1);
        assert_eq!(s.sovereign_block_count, 1);
        // Cache: one participating hit.
        assert_eq!(s.cache_participating_count, 1);
        assert_eq!(s.cache_hit_count, 1);
        assert!((s.success_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.cache_hit_rate - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn cache_savings_sums_estimate_and_tokens_over_owned_hits() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            // Two served cache hits on the tenant's key.
            ring.push_back(
                UsageEvent::success(
                    "k".into(),
                    "(cache)".into(),
                    "gpt-4o".into(),
                    4,
                    8,
                    12,
                    None,
                    false,
                )
                .with_cache_hit(Some("default".into()), 4_321),
            );
            ring.push_back(
                UsageEvent::success(
                    "k".into(),
                    "(semantic-cache)".into(),
                    "gpt-4o".into(),
                    10,
                    20,
                    30,
                    None,
                    false,
                )
                .with_cache_hit(Some("default".into()), 6_000),
            );
            // A cache miss (participating, not a hit).
            ring.push_back(
                UsageEvent::success(
                    "k".into(),
                    "openai".into(),
                    "gpt-4o".into(),
                    1,
                    2,
                    3,
                    None,
                    false,
                )
                .with_cache_status(Some("miss"), Some("default".into())),
            );
            // Another tenant's hit — excluded by key ownership.
            ring.push_back(
                UsageEvent::success(
                    "k_other".into(),
                    "(cache)".into(),
                    "gpt-4o".into(),
                    5,
                    5,
                    99,
                    None,
                    false,
                )
                .with_cache_hit(Some("default".into()), 9_999),
            );
        }

        let keys = std::collections::BTreeSet::from(["k".to_string()]);
        let report = engine.cache_savings(&keys, chrono::Duration::minutes(60));
        assert_eq!(report.cache_hits, 2);
        assert_eq!(report.cacheable_lookups, 3); // two hits + one miss
        assert_eq!(report.saved_cost_micro_usd, 4_321 + 6_000);
        assert_eq!(report.saved_tokens, 12 + 30); // served totals not re-sent
    }

    #[tokio::test]
    async fn cache_savings_empty_is_honest_zero() {
        let engine = ObservabilityEngine::new();
        let keys = std::collections::BTreeSet::from(["k".to_string()]);
        let report = engine.cache_savings(&keys, chrono::Duration::minutes(60));
        // The window is reported honestly; the savings are all zero (no hits).
        assert_eq!(report.window_secs, 3600);
        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.cacheable_lookups, 0);
        assert_eq!(report.saved_cost_micro_usd, 0);
        assert_eq!(report.saved_tokens, 0);
    }

    #[cfg(feature = "enterprise")]
    #[test]
    fn guardrails_output_denied_keeps_real_token_counts() {
        // FinOps accuracy: an output denial still consumed upstream tokens.
        let e = UsageEvent::guardrails_output_denied(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            10,
            20,
            30,
            None,
            false,
            vec![outcome()],
        );
        assert!(!e.success);
        assert_eq!(e.total_tokens, 30);
        assert_eq!(e.error.as_deref(), Some("guardrails_denied"));
    }

    // --- G2.5 (PRD-007 FR-16): cache fields ------------------------------------

    #[test]
    fn cache_fields_are_omitted_when_absent_byte_identical() {
        // AC-7 at the event layer: a request with no cache config serializes
        // WITHOUT any cache key — byte-identical to the pre-G2.5 wire shape.
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_cache_status(None, None);
        let v = serde_json::to_value(&e).unwrap();
        for absent in [
            "cache_hit",
            "cache_status",
            "cache_namespace",
            "estimated_saved_cost_micro_usd",
        ] {
            assert!(v.get(absent).is_none(), "{absent} should be omitted");
        }
    }

    #[test]
    fn cache_hit_event_records_stored_usage_and_saved_cost() {
        let e = UsageEvent::success(
            "k".into(),
            "(cache)".into(),
            "gpt-4o".into(),
            7,
            5,
            12,
            None,
            false,
        )
        .with_cache_hit(Some("default".into()), 4321);
        assert!(e.success);
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["cache_hit"], true);
        assert_eq!(v["cache_status"], "hit");
        assert_eq!(v["cache_namespace"], "default");
        assert_eq!(v["estimated_saved_cost_micro_usd"], 4321);
        assert_eq!(v["total_tokens"], 12); // stored usage, not zero
        assert_eq!(v["provider"], "(cache)");
    }

    #[test]
    fn cache_miss_and_bypass_annotations() {
        let miss = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_cache_status(Some("miss"), Some("default".into()));
        let v = serde_json::to_value(&miss).unwrap();
        assert_eq!(v["cache_hit"], false);
        assert_eq!(v["cache_status"], "miss");
        assert!(v.get("estimated_saved_cost_micro_usd").is_none());

        let bypass = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_cache_status(Some("bypass"), Some("default".into()));
        assert_eq!(bypass.cache_status.as_deref(), Some("bypass"));
        assert_eq!(bypass.cache_hit, Some(false));
    }

    // --- PRD-010 FR-17: prompt fields ------------------------------------------

    #[test]
    fn prompt_fields_omitted_on_non_prompt_events_byte_identical() {
        // A chat/embeddings success event must NOT carry any prompt_* key.
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        );
        let v = serde_json::to_value(&e).unwrap();
        for absent in ["prompt_id", "prompt_version", "prompt_label"] {
            assert!(v.get(absent).is_none(), "{absent} should be omitted");
        }
    }

    #[test]
    fn prompt_render_event_records_resolved_version_even_for_a_label() {
        // AC-10: the concrete integer version is recorded even when referenced by
        // label, and the label is carried alongside.
        let e = UsageEvent::prompt_render(
            "k".into(),
            "gpt-4o".into(),
            "prompt_greeting".into(),
            7,
            Some("prod".into()),
            None,
            false,
        );
        assert!(e.success);
        assert_eq!(e.provider, "(prompt_render)");
        assert_eq!(e.total_tokens, 0);
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["prompt_id"], "prompt_greeting");
        assert_eq!(v["prompt_version"], 7);
        assert_eq!(v["prompt_label"], "prod");
    }

    // --- PRD-010 (A/B testing): experiment annotation --------------------------

    #[test]
    fn experiment_fields_omitted_when_absent_byte_identical() {
        // A prompt.render event for a NON-experiment reference (or with_experiment
        // given None) must carry neither prompt_experiment nor prompt_variant —
        // byte-identical to the pre-A/B prompt.render wire shape.
        let e = UsageEvent::prompt_render(
            "k".into(),
            "gpt-4o".into(),
            "prompt_x".into(),
            2,
            Some("prod".into()),
            None,
            false,
        )
        .with_experiment(None);
        let v = serde_json::to_value(&e).unwrap();
        assert!(v.get("prompt_experiment").is_none());
        assert!(v.get("prompt_variant").is_none());
        // The base prompt fields are unchanged.
        assert_eq!(v["prompt_version"], 2);
        assert_eq!(v["prompt_label"], "prod");
    }

    #[test]
    fn with_experiment_attaches_name_and_served_variant() {
        let e = UsageEvent::prompt_render(
            "k".into(),
            "gpt-4o".into(),
            "prompt_x".into(),
            3,
            None,
            None,
            false,
        )
        .with_experiment(Some(("tone".into(), "casual".into())));
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["prompt_experiment"], "tone");
        assert_eq!(v["prompt_variant"], "casual");
        // The served concrete version is still the base field.
        assert_eq!(v["prompt_version"], 3);
        // An experiment-resolved render carries no static label.
        assert!(v.get("prompt_label").is_none());
    }

    #[test]
    fn experiment_fields_absent_on_non_prompt_events() {
        // A regular chat success event never gains the A/B fields.
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        );
        let v = serde_json::to_value(&e).unwrap();
        assert!(v.get("prompt_experiment").is_none());
        assert!(v.get("prompt_variant").is_none());
    }

    // --- Latency percentiles ---------------------------------------------------

    #[test]
    fn latency_field_omitted_when_absent_byte_identical() {
        // A success event with no latency attached must NOT carry `latency_ms` —
        // byte-identical to the pre-latency wire shape (A/B parity guard).
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        );
        assert!(serde_json::to_value(&e)
            .unwrap()
            .get("latency_ms")
            .is_none());
    }

    #[test]
    fn with_latency_attaches_value() {
        let e = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_latency(42);
        assert_eq!(serde_json::to_value(&e).unwrap()["latency_ms"], 42);
    }

    #[test]
    fn percentiles_empty_input_is_all_none() {
        let p = LatencyPercentiles::from_samples(&mut []);
        assert_eq!(p.count, 0);
        assert!(p.p50_ms.is_none());
        assert!(p.p99_ms.is_none());
        // Serializes to just `{"count":0}` — percentile keys omitted.
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["count"], 0);
        assert!(v.get("p50_ms").is_none());
    }

    #[test]
    fn percentiles_nearest_rank_known_values() {
        // 1..=100 → nearest-rank: p50=50, p95=95, p99=99, max=100.
        let mut s: Vec<u64> = (1..=100).collect();
        let p = LatencyPercentiles::from_samples(&mut s);
        assert_eq!(p.count, 100);
        assert_eq!(p.p50_ms, Some(50));
        assert_eq!(p.p95_ms, Some(95));
        assert_eq!(p.p99_ms, Some(99));
        assert_eq!(p.max_ms, Some(100));
    }

    #[test]
    fn percentiles_single_sample() {
        let mut s = [7u64];
        let p = LatencyPercentiles::from_samples(&mut s);
        assert_eq!(p.count, 1);
        assert_eq!(p.p50_ms, Some(7));
        assert_eq!(p.p99_ms, Some(7));
        assert_eq!(p.max_ms, Some(7));
    }

    #[tokio::test]
    async fn latency_stats_aggregates_overall_and_per_provider() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            for ms in [10u64, 20, 30] {
                ring.push_back(
                    UsageEvent::success(
                        "k".into(),
                        "openai".into(),
                        "gpt-4o".into(),
                        1,
                        2,
                        3,
                        None,
                        false,
                    )
                    .with_latency(ms),
                );
            }
            ring.push_back(
                UsageEvent::success(
                    "k".into(),
                    "anthropic".into(),
                    "claude".into(),
                    1,
                    2,
                    3,
                    None,
                    false,
                )
                .with_latency(100),
            );
            // Synthetic sentinel + a timed-less event must be excluded.
            ring.push_back(UsageEvent::sovereign_block(
                "k".into(),
                "m".into(),
                Some("IN".into()),
            ));
            ring.push_back(UsageEvent::failure(
                "k".into(),
                "openai".into(),
                "gpt-4o".into(),
                None,
                false,
                "boom".into(),
            ));
        }

        let report = engine.latency_stats();
        // Overall: 4 timed samples (10,20,30,100).
        assert_eq!(report.overall.count, 4);
        assert_eq!(report.overall.max_ms, Some(100));
        // Per-provider split; sentinel/untimed excluded.
        assert_eq!(report.by_provider.get("openai").unwrap().count, 3);
        assert_eq!(report.by_provider.get("anthropic").unwrap().count, 1);
        assert!(!report.by_provider.contains_key("(sovereign_block)"));
    }

    // --- FinOps chargeback (PRD-008 FR-24) -------------------------------------

    fn cost(micro_usd: u64, currency: &str, minor_units: u64) -> CostBreakdown {
        CostBreakdown {
            micro_usd,
            currency: currency.into(),
            minor_units,
            region: None,
        }
    }

    #[tokio::test]
    async fn chargeback_is_tenant_isolated_and_aggregates_by_model_and_key() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            // Two events on the requesting tenant's keys (k_acme_a, k_acme_b),
            // across two models and two currencies.
            ring.push_back(
                UsageEvent::success(
                    "k_acme_a".into(),
                    "openai".into(),
                    "gpt-4o".into(),
                    10,
                    5,
                    15,
                    None,
                    true,
                )
                .with_cost(cost(2_000, "USD", 200)),
            );
            ring.push_back(
                UsageEvent::success(
                    "k_acme_b".into(),
                    "gemini".into(),
                    "gemini-pro".into(),
                    20,
                    10,
                    30,
                    Some("IN".into()),
                    true,
                )
                .with_cost(cost(3_000, "INR", 24)),
            );
            // A failure on an owned key still counts as a request (not a success).
            ring.push_back(UsageEvent::failure(
                "k_acme_a".into(),
                "openai".into(),
                "gpt-4o".into(),
                None,
                false,
                "boom".into(),
            ));
            // Another tenant's key — MUST be excluded (isolation).
            ring.push_back(
                UsageEvent::success(
                    "k_other".into(),
                    "openai".into(),
                    "gpt-4o".into(),
                    99,
                    99,
                    198,
                    None,
                    true,
                )
                .with_cost(cost(9_999, "USD", 999)),
            );
            // Synthetic sentinel on an owned key — no chargeable spend, excluded.
            ring.push_back(UsageEvent::sovereign_block(
                "k_acme_a".into(),
                "m".into(),
                Some("IN".into()),
            ));
        }

        let owned: std::collections::BTreeSet<String> =
            ["k_acme_a".to_string(), "k_acme_b".to_string()]
                .into_iter()
                .collect();
        let report = engine.chargeback(&owned);

        // Window is the full ring (5); only 3 real events belong to this tenant.
        assert_eq!(report.window, 5);
        assert_eq!(report.events_matched, 3);

        // Totals: 3 requests, 2 successful; tokens summed across owned events.
        assert_eq!(report.totals.requests, 3);
        assert_eq!(report.totals.successful_requests, 2);
        assert_eq!(report.totals.prompt_tokens, 30);
        assert_eq!(report.totals.completion_tokens, 15);
        assert_eq!(report.totals.total_tokens, 45);
        // Canonical USD sum is 2_000 + 3_000 (the failure carried no cost).
        assert_eq!(report.totals.cost_micro_usd, 5_000);
        assert_eq!(report.totals.cost_by_currency.get("USD"), Some(&200));
        assert_eq!(report.totals.cost_by_currency.get("INR"), Some(&24));

        // Per-model split.
        assert_eq!(report.by_model.get("gpt-4o").unwrap().requests, 2);
        assert_eq!(
            report.by_model.get("gpt-4o").unwrap().successful_requests,
            1
        );
        assert_eq!(report.by_model.get("gemini-pro").unwrap().requests, 1);

        // Per-key split — the other tenant's key never appears.
        assert_eq!(report.by_key.get("k_acme_a").unwrap().requests, 2);
        assert_eq!(report.by_key.get("k_acme_b").unwrap().total_tokens, 30);
        assert!(!report.by_key.contains_key("k_other"));
    }

    #[tokio::test]
    async fn chargeback_empty_for_a_tenant_with_no_matching_keys() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            ring.push_back(
                UsageEvent::success(
                    "k_other".into(),
                    "openai".into(),
                    "gpt-4o".into(),
                    1,
                    1,
                    2,
                    None,
                    true,
                )
                .with_cost(cost(1_000, "USD", 100)),
            );
        }
        let none: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let report = engine.chargeback(&none);
        assert_eq!(report.window, 1);
        assert_eq!(report.events_matched, 0);
        assert_eq!(report.totals, ChargebackTotals::default());
        assert!(report.by_model.is_empty());
        assert!(report.by_key.is_empty());
    }

    #[tokio::test]
    async fn recent_events_owned_excludes_other_tenants_and_their_error_bodies() {
        // Two tenants share the cell's observability ring. `/analytics` must return
        // ONLY the caller's own events — never another tenant's key name or the raw
        // provider `error` body (which can echo prompt text). Regression for the
        // cross-tenant `/analytics` disclosure.
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            ring.push_back(UsageEvent::success(
                "tenant_a_key".into(),
                "openai".into(),
                "gpt-4o".into(),
                1,
                2,
                3,
                None,
                false,
            ));
            // Tenant B's failure carries a raw upstream body in `error`.
            ring.push_back(UsageEvent::failure(
                "tenant_b_key".into(),
                "openai".into(),
                "gpt-4o".into(),
                None,
                false,
                "openai API error (400): {\"prompt\":\"tenant B secret\"}".into(),
            ));
        }

        let mut scope = std::collections::BTreeSet::new();
        scope.insert("tenant_a_key".to_string());
        let owned = engine.recent_events_owned(&scope);

        assert_eq!(owned.len(), 1, "only tenant A's event is visible");
        assert_eq!(owned[0].virtual_key_name, "tenant_a_key");
        assert!(
            owned.iter().all(|e| e.virtual_key_name == "tenant_a_key"),
            "no other tenant's key name leaks through"
        );
        assert!(
            !owned.iter().any(|e| e
                .error
                .as_deref()
                .is_some_and(|s| s.contains("tenant B secret"))),
            "another tenant's raw provider error body must never appear"
        );

        // An empty scope (a tenant that owns no keys) sees nothing.
        let empty: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        assert!(engine.recent_events_owned(&empty).is_empty());
    }

    // --- F14 (G2.2 / ADR-021): config_ref / config_match -----------------------

    #[test]
    fn config_fields_omitted_when_absent_and_attached_when_present() {
        // Legacy (no routing config) → both keys omitted, byte-identical.
        let legacy = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_config(None, Some("ignored".into()));
        let v = serde_json::to_value(&legacy).unwrap();
        assert!(v.get("config_ref").is_none());
        assert!(v.get("config_match").is_none());

        // With a config source → both present.
        let cfg = UsageEvent::success(
            "k".into(),
            "openai".into(),
            "gpt-4o".into(),
            1,
            2,
            3,
            None,
            false,
        )
        .with_config(Some("inline"), Some("premium_users".into()));
        let v2 = serde_json::to_value(&cfg).unwrap();
        assert_eq!(v2["config_ref"], "inline");
        assert_eq!(v2["config_match"], "premium_users");
    }

    // --- recent_events (GET /v1/logs read surface) -----------------------------

    #[tokio::test]
    async fn recent_events_is_tenant_isolated_newest_first_and_bounded() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            // Oldest first into the ring (push_back), so newest-first output is the
            // reverse: k_acme_b (latency+cost) is the most-recent owned attempt.
            ring.push_back(UsageEvent::success(
                "k_acme_a".into(),
                "openai".into(),
                "gpt-4o".into(),
                10,
                5,
                15,
                None,
                false,
            ));
            // Another tenant's key — MUST be excluded (isolation).
            ring.push_back(UsageEvent::success(
                "k_other".into(),
                "openai".into(),
                "gpt-4o".into(),
                99,
                99,
                198,
                None,
                true,
            ));
            // A failure on an owned key → outcome "error".
            ring.push_back(UsageEvent::failure(
                "k_acme_a".into(),
                "anthropic".into(),
                "claude".into(),
                None,
                false,
                "boom".into(),
            ));
            // A synthetic JOIN sentinel on an owned key — excluded (not an attempt).
            ring.push_back(UsageEvent::prompt_render(
                "k_acme_a".into(),
                "gpt-4o".into(),
                "prompt_x".into(),
                1,
                None,
                None,
                false,
            ));
            // A sovereign block on an owned key — INCLUDED (a real outcome).
            ring.push_back(UsageEvent::sovereign_block(
                "k_acme_b".into(),
                "gemini-pro".into(),
                Some("IN".into()),
            ));
            // The most-recent owned success, priced + timed.
            ring.push_back(
                UsageEvent::success(
                    "k_acme_b".into(),
                    "gemini".into(),
                    "gemini-pro".into(),
                    20,
                    10,
                    30,
                    Some("IN".into()),
                    true,
                )
                .with_latency(42)
                .with_cost(cost(3_000, "INR", 24)),
            );
        }

        let owned: std::collections::BTreeSet<String> =
            ["k_acme_a".to_string(), "k_acme_b".to_string()]
                .into_iter()
                .collect();

        let rows = engine.recent_events(&owned, 200);
        // 4 owned attempts (the prompt_render sentinel + the k_other event excluded).
        assert_eq!(rows.len(), 4);
        // Newest-first: the priced k_acme_b success is first.
        assert_eq!(rows[0].virtual_key_name, "k_acme_b");
        assert_eq!(rows[0].provider, "gemini");
        assert_eq!(rows[0].outcome, "success");
        assert_eq!(rows[0].latency_ms, Some(42));
        assert_eq!(rows[0].cost_micro_usd, Some(3_000));
        // The sovereign block is INCLUDED and classified "blocked".
        assert_eq!(rows[1].provider, "(sovereign_block)");
        assert_eq!(rows[1].outcome, "blocked");
        // The provider failure is "error".
        assert_eq!(rows[2].provider, "anthropic");
        assert_eq!(rows[2].outcome, "error");
        // The oldest owned event is last.
        assert_eq!(rows[3].virtual_key_name, "k_acme_a");
        assert_eq!(rows[3].provider, "openai");
        // No other-tenant or synthetic-join row ever appears.
        assert!(rows.iter().all(|r| r.virtual_key_name != "k_other"));
        assert!(rows.iter().all(|r| r.provider != "(prompt_render)"));
        // Every row carries a synthesized, stable id.
        assert!(rows.iter().all(|r| r.id.starts_with("log_")));

        // The limit is respected (cap at 2 → the 2 newest owned rows).
        let capped = engine.recent_events(&owned, 2);
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0].virtual_key_name, "k_acme_b");
        assert_eq!(capped[1].provider, "(sovereign_block)");

        // A tenant owning no matching keys sees nothing.
        let none: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        assert!(engine.recent_events(&none, 200).is_empty());
    }

    #[cfg(feature = "enterprise")]
    #[test]
    fn log_row_reduces_guardrails_to_labels_only_no_reflection() {
        // A guardrails-denied block carries outcomes — the row must expose ONLY the
        // id + verdict label, never the detail string (no-reflection posture).
        let ev = UsageEvent::guardrails_block(
            "k".into(),
            "gpt-4o".into(),
            None,
            false,
            vec![outcome()], // has detail Some("pattern matched")
        );
        let row = LogRow::from_event(&ev);
        assert_eq!(row.outcome, "blocked");
        let labels = row.guardrails.as_ref().expect("labels present");
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].id, "c1");
        assert_eq!(labels[0].verdict, "fail");
        // Serialized row must NOT contain the matched-pattern detail anywhere.
        let v = serde_json::to_value(&row).unwrap();
        assert!(!v.to_string().contains("pattern matched"));
    }

    // --- usage_timeseries (GET /v1/finops/timeseries read surface) -------------

    #[tokio::test]
    async fn timeseries_buckets_recent_events_and_is_tenant_isolated() {
        let engine = ObservabilityEngine::new();
        let now = Utc::now();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            // Helper: an owned priced+timed success at a given age (mins ago).
            let mut at = |key: &str, mins_ago: i64, ms: u64, micro: u64, tokens: u32| {
                let mut ev = UsageEvent::success(
                    key.into(),
                    "openai".into(),
                    "gpt-4o".into(),
                    tokens,
                    0,
                    tokens,
                    None,
                    false,
                )
                .with_latency(ms)
                .with_cost(cost(micro, "USD", micro / 10));
                ev.timestamp = now - chrono::Duration::minutes(mins_ago);
                ring.push_back(ev);
            };
            // Two owned events, ~5 min apart, well inside a 60-min window.
            at("k_acme_a", 50, 100, 2_000, 10);
            at("k_acme_a", 45, 300, 3_000, 20);
            // An owned event WAY outside the window (excluded — honest, the ring may
            // not span the window).
            at("k_acme_b", 600, 999, 9_999, 99);
            // Another tenant's in-window event (excluded — isolation).
            let mut other = UsageEvent::success(
                "k_other".into(),
                "openai".into(),
                "gpt-4o".into(),
                5,
                5,
                10,
                None,
                false,
            )
            .with_cost(cost(7_777, "USD", 777));
            other.timestamp = now - chrono::Duration::minutes(40);
            ring.push_back(other);
            // A synthetic sentinel inside the window on an owned key (excluded).
            let mut sentinel =
                UsageEvent::sovereign_block("k_acme_a".into(), "m".into(), Some("IN".into()));
            sentinel.timestamp = now - chrono::Duration::minutes(30);
            ring.push_back(sentinel);
        }

        let owned: std::collections::BTreeSet<String> =
            ["k_acme_a".to_string(), "k_acme_b".to_string()]
                .into_iter()
                .collect();
        let ts = engine.usage_timeseries(&owned, chrono::Duration::minutes(60), 60);

        // 60 one-minute buckets; only the two in-window owned attempts contribute.
        assert_eq!(ts.buckets.len(), 60);
        assert_eq!(ts.bucket_secs, 60);
        assert_eq!(ts.total_events_in_window, 2);

        // Aggregate sums across buckets exclude the out-of-window owned event, the
        // other tenant's event, and the synthetic sentinel.
        let req: u64 = ts.buckets.iter().map(|b| b.requests).sum();
        let cost_sum: u64 = ts.buckets.iter().map(|b| b.cost_micro_usd).sum();
        let tok: u64 = ts.buckets.iter().map(|b| b.tokens).sum();
        assert_eq!(req, 2);
        assert_eq!(cost_sum, 2_000 + 3_000);
        assert_eq!(tok, 10 + 20);

        // Per-bucket avg latency is the mean of that bucket's timed samples.
        for b in &ts.buckets {
            if b.requests > 0 {
                assert!(b.avg_latency_ms == 100 || b.avg_latency_ms == 300);
            } else {
                assert_eq!(b.avg_latency_ms, 0);
            }
        }

        // A tenant owning no matching keys sees all-zero buckets (honest empty).
        let none: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let empty = engine.usage_timeseries(&none, chrono::Duration::minutes(60), 12);
        assert_eq!(empty.buckets.len(), 12);
        assert_eq!(empty.total_events_in_window, 0);
        assert!(empty.buckets.iter().all(|b| b.requests == 0));
    }

    // --- residency_report (GET /v1/residency/{summary,ledger} read surface) -----

    #[tokio::test]
    async fn residency_report_is_tenant_isolated_and_classifies_outcomes() {
        let engine = ObservabilityEngine::new();
        {
            let mut ring = engine.recent_events.lock().unwrap();
            // Oldest first into the ring (push_back); newest-first output reverses.

            // 1) A passthrough success on an owned key: no region, not routed.
            ring.push_back(UsageEvent::success(
                "k_acme_a".into(),
                "self_hosted".into(),
                "llama".into(),
                10,
                5,
                15,
                None,
                false,
            ));
            // 2) ANOTHER tenant's sovereign-routed success — MUST be excluded.
            ring.push_back(UsageEvent::success(
                "k_other".into(),
                "gemini".into(),
                "gemini-pro".into(),
                1,
                1,
                2,
                Some("IN".into()),
                true,
            ));
            // 3) A failure on an owned key with NO residency constraint → not a
            //    residency outcome (not_regulated, not all_failed).
            ring.push_back(UsageEvent::failure(
                "k_acme_a".into(),
                "openai".into(),
                "gpt-4o".into(),
                None,
                false,
                "boom".into(),
            ));
            // 4) A synthetic JOIN sentinel on an owned key — excluded (not an attempt).
            ring.push_back(UsageEvent::prompt_render(
                "k_acme_a".into(),
                "gpt-4o".into(),
                "prompt_x".into(),
                1,
                None,
                None,
                false,
            ));
            // 5) A residency-constrained provider failure on an owned key → all_failed.
            ring.push_back(UsageEvent::failure(
                "k_acme_b".into(),
                "gemini".into(),
                "gemini-pro".into(),
                Some("IN".into()),
                true,
                "timeout".into(),
            ));
            // 6) A sovereign BLOCK on an owned key → residency_blocked (422).
            ring.push_back(UsageEvent::sovereign_block(
                "k_acme_b".into(),
                "gpt-4o".into(),
                Some("IN".into()),
            ));
            // 7) The most-recent owned attempt: a sovereign-ROUTED success to IN.
            ring.push_back(UsageEvent::success(
                "k_acme_b".into(),
                "gemini".into(),
                "gemini-pro".into(),
                20,
                10,
                30,
                Some("IN".into()),
                true,
            ));
        }

        let owned: std::collections::BTreeSet<String> =
            ["k_acme_a".to_string(), "k_acme_b".to_string()]
                .into_iter()
                .collect();
        let (summary, rows) = engine.residency_report(&owned, 200);

        // Window is the full ring (7); 5 owned real attempts contribute (the other
        // tenant's event + the prompt_render sentinel are excluded).
        assert_eq!(summary.window, 7);
        assert_eq!(summary.total, 5);

        // Counts: regulated = events 5,6,7 (region Some OR routed) = 3.
        assert_eq!(summary.regulated_count, 3);
        // Sovereign-routed = events 7 only (5 failed, 6 blocked) = 1.
        assert_eq!(summary.sovereign_routed_count, 1);
        assert_eq!(summary.blocked_count, 1);
        assert_eq!(summary.all_failed_count, 1);

        // Percentages over total=5.
        assert!((summary.regulated_pct - 3.0 / 5.0).abs() < 1e-9);
        assert!((summary.sovereign_routed_pct - 1.0 / 5.0).abs() < 1e-9);
        assert!((summary.blocked_pct - 1.0 / 5.0).abs() < 1e-9);
        assert!((summary.all_failed_pct - 1.0 / 5.0).abs() < 1e-9);

        // by_region: "IN" for the 3 regulated, "none" for the 2 unconstrained.
        assert_eq!(summary.by_region.get("IN"), Some(&3));
        assert_eq!(summary.by_region.get("none"), Some(&2));
        // The other tenant's region never leaks.
        assert!(!summary.by_region.values().any(|&c| c > 3));

        // by_outcome labels.
        assert_eq!(summary.by_outcome.get("sovereign_routed"), Some(&1));
        assert_eq!(summary.by_outcome.get("residency_blocked"), Some(&1));
        assert_eq!(summary.by_outcome.get("all_failed"), Some(&1));
        // not_regulated = the passthrough success (1) + the unconstrained failure (1).
        assert_eq!(summary.by_outcome.get("not_regulated"), Some(&2));

        // Ledger: 5 owned rows, newest-first → event 7 first.
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].virtual_key_name, "k_acme_b");
        assert_eq!(rows[0].outcome, "sovereign_routed");
        assert_eq!(rows[0].classification, "regulated");
        assert_eq!(rows[0].required_region.as_deref(), Some("IN"));
        // A successful sovereign route DID route to IN.
        assert_eq!(rows[0].routed_region.as_deref(), Some("IN"));

        // The block: routed nowhere → routed_region None, required IN.
        let block = rows
            .iter()
            .find(|r| r.outcome == "residency_blocked")
            .expect("block row present");
        assert_eq!(block.required_region.as_deref(), Some("IN"));
        assert!(block.routed_region.is_none());

        // No other-tenant or synthetic-join row appears; every id is stable.
        assert!(rows.iter().all(|r| r.virtual_key_name != "k_other"));
        assert!(rows.iter().all(|r| r.id.starts_with("log_")));

        // The limit caps the ledger to the newest rows (summary is unaffected).
        let (capped_summary, capped_rows) = engine.residency_report(&owned, 2);
        assert_eq!(capped_rows.len(), 2);
        assert_eq!(capped_summary.total, 5);
        assert_eq!(capped_rows[0].virtual_key_name, "k_acme_b");

        // A tenant owning no matching keys sees an empty report.
        let none: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let (empty, empty_rows) = engine.residency_report(&none, 200);
        assert_eq!(empty.total, 0);
        assert_eq!(empty.regulated_pct, 0.0);
        assert!(empty.by_region.is_empty());
        assert!(empty_rows.is_empty());
    }
}
