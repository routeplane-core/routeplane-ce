//! Routing-policy engine (G2.2 / ADR-021). See the crate CLAUDE.md for the
//! boundary doctrine. Everything here is pure + allocation-bounded; the only
//! shared state is the saved-config `ArcSwap` snapshot.

use arc_swap::ArcSwap;
use routeplane_types::ChatCompletionRequest;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

// --- Limits (PRD-006 §4.5) ----------------------------------------------------

/// Inline `x-routeplane-config` size cap (whole header value).
pub const MAX_CONFIG_BYTES: usize = 8 * 1024;
/// `x-routeplane-metadata` size cap.
pub const MAX_METADATA_BYTES: usize = 4 * 1024;
const MAX_TARGETS_PER_LEVEL: usize = 16;
const MAX_FLATTENED_TARGETS: usize = 32;
const MAX_CONDITIONS: usize = 16;
const MAX_CLAUSES_PER_CONDITION: usize = 8;
const MAX_RETRY_ATTEMPTS: u32 = 5;
const MAX_NESTING_DEPTH: u32 = 2; // root + 1 pool level

/// `retry.on_status` allow-list (PRD-006 §4.1c). Retrying anything else (e.g. a
/// 401) is a key-burning loop, so it is rejected fail-closed.
const ALLOWED_RETRY_STATUS: &[u16] = &[408, 429, 500, 502, 503, 504];
/// Default retryable statuses when a `retry` omits `on_status`.
const DEFAULT_RETRY_STATUS: &[u16] = &[429, 500, 502, 503, 504];

const BACKOFF_INITIAL_MIN_MS: u64 = 1;
const BACKOFF_INITIAL_MAX_MS: u64 = 10_000;
const BACKOFF_MAX_CAP_MS: u64 = 30_000;

// --- Hedge directive bounds (ADR-057 / tail-latency hedging) -------------------
//
// Request hedging speculatively starts the NEXT eligible target while the primary
// is still in flight, so user-visible latency becomes `min(primary, hedge)`. It is
// strictly OPT-IN (the `hedge` object absent ⇒ no hedging ⇒ byte-identical to the
// sequential walk) and HARD-BOUNDED so an operator cannot fan out the whole chain
// (cost amplification) or hedge so eagerly it floods providers.
//
/// Minimum `hedge.delay_ms`. A sub-millisecond trigger is indistinguishable from
/// "hedge immediately" and would double provider load on EVERY request, so the
/// floor forces a deliberate slowness threshold.
pub const HEDGE_DELAY_MIN_MS: u64 = 1;
/// Maximum `hedge.delay_ms`. Beyond ~1 min a hedge can never beat the primary
/// inside any sane request deadline, so it is rejected fail-closed.
pub const HEDGE_DELAY_MAX_MS: u64 = 60_000;
/// Hard ceiling on `hedge.max` (extra concurrent attempts). At most 2 EXTRA
/// in-flight attempts (≤ 3 total) — this caps cost amplification structurally; a
/// larger value is rejected, never clamped, so the operator sees the bound.
pub const HEDGE_MAX_EXTRA: u8 = 2;

/// Param-shaping allow-list (PRD-006 §4.1e): the canonical optional fields of
/// `ChatCompletionRequest` EXCLUDING `messages` and `stream` (shaping either
/// would change response mode or bypass guardrails). `model` is included so a
/// target can pin a model.
const PARAM_ALLOWLIST: &[&str] = &[
    "model",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "user",
];

/// Request fields addressable by a conditional `when` clause (PRD-006 §4.2c).
/// `messages` is deliberately NOT addressable (routing must never depend on
/// prompt content).
const CONDITION_PARAM_FIELDS: &[&str] = &[
    "model",
    "user",
    "stream",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
];

// --- Cache directive bounds (PRD-007 §5.1 / G2.5) ------------------------------

/// `ttl_seconds` bounds (PRD-007 FR-3): sub-10s TTLs are thrash with no economic
/// value; >24h entries outlive the guardrail/config drift window we tolerate.
pub const CACHE_TTL_MIN_SECONDS: u64 = 10;
pub const CACHE_TTL_MAX_SECONDS: u64 = 86_400;
pub const CACHE_TTL_DEFAULT_SECONDS: u64 = 300;
/// Platform cap on a cached body (PRD-007 FR-8). The config value may only
/// LOWER it, never raise it.
pub const CACHE_MAX_RESPONSE_BYTES_CAP: u64 = 262_144;
/// `similarity_threshold` bounds (PRD-007 FR-3/FR-12). The default and floor are
/// PLACEHOLDERS pending the G3.6 eval gate — written into validation so the
/// schema is stable, carrying zero empirical claim.
pub const CACHE_SIMILARITY_MIN: f64 = 0.80;
pub const CACHE_SIMILARITY_MAX: f64 = 0.999;
pub const CACHE_SIMILARITY_DEFAULT: f64 = 0.95;

// --- Errors (PRD-006 §4.6) ----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigErrorCode {
    InvalidConfig,
    ConfigNotFound,
    ConfigTooLarge,
    InvalidMetadata,
}

impl ConfigErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConfigErrorCode::InvalidConfig => "invalid_config",
            ConfigErrorCode::ConfigNotFound => "config_not_found",
            ConfigErrorCode::ConfigTooLarge => "config_too_large",
            ConfigErrorCode::InvalidMetadata => "invalid_metadata",
        }
    }
}

/// A routing-config error. `param` is a JSON Pointer to the offending node so an
/// SDK error handler can point at it (PRD-006 §4.6). Always rendered as HTTP 400.
#[derive(Debug, Clone)]
pub struct ConfigError {
    pub code: ConfigErrorCode,
    pub message: String,
    pub param: String,
}

impl ConfigError {
    fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: ConfigErrorCode::InvalidConfig,
            message: message.into(),
            param: param.into(),
        }
    }
    fn too_large(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: ConfigErrorCode::ConfigTooLarge,
            message: message.into(),
            param: param.into(),
        }
    }
    fn not_found(id: &str) -> Self {
        Self {
            code: ConfigErrorCode::ConfigNotFound,
            message: format!("saved config '{id}' was not found"),
            param: "/".into(),
        }
    }
    fn metadata(message: impl Into<String>) -> Self {
        Self {
            code: ConfigErrorCode::InvalidMetadata,
            message: message.into(),
            param: "/".into(),
        }
    }
    pub fn code_str(&self) -> &'static str {
        self.code.as_str()
    }
    pub fn param(&self) -> &str {
        &self.param
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ConfigError {}

// --- Deterministic RNG (SplitMix64), same pattern as `router::Rng` ------------

/// Seedable RNG used for weighted nested-pool ordering AND backoff jitter.
/// Injectable so tests pin a seed and assert exact delays / orderings.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform value in `[0, bound)`. `bound` must be > 0.
    pub fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

// --- Strategy -----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyStrategy {
    Priority,
    Weighted,
    Cost,
    Latency,
    /// Load-balancing rotation; ordering is router-side (atomic cursor), ADR-021.
    RoundRobin,
    /// Fewest outstanding requests first; ordering is router-side (in-flight
    /// gauge), ADR-021.
    LeastBusy,
    Conditional,
}

impl PolicyStrategy {
    fn parse(value: &str, param: &str) -> Result<Self, ConfigError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "priority" => Ok(PolicyStrategy::Priority),
            "weighted" => Ok(PolicyStrategy::Weighted),
            "cost" => Ok(PolicyStrategy::Cost),
            "latency" => Ok(PolicyStrategy::Latency),
            "round_robin" | "roundrobin" | "round-robin" => Ok(PolicyStrategy::RoundRobin),
            "least_busy" | "leastbusy" | "least-busy" => Ok(PolicyStrategy::LeastBusy),
            "conditional" => Ok(PolicyStrategy::Conditional),
            other => Err(ConfigError::invalid(
                param,
                format!("unknown strategy.mode '{other}'"),
            )),
        }
    }

    /// The `x-routeplane-strategy`-equivalent string the proxy feeds to the
    /// router. `Conditional` resolves to a concrete branch strategy during
    /// evaluation, so it never reaches the proxy; it maps to priority defensively.
    pub fn as_router_str(&self) -> &'static str {
        match self {
            PolicyStrategy::Priority | PolicyStrategy::Conditional => "priority",
            PolicyStrategy::Weighted => "weighted",
            PolicyStrategy::Cost => "cost",
            PolicyStrategy::Latency => "latency",
            PolicyStrategy::RoundRobin => "round_robin",
            PolicyStrategy::LeastBusy => "least_busy",
        }
    }
}

// --- Backoff + retry policy ---------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    pub initial_ms: u64,
    pub max_ms: u64,
    pub jitter: bool,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            initial_ms: 200,
            max_ms: 2_000,
            jitter: true,
        }
    }
}

impl Backoff {
    /// Delay before retry number `retry_index` (0-based; 0 = first retry).
    /// `delay_n = min(initial × 2^n, max)`, full jitter ∈ [0, base] when set.
    pub fn delay(&self, retry_index: u32, rng: &mut Rng) -> Duration {
        let shift = retry_index.min(31);
        let base = self
            .initial_ms
            .saturating_mul(1u64 << shift)
            .min(self.max_ms);
        let ms = if self.jitter && base > 0 {
            rng.next_below(base + 1)
        } else {
            base
        };
        Duration::from_millis(ms)
    }
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Number of RETRIES of the same target after the first try (0–5).
    pub attempts: u32,
    /// Statuses that, when surfaced by an adapter, are retried. Transport errors
    /// and per-attempt timeouts are always retried regardless (the proxy reads
    /// the adapter's `RetryClass`); this set only gates 429/5xx/408.
    pub on_status: BTreeSet<u16>,
    pub backoff: Backoff,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 0,
            on_status: DEFAULT_RETRY_STATUS.iter().copied().collect(),
            backoff: Backoff::default(),
        }
    }
}

// --- Hedge policy (ADR-057) ---------------------------------------------------

/// A validated `hedge` directive. Absent on the config ⇒ `None` ⇒ no hedging (the
/// sequential fallback walk, byte-identical to the pre-hedge proxy). Present ⇒ the
/// proxy may start the next eligible target concurrently once the in-flight
/// attempt has run for `delay` without resolving, capped at `max_extra` EXTRA
/// concurrent attempts (≤ `max_extra + 1` total) and always inside the request
/// deadline. Hedging is ADDITIVE to failure-fallback, not a replacement for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HedgePolicy {
    /// How long an in-flight attempt may run before a hedge is started.
    pub delay: Duration,
    /// Maximum EXTRA concurrent attempts (1..=`HEDGE_MAX_EXTRA`).
    pub max_extra: u8,
}

fn parse_hedge(value: &Value, ptr: &str) -> Result<HedgePolicy, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "hedge must be a JSON object"))?;
    reject_unknown(obj, &["delay_ms", "max"], ptr)?;

    let delay_ms = match obj.get("delay_ms") {
        Some(v) => parse_positive_u64(v, &format!("{ptr}/delay_ms"))?,
        None => {
            return Err(ConfigError::invalid(
                format!("{ptr}/delay_ms"),
                "hedge.delay_ms is required (the slowness threshold before a hedge starts)",
            ))
        }
    };
    if !(HEDGE_DELAY_MIN_MS..=HEDGE_DELAY_MAX_MS).contains(&delay_ms) {
        return Err(ConfigError::invalid(
            format!("{ptr}/delay_ms"),
            format!("hedge.delay_ms must be {HEDGE_DELAY_MIN_MS}–{HEDGE_DELAY_MAX_MS}"),
        ));
    }

    let max_extra = match obj.get("max") {
        Some(v) => {
            let n = parse_u32(v, &format!("{ptr}/max"))?;
            if n == 0 {
                return Err(ConfigError::invalid(
                    format!("{ptr}/max"),
                    "hedge.max must be at least 1 (a hedge config with 0 extras is a no-op; omit `hedge` instead)",
                ));
            }
            if n > HEDGE_MAX_EXTRA as u32 {
                // Fail-closed: never silently clamp an absurd fan-out.
                return Err(ConfigError::invalid(
                    format!("{ptr}/max"),
                    format!("hedge.max must be ≤ {HEDGE_MAX_EXTRA} (extra concurrent attempts)"),
                ));
            }
            n as u8
        }
        None => 1,
    };

    Ok(HedgePolicy {
        delay: Duration::from_millis(delay_ms),
        max_extra,
    })
}

// --- Param shaping (PRD-006 §4.1e) --------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ParamShaping {
    default: Map<String, Value>,
    overrides: Map<String, Value>,
    drop: Vec<String>,
}

impl ParamShaping {
    pub fn is_noop(&self) -> bool {
        self.default.is_empty() && self.overrides.is_empty() && self.drop.is_empty()
    }

    /// Apply default (fill-if-absent) → override (set) → drop (remove) to a
    /// request. Operates on the serialized request object; on a deserialization
    /// failure (a contradictory override type) it returns the input unchanged so
    /// the request still flows (best-effort, documented). `messages`/`stream` are
    /// not in the allow-list, so shaping can never touch them.
    pub fn apply(&self, req: ChatCompletionRequest) -> ChatCompletionRequest {
        if self.is_noop() {
            return req;
        }
        let mut obj = match serde_json::to_value(&req) {
            Ok(Value::Object(m)) => m,
            _ => return req,
        };
        for (k, v) in &self.default {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &self.overrides {
            obj.insert(k.clone(), v.clone());
        }
        for k in &self.drop {
            obj.remove(k);
        }
        serde_json::from_value(Value::Object(obj)).unwrap_or(req)
    }

    /// The concrete downstream `model` this shaping pins on a target, if any.
    /// Used to enumerate a combo's target models so the combo name never reaches
    /// a provider and the eligibility gates run on the resolved models (ADR-086).
    ///
    /// Only an `override` counts as a pin. `default` is fill-if-absent
    /// (`apply` uses `entry(..).or_insert`), but a combo request ALWAYS arrives
    /// with `model` already set to the combo name — so a `default.model` never
    /// applies and the combo name would leak to the provider (a `model_not_found`
    /// at the upstream). Treating a `default`-only target as unpinned makes the
    /// self-contained load check (fail-closed, ADR-086 §A4) reject it, and keeps
    /// the anti-smuggling eligibility gates evaluating the model that is actually
    /// sent.
    pub fn pinned_model(&self) -> Option<&str> {
        self.overrides.get("model").and_then(Value::as_str)
    }
}

// --- Cache directive (PRD-007 §5.1 / G2.5) -------------------------------------

/// `cache.mode` (FR-1). `Semantic` always implies exact-first (FR-4); at G2.5
/// the semantic vector path does not exist yet (G3.6, eval-gated), so semantic
/// mode evaluates with exact-only semantics — the proxy additionally surfaces
/// the FR-11 degradation header for unentitled tenants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Simple,
    Semantic,
}

/// The validated `cache` object of a routing config — the PRD-006 §4.8 reserved
/// key, made LIVE by PRD-007 (§3 explicitly replaces the accepted-but-inert
/// posture). Parsed fail-loud (FR-3): out-of-bounds values 400 with a
/// field-precise JSON Pointer, never silently clamped.
///
/// This is config surface only — storage lives in `crates/cache`, and the
/// participation/bypass decision (regulated data, streaming, opt-in) lives in
/// the proxy (crate-boundary doctrine).
#[derive(Debug, Clone)]
pub struct CacheDirective {
    pub mode: CacheMode,
    /// TTL in seconds, ∈ [10, 86400], default 300 (FR-3).
    pub ttl_seconds: u64,
    /// Within-tenant partition, `[a-z0-9_-]{1,64}`, default "default" (FR-7).
    /// The tenant component of the key NEVER comes from here — it comes from
    /// the authenticated TenantContext, structurally (FR-7).
    pub namespace: String,
    /// FR-9/FR-18: skip the lookup, execute upstream, overwrite the entry.
    pub force_refresh: bool,
    /// Only legal with `mode: "semantic"`; ∈ [0.80, 0.999]. PLACEHOLDER until
    /// the G3.6 eval gate (FR-12) — validated for schema stability only.
    pub similarity_threshold: Option<f64>,
    /// Per-entry size ceiling; ≤ the 256 KiB platform cap, may only lower it
    /// (FR-3/FR-8).
    pub max_response_bytes: usize,
}

fn is_valid_cache_namespace(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn parse_cache_directive(value: &Value, ptr: &str) -> Result<CacheDirective, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "cache must be a JSON object"))?;
    reject_unknown(
        obj,
        &[
            "mode",
            "ttl_seconds",
            "namespace",
            "force_refresh",
            "similarity_threshold",
            "max_response_bytes",
        ],
        ptr,
    )?;

    let mode = match obj.get("mode") {
        Some(Value::String(s)) => match s.as_str() {
            "simple" => CacheMode::Simple,
            "semantic" => CacheMode::Semantic,
            other => {
                return Err(ConfigError::invalid(
                    format!("{ptr}/mode"),
                    format!("unknown cache.mode '{other}' (allowed: simple, semantic)"),
                ))
            }
        },
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/mode"),
                "cache.mode must be a string",
            ))
        }
        None => {
            return Err(ConfigError::invalid(
                format!("{ptr}/mode"),
                "cache.mode is required ('simple' or 'semantic')",
            ))
        }
    };

    let ttl_seconds = match obj.get("ttl_seconds") {
        Some(v) => {
            let n = parse_positive_u64(v, &format!("{ptr}/ttl_seconds"))?;
            if !(CACHE_TTL_MIN_SECONDS..=CACHE_TTL_MAX_SECONDS).contains(&n) {
                return Err(ConfigError::invalid(
                    format!("{ptr}/ttl_seconds"),
                    format!(
                        "cache.ttl_seconds must be {CACHE_TTL_MIN_SECONDS}–{CACHE_TTL_MAX_SECONDS}"
                    ),
                ));
            }
            n
        }
        None => CACHE_TTL_DEFAULT_SECONDS,
    };

    let namespace = match obj.get("namespace") {
        Some(Value::String(s)) => {
            if !is_valid_cache_namespace(s) {
                return Err(ConfigError::invalid(
                    format!("{ptr}/namespace"),
                    "cache.namespace must match [a-z0-9_-]{1,64}",
                ));
            }
            s.clone()
        }
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/namespace"),
                "cache.namespace must be a string",
            ))
        }
        None => "default".to_string(),
    };

    let force_refresh = match obj.get("force_refresh") {
        Some(Value::Bool(b)) => *b,
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/force_refresh"),
                "cache.force_refresh must be a boolean",
            ))
        }
        None => false,
    };

    let similarity_threshold = match obj.get("similarity_threshold") {
        Some(v) => {
            if mode != CacheMode::Semantic {
                return Err(ConfigError::invalid(
                    format!("{ptr}/similarity_threshold"),
                    "cache.similarity_threshold is only legal when mode is 'semantic'",
                ));
            }
            let f = v.as_f64().ok_or_else(|| {
                ConfigError::invalid(
                    format!("{ptr}/similarity_threshold"),
                    "cache.similarity_threshold must be a number",
                )
            })?;
            if !(CACHE_SIMILARITY_MIN..=CACHE_SIMILARITY_MAX).contains(&f) {
                return Err(ConfigError::invalid(
                    format!("{ptr}/similarity_threshold"),
                    format!(
                        "cache.similarity_threshold must be {CACHE_SIMILARITY_MIN}–{CACHE_SIMILARITY_MAX}"
                    ),
                ));
            }
            Some(f)
        }
        None => {
            if mode == CacheMode::Semantic {
                Some(CACHE_SIMILARITY_DEFAULT)
            } else {
                None
            }
        }
    };

    let max_response_bytes = match obj.get("max_response_bytes") {
        Some(v) => {
            let n = parse_positive_u64(v, &format!("{ptr}/max_response_bytes"))?;
            if n > CACHE_MAX_RESPONSE_BYTES_CAP {
                return Err(ConfigError::invalid(
                    format!("{ptr}/max_response_bytes"),
                    format!(
                        "cache.max_response_bytes may only lower the platform cap of {CACHE_MAX_RESPONSE_BYTES_CAP}"
                    ),
                ));
            }
            n as usize
        }
        None => CACHE_MAX_RESPONSE_BYTES_CAP as usize,
    };

    Ok(CacheDirective {
        mode,
        ttl_seconds,
        namespace,
        force_refresh,
        similarity_threshold,
        max_response_bytes,
    })
}

// --- Flattened plan the proxy executes ----------------------------------------

#[derive(Debug, Clone)]
pub struct TargetPlan {
    /// Provider name (`openai|anthropic|gemini|azure_openai`) or an
    /// `@integration/model` slug (resolution lands with the catalog, G2.4).
    pub provider: String,
    pub weight: Option<u32>,
    pub cost: Option<u32>,
    /// Per-attempt timeout cap (narrowed against the deadline by the proxy).
    pub timeout_ms: Option<u64>,
    pub retry: RetryPolicy,
    pub params: ParamShaping,
}

#[derive(Debug, Clone)]
pub struct CompiledPlan {
    /// Effective ordering strategy for the selected target set (never
    /// `Conditional` — that resolves to a concrete branch strategy here).
    pub strategy: PolicyStrategy,
    pub targets: Vec<TargetPlan>,
    /// Config-level request budget (narrow-only against the server deadline).
    pub request_timeout_ms: Option<u64>,
    /// Observability label (FR-9): condition index matched, or "default".
    pub matched_label: String,
    /// Tail-latency hedging directive (ADR-057). `None` ⇒ sequential fallback
    /// (byte-identical to the pre-hedge proxy).
    pub hedge: Option<HedgePolicy>,
}

// --- Internal validated tree --------------------------------------------------

#[derive(Debug, Clone)]
struct Leaf {
    name: Option<String>,
    provider: String,
    weight: Option<u32>,
    cost: Option<u32>,
    timeout_ms: Option<u64>,
    retry: Option<RetryPolicy>,
    params: ParamShaping,
}

#[derive(Debug, Clone)]
struct Pool {
    name: Option<String>,
    strategy: PolicyStrategy,
    leaves: Vec<Leaf>,
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(Leaf),
    Pool(Pool),
}

impl Node {
    fn name(&self) -> Option<&str> {
        match self {
            Node::Leaf(l) => l.name.as_deref(),
            Node::Pool(p) => p.name.as_deref(),
        }
    }
}

#[derive(Debug, Clone)]
struct Condition {
    when: Map<String, Value>,
    target: String,
}

#[derive(Debug, Clone)]
pub enum ConfigRef {
    Inline,
    Saved(String),
}

/// A validated routing config (the compiled form of one `routing` object). Parse
/// is fallible (400 on violation); `evaluate` is pure + infallible.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    strategy: PolicyStrategy,
    targets: Vec<Node>,
    conditions: Vec<Condition>,
    default_target: Option<String>,
    config_retry: RetryPolicy,
    timeout_ms: Option<u64>,
    config_ref: ConfigRef,
    cache: Option<CacheDirective>,
    hedge: Option<HedgePolicy>,
}

impl RoutingConfig {
    /// Config-level request budget; the proxy narrows the deadline by this.
    pub fn request_timeout_ms(&self) -> Option<u64> {
        self.timeout_ms
    }

    /// The validated `cache` directive (PRD-007 §5.1), when the config opts in.
    /// `None` ⇒ no cache participation of any kind (FR-2).
    pub fn cache(&self) -> Option<&CacheDirective> {
        self.cache.as_ref()
    }

    /// The validated `hedge` directive (ADR-057), when the config opts in.
    /// `None` ⇒ no hedging (sequential fallback, byte-identical).
    pub fn hedge(&self) -> Option<HedgePolicy> {
        self.hedge
    }

    pub fn config_ref_label(&self) -> &'static str {
        match self.config_ref {
            ConfigRef::Inline => "inline",
            ConfigRef::Saved(_) => "saved",
        }
    }

    /// Enumerate the concrete downstream `model` each of this config's targets
    /// pins (ADR-086 combos). One entry per leaf, in tree order; `None` for a leaf
    /// with no pinned model. A combo MUST be self-contained (no `None`) so the
    /// combo name never reaches a provider — enforced at registry load.
    pub fn target_models(&self) -> Vec<Option<String>> {
        fn collect(node: &Node, out: &mut Vec<Option<String>>) {
            match node {
                Node::Leaf(l) => out.push(l.params.pinned_model().map(str::to_string)),
                Node::Pool(p) => {
                    for l in &p.leaves {
                        out.push(l.params.pinned_model().map(str::to_string));
                    }
                }
            }
        }
        let mut out = Vec::new();
        for n in &self.targets {
            collect(n, &mut out);
        }
        out
    }

    /// Parse + validate a `routing` object into an executable config.
    pub fn parse(routing: &Value, config_ref: ConfigRef) -> Result<Self, ConfigError> {
        let obj = routing
            .as_object()
            .ok_or_else(|| ConfigError::invalid("/", "routing config must be a JSON object"))?;

        reject_unknown(
            obj,
            &[
                "strategy",
                "targets",
                "conditions",
                "default_target",
                "retry",
                "timeout_ms",
                "cache",
                "hedge",
            ],
            "",
        )?;

        let strategy = match obj.get("strategy") {
            None => PolicyStrategy::Priority,
            Some(Value::String(s)) => PolicyStrategy::parse(s, "/strategy")?,
            Some(Value::Object(m)) => {
                reject_unknown(m, &["mode"], "/strategy")?;
                let mode = m.get("mode").and_then(Value::as_str).ok_or_else(|| {
                    ConfigError::invalid("/strategy/mode", "strategy.mode must be a string")
                })?;
                PolicyStrategy::parse(mode, "/strategy/mode")?
            }
            Some(_) => {
                return Err(ConfigError::invalid(
                    "/strategy",
                    "strategy must be a string or an object",
                ))
            }
        };

        // cache (PRD-007 / G2.5): the key PRD-006 §4.8 reserved is now LIVE —
        // parsed and validated fail-loud (FR-3, field-precise pointers). PRD-007
        // §3 explicitly replaces the accepted-but-inert posture.
        let cache = match obj.get("cache") {
            Some(v) => Some(parse_cache_directive(v, "/cache")?),
            None => None,
        };

        // Config-level retry default.
        let config_retry = match obj.get("retry") {
            Some(v) => parse_retry(v, "/retry")?,
            None => RetryPolicy::default(),
        };

        let timeout_ms = match obj.get("timeout_ms") {
            Some(v) => Some(parse_positive_u64(v, "/timeout_ms")?),
            None => None,
        };

        // hedge (ADR-057): tail-latency hedging. Absent ⇒ None ⇒ no hedging.
        let hedge = match obj.get("hedge") {
            Some(v) => Some(parse_hedge(v, "/hedge")?),
            None => None,
        };

        // Targets.
        let targets_val = obj
            .get("targets")
            .and_then(Value::as_array)
            .ok_or_else(|| ConfigError::invalid("/targets", "targets must be a non-empty array"))?;
        if targets_val.is_empty() {
            return Err(ConfigError::invalid(
                "/targets",
                "targets must not be empty",
            ));
        }
        if targets_val.len() > MAX_TARGETS_PER_LEVEL {
            return Err(ConfigError::invalid(
                "/targets",
                format!("targets exceeds the limit of {MAX_TARGETS_PER_LEVEL} per level"),
            ));
        }

        let mut targets = Vec::with_capacity(targets_val.len());
        let mut flattened = 0usize;
        for (i, t) in targets_val.iter().enumerate() {
            let node = parse_node(t, &format!("/targets/{i}"), 1)?;
            flattened += match &node {
                Node::Leaf(_) => 1,
                Node::Pool(p) => p.leaves.len(),
            };
            targets.push(node);
        }
        if flattened > MAX_FLATTENED_TARGETS {
            return Err(ConfigError::invalid(
                "/targets",
                format!("flattened target count {flattened} exceeds limit {MAX_FLATTENED_TARGETS}"),
            ));
        }

        let default_target = obj
            .get("default_target")
            .and_then(Value::as_str)
            .map(str::to_string);

        let conditions = match obj.get("conditions") {
            Some(v) => parse_conditions(v, "/conditions")?,
            None => Vec::new(),
        };

        let cfg = RoutingConfig {
            strategy,
            targets,
            conditions,
            default_target,
            config_retry,
            timeout_ms,
            config_ref,
            cache,
            hedge,
        };

        cfg.validate_conditional()?;
        Ok(cfg)
    }

    /// FR-2a: conditional mode requires unique names, valid condition/default
    /// references, and a present default_target.
    fn validate_conditional(&self) -> Result<(), ConfigError> {
        if self.strategy != PolicyStrategy::Conditional {
            return Ok(());
        }
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for (i, node) in self.targets.iter().enumerate() {
            let name = node.name().ok_or_else(|| {
                ConfigError::invalid(
                    format!("/targets/{i}"),
                    "conditional routing requires every top-level target to have a unique name",
                )
            })?;
            if !names.insert(name) {
                return Err(ConfigError::invalid(
                    format!("/targets/{i}/name"),
                    format!("duplicate target name '{name}'"),
                ));
            }
        }
        let default = self.default_target.as_deref().ok_or_else(|| {
            ConfigError::invalid(
                "/default_target",
                "conditional routing requires a default_target",
            )
        })?;
        if !names.contains(default) {
            return Err(ConfigError::invalid(
                "/default_target",
                format!("default_target '{default}' does not name a target"),
            ));
        }
        for (i, c) in self.conditions.iter().enumerate() {
            if !names.contains(c.target.as_str()) {
                return Err(ConfigError::invalid(
                    format!("/conditions/{i}/target"),
                    format!("condition target '{}' does not name a target", c.target),
                ));
            }
        }
        Ok(())
    }

    /// Pure evaluation: select the branch (conditional), then flatten to an
    /// ordered target list. Never errors (a runtime type mismatch in a condition
    /// evaluates `false` → default_target, FR-2d).
    pub fn evaluate(
        &self,
        metadata: &Metadata,
        req: &ChatCompletionRequest,
        rng: &mut Rng,
    ) -> CompiledPlan {
        if self.strategy == PolicyStrategy::Conditional {
            let mut matched_label = "default".to_string();
            let mut chosen: Option<&str> = self.default_target.as_deref();
            for (i, c) in self.conditions.iter().enumerate() {
                if eval_when(&c.when, metadata, req) {
                    chosen = Some(c.target.as_str());
                    matched_label = i.to_string();
                    break;
                }
            }
            let node = chosen
                .and_then(|name| self.targets.iter().find(|n| n.name() == Some(name)))
                .or_else(|| self.targets.first());
            let (targets, strategy) = match node {
                Some(Node::Leaf(l)) => (
                    vec![leaf_to_plan(l, &self.config_retry)],
                    PolicyStrategy::Priority,
                ),
                Some(Node::Pool(p)) => (flatten_pool(p, &self.config_retry, rng), p.strategy),
                None => (Vec::new(), PolicyStrategy::Priority),
            };
            return CompiledPlan {
                strategy,
                targets,
                request_timeout_ms: self.timeout_ms,
                matched_label,
                hedge: self.hedge,
            };
        }

        // Non-conditional: flatten every top-level target in order (nested pools
        // pre-ordered by their own strategy); the root strategy then orders the
        // flat candidate list at the router.
        let mut targets = Vec::new();
        for node in &self.targets {
            match node {
                Node::Leaf(l) => targets.push(leaf_to_plan(l, &self.config_retry)),
                Node::Pool(p) => targets.extend(flatten_pool(p, &self.config_retry, rng)),
            }
        }
        CompiledPlan {
            strategy: self.strategy,
            targets,
            request_timeout_ms: self.timeout_ms,
            matched_label: "none".to_string(),
            hedge: self.hedge,
        }
    }
}

fn leaf_to_plan(leaf: &Leaf, config_retry: &RetryPolicy) -> TargetPlan {
    TargetPlan {
        provider: leaf.provider.clone(),
        weight: leaf.weight,
        cost: leaf.cost,
        timeout_ms: leaf.timeout_ms,
        retry: leaf.retry.clone().unwrap_or_else(|| config_retry.clone()),
        params: leaf.params.clone(),
    }
}

/// Order a nested pool's leaves by its strategy. Policy has no live health/EWMA,
/// so `cost` orders by declared cost, `weighted` draws via the injected RNG, and
/// `priority`/`latency` keep declared order (latency is honored at the top level
/// by the router, which has the EWMA — documented limitation for nested pools).
fn flatten_pool(pool: &Pool, config_retry: &RetryPolicy, rng: &mut Rng) -> Vec<TargetPlan> {
    let mut leaves: Vec<&Leaf> = pool.leaves.iter().collect();
    match pool.strategy {
        PolicyStrategy::Cost => {
            leaves.sort_by_key(|l| l.cost.unwrap_or(100));
        }
        PolicyStrategy::Weighted => {
            leaves = weighted_order(leaves, rng);
        }
        _ => {}
    }
    leaves
        .into_iter()
        .map(|l| leaf_to_plan(l, config_retry))
        .collect()
}

fn weighted_order<'a>(mut pool: Vec<&'a Leaf>, rng: &mut Rng) -> Vec<&'a Leaf> {
    let mut ordered = Vec::with_capacity(pool.len());
    while !pool.is_empty() {
        let weights: Vec<u64> = pool
            .iter()
            .map(|l| l.weight.unwrap_or(1).max(1) as u64)
            .collect();
        let total: u64 = weights.iter().sum();
        let mut pick = rng.next_below(total.max(1));
        let mut chosen = 0;
        for (i, w) in weights.iter().enumerate() {
            if pick < *w {
                chosen = i;
                break;
            }
            pick -= *w;
        }
        ordered.push(pool.remove(chosen));
    }
    ordered
}

// --- Node / retry / condition parsing -----------------------------------------

fn parse_node(value: &Value, ptr: &str, depth: u32) -> Result<Node, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "target must be a JSON object"))?;

    let is_leaf = obj.contains_key("provider");
    let is_nested = obj.contains_key("targets");
    if is_leaf && is_nested {
        return Err(ConfigError::invalid(
            ptr,
            "a target may not be both a leaf (provider) and a nested pool (targets)",
        ));
    }

    if is_nested {
        if depth >= MAX_NESTING_DEPTH {
            return Err(ConfigError::invalid(
                ptr,
                format!("nesting exceeds the maximum depth of {MAX_NESTING_DEPTH}"),
            ));
        }
        reject_unknown(obj, &["name", "strategy", "targets"], ptr)?;
        let strategy = match obj.get("strategy") {
            None => PolicyStrategy::Priority,
            Some(Value::String(s)) => PolicyStrategy::parse(s, &format!("{ptr}/strategy"))?,
            Some(Value::Object(m)) => {
                reject_unknown(m, &["mode"], &format!("{ptr}/strategy"))?;
                let mode = m.get("mode").and_then(Value::as_str).ok_or_else(|| {
                    ConfigError::invalid(
                        format!("{ptr}/strategy/mode"),
                        "strategy.mode must be a string",
                    )
                })?;
                PolicyStrategy::parse(mode, &format!("{ptr}/strategy/mode"))?
            }
            Some(_) => {
                return Err(ConfigError::invalid(
                    format!("{ptr}/strategy"),
                    "strategy must be a string or object",
                ))
            }
        };
        if strategy == PolicyStrategy::Conditional {
            return Err(ConfigError::invalid(
                format!("{ptr}/strategy"),
                "a nested pool cannot use conditional strategy",
            ));
        }
        let name = obj.get("name").and_then(Value::as_str).map(str::to_string);
        let children = obj
            .get("targets")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ConfigError::invalid(format!("{ptr}/targets"), "nested targets must be an array")
            })?;
        if children.is_empty() {
            return Err(ConfigError::invalid(
                format!("{ptr}/targets"),
                "nested targets must not be empty",
            ));
        }
        if children.len() > MAX_TARGETS_PER_LEVEL {
            return Err(ConfigError::invalid(
                format!("{ptr}/targets"),
                format!("nested targets exceed the limit of {MAX_TARGETS_PER_LEVEL}"),
            ));
        }
        let mut leaves = Vec::with_capacity(children.len());
        for (i, c) in children.iter().enumerate() {
            // depth+1: children of a pool must be leaves (depth budget exhausted).
            match parse_node(c, &format!("{ptr}/targets/{i}"), depth + 1)? {
                Node::Leaf(l) => leaves.push(l),
                Node::Pool(_) => {
                    return Err(ConfigError::invalid(
                        format!("{ptr}/targets/{i}"),
                        "nested pools may only contain leaf targets",
                    ))
                }
            }
        }
        return Ok(Node::Pool(Pool {
            name,
            strategy,
            leaves,
        }));
    }

    // Leaf.
    reject_unknown(
        obj,
        &[
            "name",
            "provider",
            "weight",
            "cost",
            "timeout_ms",
            "retry",
            "params",
        ],
        ptr,
    )?;
    let provider = obj
        .get("provider")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ConfigError::invalid(ptr, "leaf target requires a non-empty 'provider'"))?
        .to_string();
    let name = obj.get("name").and_then(Value::as_str).map(str::to_string);
    let weight = match obj.get("weight") {
        Some(v) => Some(parse_u32(v, &format!("{ptr}/weight"))?),
        None => None,
    };
    let cost = match obj.get("cost") {
        Some(v) => Some(parse_u32(v, &format!("{ptr}/cost"))?),
        None => None,
    };
    let timeout_ms = match obj.get("timeout_ms") {
        Some(v) => Some(parse_positive_u64(v, &format!("{ptr}/timeout_ms"))?),
        None => None,
    };
    let retry = match obj.get("retry") {
        Some(v) => Some(parse_retry(v, &format!("{ptr}/retry"))?),
        None => None,
    };
    let params = match obj.get("params") {
        Some(v) => parse_params(v, &format!("{ptr}/params"))?,
        None => ParamShaping::default(),
    };
    Ok(Node::Leaf(Leaf {
        name,
        provider,
        weight,
        cost,
        timeout_ms,
        retry,
        params,
    }))
}

fn parse_retry(value: &Value, ptr: &str) -> Result<RetryPolicy, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "retry must be a JSON object"))?;
    reject_unknown(obj, &["attempts", "on_status", "backoff"], ptr)?;

    let attempts = match obj.get("attempts") {
        Some(v) => parse_u32(v, &format!("{ptr}/attempts"))?,
        None => 1,
    };
    if attempts > MAX_RETRY_ATTEMPTS {
        return Err(ConfigError::invalid(
            format!("{ptr}/attempts"),
            format!("retry.attempts {attempts} exceeds the maximum of {MAX_RETRY_ATTEMPTS}"),
        ));
    }

    let on_status = match obj.get("on_status") {
        None => DEFAULT_RETRY_STATUS.iter().copied().collect(),
        Some(Value::Array(arr)) => {
            let mut set = BTreeSet::new();
            for (i, code) in arr.iter().enumerate() {
                let c = code.as_u64().ok_or_else(|| {
                    ConfigError::invalid(
                        format!("{ptr}/on_status/{i}"),
                        "on_status entries must be integers",
                    )
                })? as u16;
                if !ALLOWED_RETRY_STATUS.contains(&c) {
                    return Err(ConfigError::invalid(
                        format!("{ptr}/on_status/{i}"),
                        format!(
                            "retry.on_status contains {c}, which is not retryable (allowed: {ALLOWED_RETRY_STATUS:?})"
                        ),
                    ));
                }
                set.insert(c);
            }
            set
        }
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/on_status"),
                "on_status must be an array of status codes",
            ))
        }
    };

    let backoff = match obj.get("backoff") {
        Some(v) => parse_backoff(v, &format!("{ptr}/backoff"))?,
        None => Backoff::default(),
    };

    Ok(RetryPolicy {
        attempts,
        on_status,
        backoff,
    })
}

fn parse_backoff(value: &Value, ptr: &str) -> Result<Backoff, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "backoff must be a JSON object"))?;
    reject_unknown(obj, &["initial_ms", "max_ms", "jitter"], ptr)?;
    let initial_ms = match obj.get("initial_ms") {
        Some(v) => parse_positive_u64(v, &format!("{ptr}/initial_ms"))?,
        None => Backoff::default().initial_ms,
    };
    if !(BACKOFF_INITIAL_MIN_MS..=BACKOFF_INITIAL_MAX_MS).contains(&initial_ms) {
        return Err(ConfigError::invalid(
            format!("{ptr}/initial_ms"),
            format!("backoff.initial_ms must be {BACKOFF_INITIAL_MIN_MS}–{BACKOFF_INITIAL_MAX_MS}"),
        ));
    }
    let max_ms = match obj.get("max_ms") {
        Some(v) => parse_positive_u64(v, &format!("{ptr}/max_ms"))?,
        None => Backoff::default().max_ms.max(initial_ms),
    };
    if max_ms > BACKOFF_MAX_CAP_MS {
        return Err(ConfigError::invalid(
            format!("{ptr}/max_ms"),
            format!("backoff.max_ms must be ≤ {BACKOFF_MAX_CAP_MS}"),
        ));
    }
    if max_ms < initial_ms {
        return Err(ConfigError::invalid(
            format!("{ptr}/max_ms"),
            "backoff.max_ms must be ≥ initial_ms",
        ));
    }
    let jitter = match obj.get("jitter") {
        Some(Value::Bool(b)) => *b,
        None => true,
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/jitter"),
                "backoff.jitter must be a boolean",
            ))
        }
    };
    Ok(Backoff {
        initial_ms,
        max_ms,
        jitter,
    })
}

fn parse_params(value: &Value, ptr: &str) -> Result<ParamShaping, ConfigError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::invalid(ptr, "params must be a JSON object"))?;
    reject_unknown(obj, &["default", "override", "drop"], ptr)?;

    let default = parse_param_map(obj.get("default"), &format!("{ptr}/default"))?;
    let overrides = parse_param_map(obj.get("override"), &format!("{ptr}/override"))?;
    let drop = match obj.get("drop") {
        None => Vec::new(),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, k) in arr.iter().enumerate() {
                let key = k.as_str().ok_or_else(|| {
                    ConfigError::invalid(format!("{ptr}/drop/{i}"), "drop entries must be strings")
                })?;
                check_allowlisted(key, &format!("{ptr}/drop/{i}"))?;
                out.push(key.to_string());
            }
            out
        }
        Some(_) => {
            return Err(ConfigError::invalid(
                format!("{ptr}/drop"),
                "params.drop must be an array of field names",
            ))
        }
    };

    // FR-1e: a key appearing in both override and drop is a contradiction.
    for k in &drop {
        if overrides.contains_key(k) {
            return Err(ConfigError::invalid(
                format!("{ptr}/drop"),
                format!("'{k}' appears in both override and drop"),
            ));
        }
    }

    Ok(ParamShaping {
        default,
        overrides,
        drop,
    })
}

fn parse_param_map(value: Option<&Value>, ptr: &str) -> Result<Map<String, Value>, ConfigError> {
    match value {
        None => Ok(Map::new()),
        Some(Value::Object(m)) => {
            for k in m.keys() {
                check_allowlisted(k, &format!("{ptr}/{k}"))?;
            }
            Ok(m.clone())
        }
        Some(_) => Err(ConfigError::invalid(ptr, "must be a JSON object of params")),
    }
}

fn check_allowlisted(key: &str, ptr: &str) -> Result<(), ConfigError> {
    if PARAM_ALLOWLIST.contains(&key) {
        Ok(())
    } else {
        Err(ConfigError::invalid(
            ptr,
            format!("'{key}' is not a shapeable param (allowed: {PARAM_ALLOWLIST:?})"),
        ))
    }
}

fn parse_conditions(value: &Value, ptr: &str) -> Result<Vec<Condition>, ConfigError> {
    let arr = value
        .as_array()
        .ok_or_else(|| ConfigError::invalid(ptr, "conditions must be an array"))?;
    if arr.len() > MAX_CONDITIONS {
        return Err(ConfigError::invalid(
            ptr,
            format!("conditions exceeds the limit of {MAX_CONDITIONS}"),
        ));
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, c) in arr.iter().enumerate() {
        let cptr = format!("{ptr}/{i}");
        let obj = c
            .as_object()
            .ok_or_else(|| ConfigError::invalid(&cptr, "condition must be an object"))?;
        reject_unknown(obj, &["when", "target"], &cptr)?;
        let target = obj
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ConfigError::invalid(
                    format!("{cptr}/target"),
                    "condition.target must be a string",
                )
            })?
            .to_string();
        let when = obj.get("when").and_then(Value::as_object).ok_or_else(|| {
            ConfigError::invalid(format!("{cptr}/when"), "condition.when must be an object")
        })?;
        if when.len() > MAX_CLAUSES_PER_CONDITION {
            return Err(ConfigError::invalid(
                format!("{cptr}/when"),
                format!("when exceeds {MAX_CLAUSES_PER_CONDITION} clauses"),
            ));
        }
        validate_when(when, &format!("{cptr}/when"))?;
        out.push(Condition {
            when: when.clone(),
            target,
        });
    }
    Ok(out)
}

/// Static validation of a `when` clause (FR-2c/d): legal paths + operators, and
/// statically-detectable type mismatches (ordering op against a non-number).
fn validate_when(when: &Map<String, Value>, ptr: &str) -> Result<(), ConfigError> {
    for (path, spec) in when {
        let pptr = format!("{ptr}/{path}");
        validate_path(path, &pptr)?;
        let ops = spec.as_object().ok_or_else(|| {
            ConfigError::invalid(&pptr, "a clause must be an object of operator → value")
        })?;
        for (op, val) in ops {
            let optr = format!("{pptr}/{op}");
            match op.as_str() {
                "eq" | "ne" => {}
                "in" | "nin" => {
                    if !val.is_array() {
                        return Err(ConfigError::invalid(
                            &optr,
                            format!("operator '{op}' requires an array value"),
                        ));
                    }
                }
                "gt" | "gte" | "lt" | "lte" => {
                    if !val.is_number() {
                        return Err(ConfigError::invalid(
                            &optr,
                            format!("operator '{op}' requires a numeric value"),
                        ));
                    }
                }
                "exists" => {
                    if !val.is_boolean() {
                        return Err(ConfigError::invalid(
                            &optr,
                            "operator 'exists' requires a boolean value",
                        ));
                    }
                }
                other => {
                    return Err(ConfigError::invalid(
                        &optr,
                        format!("unknown operator '{other}'"),
                    ))
                }
            }
        }
    }
    Ok(())
}

fn validate_path(path: &str, ptr: &str) -> Result<(), ConfigError> {
    if let Some(field) = path.strip_prefix("params.") {
        if !CONDITION_PARAM_FIELDS.contains(&field) {
            return Err(ConfigError::invalid(
                ptr,
                format!("params.{field} is not addressable (allowed: {CONDITION_PARAM_FIELDS:?})"),
            ));
        }
        Ok(())
    } else if path.strip_prefix("metadata.").is_some() {
        Ok(())
    } else {
        Err(ConfigError::invalid(
            ptr,
            format!("path '{path}' must start with 'metadata.' or 'params.'"),
        ))
    }
}

// --- Conditional evaluation ---------------------------------------------------

fn eval_when(when: &Map<String, Value>, metadata: &Metadata, req: &ChatCompletionRequest) -> bool {
    for (path, spec) in when {
        let actual = resolve_path(path, metadata, req);
        let Some(ops) = spec.as_object() else {
            return false;
        };
        for (op, expected) in ops {
            if !eval_op(op, actual.as_ref(), expected) {
                return false;
            }
        }
    }
    true
}

fn resolve_path(path: &str, metadata: &Metadata, req: &ChatCompletionRequest) -> Option<Value> {
    if let Some(key) = path.strip_prefix("metadata.") {
        metadata.0.get(key).cloned()
    } else if let Some(field) = path.strip_prefix("params.") {
        param_value(field, req)
    } else {
        None
    }
}

fn param_value(field: &str, req: &ChatCompletionRequest) -> Option<Value> {
    match field {
        "model" => Some(Value::String(req.model.clone())),
        "user" => req.user.clone().map(Value::String),
        "stream" => req.stream.map(Value::Bool),
        "max_tokens" => req.max_tokens.map(Value::from),
        "max_completion_tokens" => req.max_completion_tokens.map(Value::from),
        "temperature" => req
            .temperature
            .and_then(|t| serde_json::Number::from_f64(t as f64).map(Value::Number)),
        _ => None,
    }
}

fn eval_op(op: &str, actual: Option<&Value>, expected: &Value) -> bool {
    if op == "exists" {
        let want = expected.as_bool().unwrap_or(false);
        return actual.is_some() == want;
    }
    let Some(actual) = actual else {
        // Missing value: fail-safe to false (toward default_target).
        return false;
    };
    match op {
        "eq" => actual == expected,
        "ne" => actual != expected,
        "in" => expected
            .as_array()
            .map(|a| a.iter().any(|v| v == actual))
            .unwrap_or(false),
        "nin" => expected
            .as_array()
            .map(|a| !a.iter().any(|v| v == actual))
            .unwrap_or(false),
        "gt" | "gte" | "lt" | "lte" => match (actual.as_f64(), expected.as_f64()) {
            (Some(a), Some(b)) => match op {
                "gt" => a > b,
                "gte" => a >= b,
                "lt" => a < b,
                _ => a <= b,
            },
            _ => false, // runtime type mismatch → false (FR-2d)
        },
        _ => false,
    }
}

// --- Metadata (x-routeplane-metadata) -----------------------------------------

/// Conditional-routing input: a flat map of scalar values. Gateway-local; never
/// forwarded upstream, never masked (documented client contract: no regulated PII).
#[derive(Debug, Clone, Default)]
pub struct Metadata(pub BTreeMap<String, Value>);

/// Parse the `x-routeplane-metadata` header. Absent/empty → empty map.
pub fn parse_metadata(header: Option<&str>) -> Result<Metadata, ConfigError> {
    let Some(raw) = header else {
        return Ok(Metadata::default());
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(Metadata::default());
    }
    if raw.len() > MAX_METADATA_BYTES {
        return Err(ConfigError::too_large(
            "/",
            format!("x-routeplane-metadata exceeds {MAX_METADATA_BYTES} bytes"),
        ));
    }
    let value: Value = serde_json::from_str(raw).map_err(|e| {
        ConfigError::metadata(format!("x-routeplane-metadata is not valid JSON: {e}"))
    })?;
    let obj = value
        .as_object()
        .ok_or_else(|| ConfigError::metadata("x-routeplane-metadata must be a JSON object"))?;
    let mut map = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_)) {
            return Err(ConfigError::metadata(format!(
                "metadata.{k} must be a string, number, or boolean"
            )));
        }
        map.insert(k.clone(), v.clone());
    }
    Ok(Metadata(map))
}

// --- Saved configs (ArcSwap snapshot, ADR-021 §3) -----------------------------

pub type PolicyRegistry = std::collections::HashMap<String, Arc<RoutingConfig>>;
pub type SharedPolicyRegistry = Arc<ArcSwap<PolicyRegistry>>;

pub fn new_shared_registry(reg: PolicyRegistry) -> SharedPolicyRegistry {
    Arc::new(ArcSwap::from_pointee(reg))
}

#[derive(Debug)]
pub enum RegistryLoadError {
    Read {
        path: String,
        source: std::io::Error,
    },
    Parse {
        path: String,
        source: serde_json::Error,
    },
    Config {
        id: String,
        source: ConfigError,
    },
    BadShape {
        path: String,
        reason: String,
    },
}

impl std::fmt::Display for RegistryLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryLoadError::Read { path, source } => {
                write!(f, "cannot read routing policies '{path}': {source}")
            }
            RegistryLoadError::Parse { path, source } => {
                write!(f, "cannot parse routing policies '{path}': {source}")
            }
            RegistryLoadError::Config { id, source } => {
                write!(f, "saved config '{id}' is invalid: {source}")
            }
            RegistryLoadError::BadShape { path, reason } => {
                write!(f, "routing policies '{path}' has the wrong shape: {reason}")
            }
        }
    }
}

impl std::error::Error for RegistryLoadError {}

/// Load + validate `configs/routing-policies.json` at startup. Shape:
/// `{ "configs": [ { "id": "cfg_x", "routing": { ... } }, ... ] }`.
/// An invalid file refuses startup (fail-closed, ADR-021 §3) — the caller exits.
pub fn load_registry_from_file(path: &str) -> Result<PolicyRegistry, RegistryLoadError> {
    let content = std::fs::read_to_string(path).map_err(|source| RegistryLoadError::Read {
        path: path.to_string(),
        source,
    })?;
    let value: Value =
        serde_json::from_str(&content).map_err(|source| RegistryLoadError::Parse {
            path: path.to_string(),
            source,
        })?;
    let configs = value
        .get("configs")
        .and_then(Value::as_array)
        .ok_or_else(|| RegistryLoadError::BadShape {
            path: path.to_string(),
            reason: "expected a top-level 'configs' array".into(),
        })?;
    let mut registry = PolicyRegistry::new();
    for (i, entry) in configs.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| s.starts_with("cfg_"))
            .ok_or_else(|| RegistryLoadError::BadShape {
                path: path.to_string(),
                reason: format!("configs[{i}].id must be a 'cfg_'-prefixed string"),
            })?
            .to_string();
        let routing = entry
            .get("routing")
            .ok_or_else(|| RegistryLoadError::BadShape {
                path: path.to_string(),
                reason: format!("configs[{i}] is missing a 'routing' object"),
            })?;
        let cfg =
            RoutingConfig::parse(routing, ConfigRef::Saved(id.clone())).map_err(|source| {
                RegistryLoadError::Config {
                    id: id.clone(),
                    source,
                }
            })?;
        let arc_cfg = Arc::new(cfg);
        // ADR-086: an optional public `combo` name makes this config addressable
        // via the OpenAI `model` field and surfaces it in `/v1/models`. Stored
        // under a reserved `combo:` namespace so a raw `cfg_` id can NEVER be
        // addressed via the `model` field (that would bypass the RoutingPolicy
        // gate — an escalation). A combo must be self-contained: every target pins
        // a concrete downstream model, so the combo name never reaches a provider.
        if let Some(combo_raw) = entry.get("combo").and_then(Value::as_str) {
            let combo = combo_raw.trim();
            if combo.is_empty() || combo.starts_with("cfg_") || combo.starts_with("combo:") {
                return Err(RegistryLoadError::BadShape {
                    path: path.to_string(),
                    reason: format!(
                        "configs[{i}].combo '{combo}' must be non-empty and not 'cfg_'/'combo:'-prefixed"
                    ),
                });
            }
            if arc_cfg.target_models().iter().any(Option::is_none) {
                return Err(RegistryLoadError::BadShape {
                    path: path.to_string(),
                    reason: format!(
                        "combo '{combo}' has a target with no pinned model (ADR-086 self-contained rule)"
                    ),
                });
            }
            let key = format!("combo:{combo}");
            if registry.contains_key(&key) {
                return Err(RegistryLoadError::BadShape {
                    path: path.to_string(),
                    reason: format!("duplicate combo name '{combo}'"),
                });
            }
            registry.insert(key, Arc::clone(&arc_cfg));
        }
        registry.insert(id, arc_cfg);
    }
    Ok(registry)
}

/// The registry key namespace a `model`-field combo resolves under (ADR-086).
/// A client `model: "<name>"` resolves against `combo:<name>` — never a raw
/// `cfg_<id>` — so a saved routing config can never be addressed (and its
/// RoutingPolicy gate bypassed) via the `model` field.
pub fn combo_registry_key(name: &str) -> String {
    format!("combo:{name}")
}

/// Iterate the combo names registered in a snapshot (the `combo:`-namespaced
/// keys), for surfacing in `/v1/models` (ADR-086). Additive: an empty registry
/// yields nothing, so the models catalog is byte-identical when no combos exist.
pub fn combo_names(registry: &PolicyRegistry) -> impl Iterator<Item = &str> {
    registry.keys().filter_map(|k| k.strip_prefix("combo:"))
}

/// Resolve the `x-routeplane-config` header into an executable config (ADR-021 §6).
/// `{`-prefixed → inline envelope JSON whose `routing` section is parsed; `cfg_`
/// prefixed → saved reference; anything else → 400. The header may carry sibling
/// `guardrails` sections (G2.6); the `cache` directive rides INSIDE the `routing`
/// object (PRD-006 §4.8 / PRD-007 §5.1). Absence of a `routing` section → None
/// (legacy routing, byte-identical).
pub fn resolve_routing_config(
    header: Option<&str>,
    registry: &SharedPolicyRegistry,
) -> Result<Option<Arc<RoutingConfig>>, ConfigError> {
    let Some(raw) = header else {
        return Ok(None);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    if raw.len() > MAX_CONFIG_BYTES {
        return Err(ConfigError::too_large(
            "/",
            format!("x-routeplane-config exceeds {MAX_CONFIG_BYTES} bytes"),
        ));
    }
    if raw.starts_with('{') {
        let value: Value = serde_json::from_str(raw).map_err(|e| {
            ConfigError::invalid("/", format!("x-routeplane-config is not valid JSON: {e}"))
        })?;
        // Envelope-level deny-unknown (security review note: PRD-006's §4.1
        // example is flat — a flat config would otherwise be SILENTLY ignored,
        // dropping the client's retry/timeout policy; fail loud instead).
        if let Some(obj) = value.as_object() {
            for key in obj.keys() {
                if !matches!(key.as_str(), "routing" | "guardrails") {
                    return Err(ConfigError::invalid(
                        format!("/{key}"),
                        format!(
                            "unknown x-routeplane-config section '{key}' (known: routing, guardrails)"
                        ),
                    ));
                }
            }
        }
        match value.get("routing") {
            None => Ok(None),
            Some(routing) => Ok(Some(Arc::new(RoutingConfig::parse(
                routing,
                ConfigRef::Inline,
            )?))),
        }
    } else if raw.starts_with("cfg_") {
        // The cfg_<id>@<version> grammar is reserved until G3.3.
        if raw.contains('@') {
            return Err(ConfigError::not_found(raw));
        }
        let snapshot = registry.load();
        match snapshot.get(raw) {
            Some(cfg) => Ok(Some(cfg.clone())),
            None => Err(ConfigError::not_found(raw)),
        }
    } else {
        Err(ConfigError::invalid(
            "/",
            "x-routeplane-config must be inline JSON ('{...}') or a saved-config id ('cfg_...')",
        ))
    }
}

// --- Small typed-parse helpers ------------------------------------------------

fn parse_u32(value: &Value, ptr: &str) -> Result<u32, ConfigError> {
    value
        .as_u64()
        .filter(|n| *n <= u32::MAX as u64)
        .map(|n| n as u32)
        .ok_or_else(|| ConfigError::invalid(ptr, "must be a non-negative integer"))
}

fn parse_positive_u64(value: &Value, ptr: &str) -> Result<u64, ConfigError> {
    value
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| ConfigError::invalid(ptr, "must be a positive integer"))
}

/// Reject any key not in `allowed` (deny_unknown_fields, ADR-021 §1).
fn reject_unknown(
    obj: &Map<String, Value>,
    allowed: &[&str],
    base_ptr: &str,
) -> Result<(), ConfigError> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(ConfigError::invalid(
                format!("{base_ptr}/{key}"),
                format!("unknown field '{key}'"),
            ));
        }
    }
    Ok(())
}

// =============================== tests ========================================

#[cfg(test)]
mod tests {
    use super::*;
    use routeplane_types::{ChatCompletionRequest, Message};

    fn req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "hi".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            }],
            temperature: None,
            top_p: None,
            stream: None,
            max_tokens: None,
            stop: None,
            n: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            ..Default::default()
        }
    }

    fn parse(json: &str) -> Result<RoutingConfig, ConfigError> {
        let v: Value = serde_json::from_str(json).unwrap();
        RoutingConfig::parse(&v, ConfigRef::Inline)
    }

    #[test]
    fn minimal_priority_chain_parses_and_flattens_in_order() {
        let cfg = parse(r#"{"strategy":{"mode":"priority"},"targets":[{"provider":"openai"},{"provider":"anthropic"}]}"#).unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert_eq!(plan.strategy, PolicyStrategy::Priority);
        let names: Vec<_> = plan.targets.iter().map(|t| t.provider.as_str()).collect();
        assert_eq!(names, vec!["openai", "anthropic"]);
        // No retry by default.
        assert_eq!(plan.targets[0].retry.attempts, 0);
    }

    #[test]
    fn strategy_accepts_bare_string() {
        let cfg = parse(r#"{"strategy":"cost","targets":[{"provider":"openai"}]}"#).unwrap();
        assert_eq!(cfg.strategy, PolicyStrategy::Cost);
    }

    #[test]
    fn unknown_top_level_key_is_rejected() {
        let err = parse(r#"{"targets":[{"provider":"openai"}],"bogus":1}"#).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::InvalidConfig);
        assert_eq!(err.param, "/bogus");
    }

    #[test]
    fn retry_on_status_401_is_rejected_with_pointer() {
        let err = parse(
            r#"{"targets":[{"provider":"openai","retry":{"attempts":1,"on_status":[401]}}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::InvalidConfig);
        assert_eq!(err.param, "/targets/0/retry/on_status/0");
        assert!(err.message.contains("401"));
    }

    #[test]
    fn retry_attempts_over_five_is_rejected() {
        let err =
            parse(r#"{"targets":[{"provider":"openai","retry":{"attempts":6}}]}"#).unwrap_err();
        assert_eq!(err.param, "/targets/0/retry/attempts");
    }

    #[test]
    fn per_target_retry_overrides_config_default() {
        let cfg = parse(
            r#"{"retry":{"attempts":1,"on_status":[429]},
                "targets":[
                  {"provider":"openai","retry":{"attempts":3,"on_status":[503]}},
                  {"provider":"anthropic"}
                ]}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        // openai uses its override; anthropic inherits the config-level default.
        assert_eq!(plan.targets[0].retry.attempts, 3);
        assert!(plan.targets[0].retry.on_status.contains(&503));
        assert_eq!(plan.targets[1].retry.attempts, 1);
        assert!(plan.targets[1].retry.on_status.contains(&429));
    }

    #[test]
    fn param_override_and_drop_collision_is_rejected() {
        let err = parse(
            r#"{"targets":[{"provider":"openai","params":{"override":{"top_p":0.2},"drop":["top_p"]}}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/targets/0/params/drop");
    }

    #[test]
    fn non_allowlisted_param_is_rejected() {
        let err =
            parse(r#"{"targets":[{"provider":"openai","params":{"override":{"messages":[]}}}]}"#)
                .unwrap_err();
        assert_eq!(err.param, "/targets/0/params/override/messages");
    }

    #[test]
    fn param_shaping_default_override_drop_apply() {
        let cfg = parse(
            r#"{"targets":[{"provider":"openai","params":{
                "default":{"max_tokens":1024},
                "override":{"temperature":0.2},
                "drop":["top_p"]
            }}]}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        let mut base = req();
        base.temperature = Some(0.9);
        base.top_p = Some(0.5);
        let shaped = plan.targets[0].params.apply(base);
        assert_eq!(shaped.max_tokens, Some(1024)); // default filled
        assert_eq!(shaped.temperature, Some(0.2)); // override forced
        assert_eq!(shaped.top_p, None); // dropped
    }

    #[test]
    fn max_completion_tokens_is_shapeable_and_condition_readable() {
        // The reasoning-model cap must be shapeable exactly like `max_tokens`
        // (else a client-sent max_completion_tokens outflanks an operator cost
        // cap) and readable by `when` conditions.
        let cfg = parse(
            r#"{"targets":[{"provider":"openai","params":{
                "override":{"max_completion_tokens":100},
                "drop":["max_tokens"]
            }}]}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        let mut base = req();
        base.max_tokens = Some(4096);
        base.max_completion_tokens = Some(100_000);
        let shaped = plan.targets[0].params.apply(base);
        assert_eq!(shaped.max_completion_tokens, Some(100)); // override forced
        assert_eq!(shaped.max_tokens, None); // dropped

        // Condition path: params.max_completion_tokens is addressable and
        // evaluates (mirrors the params.max_tokens condition contract).
        let cfg = parse(
            r#"{"strategy":"conditional",
                "targets":[{"name":"a","provider":"anthropic"},{"name":"b","provider":"openai"}],
                "conditions":[{"when":{"params.max_completion_tokens":{"gt":1000}},"target":"a"}],
                "default_target":"b"}"#,
        )
        .unwrap();
        let mut big = req();
        big.max_completion_tokens = Some(100_000);
        let plan = cfg.evaluate(&Metadata::default(), &big, &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "anthropic");
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "openai"); // absent → default branch
    }

    #[test]
    fn default_does_not_overwrite_present_value() {
        let cfg = parse(
            r#"{"targets":[{"provider":"openai","params":{"default":{"max_tokens":1024}}}]}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        let mut base = req();
        base.max_tokens = Some(50);
        let shaped = plan.targets[0].params.apply(base);
        assert_eq!(shaped.max_tokens, Some(50)); // default did not clobber
    }

    #[test]
    fn nesting_depth_over_two_is_rejected() {
        let err = parse(
            r#"{"targets":[{"name":"p","strategy":"priority","targets":[
                {"name":"q","strategy":"priority","targets":[{"provider":"openai"}]}
            ]}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::InvalidConfig);
    }

    #[test]
    fn nested_pool_flattens_in_order() {
        let cfg = parse(
            r#"{"strategy":"priority","targets":[
                {"provider":"openai"},
                {"name":"pool","strategy":"priority","targets":[
                    {"provider":"anthropic"},{"provider":"gemini"}
                ]}
            ]}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        let names: Vec<_> = plan.targets.iter().map(|t| t.provider.as_str()).collect();
        assert_eq!(names, vec!["openai", "anthropic", "gemini"]);
    }

    #[test]
    fn conditional_first_match_wins_then_default() {
        let cfg = parse(
            r#"{"strategy":{"mode":"conditional"},
                "targets":[
                  {"name":"primary","provider":"openai"},
                  {"name":"fallback","provider":"anthropic"}
                ],
                "conditions":[
                  {"when":{"metadata.team":{"eq":"fraud-ml"}},"target":"primary"}
                ],
                "default_target":"fallback"}"#,
        )
        .unwrap();

        // metadata matches → primary.
        let meta = parse_metadata(Some(r#"{"team":"fraud-ml"}"#)).unwrap();
        let plan = cfg.evaluate(&meta, &req(), &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "openai");
        assert_eq!(plan.matched_label, "0");

        // absent metadata → default (fallback).
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "anthropic");
        assert_eq!(plan.matched_label, "default");
    }

    #[test]
    fn conditional_requires_default_and_names() {
        // Missing default_target.
        let err = parse(
            r#"{"strategy":"conditional","targets":[{"name":"a","provider":"openai"}],
                "conditions":[{"when":{"metadata.x":{"eq":"y"}},"target":"a"}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/default_target");

        // Dangling condition target.
        let err = parse(
            r#"{"strategy":"conditional","targets":[{"name":"a","provider":"openai"}],
                "conditions":[{"when":{"metadata.x":{"eq":"y"}},"target":"nope"}],
                "default_target":"a"}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/conditions/0/target");
    }

    #[test]
    fn condition_param_in_operator() {
        let cfg = parse(
            r#"{"strategy":"conditional",
                "targets":[{"name":"a","provider":"openai"},{"name":"b","provider":"anthropic"}],
                "conditions":[{"when":{"params.model":{"in":["gpt-4o","gpt-4o-mini"]}},"target":"a"}],
                "default_target":"b"}"#,
        )
        .unwrap();
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "openai");
    }

    #[test]
    fn static_type_mismatch_in_ordering_op_is_rejected() {
        let err = parse(
            r#"{"strategy":"conditional",
                "targets":[{"name":"a","provider":"openai"}],
                "conditions":[{"when":{"params.max_tokens":{"gt":"big"}},"target":"a"}],
                "default_target":"a"}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/conditions/0/when/params.max_tokens/gt");
    }

    #[test]
    fn runtime_type_mismatch_evaluates_false_not_error() {
        // gt against a string metadata value → false → default branch.
        let cfg = parse(
            r#"{"strategy":"conditional",
                "targets":[{"name":"a","provider":"openai"},{"name":"b","provider":"anthropic"}],
                "conditions":[{"when":{"metadata.score":{"gt":10}},"target":"a"}],
                "default_target":"b"}"#,
        )
        .unwrap();
        let meta = parse_metadata(Some(r#"{"score":"high"}"#)).unwrap();
        let plan = cfg.evaluate(&meta, &req(), &mut Rng::seeded(1));
        assert_eq!(plan.targets[0].provider, "anthropic");
    }

    // --- cache directive (PRD-007 §5.1 — replaces the §4.8 inert posture) -----

    #[test]
    fn cache_directive_parses_with_documented_defaults() {
        let cfg =
            parse(r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple"}}"#).unwrap();
        let d = cfg.cache().expect("cache directive present");
        assert_eq!(d.mode, CacheMode::Simple);
        assert_eq!(d.ttl_seconds, CACHE_TTL_DEFAULT_SECONDS);
        assert_eq!(d.namespace, "default");
        assert!(!d.force_refresh);
        assert!(d.similarity_threshold.is_none());
        assert_eq!(d.max_response_bytes, CACHE_MAX_RESPONSE_BYTES_CAP as usize);
        // No cache key → no directive (FR-2).
        let plain = parse(r#"{"targets":[{"provider":"openai"}]}"#).unwrap();
        assert!(plain.cache().is_none());
    }

    #[test]
    fn cache_mode_is_required_and_object_shape_enforced() {
        // PRD-007 FR-1 makes `mode` required — the PRD-006 §4.8 accepted-but-
        // inert `{}` posture is explicitly replaced (PRD-007 §3).
        let err = parse(r#"{"targets":[{"provider":"openai"}],"cache":{}}"#).unwrap_err();
        assert_eq!(err.param, "/cache/mode");
        let err = parse(r#"{"targets":[{"provider":"openai"}],"cache":"simple"}"#).unwrap_err();
        assert_eq!(err.param, "/cache");
        let err =
            parse(r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"turbo"}}"#).unwrap_err();
        assert_eq!(err.param, "/cache/mode");
    }

    #[test]
    fn cache_bounds_are_fail_loud_with_field_precise_pointers() {
        // ttl below floor / above ceiling (FR-3: never silently clamped).
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","ttl_seconds":5}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/ttl_seconds");
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","ttl_seconds":90000}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/ttl_seconds");
        // namespace charset.
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","namespace":"Bad NS"}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/namespace");
        // max_response_bytes may only LOWER the platform cap.
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","max_response_bytes":999999}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/max_response_bytes");
        // unknown field.
        let err =
            parse(r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","bogus":1}}"#)
                .unwrap_err();
        assert_eq!(err.param, "/cache/bogus");
    }

    #[test]
    fn cache_similarity_threshold_rules() {
        // Only legal with semantic mode (FR-1).
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","similarity_threshold":0.95}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/similarity_threshold");
        // Out of [0.80, 0.999] (FR-3).
        let err = parse(
            r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"semantic","similarity_threshold":0.5}}"#,
        )
        .unwrap_err();
        assert_eq!(err.param, "/cache/similarity_threshold");
        // Semantic defaults the placeholder threshold (FR-12: placeholder, not
        // an empirical claim).
        let cfg =
            parse(r#"{"targets":[{"provider":"openai"}],"cache":{"mode":"semantic"}}"#).unwrap();
        let d = cfg.cache().unwrap();
        assert_eq!(d.mode, CacheMode::Semantic);
        assert_eq!(d.similarity_threshold, Some(CACHE_SIMILARITY_DEFAULT));
    }

    // --- backoff math (injectable Rng, deterministic) -------------------------

    #[test]
    fn backoff_no_jitter_is_exponential_capped() {
        let b = Backoff {
            initial_ms: 100,
            max_ms: 500,
            jitter: false,
        };
        let mut rng = Rng::seeded(1);
        assert_eq!(b.delay(0, &mut rng), Duration::from_millis(100));
        assert_eq!(b.delay(1, &mut rng), Duration::from_millis(200));
        assert_eq!(b.delay(2, &mut rng), Duration::from_millis(400));
        assert_eq!(b.delay(3, &mut rng), Duration::from_millis(500)); // capped
    }

    #[test]
    fn backoff_full_jitter_within_bounds_and_deterministic() {
        let b = Backoff {
            initial_ms: 100,
            max_ms: 1000,
            jitter: true,
        };
        let mut a = Rng::seeded(42);
        let mut c = Rng::seeded(42);
        for n in 0..4 {
            let da = b.delay(n, &mut a);
            let dc = b.delay(n, &mut c);
            assert_eq!(da, dc); // same seed → same delay
            let base = (100u64 << n).min(1000);
            assert!(da.as_millis() as u64 <= base);
        }
    }

    // --- hedge directive (ADR-057) -------------------------------------------

    #[test]
    fn hedge_absent_is_none_byte_identical() {
        let cfg = parse(r#"{"targets":[{"provider":"openai"},{"provider":"anthropic"}]}"#).unwrap();
        assert!(cfg.hedge().is_none());
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert!(plan.hedge.is_none());
    }

    #[test]
    fn hedge_parses_with_defaults_and_flows_to_plan() {
        let cfg = parse(
            r#"{"targets":[{"provider":"openai"},{"provider":"anthropic"}],"hedge":{"delay_ms":50}}"#,
        )
        .unwrap();
        let h = cfg.hedge().expect("hedge present");
        assert_eq!(h.delay, Duration::from_millis(50));
        assert_eq!(h.max_extra, 1); // default
        let plan = cfg.evaluate(&Metadata::default(), &req(), &mut Rng::seeded(1));
        assert_eq!(plan.hedge, Some(h));
    }

    #[test]
    fn hedge_max_is_honored_and_capped_fail_closed() {
        let cfg = parse(
            r#"{"targets":[{"provider":"openai"},{"provider":"anthropic"}],"hedge":{"delay_ms":10,"max":2}}"#,
        )
        .unwrap();
        assert_eq!(cfg.hedge().unwrap().max_extra, 2);
        // Above the hard cap → rejected, never clamped (fail-closed).
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"delay_ms":10,"max":3}}"#)
            .unwrap_err();
        assert_eq!(err.param, "/hedge/max");
        // max:0 is a no-op config → rejected (omit `hedge` instead).
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"delay_ms":10,"max":0}}"#)
            .unwrap_err();
        assert_eq!(err.param, "/hedge/max");
    }

    #[test]
    fn hedge_delay_bounds_and_required_field_are_fail_loud() {
        // delay_ms required.
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"max":1}}"#).unwrap_err();
        assert_eq!(err.param, "/hedge/delay_ms");
        // delay_ms above ceiling.
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"delay_ms":120000}}"#)
            .unwrap_err();
        assert_eq!(err.param, "/hedge/delay_ms");
        // delay_ms zero (not a positive integer).
        let err =
            parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"delay_ms":0}}"#).unwrap_err();
        assert_eq!(err.param, "/hedge/delay_ms");
        // unknown field.
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":{"delay_ms":10,"bogus":1}}"#)
            .unwrap_err();
        assert_eq!(err.param, "/hedge/bogus");
        // wrong shape.
        let err = parse(r#"{"targets":[{"provider":"openai"}],"hedge":"fast"}"#).unwrap_err();
        assert_eq!(err.param, "/hedge");
    }

    #[test]
    fn metadata_rejects_non_scalar_and_too_large() {
        assert!(parse_metadata(Some(r#"{"a":{"nested":1}}"#)).is_err());
        let big = format!("{{\"a\":\"{}\"}}", "x".repeat(MAX_METADATA_BYTES));
        let err = parse_metadata(Some(&big)).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::ConfigTooLarge);
    }

    #[test]
    fn saved_registry_load_and_resolve() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_policies_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"configs":[{"id":"cfg_fast","routing":{"strategy":"cost","targets":[{"provider":"gemini"}]}}]}"#,
        )
        .unwrap();
        let reg = load_registry_from_file(path.to_str().unwrap()).unwrap();
        let shared = new_shared_registry(reg);
        let resolved = resolve_routing_config(Some("cfg_fast"), &shared)
            .unwrap()
            .unwrap();
        assert_eq!(resolved.strategy, PolicyStrategy::Cost);
        // Unknown id → config_not_found.
        let err = resolve_routing_config(Some("cfg_nope"), &shared).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::ConfigNotFound);
        // Reserved @version grammar.
        let err = resolve_routing_config(Some("cfg_fast@2"), &shared).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::ConfigNotFound);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn resolve_discriminates_inline_saved_and_garbage() {
        let shared = new_shared_registry(PolicyRegistry::new());
        // Envelope without a routing section → None (legacy path).
        assert!(
            resolve_routing_config(Some(r#"{"guardrails":{}}"#), &shared)
                .unwrap()
                .is_none()
        );
        // Garbage discriminator.
        let err = resolve_routing_config(Some("garbage"), &shared).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::InvalidConfig);
        // Oversize.
        let big = format!(
            "{{\"routing\":{{\"x\":\"{}\"}}}}",
            "y".repeat(MAX_CONFIG_BYTES)
        );
        let err = resolve_routing_config(Some(&big), &shared).unwrap_err();
        assert_eq!(err.code, ConfigErrorCode::ConfigTooLarge);
    }

    // --- ADR-086: combo-as-model-id ------------------------------------------

    fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("rp_combo_{}_{}.json", name, std::process::id()));
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn combo_loads_addressable_by_name_and_enumerates_targets() {
        let path = write_tmp(
            "ok",
            r#"{"configs":[{"id":"cfg_fast","combo":"fast","routing":{"strategy":"cost","targets":[
                {"provider":"openai","params":{"override":{"model":"gpt-4o"}}},
                {"provider":"anthropic","params":{"override":{"model":"claude-3-5-sonnet"}}}
            ]}}]}"#,
        );
        let reg = load_registry_from_file(path.to_str().unwrap()).unwrap();
        // Addressable under the `combo:` namespace AND still by its cfg_ id.
        assert!(reg.contains_key("combo:fast"));
        assert!(reg.contains_key("cfg_fast"));
        // Escalation guard: a raw cfg_ id is NOT addressable via the combo namespace.
        assert!(!reg.contains_key("combo:cfg_fast"));
        // combo_names surfaces exactly the public name (for /v1/models).
        assert_eq!(combo_names(&reg).collect::<Vec<_>>(), vec!["fast"]);
        // target_models enumerates the pinned downstream models (order preserved).
        let models: Vec<String> = reg
            .get("combo:fast")
            .unwrap()
            .target_models()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            models,
            vec!["gpt-4o".to_string(), "claude-3-5-sonnet".to_string()]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn combo_must_be_self_contained() {
        // A target with no pinned model ⇒ the combo name could reach a provider ⇒ reject.
        let path = write_tmp(
            "unpinned",
            r#"{"configs":[{"id":"cfg_x","combo":"loose","routing":{"targets":[{"provider":"openai"}]}}]}"#,
        );
        let err = load_registry_from_file(path.to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, RegistryLoadError::BadShape { .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn param_shaping_default_model_is_not_a_pin() {
        // A `default.model` fills-if-absent (`apply` uses entry().or_insert), but a
        // combo request always arrives with `model` = the combo name, so the default
        // never applies and the combo name would leak. Only an `override` counts as a
        // pin (ADR-086 §A4). Isolate the exact defect at the `ParamShaping` level.
        let default_only = ParamShaping {
            default: serde_json::json!({ "model": "gpt-4o" })
                .as_object()
                .unwrap()
                .clone(),
            overrides: Map::new(),
            drop: Vec::new(),
        };
        assert_eq!(default_only.pinned_model(), None);

        let override_pin = ParamShaping {
            default: Map::new(),
            overrides: serde_json::json!({ "model": "gpt-4o" })
                .as_object()
                .unwrap()
                .clone(),
            drop: Vec::new(),
        };
        assert_eq!(override_pin.pinned_model(), Some("gpt-4o"));
    }

    #[test]
    fn combo_default_pinned_model_is_rejected() {
        // Regression (internal bug sweep): a combo target that pins its model ONLY
        // via `params.default.model` used to pass the self-contained load check
        // (pinned_model accepted `default`), yet `apply` never overwrites the combo
        // name already sitting in `model` — so the combo name "fast" would reach the
        // provider (upstream model_not_found), the exact leak ADR-086 §A4 forbids.
        // It must now fail startup fail-closed, identical to an unpinned target.
        let path = write_tmp(
            "default_pin",
            r#"{"configs":[{"id":"cfg_fast","combo":"fast","routing":{"targets":[
                {"provider":"openai","params":{"default":{"model":"gpt-4o"}}}
            ]}}]}"#,
        );
        let err = load_registry_from_file(path.to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, RegistryLoadError::BadShape { .. }),
            "a default-only pinned combo target must be rejected, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn combo_name_rejects_reserved_prefixes_and_duplicates() {
        for bad in [r#""cfg_shadow""#, r#""combo:x""#, r#""""#] {
            let path = write_tmp(
                "reserved",
                &format!(
                    r#"{{"configs":[{{"id":"cfg_a","combo":{bad},"routing":{{"targets":[{{"provider":"openai","params":{{"override":{{"model":"gpt-4o"}}}}}}]}}}}]}}"#
                ),
            );
            assert!(
                matches!(
                    load_registry_from_file(path.to_str().unwrap()),
                    Err(RegistryLoadError::BadShape { .. })
                ),
                "combo name {bad} should be rejected"
            );
            let _ = std::fs::remove_file(&path);
        }
        // Duplicate combo names across two entries ⇒ reject.
        let path = write_tmp(
            "dup",
            r#"{"configs":[
                {"id":"cfg_a","combo":"dup","routing":{"targets":[{"provider":"openai","params":{"override":{"model":"gpt-4o"}}}]}},
                {"id":"cfg_b","combo":"dup","routing":{"targets":[{"provider":"anthropic","params":{"override":{"model":"claude-3-5-sonnet"}}}]}}
            ]}"#,
        );
        assert!(matches!(
            load_registry_from_file(path.to_str().unwrap()),
            Err(RegistryLoadError::BadShape { .. })
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_combo_field_is_byte_identical_load() {
        // A config with no `combo` field loads exactly as before (no combo: keys).
        let path = write_tmp(
            "plain",
            r#"{"configs":[{"id":"cfg_plain","routing":{"targets":[{"provider":"gemini"}]}}]}"#,
        );
        let reg = load_registry_from_file(path.to_str().unwrap()).unwrap();
        assert!(reg.contains_key("cfg_plain"));
        assert_eq!(combo_names(&reg).count(), 0);
        let _ = std::fs::remove_file(&path);
    }
}
