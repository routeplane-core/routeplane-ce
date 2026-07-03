//! Community Edition slot types (PRD-047 / ADR-088 Phase 1).
//!
//! Compiled ONLY under `--no-default-features` (`cfg(not(feature =
//! "enterprise"))`). Each item here mirrors the *shape* of a moat-crate type the
//! shared wiring names, so the binary compiles with the five moat crates
//! (`routeplane-ledger`, `routeplane-telemetry`, `routeplane-export`,
//! `routeplane-semantic-cache`, `routeplane-mcp`) **absent from the dependency
//! graph entirely** — verified by the CI `cargo tree` absence gate.
//!
//! Design rules (the `DistributedLimiterHandle` precedent, `proxy.rs`):
//! - **Slot types are uninhabited** where the CE build can never hold a live
//!   handle (`LedgerHandle`, `TelemetryHandle` are empty enums — the `Some`
//!   arms are statically dead, `Option<T>` construction sites stay uniform).
//! - **Inert stand-ins** where the wiring calls methods (`ExportHandle` is the
//!   permanently-disabled handle; `SemanticCache` never hits and never stores).
//! - **Signatures mirror the real crates exactly** so call sites compile
//!   unchanged in both variants; behavior on the CE build is the moat's
//!   ship-dark default (handle `None` / export disabled / semantic inert) —
//!   byte-identical to an enterprise build with every moat env flag unset.
#![allow(dead_code)] // mirrors enterprise shapes; some fields exist only so statically-dead call sites compile

/// CE slot for `routeplane_ledger::LedgerHandle` — uninhabited: the CE build
/// can never construct one, so `AppState.ledger` is permanently `None` and the
/// `ledger_sink::record_*` no-op twins never see a handle.
#[derive(Clone)]
pub enum LedgerHandle {}

/// CE slot for `routeplane_telemetry::TelemetryHandle` — uninhabited: the CE
/// build can never construct one, so `AppState.telemetry` is permanently `None`
/// and the (compiled-out) durable-telemetry record block never runs.
#[derive(Clone)]
pub enum TelemetryHandle {}

// ---------------------------------------------------------------------------
// routeplane-export stand-ins (`use ... as export_api` seam in proxy.rs)
// ---------------------------------------------------------------------------

/// CE mirror of the `routeplane_export` surface the binary names. Imported as
/// `export_api` (the same alias the enterprise build gives the real crate), so
/// the emit/export funnel bodies compile unchanged; `is_enabled()` is
/// permanently `false`, making every guarded fan-out block statically dead at
/// runtime — exactly the enterprise ship-dark (no `RP_EXPORT_*`) behavior.
pub mod export_api {
    /// Opaque no-op event — [`usage_event`]/[`security_event`] construct it,
    /// [`ExportHandle::try_export`] drops it.
    pub struct ExportEvent;

    /// Label-only guardrail verdict placeholder (never serialized in CE).
    #[derive(Clone)]
    pub struct GuardrailVerdict;

    /// The permanently-disabled export handle (mirror of
    /// `routeplane_export::ExportHandle::disabled()`).
    #[derive(Clone)]
    pub struct ExportHandle;

    impl ExportHandle {
        /// The CE handle is always the disabled no-op.
        pub fn disabled() -> Self {
            Self
        }

        /// Never enabled on the CE build.
        pub fn is_enabled(&self) -> bool {
            false
        }

        /// Drop the event (all call sites are behind `is_enabled()`).
        pub fn try_export(&self, _event: ExportEvent) {}

        /// A disabled handle never attempts a send, so it never drops one.
        pub fn dropped_total(&self) -> u64 {
            0
        }
    }

    /// Mirror of `routeplane_export::guardrail_verdict` (label-only mapper).
    pub fn guardrail_verdict(
        _id: &str,
        _check_type: &str,
        _hook: &str,
        _action: &str,
        _verdict: &str,
    ) -> GuardrailVerdict {
        GuardrailVerdict
    }

    /// Mirror of `routeplane_export::usage_event`.
    #[allow(clippy::too_many_arguments)]
    pub fn usage_event(
        _timestamp: String,
        _success: bool,
        _error: Option<&str>,
        _virtual_key_name: &str,
        _provider: &str,
        _model: &str,
        _region: Option<&str>,
        _prompt_tokens: u32,
        _completion_tokens: u32,
        _total_tokens: u32,
        _guardrails: Option<&[GuardrailVerdict]>,
    ) -> ExportEvent {
        ExportEvent
    }

    /// Mirror of `routeplane_export::security_event`.
    pub fn security_event(
        _timestamp: String,
        _category: &str,
        _outcome_code: &str,
        _count: Option<u64>,
        _detail_code: Option<&str>,
        _tenant_id: Option<&str>,
    ) -> ExportEvent {
        ExportEvent
    }
}

// ---------------------------------------------------------------------------
// routeplane-semantic-cache stand-ins
// ---------------------------------------------------------------------------

/// CE mirror of `routeplane_semantic_cache::SemanticKey` (shape only — the CE
/// plan builder can never produce a `SemanticPlan::Active`, so no key is ever
/// hashed or stored).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticKey;

impl SemanticKey {
    /// Mirror of the real constructor; the components are ignored.
    pub fn new(
        _tenant_id: &str,
        _namespace: &str,
        _model: &str,
        _provider_chain: &[String],
    ) -> Self {
        Self
    }
}

/// CE mirror of `routeplane_semantic_cache::SemanticEntry` — field-for-field
/// identical so the (statically dead) write-behind insert literal compiles.
#[derive(Debug, Clone)]
pub struct SemanticEntry {
    pub embedding: Vec<f32>,
    pub body: bytes::Bytes,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub inserted_at_ms: u64,
    pub ttl_ms: u64,
}

/// CE mirror of `routeplane_semantic_cache::SemanticHit` (never produced —
/// [`SemanticCache::lookup_with_threshold`] always misses).
#[derive(Debug)]
pub struct SemanticHit {
    pub entry: std::sync::Arc<SemanticEntry>,
    pub similarity: f32,
}

/// Inert CE semantic cache: lookups always miss, inserts drop the entry. A
/// misconfigured `mode:"semantic"` directive on the CE build therefore degrades
/// to exact-only semantics (and the proxy additionally forces the semantic plan
/// off under `cfg(not(feature = "enterprise"))`, so no embedding call is ever
/// made).
pub struct SemanticCache {
    threshold: f32,
}

impl SemanticCache {
    /// Mirror of the real constructor (threshold clamped like the real cache;
    /// the capacity is ignored — nothing is ever stored).
    pub fn new(threshold: f32, _max_entries: usize) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Mirror of `SemanticCache::from_env` (same env knobs, inert result).
    pub fn from_env() -> Self {
        let threshold = std::env::var("ROUTEPLANE_SEMANTIC_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.95);
        Self::new(threshold, 0)
    }

    /// Always a miss on the CE build.
    pub fn lookup_with_threshold(
        &self,
        _key: &SemanticKey,
        _embedding: &[f32],
        _threshold: f32,
    ) -> Option<SemanticHit> {
        None
    }

    /// Drop the entry (statically-dead call site — the CE plan is never Active).
    pub fn insert(&self, _key: SemanticKey, _entry: SemanticEntry) {}

    /// The configured default similarity threshold (read at plan build).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Mirror of `routeplane_semantic_cache::request_text_for_embedding`. Only
/// reachable from the statically-dead `SemanticPlan::Active` path on the CE
/// build, so it returns an empty string rather than duplicating the real
/// canonicalization.
pub fn request_text_for_embedding(_req: &routeplane_types::ChatCompletionRequest) -> String {
    String::new()
}
