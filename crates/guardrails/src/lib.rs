//! Routeplane Guardrails — Community Edition core (PRD-047 / ADR-088).
//!
//! This crate holds ONLY the CE-safe surface of the guardrail system:
//!   * the **closed vocabulary** the always-on data-plane path names —
//!     [`Hook`], [`CheckAction`], [`Verdict`], [`CheckOutcome`],
//!     [`ConfigSource`], [`ParseError`] (observability records `CheckOutcome`
//!     UNCONDITIONALLY, so these must live here, in the always-present crate);
//!   * the deterministic **detector library** ([`detect`]) — basic PII/secret
//!     masking (`redact`), `scan_pii`/`scan_secrets`, `KeywordMatcher`, the
//!     Verhoeff/Luhn/mod-97 validators, invisible-unicode handling, and the
//!     pure detector primitives (`detect_injection`, `detect_system_prompt_leak`,
//!     `leak_span_bucket`); and
//!   * the pure JSON-schema-subset validator ([`schema`]).
//!
//! The **advanced threat-detection MOAT** (the declarative check ENGINE, the
//! off-path cheap-gates-expensive pipeline, ML/ONNX detectors, moderation,
//! webhook checks + vendor packs, and reversible tokenization) lives in the
//! sibling `routeplane-guardrails-advanced` crate, which depends on THIS crate
//! and re-exports every name here — the same leaf/moat split the ledger uses
//! (`routeplane-ledger` over `routeplane-ledger-vocab`). CE builds
//! (`--no-default-features`) ship this crate only; the advanced crate is absent
//! from the dependency graph entirely.

use serde::{Deserialize, Serialize};

pub mod detect;
pub mod schema;

/// The evaluation point a check is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hook {
    BeforeRequest,
    AfterRequest,
}

/// What a failed check does. `Deny` rejects the request (HTTP 446 at the
/// proxy); `Observe` records the verdict in the usage event and never blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckAction {
    /// Default: a declared constraint blocks on failure (explicit security
    /// posture; opt into `observe` to monitor without blocking).
    #[default]
    Deny,
    Observe,
}

/// A check's verdict. `Error` (e.g. webhook unreachable) is treated like `Fail`
/// for deny-action checks — security decisions fail CLOSED.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    Error,
}

/// One evaluated check result — serialized into the 446 error detail and into
/// the usage event (`UsageEvent.guardrails`). Never contains matched input
/// text, only metadata — a guardrail report must not become an exfil channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckOutcome {
    pub id: String,
    pub check_type: String,
    pub hook: Hook,
    pub action: CheckAction,
    pub verdict: Verdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl CheckOutcome {
    /// True when this outcome should block the request: a deny-action check
    /// that did not pass (Fail OR Error — fail-closed).
    pub fn is_blocking(&self) -> bool {
        self.action == CheckAction::Deny && self.verdict != Verdict::Pass
    }
}

/// Where a config came from. `Inline` (the `x-routeplane-config` header) is
/// REQUEST input and may not configure webhook checks — the webhook URL must
/// come from tenant config, or any caller could turn the gateway into an SSRF /
/// exfiltration proxy by pointing a webhook at an attacker URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    Tenant,
    Inline,
}

/// Config parse/compile error. Hand-rolled (no `thiserror` — same frugality
/// stance as `AuthLoadError`).
#[derive(Debug)]
pub struct ParseError(String);

impl ParseError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid guardrails config: {}", self.0)
    }
}

impl std::error::Error for ParseError {}
