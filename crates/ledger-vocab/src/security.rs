//! Security-event vocabulary for the sovereign audit ledger (R0.3).
//!
//! Routing-DECISION audit (see [`crate::entry`]) records who routed what, where.
//! A SECURITY event records who was *refused* and why: an auth failure, a
//! brute-force throttle trip, a guardrail deny, a residency rejection, a
//! rate-limit / budget breach, an MCP authorize/egress denial. Putting these
//! into the SAME hash-chained, signable, in-region evidence ([ADR-019]) makes
//! the security posture tamper-evident and offline-verifiable via
//! `routeplane-ledger`'s `verify_entries` — one integrity surface, not a second
//! log.
//!
//! ## Why this needs no schema change
//! Security events reuse the existing [`DecisionDraft`] skeleton plus `ext`
//! **unchanged** (the same move `routeplane-ledger`'s `governance` makes for
//! flag events), so the integrity core stays a single reviewed surface. They
//! form their own chain (`profile_id = "security"`), isolated from the decision
//! and governance chains — the DPDP artifact generator filters by profile and
//! never sees them.
//!
//! ## The no-raw-PII guarantee holds STRUCTURALLY (the binding R0.3 constraint)
//! A [`SecurityEvent`] carries ONLY:
//! - a coarse **category** ([`SecurityCategory`]) — a closed-vocabulary code,
//! - an **outcome** ([`SecurityOutcome`]) — `allow` / `deny` / `throttle`,
//! - an optional **count** (e.g. the post-increment auth-failure count, or the
//!   number of guardrail checks that fired) as an opaque `u64`,
//! - an optional **detail code** — a closed-vocabulary [`CodeStr`], never free
//!   text and never a matched value.
//!
//! It carries NO matched bytes, NO source IP (the IP is never echoed to clients
//! and is not placed in the ledger), and NO key material. Every field maps onto
//! a validated newtype (`Label` / `CodeStr` / `ExtValue::{Bool,Count,Code}`), so
//! raw PII / secrets are unrepresentable by construction — identical to the
//! decision and governance entries.

use std::collections::BTreeMap;

use crate::entry::{
    BoundedText, CodeStr, DecisionDraft, ExtKey, ExtValue, Label, LabelError, Outcome, UsageTotals,
};

/// The security chain's profile id — security events form their own
/// `(tenant, "security")` chain, never mixed with decision / governance chains.
pub const SECURITY_PROFILE_ID: &str = "security";
/// Security schema/profile version (independent of the decision profiles).
pub const SECURITY_PROFILE_VERSION: u32 = 1;
/// Sentinel `tenant_id` for an event not attributable to one tenant (e.g. an
/// auth failure where the key never resolved to a tenant).
pub const SECURITY_GLOBAL_SCOPE: &str = "_global";

/// The coarse category of a security event. Each maps to a stable, closed-vocab
/// snake_case identifier carried as a [`Label`] in `data_classes` (≤32 chars).
/// An exhaustive match means a new variant forces a reviewed code choice rather
/// than silently widening the vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityCategory {
    /// A request failed authentication (unknown/invalid key, non-active tenant,
    /// missing credential).
    AuthFailure,
    /// The auth-failure brute-force limiter tripped for a source.
    AuthThrottle,
    /// An inline guardrail check denied the request (input or output hook).
    GuardrailDeny,
    /// Sovereign residency refused the request (no resident provider for the
    /// required region).
    ResidencyBlock,
    /// A rate-limit threshold was breached (requests or tokens).
    RateLimit,
    /// A budget threshold was breached (cost or tokens).
    BudgetBreach,
    /// An MCP tool-call authorization was denied.
    McpAuthorizeDeny,
    /// A NON-blocking behavioral-anomaly FLAG on an *allowed* MCP tool call
    /// (Ring-2 enrichment, [ADR-016]/[ADR-055]): the call deviated from the
    /// agent's learned baseline but was permitted to proceed. Distinct from
    /// [`SecurityCategory::McpAuthorizeDeny`] so an allowed-but-flagged call is
    /// never bucketed with genuine authorize denials on the dashboard / SIEM
    /// export (it is recorded with an `allow` outcome, not `deny`).
    McpAnomalyFlag,
    /// An MCP egress firewall denied an outbound call.
    McpEgressDeny,
    /// An agent exceeded its configured per-agent MCP tool-call quota
    /// (agent-governance rate ceiling, [ADR-016]/[ADR-017]) — the call was
    /// throttled at the enforcement point (429-style, fail-closed).
    McpQuotaExceeded,
    /// An MCP tool RESULT exceeded the configured maximum size and was withheld
    /// before it could re-enter the model's context ([ADR-016]/[ADR-018],
    /// agent-governance). An oversized result is a cost-DoS + context-stuffing /
    /// indirect-injection vector; the cap fails closed (the result is denied,
    /// never scanned, never reflected). The detail code carries only a coarse
    /// size bucket — never the oversized bytes.
    McpResultTooLarge,
    /// The model's OUTPUT leaked (a verbatim span of) the request's system prompt
    /// — OWASP LLM07 system-prompt leakage detected on the after-response path.
    SystemPromptLeak,
    /// The model emitted an OpenAI-style function/tool call whose name is not
    /// permitted by the tenant's `tool_policy` allow/deny governance (moat /
    /// agent-governance, [ADR-016]/[ADR-017]) — caught on the after-response path.
    /// Distinct from [`SecurityCategory::McpAuthorizeDeny`] (MCP-server tool
    /// calls); this governs the function calls in the chat-completions response.
    ToolCallDenied,
    /// The org compliance-framework gate ([ADR-035] §4) excluded a model: the
    /// requested model's `compliance_restrictions` intersect the tenant's
    /// `compliance_frameworks`. A `Deny` outcome is a `strict`-mode `403
    /// model_compliance_excluded` block (default-deny, before dispatch); an
    /// `Allow` outcome is a `warn`-mode flag (the request routed but the
    /// intersection is recorded for audit). The detail code carries a framework
    /// NAME (a §5 registry identifier — config, never user content).
    ComplianceBlock,
}

impl SecurityCategory {
    /// Stable, closed-vocab snake_case id (≤32 chars, `Label`-valid) — recorded
    /// in `data_classes` (it is the security analogue of governance's flag key).
    pub fn label(self) -> &'static str {
        match self {
            SecurityCategory::AuthFailure => "auth_failure",
            SecurityCategory::AuthThrottle => "auth_throttle",
            SecurityCategory::GuardrailDeny => "guardrail_deny",
            SecurityCategory::ResidencyBlock => "residency_block",
            SecurityCategory::RateLimit => "rate_limit",
            SecurityCategory::BudgetBreach => "budget_breach",
            SecurityCategory::McpAuthorizeDeny => "mcp_authorize_deny",
            SecurityCategory::McpAnomalyFlag => "mcp_anomaly_flag",
            SecurityCategory::McpEgressDeny => "mcp_egress_deny",
            SecurityCategory::McpQuotaExceeded => "mcp_quota_exceeded",
            SecurityCategory::McpResultTooLarge => "mcp_result_too_large",
            SecurityCategory::SystemPromptLeak => "system_prompt_leak",
            SecurityCategory::ToolCallDenied => "tool_call_denied",
            SecurityCategory::ComplianceBlock => "compliance_block",
        }
    }
}

/// The outcome of the security decision. Closed vocabulary; recorded as a
/// `CodeStr` in `ext["sec.outcome"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityOutcome {
    Allow,
    Deny,
    Throttle,
}

impl SecurityOutcome {
    /// Stable ≤16-char code (`CodeStr`-valid) for `ext["sec.outcome"]`.
    pub fn code(self) -> &'static str {
        match self {
            SecurityOutcome::Allow => "allow",
            SecurityOutcome::Deny => "deny",
            SecurityOutcome::Throttle => "throttle",
        }
    }
}

/// A normalized security event, decoupled from any wire/source shape. Built at a
/// decision point in the data plane and turned into a chainable
/// [`DecisionDraft`] by [`SecurityEvent::to_draft`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityEvent {
    /// Opaque correlation id (the request id, or a synthesized id for events
    /// that have no request context such as a pre-key-lookup auth failure).
    pub event_id: String,
    /// RFC3339 UTC timestamp; stored verbatim (deterministic hash).
    pub timestamp: String,
    /// What kind of security decision this records.
    pub category: SecurityCategory,
    /// The decision outcome.
    pub outcome: SecurityOutcome,
    /// An opaque count for the event — e.g. the post-increment auth-failure
    /// count for the source, or the number of blocking guardrail checks. NEVER
    /// a value derived from matched content. `None` ⇒ not applicable.
    pub count: Option<u64>,
    /// An optional closed-vocabulary detail code (e.g. the limit kind
    /// `"requests"`/`"tokens"`, or a guardrail hook `"before"`/`"after"`). NEVER
    /// free text, NEVER a matched value — sanitized into a `CodeStr`.
    pub detail_code: Option<String>,
    /// Affected tenant id, or `None` for a non-attributable event (→ `_global`).
    pub tenant_id: Option<String>,
}

impl SecurityEvent {
    /// Build the chainable [`DecisionDraft`] for this security event. Infallible
    /// for the closed-vocabulary category labels (they are all valid `Label`s),
    /// but returns a `Result` to mirror `routeplane-ledger`'s
    /// `governance::FlagChangeEvent` and stay fail-safe if the label vocabulary
    /// is ever widened incorrectly.
    pub fn to_draft(&self) -> Result<DecisionDraft, LabelError> {
        let category = Label::new(self.category.label())?;

        let mut ext: BTreeMap<ExtKey, ExtValue> = BTreeMap::new();
        ext.insert(
            ExtKey::known("sec.outcome"),
            ExtValue::Code(CodeStr::new(self.outcome.code())?),
        );
        if let Some(n) = self.count {
            ext.insert(ExtKey::known("sec.count"), ExtValue::Count(n));
        }
        if let Some(detail) = &self.detail_code {
            // Sanitized: a closed-vocab code is expected, but the lossy
            // constructor guarantees no PII-shaped residue can ever round-trip.
            ext.insert(
                ExtKey::known("sec.detail"),
                ExtValue::Code(CodeStr::sanitized(detail)),
            );
        }

        Ok(DecisionDraft {
            tenant_id: self
                .tenant_id
                .clone()
                .unwrap_or_else(|| SECURITY_GLOBAL_SCOPE.to_string()),
            request_id: self.event_id.clone(),
            timestamp: self.timestamp.clone(),
            provider: None,
            // Required skeleton field; a constant marks this as a non-routing entry.
            model: BoundedText::sanitized("security-event"),
            region: None,
            contains_regulated_data: false,
            data_classes: vec![category],
            profile_id: Label::known(SECURITY_PROFILE_ID),
            profile_version: SECURITY_PROFILE_VERSION,
            residency_required: false,
            required_regions: vec![],
            sovereign_routed: false,
            client_override_applied: false,
            // Map the security outcome onto the skeleton `Outcome`: a denied/
            // throttled security decision is recorded faithfully as a block;
            // an `allow` event (rare — reserved for "checked, permitted") as Ok.
            outcome: match self.outcome {
                SecurityOutcome::Allow => Outcome::Ok,
                SecurityOutcome::Deny | SecurityOutcome::Throttle => Outcome::ResidencyBlocked,
            },
            usage: UsageTotals::default(),
            classifier_version: BoundedText::sanitized("routeplane-security/1"),
            classifier_confidence: None,
            ext,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        category: SecurityCategory,
        outcome: SecurityOutcome,
        count: Option<u64>,
        detail: Option<&str>,
        tenant: Option<&str>,
    ) -> SecurityEvent {
        SecurityEvent {
            event_id: "req_abc".into(),
            timestamp: "2026-06-26T10:00:00+00:00".into(),
            category,
            outcome,
            count,
            detail_code: detail.map(str::to_string),
            tenant_id: tenant.map(str::to_string),
        }
    }

    #[test]
    fn every_category_label_is_a_valid_label() {
        // The whole no-raw-PII guarantee for the category rides on this: every
        // category must construct into a closed-vocab Label.
        for cat in [
            SecurityCategory::AuthFailure,
            SecurityCategory::AuthThrottle,
            SecurityCategory::GuardrailDeny,
            SecurityCategory::ResidencyBlock,
            SecurityCategory::RateLimit,
            SecurityCategory::BudgetBreach,
            SecurityCategory::McpAuthorizeDeny,
            SecurityCategory::McpAnomalyFlag,
            SecurityCategory::McpEgressDeny,
            SecurityCategory::McpQuotaExceeded,
            SecurityCategory::McpResultTooLarge,
            SecurityCategory::SystemPromptLeak,
            SecurityCategory::ToolCallDenied,
            SecurityCategory::ComplianceBlock,
        ] {
            assert!(Label::new(cat.label()).is_ok(), "{cat:?}");
        }
    }

    #[test]
    fn auth_failure_draft_carries_category_outcome_and_count() {
        let draft = event(
            SecurityCategory::AuthFailure,
            SecurityOutcome::Deny,
            Some(3),
            None,
            None,
        )
        .to_draft()
        .expect("valid security event");
        assert_eq!(draft.data_classes[0].as_str(), "auth_failure");
        assert_eq!(draft.profile_id.as_str(), SECURITY_PROFILE_ID);
        assert_eq!(draft.tenant_id, SECURITY_GLOBAL_SCOPE);
        assert_eq!(draft.outcome, Outcome::ResidencyBlocked);
        assert_eq!(
            draft.ext.get(&ExtKey::known("sec.outcome")),
            Some(&ExtValue::Code(CodeStr::new("deny").unwrap()))
        );
        assert_eq!(
            draft.ext.get(&ExtKey::known("sec.count")),
            Some(&ExtValue::Count(3))
        );
    }

    #[test]
    fn mcp_anomaly_flag_is_a_distinct_non_deny_category() {
        // Regression (routeplane bug-sweep): a NON-blocking anomaly FLAG on an
        // ALLOWED MCP tool call must NOT be recorded under the deny category
        // (McpAuthorizeDeny) — that inflated the authorize-deny bucket with calls
        // that were actually allowed. It has its own label AND, recorded with an
        // `Allow` outcome, maps to a NON-block skeleton outcome.
        assert_eq!(SecurityCategory::McpAnomalyFlag.label(), "mcp_anomaly_flag");
        assert_ne!(
            SecurityCategory::McpAnomalyFlag.label(),
            SecurityCategory::McpAuthorizeDeny.label()
        );
        let draft = event(
            SecurityCategory::McpAnomalyFlag,
            SecurityOutcome::Allow,
            None,
            Some("anomaly_flag"),
            Some("t_1"),
        )
        .to_draft()
        .expect("valid security event");
        assert_eq!(draft.data_classes[0].as_str(), "mcp_anomaly_flag");
        // Allow ⇒ Outcome::Ok, NOT a block — so dashboards/exports that bucket by
        // outcome do not count it as an enforcement denial.
        assert_eq!(draft.outcome, Outcome::Ok);
        assert_eq!(
            draft.ext.get(&ExtKey::known("sec.outcome")),
            Some(&ExtValue::Code(CodeStr::new("allow").unwrap()))
        );
    }

    #[test]
    fn no_raw_pii_survives_serialization() {
        // The `detail_code` is a CLOSED VOCABULARY at every call site; the
        // `CodeStr::sanitized` constructor is structural defense in depth. Even
        // when a caller offers PII-shaped detail, the serialized draft cannot
        // carry the `@`/whitespace/length that PII (email, Aadhaar, phone) needs:
        // the `[A-Za-z0-9_-]{1,16}` charset makes those residues unrepresentable.
        let draft = SecurityEvent {
            event_id: "req_1".into(),
            timestamp: "2026-06-26T10:00:00+00:00".into(),
            category: SecurityCategory::GuardrailDeny,
            outcome: SecurityOutcome::Deny,
            count: Some(2),
            detail_code: Some("a@b.com 4321 4321 4321 +91 98765 43210".into()),
            tenant_id: Some("t_1".into()),
        }
        .to_draft()
        .unwrap();
        let json = serde_json::to_string(&draft).unwrap();
        assert!(!json.contains('@'), "no email residue");
        assert!(!json.contains(' '), "no whitespace (PII-shaped) in codes");
        assert!(!json.contains("4321 4321"), "no spaced-digit residue");
        // And the detail code is bounded to the CodeStr 16-char cap (truncated).
        if let Some(ExtValue::Code(c)) = draft.ext.get(&ExtKey::known("sec.detail")) {
            assert!(c.as_str().len() <= CodeStr::MAX_LEN);
        } else {
            panic!("expected a sanitized detail code");
        }
    }
}
