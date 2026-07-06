//! The ledger entry vocabulary (PRD-001 FR-10/11/12, 001-A/E).
//!
//! ## The no-raw-PII guarantee is STRUCTURAL (FR-2, NFR-9, AC-7)
//! There is no field in [`DecisionDraft`] (or the chained `LedgerEntry` built
//! from it in `routeplane-ledger`) that can hold message text. Every
//! string-ish field is a validated/sanitizing newtype:
//! - [`Label`] — data-class / profile identifiers: `[a-z][a-z0-9_]{0,31}`.
//!   An Aadhaar (digits), PAN (uppercase), email (`@`) or phone (`+`/digits)
//!   cannot be constructed into one.
//! - [`ExtKey`] / [`ExtValue`] — the FR-11 profile extension map: namespaced
//!   keys, values restricted to bool / u64 / short codes. No free text.
//! - [`CodeStr`] — region/provider codes, ≤16 chars `[A-Za-z0-9_-]`.
//! - [`BoundedText`] — `model` / `classifier_version`: ≤64 chars, no `@`, no
//!   whitespace. `model` is the one client-influenced skeleton field (FR-10
//!   requires it); sanitization bounds, but cannot eliminate, what a client can
//!   put there — disclosed in the engineering notes.
//!
//! ## Integrity scope (001-E)
//! `usage` rides in the entry but is EXCLUDED from the canonical hash payload
//! (`routeplane-ledger`'s `chain.rs`): token counts are not yet part of the
//! signed-evidence claim.

use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

/// Compliance profile #0 — DPDP/RBI (PRD-000 §6, PRD-002 §5).
pub const PROFILE_0_ID: &str = "dpdp_rbi";
pub const PROFILE_0_VERSION: u32 = 1;

/// Validation failure for a bounded schema type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelError {
    pub reason: &'static str,
}

impl std::fmt::Display for LabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid ledger label: {}", self.reason)
    }
}

impl std::error::Error for LabelError {}

fn valid_label_segment(s: &str, max: usize) -> Result<(), LabelError> {
    if s.is_empty() {
        return Err(LabelError { reason: "empty" });
    }
    if s.len() > max {
        return Err(LabelError { reason: "too long" });
    }
    let Some(first) = s.chars().next() else {
        return Err(LabelError { reason: "empty" });
    };
    if !first.is_ascii_lowercase() {
        return Err(LabelError {
            reason: "must start with a lowercase ascii letter",
        });
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(LabelError {
            reason: "only [a-z0-9_] allowed",
        });
    }
    Ok(())
}

/// A closed-vocabulary identifier (data class, profile id). See module docs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Label(String);

impl Label {
    pub const MAX_LEN: usize = 32;

    pub fn new(label: impl Into<String>) -> Result<Self, LabelError> {
        let label = label.into();
        valid_label_segment(&label, Self::MAX_LEN)?;
        Ok(Self(label))
    }

    /// Infallible constructor for compile-time-known labels. Falls back to
    /// `"invalid"` instead of panicking (never on a request thread) — the
    /// fallback is unreachable for valid literals and unit-tested as such.
    pub fn known(label: &'static str) -> Self {
        Self::new(label).unwrap_or_else(|_| Self("invalid".to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Label {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Label::new(String::deserialize(d)?).map_err(D::Error::custom)
    }
}

/// Namespaced FR-11 extension key, e.g. `dpdp.personal_data_present`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ExtKey(String);

impl ExtKey {
    pub const MAX_LEN: usize = 64;

    pub fn new(key: impl Into<String>) -> Result<Self, LabelError> {
        let key = key.into();
        if key.len() > Self::MAX_LEN {
            return Err(LabelError { reason: "too long" });
        }
        let segments: Vec<&str> = key.split('.').collect();
        if segments.len() < 2 {
            return Err(LabelError {
                reason: "extension keys must be profile-namespaced (e.g. dpdp.x)",
            });
        }
        for seg in segments {
            valid_label_segment(seg, Label::MAX_LEN)?;
        }
        Ok(Self(key))
    }

    /// Infallible constructor for known literals (see [`Label::known`]).
    pub fn known(key: &'static str) -> Self {
        Self::new(key).unwrap_or_else(|_| Self("invalid.invalid".to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ExtKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ExtKey::new(String::deserialize(d)?).map_err(D::Error::custom)
    }
}

/// Short code (region / provider name): ≤16 chars of `[A-Za-z0-9_-]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CodeStr(String);

impl CodeStr {
    pub const MAX_LEN: usize = 16;

    fn is_code_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '-'
    }

    pub fn new(code: impl Into<String>) -> Result<Self, LabelError> {
        let code = code.into();
        if code.is_empty() {
            return Err(LabelError { reason: "empty" });
        }
        if code.len() > Self::MAX_LEN {
            return Err(LabelError { reason: "too long" });
        }
        if !code.chars().all(Self::is_code_char) {
            return Err(LabelError {
                reason: "only [A-Za-z0-9_-] allowed",
            });
        }
        Ok(Self(code))
    }

    /// Lossy constructor for client-supplied input (region headers): keeps only
    /// code chars, truncates, never fails — PII-shaped input collapses to a
    /// harmless residue and can never round-trip back to the original value.
    pub fn sanitized(input: &str) -> Self {
        let cleaned: String = input
            .chars()
            .filter(|c| Self::is_code_char(*c))
            .take(Self::MAX_LEN)
            .collect();
        if cleaned.is_empty() {
            Self("invalid".to_string())
        } else {
            Self(cleaned)
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CodeStr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        CodeStr::new(String::deserialize(d)?).map_err(D::Error::custom)
    }
}

/// Bounded technical text (`model`, `classifier_version`): ≤64 chars of
/// `[A-Za-z0-9._:/+-]` — no `@`, no whitespace, no control characters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct BoundedText(String);

impl BoundedText {
    pub const MAX_LEN: usize = 64;

    fn is_allowed(c: char) -> bool {
        c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':' | '/' | '+')
    }

    fn validate(s: &str) -> Result<(), LabelError> {
        if s.is_empty() {
            return Err(LabelError { reason: "empty" });
        }
        if s.len() > Self::MAX_LEN {
            return Err(LabelError { reason: "too long" });
        }
        if !s.chars().all(Self::is_allowed) {
            return Err(LabelError {
                reason: "only [A-Za-z0-9._:/+-] allowed",
            });
        }
        Ok(())
    }

    /// Lossy constructor (see [`CodeStr::sanitized`]).
    pub fn sanitized(input: &str) -> Self {
        let cleaned: String = input
            .chars()
            .filter(|c| Self::is_allowed(*c))
            .take(Self::MAX_LEN)
            .collect();
        if cleaned.is_empty() {
            Self("unknown".to_string())
        } else {
            Self(cleaned)
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for BoundedText {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        BoundedText::validate(&raw).map_err(D::Error::custom)?;
        Ok(Self(raw))
    }
}

/// FR-11 extension value: bool / count / short code. No free text by type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtValue {
    Bool(bool),
    Count(u64),
    Code(CodeStr),
}

/// FR-1/FR-4 outcome of the routing decision — recorded faithfully, including
/// failures (an optimistic compliant-looking entry would be worse than none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    ResidencyBlocked,
    AllFailed,
    /// A stream started (headers + zero-or-more chunks were sent) but ended on
    /// a mid-stream provider error or idle timeout instead of a clean terminal
    /// event. Recorded faithfully — logging a truncated answer as `Ok` is the
    /// exact optimistic-entry failure mode this enum's doc warns against.
    StreamTruncated,
}

/// Token usage. In the entry, OUTSIDE the canonical hash payload (001-E).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageTotals {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// What the data plane emits per decision; the ledger writer turns it into a
/// chained `LedgerEntry`. Serializable because the spill WAL persists drafts
/// (001-C).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionDraft {
    pub tenant_id: String,
    pub request_id: String,
    /// RFC3339 UTC, set at decision time; stored verbatim (deterministic hash).
    pub timestamp: String,
    /// Provider that served the request (None for blocked / all-failed).
    pub provider: Option<CodeStr>,
    pub model: BoundedText,
    /// Region of the chosen route (None when blocked / unknown).
    pub region: Option<CodeStr>,
    pub contains_regulated_data: bool,
    pub data_classes: Vec<Label>,
    pub profile_id: Label,
    pub profile_version: u32,
    pub residency_required: bool,
    pub required_regions: Vec<CodeStr>,
    pub sovereign_routed: bool,
    pub client_override_applied: bool,
    pub outcome: Outcome,
    pub usage: UsageTotals,
    pub classifier_version: BoundedText,
    /// Reserved (001-A): the classifier is boolean today; nullable confidence
    /// is in the schema so no migration is needed later (FR-12).
    pub classifier_confidence: Option<f64>,
    pub ext: BTreeMap<ExtKey, ExtValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn label_rejects_pii_shaped_values() {
        // PAN (uppercase), email (@/.), Aadhaar (digits/spaces), phone (+):
        for bad in [
            "ABCDE1234F",
            "a@b.com",
            "2341 2341 2346",
            "9876543210",
            "+91 98765 43210",
            "",
            "a-very-long-label-exceeding-the-thirty-two-char-cap",
        ] {
            assert!(Label::new(bad).is_err(), "{bad:?} must be rejected");
        }
        for good in ["aadhaar", "pan", "email", "phone", "dpdp_rbi", "stub_v0"] {
            assert!(Label::new(good).is_ok(), "{good:?} must be accepted");
        }
        // The infallible fallback never fires for valid literals.
        assert_eq!(Label::known("aadhaar").as_str(), "aadhaar");
    }

    #[test]
    fn ext_keys_must_be_namespaced() {
        assert!(ExtKey::new("dpdp.personal_data_present").is_ok());
        assert!(ExtKey::new("rbi.localization_relevant").is_ok());
        assert!(ExtKey::new("unnamespaced").is_err());
        assert!(ExtKey::new("Has.Uppercase").is_err());
        assert!(ExtKey::new("a@b.c").is_err());
    }

    #[test]
    fn codestr_sanitization_is_lossy_and_safe() {
        assert_eq!(CodeStr::sanitized("IN").as_str(), "IN");
        assert_eq!(CodeStr::sanitized("azure_openai").as_str(), "azure_openai");
        let dirty = CodeStr::sanitized("IN'; DROP TABLE x; a@b.com");
        assert!(!dirty.as_str().contains('@'));
        assert!(!dirty.as_str().contains(' '));
        assert!(dirty.as_str().len() <= CodeStr::MAX_LEN);
        assert_eq!(CodeStr::sanitized("\u{0}\u{1}").as_str(), "invalid");
    }

    #[test]
    fn bounded_text_strips_email_shape_and_truncates() {
        let t = BoundedText::sanitized("mail me at a@b.com please");
        assert!(!t.as_str().contains('@'));
        assert!(!t.as_str().contains(' '));
        assert!(t.as_str().len() <= BoundedText::MAX_LEN);
        assert_eq!(BoundedText::sanitized("").as_str(), "unknown");
    }

    proptest! {
        // The no-raw-PII property: whatever string is offered, an accepted
        // Label can only ever be a short lowercase identifier.
        #[test]
        fn label_accepts_only_the_closed_charset(s in ".*") {
            if let Ok(label) = Label::new(s) {
                prop_assert!(label.as_str().len() <= Label::MAX_LEN);
                let mut chars = label.as_str().chars();
                prop_assert!(chars.next().map(|c| c.is_ascii_lowercase()).unwrap_or(false));
                prop_assert!(label.as_str().chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'));
            }
        }

        // Sanitized codes never retain PII-bearing punctuation.
        #[test]
        fn sanitized_code_never_contains_separators(s in ".*") {
            let code = CodeStr::sanitized(&s);
            prop_assert!(code.as_str().chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
            prop_assert!(code.as_str().len() <= CodeStr::MAX_LEN);
        }
    }
}
