//! The sovereign-audit ledger's closed-vocabulary types (PRD-001 FR-10/FR-11,
//! R0.3) — extracted from `routeplane-ledger` as a dependency leaf (PRD-047 /
//! ADR-088 CE enabler).
//!
//! This crate holds ONLY the pure vocabulary: the validated newtypes that make
//! raw PII structurally unrepresentable ([`Label`], [`ExtKey`]/[`ExtValue`],
//! [`CodeStr`], [`BoundedText`]), the routing-decision vocabulary
//! ([`Outcome`], [`UsageTotals`], [`DecisionDraft`]) and the security-event
//! vocabulary ([`SecurityCategory`], [`SecurityOutcome`], [`SecurityEvent`]).
//! No hash chain, no signer, no store, no network — the moat engine lives in
//! `routeplane-ledger`, which depends on this crate and re-exports every name
//! here, so its consumers see zero diff.

pub mod entry;
pub mod security;

pub use entry::{
    BoundedText, CodeStr, DecisionDraft, ExtKey, ExtValue, Label, LabelError, Outcome, UsageTotals,
    PROFILE_0_ID, PROFILE_0_VERSION,
};
pub use security::{
    SecurityCategory, SecurityEvent, SecurityOutcome, SECURITY_GLOBAL_SCOPE, SECURITY_PROFILE_ID,
    SECURITY_PROFILE_VERSION,
};
