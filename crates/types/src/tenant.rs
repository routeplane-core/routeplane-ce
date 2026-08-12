//! The tenant identity newtype — the workspace's single definition of "which
//! customer is this".
//!
//! Tenant identity is authority, so it must be validated once and remain
//! byte-exact. Mutable display names and lossy normalization are never accepted
//! as substitutes.
//!
//! # The charset makes storage folds injective
//!
//! Storage component sanitizers preserve `[A-Za-z0-9._-]` and fold other bytes
//! to `_`.
//!
//! On the subset `[A-Za-z0-9_-]` — exactly this type's charset — that fold is the
//! **identity function**, and therefore injective. Admitting `.` would break it:
//! `t.acme` and `t_acme` both fold to `t_acme` and share one partition, one blob,
//! one hash chain.
//!
//! This type therefore admits the largest subset on which those sanitizers are
//! collision-free. Widening it requires proving every storage key stays
//! injective.
//!
//! # Reject, never coerce
//!
//! [`TenantId::new`] REFUSES a bad value rather than mapping it to something
//! adjacent. A sanitizer that coerces is exactly how two tenants end up sharing a
//! key without anyone noticing — the refusal is the whole point.
//!
//! The private field plus fallible `new` keep every construction path
//! validating. It deliberately does NOT mirror [`crate::Region`], which has a
//! public field and an infallible constructor and therefore validates nothing.

use serde::{Deserialize, Deserializer, Serialize};

/// Why a value was rejected by one of the validated newtypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for ValueError {}

/// A validated tenant identity.
///
/// Construct with [`TenantId::new`]. The inner value is private so the
/// invariant cannot be bypassed by field access.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TenantId(String);

impl TenantId {
    pub const MAX_LEN: usize = 64;

    /// Validate and wrap. `[A-Za-z0-9_-]`, 1–64 bytes, no trim, no case-fold.
    ///
    /// No trimming or case-folding on purpose: `t_acme` and `T_acme` are
    /// DIFFERENT tenants, and silently folding them would be the same class of
    /// bug this type exists to prevent.
    pub fn new(id: impl Into<String>) -> Result<Self, ValueError> {
        let id = id.into();
        if id.is_empty() || id.len() > Self::MAX_LEN {
            return Err(ValueError {
                field: "tenant_id",
                reason: "must be 1-64 characters",
            });
        }
        if !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        {
            return Err(ValueError {
                field: "tenant_id",
                reason: "only [A-Za-z0-9_-] allowed",
            });
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for TenantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for TenantId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        TenantId::new(String::deserialize(d)?).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The charset is chosen so `sanitize_component` (which preserves
    /// `[A-Za-z0-9._-]` and folds everything else to `_`) is the IDENTITY on it.
    /// If this ever passes for a value containing `.`, the ledger, telemetry and
    /// export partition keys stop being injective and the defect class is back.
    #[test]
    fn charset_is_exactly_the_set_on_which_sanitize_component_is_injective() {
        // The fold maps every one of these to the SAME string, so admitting any
        // of them would let two tenants share one partition.
        for folds_together in ["t.acme", "t acme", "t:acme", "t/acme", "t+acme"] {
            assert!(
                TenantId::new(folds_together).is_err(),
                "{folds_together:?} folds onto t_acme and must be refused"
            );
        }
        // `t_acme` is the value they all fold TO — it must remain valid, or the
        // refusal above would be pointless.
        assert!(TenantId::new("t_acme").is_ok());
    }

    #[test]
    fn accepts_what_the_control_plane_mints() {
        // gen_tenant_id() = "t_" + 32 lowercase hex.
        assert!(TenantId::new("t_eace298696af41698f15b8aa5b3418ba").is_ok());
        assert!(TenantId::new("t_paid_demo").is_ok());
        assert!(TenantId::new("Tenant-1").is_ok());
    }

    /// The exact shape that the display-name fallback produced, and the reason
    /// this type exists.
    #[test]
    fn refuses_a_display_name() {
        assert!(TenantId::new("Default Development Key").is_err());
        assert!(TenantId::new("Garth Prod").is_err());
    }

    #[test]
    fn refuses_empty_and_overlong() {
        assert!(TenantId::new("").is_err());
        assert!(TenantId::new("a".repeat(TenantId::MAX_LEN)).is_ok());
        assert!(TenantId::new("a".repeat(TenantId::MAX_LEN + 1)).is_err());
    }

    /// No case-folding: these are two different tenants, not one.
    #[test]
    fn case_is_significant() {
        let lower = TenantId::new("t_acme").unwrap();
        let upper = TenantId::new("T_acme").unwrap();
        assert_ne!(lower, upper);
    }

    #[test]
    fn serde_round_trips_transparently_and_validates_on_the_way_in() {
        let t = TenantId::new("t_acme").unwrap();
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#""t_acme""#, "transparent: a bare JSON string");
        assert_eq!(serde_json::from_str::<TenantId>(&json).unwrap(), t);
        // Deserialize routes through new(), so the wire cannot smuggle one in.
        assert!(serde_json::from_str::<TenantId>(r#""bad tenant""#).is_err());
    }
}
