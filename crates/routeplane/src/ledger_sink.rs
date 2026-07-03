//! Data-plane → ledger seam (PRD-001 FR-16, ADR-012 §6.2).
//!
//! [`record_decision`] is the ONLY ledger touchpoint on the request path:
//! when the process-wide handle is `None` (ship-dark default) or the tenant
//! lacks `Feature::AuditLedger`, it returns before the draft closure is even
//! invoked — zero allocation, zero work, byte-identical behavior (AC-6).
//! When both gates pass, the cost is one bounded `try_send` (NFR-1/2).

use routeplane_entitlements::CapabilitySet;
#[cfg(feature = "enterprise")]
use routeplane_entitlements::Feature;
// The handle type itself is part of the seam (PRD-047 / ADR-088): the
// enterprise build re-exports the real moat handle; the CE build re-exports the
// uninhabited `ce_stubs` slot, so `AppState.ledger`, the auth-seam
// `SharedLedgerHandle`, and every `record_*` signature stay textually identical
// across the two variants (`Option<LedgerHandle>` — permanently `None` on CE).
#[cfg(not(feature = "enterprise"))]
pub use crate::ce_stubs::LedgerHandle;
#[cfg(feature = "enterprise")]
pub use routeplane_ledger::LedgerHandle;
// The closed-vocabulary ledger types come from the dependency-leaf vocab crate
// (PRD-047 / ADR-088 CE enabler) and are RE-EXPORTED here: handler modules
// (embeddings/audio/images/rerank/moderations/proxy) import them via
// `crate::ledger_sink::{...}`, so this module is the binary's ONE ledger seam
// — the vocab types re-exported here, the records through the `record_*` fns
// below. The vocab crate is ALWAYS a dependency (types are not the moat), so
// these re-exports are identical on both build variants.
pub use routeplane_ledger_vocab::{
    BoundedText, CodeStr, DecisionDraft, ExtKey, ExtValue, Label, Outcome, SecurityCategory,
    SecurityEvent, SecurityOutcome, UsageTotals, PROFILE_0_ID, PROFILE_0_VERSION,
};
use routeplane_residency::{Classification, EntityType, CLASSIFIER_VERSION};

/// Gate + record. The draft is built lazily so an unentitled request never
/// pays for construction.
#[cfg(feature = "enterprise")]
pub fn record_decision<F>(ledger: &Option<LedgerHandle>, capabilities: &CapabilitySet, build: F)
where
    F: FnOnce() -> DecisionDraft,
{
    let Some(handle) = ledger else { return };
    if !capabilities.active(Feature::AuditLedger) {
        return;
    }
    handle.record(build());
}

/// CE no-op twin (PRD-047 / ADR-088): identical signature, and the same
/// zero-cost shape as the enterprise ship-dark path — the handle is the
/// uninhabited CE slot (only ever `None`), so the draft closure never runs.
#[cfg(not(feature = "enterprise"))]
pub fn record_decision<F>(_ledger: &Option<LedgerHandle>, _capabilities: &CapabilitySet, _build: F)
where
    F: FnOnce() -> DecisionDraft,
{
}

/// Gate + record a SECURITY event (R0.3), capability-gated EXACTLY like
/// [`record_decision`]: an absent handle OR a tenant without
/// [`Feature::AuditLedger`] returns before the event closure runs — zero
/// allocation, zero work, byte-identical. When both gates pass, the cost is one
/// bounded `try_send` (the same hot-path discipline as `record_decision`).
///
/// Used at the proxy-side decision points where a resolved [`CapabilitySet`] is
/// in hand (guardrail deny, residency block, rate/budget breach, MCP denial).
#[cfg(feature = "enterprise")]
pub fn record_security<F>(ledger: &Option<LedgerHandle>, capabilities: &CapabilitySet, build: F)
where
    F: FnOnce() -> SecurityEvent,
{
    let Some(handle) = ledger else { return };
    if !capabilities.active(Feature::AuditLedger) {
        return;
    }
    record_security_event(handle, build());
}

/// CE no-op twin: identical signature; the event closure never runs (the
/// uninhabited CE handle can only ever be `None`).
#[cfg(not(feature = "enterprise"))]
pub fn record_security<F>(_ledger: &Option<LedgerHandle>, _capabilities: &CapabilitySet, _build: F)
where
    F: FnOnce() -> SecurityEvent,
{
}

/// Record a SECURITY event at the AUTH seam, where NO tenant context exists yet
/// (the failure happens before — or because — key resolution failed). Gated on
/// handle presence ONLY: there is no `CapabilitySet` to consult, so this records
/// a `_global`-scoped event whenever the audit ledger is enabled at all. An
/// absent handle (ship-dark default) is a zero-work no-op — byte-identical.
#[cfg(feature = "enterprise")]
pub fn record_security_global<F>(ledger: &Option<LedgerHandle>, build: F)
where
    F: FnOnce() -> SecurityEvent,
{
    let Some(handle) = ledger else { return };
    record_security_event(handle, build());
}

/// CE no-op twin: identical signature; the event closure never runs.
#[cfg(not(feature = "enterprise"))]
pub fn record_security_global<F>(_ledger: &Option<LedgerHandle>, _build: F)
where
    F: FnOnce() -> SecurityEvent,
{
}

/// Shared sink: turn a [`SecurityEvent`] into a draft and `record` it. A draft
/// that fails to build (a malformed category label — unreachable for the
/// closed-vocabulary categories) is dropped rather than panicking the request
/// thread (fail-safe: no entry beats a bad entry).
#[cfg(feature = "enterprise")]
fn record_security_event(handle: &LedgerHandle, event: SecurityEvent) {
    match event.to_draft() {
        Ok(draft) => handle.record(draft),
        Err(e) => tracing::warn!("dropping malformed security ledger event: {e}"),
    }
}

/// Construct a [`SecurityEvent`] at a decision point. Keeps the (timestamp,
/// closed-vocab field) construction in one reviewed place.
pub fn security_event(
    request_id: &str,
    tenant_id: Option<&str>,
    category: SecurityCategory,
    outcome: SecurityOutcome,
    count: Option<u64>,
    detail_code: Option<&str>,
) -> SecurityEvent {
    SecurityEvent {
        event_id: request_id.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        category,
        outcome,
        count,
        detail_code: detail_code.map(str::to_string),
        tenant_id: tenant_id.map(str::to_string),
    }
}

fn data_class_label(entity: &EntityType) -> Label {
    match entity {
        EntityType::Email => Label::known("email"),
        EntityType::Phone => Label::known("phone"),
        EntityType::Aadhaar => Label::known("aadhaar"),
        EntityType::Pan => Label::known("pan"),
        EntityType::Ssn => Label::known("ssn"),
        EntityType::Iban => Label::known("iban"),
        EntityType::Cpf => Label::known("cpf"),
        EntityType::Nric => Label::known("nric"),
        EntityType::EmiratesId => Label::known("emirates_id"),
        EntityType::SaudiId => Label::known("saudi_id"),
        EntityType::Ifsc => Label::known("ifsc"),
        EntityType::Tfn => Label::known("tfn"),
        EntityType::MyNumber => Label::known("my_number"),
    }
}

/// Build one FR-10 decision draft. Records classification LABELS and the
/// sovereign decision — never values, never message text (the ledger types
/// make that structurally impossible). Provider error strings are deliberately
/// NOT recorded (a provider error can echo prompt text).
#[allow(clippy::too_many_arguments)]
pub fn decision_draft(
    tenant_id: &str,
    request_id: &str,
    model: &str,
    provider: Option<&str>,
    route_region: Option<&str>,
    classification: &Classification,
    required_region: Option<&str>,
    sovereign_routed: bool,
    client_provider_requested: bool,
    outcome: Outcome,
    usage: UsageTotals,
) -> DecisionDraft {
    let mut ext = std::collections::BTreeMap::new();
    // Profile-#0 evidence extension (PRD-001 FR-11) — namespaced, never generic.
    ext.insert(
        ExtKey::known("dpdp.personal_data_present"),
        ExtValue::Bool(classification.contains_personal_data),
    );
    ext.insert(
        ExtKey::known("rbi.localization_relevant"),
        ExtValue::Bool(required_region == Some("IN")),
    );
    DecisionDraft {
        tenant_id: tenant_id.to_string(),
        request_id: request_id.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        provider: provider.map(CodeStr::sanitized),
        model: BoundedText::sanitized(model),
        region: route_region.map(CodeStr::sanitized),
        contains_regulated_data: classification.contains_personal_data,
        data_classes: classification
            .entities
            .iter()
            .map(data_class_label)
            .collect(),
        profile_id: Label::known(PROFILE_0_ID),
        profile_version: PROFILE_0_VERSION,
        residency_required: required_region.is_some(),
        required_regions: required_region
            .map(|r| vec![CodeStr::sanitized(r)])
            .unwrap_or_default(),
        sovereign_routed,
        client_override_applied: sovereign_routed && client_provider_requested,
        outcome,
        usage,
        classifier_version: BoundedText::sanitized(CLASSIFIER_VERSION),
        classifier_confidence: None, // reserved (001-A)
        ext,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use routeplane_entitlements::Tier;
    // The chain-coupled tests below spin a REAL ledger writer, so they exist
    // only on the enterprise build; the gating/no-op tests run on both.
    #[cfg(feature = "enterprise")]
    use routeplane_ledger::{spawn_ledger, InMemoryLedgerStore, LedgerConfig, LedgerStore};
    use routeplane_residency::ResidencyEngine;
    use std::cell::Cell;
    use std::collections::BTreeSet;
    #[cfg(feature = "enterprise")]
    use std::sync::Arc;
    #[cfg(feature = "enterprise")]
    use std::time::Duration;

    fn pii_classification() -> Classification {
        ResidencyEngine::new().classify("PAN ABCDE1234F, mail a@b.com")
    }

    fn draft() -> DecisionDraft {
        decision_draft(
            "t_1",
            "req_x",
            "gpt-4o",
            Some("azure_openai"),
            Some("IN"),
            &pii_classification(),
            Some("IN"),
            true,
            true,
            Outcome::Ok,
            UsageTotals::default(),
        )
    }

    /// AC-6 hermetic analogue: capability off (or handle absent) ⇒ the draft
    /// builder is never invoked and nothing reaches the channel — the request
    /// path is byte-identical to a build without the ledger.
    #[cfg(feature = "enterprise")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ledger_is_dark_unless_capability_active() {
        let store = Arc::new(InMemoryLedgerStore::new());
        let cfg = LedgerConfig {
            wal_dir: std::env::temp_dir()
                .join(format!("rp_sink_{}", std::process::id()))
                .join(uuid::Uuid::new_v4().simple().to_string()),
            drain_batch: 1,
            drain_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let ledger = Some(spawn_ledger(cfg, store.clone(), None).unwrap());

        let free = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        let entitled = CapabilitySet::resolve(
            Tier::Free,
            &BTreeSet::from([Feature::AuditLedger]),
            &BTreeSet::new(),
        );

        let called = Cell::new(false);
        record_decision(&ledger, &free, || {
            called.set(true);
            draft()
        });
        assert!(!called.get(), "unentitled: builder must not run");
        record_decision(&None, &entitled, || {
            called.set(true);
            draft()
        });
        assert!(!called.get(), "no handle: builder must not run");

        record_decision(&ledger, &entitled, draft);
        let from = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let to = chrono::DateTime::parse_from_rfc3339("2100-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // The entry lands in ~ms once the detached writer task is scheduled, and
        // this loop breaks the instant it does — so the budget only bites under
        // pathological CPU starvation. Under `cargo test --all` (the CI command)
        // every test binary runs in parallel AND each multi-threads, heavily
        // oversubscribing the runner, and `Instant` advances on wall clock even
        // when this process is starved of CPU. A 5s budget flaked there (#141);
        // 30s is generous headroom with zero happy-path cost (it never waits the
        // full budget on success). NOT a `worker_threads` bump (more threads
        // worsen the oversubscription) and NOT `serial` (cross-binary
        // oversubscription dominates, and serial_test isn't a workspace dep).
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let got = store
                .read_entries("t_1", "dpdp_rbi", from, to)
                .await
                .unwrap();
            if got.len() == 1 {
                assert_eq!(got[0].request_id, "req_x");
                break;
            }
            assert!(std::time::Instant::now() < deadline, "entry never landed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// FR-2/AC-7: the draft built from a PII-bearing request carries class
    /// labels, never the values.
    #[test]
    fn decision_draft_carries_labels_not_values() {
        let d = draft();
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("ABCDE1234F"));
        assert!(!json.contains("a@b.com"));
        assert!(json.contains("\"pan\""));
        assert!(json.contains("\"email\""));
        assert!(d.contains_regulated_data);
        assert!(d.client_override_applied);
        assert_eq!(d.required_regions[0].as_str(), "IN");
    }

    // --- R0.3: security-event recording ----------------------------------------

    /// `record_security` is gated EXACTLY like `record_decision`: an unentitled
    /// tenant or an absent handle never invokes the event builder (zero work,
    /// byte-identical disabled path).
    #[test]
    fn record_security_is_dark_unless_capability_active() {
        let free = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        let called = Cell::new(false);
        // No handle ⇒ builder never runs.
        record_security(&None, &free, || {
            called.set(true);
            security_event(
                "req_x",
                Some("t_1"),
                SecurityCategory::GuardrailDeny,
                SecurityOutcome::Deny,
                Some(1),
                Some("before"),
            )
        });
        assert!(!called.get(), "no handle: builder must not run");
    }

    /// An entitled tenant's security event lands on the `security` profile chain
    /// (not the `dpdp_rbi` decision chain), and the auth-seam `_global` event
    /// records on handle presence alone.
    #[cfg(feature = "enterprise")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn security_events_land_on_the_security_chain() {
        let store = Arc::new(InMemoryLedgerStore::new());
        let cfg = LedgerConfig {
            wal_dir: std::env::temp_dir()
                .join(format!("rp_sec_{}", std::process::id()))
                .join(uuid::Uuid::new_v4().simple().to_string()),
            drain_batch: 1,
            drain_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let ledger = Some(spawn_ledger(cfg, store.clone(), None).unwrap());
        let entitled = CapabilitySet::resolve(
            Tier::Free,
            &BTreeSet::from([Feature::AuditLedger]),
            &BTreeSet::new(),
        );

        // Proxy-side (capability-gated) residency-block security event.
        record_security(&ledger, &entitled, || {
            security_event(
                "req_sec",
                Some("t_sec"),
                SecurityCategory::ResidencyBlock,
                SecurityOutcome::Deny,
                None,
                Some("IN"),
            )
        });
        // Auth-seam (handle-only) global auth-failure event.
        record_security_global(&ledger, || {
            security_event(
                "authsec_1",
                None,
                SecurityCategory::AuthFailure,
                SecurityOutcome::Deny,
                Some(4),
                None,
            )
        });

        let from = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let to = chrono::DateTime::parse_from_rfc3339("2100-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let tenant_scoped = store
                .read_entries("t_sec", "security", from, to)
                .await
                .unwrap();
            let global_scoped = store
                .read_entries("_global", "security", from, to)
                .await
                .unwrap();
            if tenant_scoped.len() == 1 && global_scoped.len() == 1 {
                assert_eq!(tenant_scoped[0].request_id, "req_sec");
                assert_eq!(tenant_scoped[0].data_classes[0].as_str(), "residency_block");
                assert_eq!(global_scoped[0].request_id, "authsec_1");
                assert_eq!(global_scoped[0].data_classes[0].as_str(), "auth_failure");
                // The DPDP decision chain saw NONE of these.
                let decision = store
                    .read_entries("t_sec", "dpdp_rbi", from, to)
                    .await
                    .unwrap();
                assert!(
                    decision.is_empty(),
                    "security events never pollute the decision chain"
                );
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "security entries never landed"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
