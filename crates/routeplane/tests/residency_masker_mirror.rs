//! The sovereign-classifier ⇄ guardrail-masker mirror invariant.
//!
//! `routeplane_residency` decides whether text carries regulated personal data
//! (and therefore whether routing hard-locks to a region).
//! `routeplane_guardrails::detect::redact` decides whether that same text is
//! redacted before it reaches a provider. The two are separate crates with
//! separate recognizer sets, and they have drifted repeatedly — the Aadhaar,
//! phone, and IBAN comments in `detect.rs` each record one such episode.
//!
//! **The failure mode is always the same shape and always one-directional:** when
//! the classifier recognises something the masker does not, the request is
//! flagged as personal data, region-locked for routing, and then egressed to the
//! provider IN CLEARTEXT. The reverse (masker wider than classifier) is merely
//! conservative and costs nothing but a redaction.
//!
//! Those drifts were caught by review, and only after shipping. This file pins
//! the invariant as an executable test instead, for the recognizers where the two
//! crates are supposed to agree exactly.
//!
//! Scope note: this is deliberately NOT a claim that every classifier entity has
//! a masker twin — several (e.g. `Phone`) use intentionally different shapes for
//! documented reasons. It covers the India set, where exact agreement IS the
//! contract.

use routeplane_guardrails::detect::redact;
use routeplane_residency::{EntityType, ResidencyEngine};

/// Assert the classifier and the masker reach the same verdict on `text`.
fn assert_agrees(text: &str, entity: EntityType, mask_token: &str) {
    let classified = ResidencyEngine::new()
        .classify(text)
        .entities
        .contains(&entity);
    let masked = redact(text).contains(mask_token);
    assert_eq!(
        classified, masked,
        "classifier/masker disagree on {text:?} — classified={classified} masked={masked}. \
         If classified=true and masked=false this is a LEAK: the request is region-locked \
         as personal data and then sent to the provider in cleartext."
    );
}

#[test]
fn ifsc_classifier_and_masker_agree() {
    // IFSC is shape-only on BOTH sides (the RBI publishes no check digit). A cue
    // gate — requiring an IFSC/NEFT/RTGS/IMPS word nearby — was briefly added to
    // the masker alone, which made the bare-IFSC case classify-but-not-mask.
    for s in [
        "transfer to HDFC0001234 please", // no cue word at all — the regression case
        "IFSC code HDFC0001234 for the transfer",
        "NEFT to SBIN0000456 today",
        "branch HDFC0000043 NEFT", // the residency crate's own fixture
        "token HDFCX000043 here",  // no reserved `0` in slot 5 -> neither side fires
        "nothing sensitive in this sentence",
    ] {
        assert_agrees(s, EntityType::Ifsc, "[IFSC_MASKED]");
    }
}

#[test]
fn gstin_classifier_and_masker_agree() {
    // GSTIN carries a mod-36 check character, so both sides gate on it and a
    // corrupted check character must fire on NEITHER.
    for s in [
        "GSTIN 27AAPFU0939F1ZV on the invoice", // published, valid
        "GSTIN 29AAGCB7383J1Z4 filed",          // published, valid
        "bill to 07AAACS1429B1ZX",              // published, valid, no cue word
        "GSTIN 27AAPFU0939F1ZQ filed",          // check character flipped -> neither
        "order 27AAPFU0939F1Z alone",           // 14 chars -> neither
        "nothing sensitive in this sentence",
    ] {
        assert_agrees(s, EntityType::Gstin, "[GSTIN_MASKED]");
    }
}

#[test]
fn aadhaar_classifier_and_masker_agree() {
    // Verhoeff-gated on both sides. Included because this pair is what the
    // separator-class drift (`[\s.-]{0,2}`) broke once already.
    for s in [
        "id 2341 2341 2346 ok",   // valid, single spaces
        "id 2341  2341  2346 ok", // valid, double spaces — the drift case
        "id 234123412346 ok",     // valid, compact
        "id 2341 2341 2348 ok",   // check digit flipped -> neither
        "nothing sensitive in this sentence",
    ] {
        assert_agrees(s, EntityType::Aadhaar, "[AADHAAR_MASKED]");
    }
}

#[test]
fn a_classified_india_request_never_egresses_unmasked() {
    // The property stated end-to-end, over the whole India set at once: if the
    // classifier says "this is personal data", redact() must have changed the
    // text. This is the invariant a reviewer actually cares about, independent of
    // which recognizer fired.
    for s in [
        "transfer to HDFC0001234 please",
        "GSTIN 27AAPFU0939F1ZV on the invoice",
        "id 2341 2341 2346 ok",
        "my PAN is ABCDE1234F",
        "call me on +91 98765 43210",
    ] {
        let c = ResidencyEngine::new().classify(s);
        assert!(
            c.contains_personal_data,
            "fixture {s:?} was supposed to classify as personal data"
        );
        assert_ne!(
            redact(s).as_ref(),
            s,
            "{s:?} is classified as India personal data but egresses byte-identical — \
             region-locked and unmasked is the leak this test exists to catch"
        );
    }
}
