//! Guardrail efficacy gate (devsecops-pipeline.md §7.2 / PRD-036 §9, ADR-032).
//!
//! Scores the deterministic detector library (`routeplane_guardrails::detect`,
//! ADR-031) against the CODEOWNER-gated corpus under
//! `tests/guardrail_eval/corpus/` and asserts the manifest floors — recall
//! (mask/flag rate on positives, security-critical) and precision (1 - FP rate
//! on negatives, product-quality) per category bucket.
//!
//! `known_gaps.jsonl` is report-only (excluded from pass/fail), exactly as the
//! manifest mandates. This realizes the harness the corpus manifest declared as
//! "the next §7.2 work item".
//!
//! Floors below are the `manifest.toml` thresholds verbatim; `manifest.toml`
//! stays the documented source of truth (a future revision can parse it once a
//! `toml` dev-dependency is admitted — kept dependency-free here on purpose).

use routeplane_guardrails::detect::{detect_injection, scan_pii};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

// --- manifest.toml floors (verbatim) ------------------------------------------
const PII_EMAIL_PHONE_RECALL_FLOOR: f64 = 0.98;
const PII_EMAIL_PHONE_PRECISION_FLOOR: f64 = 0.95;
const PII_DPDP_RECALL_FLOOR: f64 = 0.95;
const PII_DPDP_PRECISION_FLOOR: f64 = 0.90;
const INJECTION_RECALL_FLOOR: f64 = 0.90;
const INJECTION_PRECISION_FLOOR: f64 = 0.85;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/guardrail_eval/corpus")
}

fn load(rel: &str) -> Vec<Value> {
    let path = corpus_dir().join(rel);
    let body = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("parse {rel}: {e}\n{l}")))
        .collect()
}

fn expected_categories(rec: &Value) -> Vec<String> {
    rec["expect"]["masked"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// recall over a positive set, where a record is a "hit" iff every expected PII
/// category is detected by `scan_pii`.
fn pii_recall(records: &[&Value]) -> f64 {
    if records.is_empty() {
        return 1.0;
    }
    let hits = records
        .iter()
        .filter(|r| {
            let got = scan_pii(r["input"].as_str().unwrap_or_default());
            expected_categories(r)
                .iter()
                .all(|cat| got.iter().any(|g| g == cat))
        })
        .count();
    hits as f64 / records.len() as f64
}

#[test]
fn guardrail_corpus_meets_manifest_floors() {
    // ---- PII ----
    let positives = load("pii/positives.jsonl");
    let negatives = load("pii/negatives.jsonl");

    let is_email_phone = |r: &&Value| {
        expected_categories(r)
            .iter()
            .any(|c| c == "email" || c == "phone")
    };
    let is_dpdp = |r: &&Value| {
        expected_categories(r)
            .iter()
            .any(|c| c == "aadhaar" || c == "pan")
    };

    let ep: Vec<&Value> = positives.iter().filter(is_email_phone).collect();
    let dpdp: Vec<&Value> = positives.iter().filter(is_dpdp).collect();
    let ep_recall = pii_recall(&ep);
    let dpdp_recall = pii_recall(&dpdp);

    // precision: a negative is a false positive if ANY PII is detected.
    let fp = negatives
        .iter()
        .filter(|r| !scan_pii(r["input"].as_str().unwrap_or_default()).is_empty())
        .count();
    let pii_precision = 1.0 - (fp as f64 / negatives.len().max(1) as f64);

    // ---- injection ----
    let attacks = load("injection/attacks.jsonl");
    let benign = load("injection/benign.jsonl");
    let inj_hits = attacks
        .iter()
        .filter(|r| detect_injection(r["input"].as_str().unwrap_or_default()))
        .count();
    let inj_recall = inj_hits as f64 / attacks.len().max(1) as f64;
    let inj_fp = benign
        .iter()
        .filter(|r| detect_injection(r["input"].as_str().unwrap_or_default()))
        .count();
    let inj_precision = 1.0 - (inj_fp as f64 / benign.len().max(1) as f64);

    // ---- report (always printed with `--nocapture`) ----
    eprintln!("guardrail eval (PRD-036 §9):");
    eprintln!("  pii_email_phone  recall={ep_recall:.3} precision={pii_precision:.3} (floors {PII_EMAIL_PHONE_RECALL_FLOOR}/{PII_EMAIL_PHONE_PRECISION_FLOOR})");
    eprintln!("  pii_dpdp         recall={dpdp_recall:.3} precision={pii_precision:.3} (floors {PII_DPDP_RECALL_FLOOR}/{PII_DPDP_PRECISION_FLOOR})");
    eprintln!("  injection        recall={inj_recall:.3} precision={inj_precision:.3} (floors {INJECTION_RECALL_FLOOR}/{INJECTION_PRECISION_FLOOR})");

    // ---- gate (the hard floors) ----
    assert!(
        ep_recall >= PII_EMAIL_PHONE_RECALL_FLOOR,
        "pii_email_phone recall {ep_recall:.3} < floor {PII_EMAIL_PHONE_RECALL_FLOOR}"
    );
    assert!(
        pii_precision >= PII_EMAIL_PHONE_PRECISION_FLOOR,
        "pii precision {pii_precision:.3} < floor {PII_EMAIL_PHONE_PRECISION_FLOOR}"
    );
    assert!(
        dpdp_recall >= PII_DPDP_RECALL_FLOOR,
        "pii_dpdp recall {dpdp_recall:.3} < floor {PII_DPDP_RECALL_FLOOR}"
    );
    assert!(
        pii_precision >= PII_DPDP_PRECISION_FLOOR,
        "pii precision {pii_precision:.3} < floor {PII_DPDP_PRECISION_FLOOR}"
    );
    assert!(
        inj_recall >= INJECTION_RECALL_FLOOR,
        "injection recall {inj_recall:.3} < floor {INJECTION_RECALL_FLOOR}"
    );
    assert!(
        inj_precision >= INJECTION_PRECISION_FLOOR,
        "injection precision {inj_precision:.3} < floor {INJECTION_PRECISION_FLOOR}"
    );
}

/// The manifest mandates `known_gaps.jsonl` is report-only — never scored. This
/// guards that contract: the gaps file exists and is NOT part of the scored
/// positives (so a known miss can never red the gate, nor silently disappear).
#[test]
fn known_gaps_are_excluded_from_scoring() {
    let gaps = load("pii/known_gaps.jsonl");
    assert!(
        !gaps.is_empty(),
        "known_gaps.jsonl should document current misses"
    );
    let positive_ids: Vec<String> = load("pii/positives.jsonl")
        .iter()
        .map(|r| r["id"].as_str().unwrap_or_default().to_string())
        .collect();
    for g in &gaps {
        let id = g["id"].as_str().unwrap_or_default();
        assert!(
            !positive_ids.contains(&id.to_string()),
            "known-gap {id} must not also be in the scored positives"
        );
        assert!(g.get("gap").is_some(), "known-gap {id} must document why");
    }
}
