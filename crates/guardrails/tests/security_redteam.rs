//! Automated security red-team / eval gate (ADR-048 Ring 5, ADR-032 §1 quality
//! gate). This is the PER-PR FAST gate: it scores the deterministic detector
//! library (`routeplane_guardrails::detect`, ADR-031) against the
//! OWASP-LLM-Top-10 adversarial corpus under `tests/security/corpus/` and
//! asserts the precision/recall floors declared in that corpus's `manifest.toml`.
//!
//! It is the cheap, deterministic, network-free, model-free counterpart to the
//! NIGHTLY off-box red-team tooling (garak / promptfoo) wired in
//! `.github/workflows/security-redteam-nightly.yml`, which draws from the SAME
//! corpus but exercises a live gateway. Same corpus -> same verdict => never
//! flaky (the ADR-032 rule). Heavy red-team stays on the schedule; this stays
//! on every PR.
//!
//! Scope (do NOT cross): this test only CONSUMES the existing detector surface
//! (`detect_injection` / `scan_secrets` / `contains_invisible_unicode`). It does
//! not modify the detectors. If a detector genuinely misses an adversarial case,
//! the case is recorded in the corpus's `known_gaps.jsonl` (report-only,
//! excluded from scoring) and surfaced as a finding for a later ring — the gate
//! is never weakened to pass, and the detectors are never weakened to pass it.
//!
//! Floors below are the `tests/security/corpus/manifest.toml` thresholds
//! verbatim (kept dependency-free here on purpose, mirroring `guardrail_eval.rs`
//! — `manifest.toml` stays the documented source of truth).

use routeplane_guardrails::detect::{contains_invisible_unicode, detect_injection, scan_secrets};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

// --- manifest.toml floors (verbatim) -----------------------------------------
const INJECTION_RECALL_FLOOR: f64 = 0.95;
const INJECTION_PRECISION_FLOOR: f64 = 0.85;
const SMUGGLING_RECALL_FLOOR: f64 = 1.00;
const SMUGGLING_PRECISION_FLOOR: f64 = 1.00;
const SECRETS_RECALL_FLOOR: f64 = 1.00;
const SECRETS_PRECISION_FLOOR: f64 = 1.00;

fn corpus_dir() -> PathBuf {
    // crates/guardrails -> ../../tests/security/corpus (repo-root corpus). The
    // detector library this gate measures lives in THIS crate
    // (routeplane_guardrails), so the gate is hosted here and stays decoupled
    // from the binary crate's compile state — a security gate must not be coupled
    // to an unrelated lib build (crates/guardrails/CLAUDE.md: efficacy is gated
    // by tests over this crate).
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/security/corpus")
}

fn load(rel: &str) -> Vec<Value> {
    let path = corpus_dir().join(rel);
    let body = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("parse {rel}: {e}\n{l}")))
        .collect()
}

fn input(rec: &Value) -> &str {
    rec["input"].as_str().unwrap_or_default()
}

fn id(rec: &Value) -> &str {
    rec["id"].as_str().unwrap_or_default()
}

/// The deterministic decision the gateway's hot path can make on a prompt today
/// (ADR-031): an injection signature fires, OR an invisible/smuggling char is
/// present, OR an embedded secret is detected in the prompt. Mirrors the inline
/// guardrail track — nothing here calls a model or the network.
fn injection_flagged(text: &str) -> bool {
    detect_injection(text) || contains_invisible_unicode(text) || !scan_secrets(text).is_empty()
}

fn pct(hits: usize, total: usize) -> f64 {
    if total == 0 {
        return 1.0;
    }
    hits as f64 / total as f64
}

#[test]
fn redteam_corpus_meets_manifest_floors() {
    // ---- injection (prompt-injection / jailbreak / exfil-intent / tool-poison) ----
    let attacks = load("injection/attacks.jsonl");
    let benign = load("injection/benign.jsonl");
    let inj_hits = attacks
        .iter()
        .filter(|r| injection_flagged(input(r)))
        .count();
    let inj_recall = pct(inj_hits, attacks.len());
    let inj_fp = benign
        .iter()
        .filter(|r| injection_flagged(input(r)))
        .count();
    let inj_precision = 1.0 - pct(inj_fp, benign.len());

    // ---- unicode / invisible-char smuggling ----
    let smg_atk = load("smuggling/attacks.jsonl");
    let smg_ben = load("smuggling/benign.jsonl");
    let smg_hits = smg_atk
        .iter()
        .filter(|r| contains_invisible_unicode(input(r)))
        .count();
    let smg_recall = pct(smg_hits, smg_atk.len());
    let smg_fp = smg_ben
        .iter()
        .filter(|r| contains_invisible_unicode(input(r)))
        .count();
    let smg_precision = 1.0 - pct(smg_fp, smg_ben.len());

    // ---- secrets exfiltration (DLP must catch the embedded credential) ----
    // recall: every EXPECTED secret category must be detected by scan_secrets.
    let sec_atk = load("secrets_exfil/attacks.jsonl");
    let sec_ben = load("secrets_exfil/benign.jsonl");
    let sec_hits = sec_atk
        .iter()
        .filter(|r| {
            let got = scan_secrets(input(r));
            r["expect"]["secrets"]
                .as_array()
                .map(|exp| {
                    exp.iter()
                        .filter_map(|v| v.as_str())
                        .all(|cat| got.contains(&cat))
                })
                .unwrap_or(false)
        })
        .count();
    let sec_recall = pct(sec_hits, sec_atk.len());
    // precision: a benign record is a false positive if ANY secret is detected.
    let sec_fp = sec_ben
        .iter()
        .filter(|r| !scan_secrets(input(r)).is_empty())
        .count();
    let sec_precision = 1.0 - pct(sec_fp, sec_ben.len());

    // ---- report (visible with --nocapture; the nightly + perf-quality panels read this) ----
    eprintln!("security red-team eval (ADR-048 / OWASP-LLM-Top-10 corpus):");
    eprintln!(
        "  injection      recall={inj_recall:.3} precision={inj_precision:.3} (floors {INJECTION_RECALL_FLOOR}/{INJECTION_PRECISION_FLOOR})  [{}/{} attacks, {} benign FP]",
        inj_hits, attacks.len(), inj_fp
    );
    eprintln!(
        "  smuggling      recall={smg_recall:.3} precision={smg_precision:.3} (floors {SMUGGLING_RECALL_FLOOR}/{SMUGGLING_PRECISION_FLOOR})  [{}/{} attacks, {} benign FP]",
        smg_hits, smg_atk.len(), smg_fp
    );
    eprintln!(
        "  secrets_exfil  recall={sec_recall:.3} precision={sec_precision:.3} (floors {SECRETS_RECALL_FLOOR}/{SECRETS_PRECISION_FLOOR})  [{}/{} attacks, {} benign FP]",
        sec_hits, sec_atk.len(), sec_fp
    );

    // ---- gate (hard floors) ----
    assert!(
        inj_recall >= INJECTION_RECALL_FLOOR,
        "injection recall {inj_recall:.3} < floor {INJECTION_RECALL_FLOOR} — a scored attack is no longer caught; add it to injection/known_gaps.jsonl (with a `gap`) and open a finding, do NOT weaken the detector or the floor"
    );
    assert!(
        inj_precision >= INJECTION_PRECISION_FLOOR,
        "injection precision {inj_precision:.3} < floor {INJECTION_PRECISION_FLOOR} — a benign prompt is being flagged"
    );
    assert!(
        smg_recall >= SMUGGLING_RECALL_FLOOR,
        "smuggling recall {smg_recall:.3} < floor {SMUGGLING_RECALL_FLOOR}"
    );
    assert!(
        smg_precision >= SMUGGLING_PRECISION_FLOOR,
        "smuggling precision {smg_precision:.3} < floor {SMUGGLING_PRECISION_FLOOR}"
    );
    assert!(
        sec_recall >= SECRETS_RECALL_FLOOR,
        "secrets_exfil recall {sec_recall:.3} < floor {SECRETS_RECALL_FLOOR} — a credential would reach the provider/response in the clear (DLP leak)"
    );
    assert!(
        sec_precision >= SECRETS_PRECISION_FLOOR,
        "secrets_exfil precision {sec_precision:.3} < floor {SECRETS_PRECISION_FLOOR}"
    );
}

/// Guards the report-only contract: known-gap records exist, document WHY, and
/// are NOT also in the scored attacks (so a known miss can never red the gate,
/// nor silently vanish). Mirrors `guardrail_eval::known_gaps_are_excluded`.
#[test]
fn redteam_known_gaps_are_excluded_from_scoring() {
    let gaps = load("injection/known_gaps.jsonl");
    assert!(
        !gaps.is_empty(),
        "injection/known_gaps.jsonl should document the current deterministic-tier misses (the novel-paraphrase class routed off-path per ADR-031)"
    );
    let scored_ids: Vec<String> = load("injection/attacks.jsonl")
        .iter()
        .map(|r| id(r).to_string())
        .collect();
    for g in &gaps {
        let gid = id(g);
        assert!(
            !scored_ids.contains(&gid.to_string()),
            "known-gap {gid} must not also be in the scored attacks"
        );
        assert!(g.get("gap").is_some(), "known-gap {gid} must document why");
    }
}
