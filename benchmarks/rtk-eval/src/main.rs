//! RTK trace-replay eval — the launch credibility harness for the
//! "RTK token compression" headline (PRD-051 FR-6).
//!
//! What it does:
//! 1. Loads the committed corpus of coding-agent conversations
//!    (`corpus/traces.jsonl` — provenance in `corpus/README.md`).
//! 2. Applies `routeplane_rtk::compress` to every `tool`-role message,
//!    **exactly as the gateway hot path does** (`crates/routeplane/src/proxy.rs`
//!    applies `message.content.map_text(routeplane_rtk::compress)` when
//!    `Feature::TokenCompression` is active): string content is compressed
//!    whole; array-of-parts content has each `{"type":"text"}` part compressed.
//! 3. Counts tokens BEFORE and AFTER with two real tokenizers
//!    (tiktoken o200k_base — GPT-4o/o-series — and cl100k_base — GPT-4/3.5),
//!    plus raw bytes (what the crate natively optimizes).
//! 4. Writes `RESULTS.md` (per-trace, aggregate, p50/p90, per-filter) and
//!    prints the aggregate to stdout.
//!
//! Run from `benchmarks/`:
//! ```text
//! cargo run --release -p rtk-eval
//! ```
//! Optional args: `--corpus <path>` `--out <path>`.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Tokenizer provenance cited in RESULTS.md. Keep in lockstep with the exact
/// pin in `benchmarks/Cargo.toml` (`tiktoken-rs = "=0.12.0"`).
const TOKENIZER_CRATE: &str = "tiktoken-rs v0.12.0";

#[derive(Deserialize)]
struct Trace {
    id: String,
    category: String,
    source: String,
    #[serde(default)]
    #[allow(dead_code)] // corpus documentation field, not used in scoring
    description: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
}

/// Extract the text units of a message content value, mirroring
/// `routeplane_types::MessageContent`: a JSON string is one unit; an array of
/// parts contributes one unit per `{"type":"text","text":...}` part
/// (non-text parts — images etc. — are untouched by `map_text` and carry no
/// countable text).
fn text_units(content: &serde_json::Value) -> Vec<String> {
    match content {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Label for the per-filter breakdown. `detect_filter` is the same routine
/// `compress` uses internally to pick a filter.
fn filter_label(text: &str) -> &'static str {
    if text.len() < 256 {
        return "(below 256B floor — passthrough)";
    }
    match routeplane_rtk::detect_filter(text) {
        Some(routeplane_rtk::Filter::GitDiff) => "git-diff",
        Some(routeplane_rtk::Filter::GitStatus) => "git-status",
        Some(routeplane_rtk::Filter::Grep) => "grep",
        Some(routeplane_rtk::Filter::Find) => "find",
        Some(routeplane_rtk::Filter::Ls) => "ls",
        Some(routeplane_rtk::Filter::Tree) => "tree",
        Some(routeplane_rtk::Filter::DedupLog) => "dedup-log",
        Some(routeplane_rtk::Filter::SmartTruncate) => "smart-truncate",
        Some(routeplane_rtk::Filter::ReadNumbered) => "read-numbered",
        Some(routeplane_rtk::Filter::SearchList) => "search-list",
        Some(routeplane_rtk::Filter::BuildOutput) => "build-output",
        None => "(no filter detected — passthrough)",
    }
}

struct Tokenizers {
    o200k: tiktoken_rs::CoreBPE,
    cl100k: tiktoken_rs::CoreBPE,
}

impl Tokenizers {
    fn count(&self, text: &str) -> (usize, usize) {
        (
            self.o200k.encode_ordinary(text).len(),
            self.cl100k.encode_ordinary(text).len(),
        )
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    bytes_before: usize,
    bytes_after: usize,
    o200k_before: usize,
    o200k_after: usize,
    cl100k_before: usize,
    cl100k_after: usize,
    units: usize,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.bytes_before += other.bytes_before;
        self.bytes_after += other.bytes_after;
        self.o200k_before += other.o200k_before;
        self.o200k_after += other.o200k_after;
        self.cl100k_before += other.cl100k_before;
        self.cl100k_after += other.cl100k_after;
        self.units += other.units;
    }
}

fn reduction(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        100.0 * (before as f64 - after as f64) / before as f64
    }
}

/// Nearest-rank percentile (p in 0..=100) over an unsorted slice.
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in reductions"));
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

struct TraceResult {
    id: String,
    category: String,
    source: String,
    tool_msgs: usize,
    tool: Tally,
    conversation: Tally,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut corpus_path = manifest.join("corpus/traces.jsonl");
    let mut out_path = manifest.join("RESULTS.md");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => corpus_path = PathBuf::from(args.next().expect("--corpus needs a path")),
            "--out" => out_path = PathBuf::from(args.next().expect("--out needs a path")),
            other => panic!("unknown arg: {other} (supported: --corpus <path>, --out <path>)"),
        }
    }

    let raw = std::fs::read(&corpus_path)
        .unwrap_or_else(|e| panic!("cannot read corpus {}: {e}", corpus_path.display()));
    let corpus_sha = sha256_hex(&raw);
    let raw_text = String::from_utf8(raw).expect("corpus must be UTF-8");

    let tokenizers = Tokenizers {
        o200k: tiktoken_rs::o200k_base().expect("load o200k_base"),
        cl100k: tiktoken_rs::cl100k_base().expect("load cl100k_base"),
    };

    let mut traces: Vec<TraceResult> = Vec::new();
    let mut per_filter: BTreeMap<&'static str, Tally> = BTreeMap::new();

    for (lineno, line) in raw_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let trace: Trace = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {}: bad JSON: {e}", lineno + 1));

        let mut tool = Tally::default();
        let mut conversation = Tally::default();
        let mut tool_msgs = 0usize;

        for msg in &trace.messages {
            let units = text_units(&msg.content);
            if msg.role == "tool" {
                tool_msgs += 1;
                for text in &units {
                    // EXACTLY the hot-path transform: proxy.rs runs
                    // `content.map_text(routeplane_rtk::compress)` per tool message.
                    let compressed = routeplane_rtk::compress(text);
                    let (o_b, c_b) = tokenizers.count(text);
                    let (o_a, c_a) = tokenizers.count(&compressed);
                    let t = Tally {
                        bytes_before: text.len(),
                        bytes_after: compressed.len(),
                        o200k_before: o_b,
                        o200k_after: o_a,
                        cl100k_before: c_b,
                        cl100k_after: c_a,
                        units: 1,
                    };
                    tool.add(&t);
                    conversation.add(&t);
                    per_filter.entry(filter_label(text)).or_default().add(&t);
                }
            } else {
                for text in &units {
                    let (o, c) = tokenizers.count(text);
                    let t = Tally {
                        bytes_before: text.len(),
                        bytes_after: text.len(),
                        o200k_before: o,
                        o200k_after: o,
                        cl100k_before: c,
                        cl100k_after: c,
                        units: 1,
                    };
                    conversation.add(&t);
                }
            }
        }

        traces.push(TraceResult {
            id: trace.id,
            category: trace.category,
            source: trace.source,
            tool_msgs,
            tool,
            conversation,
        });
    }

    assert!(!traces.is_empty(), "corpus contained no traces");

    let mut total_tool = Tally::default();
    let mut total_conv = Tally::default();
    for t in &traces {
        total_tool.add(&t.tool);
        total_conv.add(&t.conversation);
    }

    let tool_reductions: Vec<f64> = traces
        .iter()
        .map(|t| reduction(t.tool.o200k_before, t.tool.o200k_after))
        .collect();
    let conv_reductions: Vec<f64> = traces
        .iter()
        .map(|t| reduction(t.conversation.o200k_before, t.conversation.o200k_after))
        .collect();

    let report = render_report(
        &traces,
        &per_filter,
        &total_tool,
        &total_conv,
        &tool_reductions,
        &conv_reductions,
        &corpus_sha,
    );
    std::fs::write(&out_path, &report)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out_path.display()));

    println!(
        "traces: {} | tool messages: {} ({} text units)",
        traces.len(),
        traces.iter().map(|t| t.tool_msgs).sum::<usize>(),
        total_tool.units,
    );
    println!(
        "tool-message input tokens  (o200k):  {} -> {}  ({:.1}% reduction)",
        total_tool.o200k_before,
        total_tool.o200k_after,
        reduction(total_tool.o200k_before, total_tool.o200k_after)
    );
    println!(
        "tool-message input tokens  (cl100k): {} -> {}  ({:.1}% reduction)",
        total_tool.cl100k_before,
        total_tool.cl100k_after,
        reduction(total_tool.cl100k_before, total_tool.cl100k_after)
    );
    println!(
        "whole-conversation tokens  (o200k):  {} -> {}  ({:.1}% reduction)",
        total_conv.o200k_before,
        total_conv.o200k_after,
        reduction(total_conv.o200k_before, total_conv.o200k_after)
    );
    println!(
        "bytes (tool messages):               {} -> {}  ({:.1}% reduction)",
        total_tool.bytes_before,
        total_tool.bytes_after,
        reduction(total_tool.bytes_before, total_tool.bytes_after)
    );
    println!(
        "per-trace tool reduction (o200k): p50 {:.1}%  p90 {:.1}%  min {:.1}%  max {:.1}%",
        percentile(&tool_reductions, 50.0),
        percentile(&tool_reductions, 90.0),
        tool_reductions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        tool_reductions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    println!("report written to {}", out_path.display());
}

#[allow(clippy::too_many_arguments)]
fn render_report(
    traces: &[TraceResult],
    per_filter: &BTreeMap<&'static str, Tally>,
    total_tool: &Tally,
    total_conv: &Tally,
    tool_reductions: &[f64],
    conv_reductions: &[f64],
    corpus_sha: &str,
) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# RTK token-compression eval — measured results\n");
    let _ = writeln!(
        md,
        "> Generated by `cargo run --release -p rtk-eval` (benchmarks workspace, \
         `benchmarks/rtk-eval/`). Do not edit by hand — re-run the command.\n"
    );

    let _ = writeln!(md, "## Methodology\n");
    let _ = writeln!(
        md,
        "- **What is measured**: the exact hot-path transform the gateway applies when \
         `Feature::TokenCompression` is active — `routeplane_rtk::compress` over every \
         `tool`-role message's text content (`crates/routeplane/src/proxy.rs`), including \
         its production fail-safes (inputs < 256 bytes skipped; output never empty, never \
         longer than the input; a filter result larger than 95% of the input is discarded \
         in favor of the original)."
    );
    let _ = writeln!(
        md,
        "- **Corpus**: `corpus/traces.jsonl` (sha256 `{corpus_sha}`), {} coding-agent \
         conversations. Provenance, identity scrub, and regeneration are documented in \
         [`corpus/README.md`](corpus/README.md) — real command outputs captured against the \
         Routeplane repository itself, plus a clearly-labeled synthetic bucket \
         (`source = \"synthetic-generator\"`).",
        traces.len()
    );
    let _ = writeln!(
        md,
        "- **Tokenizer**: {TOKENIZER_CRATE} (`encode_ordinary`), reported for **o200k_base** \
         (GPT-4o / o-series) and **cl100k_base** (GPT-4 / GPT-3.5). Raw bytes are also \
         reported — bytes are what the crate natively optimizes and are tokenizer-neutral."
    );
    let _ = writeln!(
        md,
        "- **Token accounting**: content-text tokens only. Per-message chat scaffolding \
         (role tags, tool_call envelopes) is constant before/after compression, so it is \
         excluded from both sides; including it would slightly dilute the whole-conversation \
         percentage but cannot change the absolute token savings."
    );
    let _ = writeln!(
        md,
        "- **Two honest denominators**: *tool-message reduction* (savings on the content RTK \
         actually touches) and *whole-conversation reduction* (the same savings diluted by \
         system/user/assistant text — what a full request body sees). The headline claim \
         should be read against the corpus mix; both numbers are below.\n"
    );

    let _ = writeln!(md, "## Aggregate (micro-average over the whole corpus)\n");
    let _ = writeln!(
        md,
        "| Metric | Before | After | Reduction |\n|---|---:|---:|---:|"
    );
    let rows: [(&str, usize, usize); 5] = [
        (
            "Tool-message tokens (o200k_base)",
            total_tool.o200k_before,
            total_tool.o200k_after,
        ),
        (
            "Tool-message tokens (cl100k_base)",
            total_tool.cl100k_before,
            total_tool.cl100k_after,
        ),
        (
            "Tool-message bytes",
            total_tool.bytes_before,
            total_tool.bytes_after,
        ),
        (
            "Whole-conversation tokens (o200k_base)",
            total_conv.o200k_before,
            total_conv.o200k_after,
        ),
        (
            "Whole-conversation tokens (cl100k_base)",
            total_conv.cl100k_before,
            total_conv.cl100k_after,
        ),
    ];
    for (label, before, after) in rows {
        let _ = writeln!(
            md,
            "| {label} | {before} | {after} | **{:.1}%** |",
            reduction(before, after)
        );
    }

    let _ = writeln!(md, "\n## Distribution across traces (o200k_base)\n");
    let _ = writeln!(
        md,
        "| Statistic | Tool-message reduction | Whole-conversation reduction |\n|---|---:|---:|"
    );
    for (label, p) in [("p50", 50.0), ("p90", 90.0)] {
        let _ = writeln!(
            md,
            "| {label} | {:.1}% | {:.1}% |",
            percentile(tool_reductions, p),
            percentile(conv_reductions, p)
        );
    }
    let _ = writeln!(
        md,
        "| min | {:.1}% | {:.1}% |",
        tool_reductions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        conv_reductions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    );
    let _ = writeln!(
        md,
        "| max | {:.1}% | {:.1}% |",
        tool_reductions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        conv_reductions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    let _ = writeln!(
        md,
        "\n## Per-filter breakdown (tool messages, o200k_base)\n"
    );
    let _ = writeln!(
        md,
        "Filter attribution via `routeplane_rtk::detect_filter` — the same routine \
         `compress` uses internally. Passthrough rows are counted honestly: they are \
         tool outputs RTK leaves untouched.\n"
    );
    let _ = writeln!(
        md,
        "| Detected filter | Text units | Tokens before | Tokens after | Reduction | Bytes before | Bytes after |\n|---|---:|---:|---:|---:|---:|---:|"
    );
    for (label, t) in per_filter {
        let _ = writeln!(
            md,
            "| {label} | {} | {} | {} | **{:.1}%** | {} | {} |",
            t.units,
            t.o200k_before,
            t.o200k_after,
            reduction(t.o200k_before, t.o200k_after),
            t.bytes_before,
            t.bytes_after,
        );
    }

    let _ = writeln!(
        md,
        "\n## Interpretation — read this before quoting a number\n"
    );
    let _ = writeln!(
        md,
        "- **RTK filters are deliberately lossy summarizers**, not entropy coding. A high \
         percentage means the filter kept a head/tail skeleton plus a summary line \
         (e.g. `read-numbered` keeps the first 10 + last 10 lines of a long file read; \
         `find` collapses deep listings to a pruned-items summary). Whether the agent \
         still completes its task on the compressed view is a product question, \
         **out of scope for this harness** — this harness measures exactly what the \
         gateway transform saves, no more."
    );
    let _ = writeln!(
        md,
        "- **The micro-average is dominated by large file reads** (see the per-filter \
         table: `read-numbered` carries most of the before-tokens). For a single \
         quotable figure prefer the **per-trace median** and the **mixed-session \
         trace** — a multi-round debug session lands near the low end, single large \
         reads near the high end."
    );
    let _ = writeln!(
        md,
        "- **Several real shapes get 0% by design**: outputs under the 256-byte floor, \
         insertion-only diffs with no context to strip, `tree` output already within \
         the kept depth, and unrecognized shapes all pass through via the fail-safes. \
         Those zeros are counted in every aggregate above — nothing is excluded."
    );
    let _ = writeln!(
        md,
        "- **Corpus mix drives the headline.** This corpus is tool_result-heavy by \
         construction (that is RTK's target workload). Conversations dominated by \
         human/assistant prose will see proportionally less: the whole-conversation \
         number converges to the tool-message number only when tool output dominates, \
         as it does in real coding-agent transcripts.\n"
    );

    let _ = writeln!(md, "\n## Per-trace results (o200k_base)\n");
    let _ = writeln!(
        md,
        "| Trace | Category | Source | Tool msgs | Tool tokens before | Tool tokens after | Tool reduction | Conversation reduction |\n|---|---|---|---:|---:|---:|---:|---:|"
    );
    for t in traces {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | **{:.1}%** | {:.1}% |",
            t.id,
            t.category,
            t.source,
            t.tool_msgs,
            t.tool.o200k_before,
            t.tool.o200k_after,
            reduction(t.tool.o200k_before, t.tool.o200k_after),
            reduction(t.conversation.o200k_before, t.conversation.o200k_after),
        );
    }

    let _ = writeln!(md, "\n## Reproduce\n");
    let _ = writeln!(md, "```bash");
    let _ = writeln!(md, "cd benchmarks");
    let _ = writeln!(md, "cargo run --release -p rtk-eval");
    let _ = writeln!(md, "# or: just eval");
    let _ = writeln!(md, "```");
    let _ = writeln!(
        md,
        "\nThe corpus is frozen and committed; the eval is deterministic (no network, no \
         randomness, no clock dependence), so re-running on the same corpus reproduces this \
         file byte-for-byte apart from nothing — there is no timestamp in this report by design."
    );

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_text() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/traces.jsonl");
        std::fs::read_to_string(path).expect("committed corpus must exist")
    }

    #[test]
    fn percentile_nearest_rank() {
        let v = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile(&v, 50.0), 20.0);
        assert_eq!(percentile(&v, 90.0), 40.0);
        assert_eq!(percentile(&v, 100.0), 40.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn text_units_mirror_message_content_semantics() {
        // String content = one unit.
        let s = serde_json::json!("hello world");
        assert_eq!(text_units(&s), vec!["hello world".to_string()]);
        // Parts content = one unit per text part; non-text parts skipped.
        let parts = serde_json::json!([
            {"type": "text", "text": "alpha"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
            {"type": "text", "text": "beta"}
        ]);
        assert_eq!(
            text_units(&parts),
            vec!["alpha".to_string(), "beta".to_string()]
        );
        // Null content (assistant tool_call messages) = no units.
        assert!(text_units(&serde_json::Value::Null).is_empty());
    }

    /// Identity gate: the committed corpus must contain zero personal names,
    /// zero usernames, zero email-like tokens. This is a launch requirement
    /// (identity-clean trace data), enforced in CI via `cargo test`.
    #[test]
    fn corpus_is_identity_clean() {
        let text = corpus_text();
        let lower = text.to_lowercase();
        // Generic patterns only — operator-identifying tokens live in the
        // gitignored corpus/denylist.local (or RTK_EVAL_DENYLIST env, comma-
        // separated) so the guard never republishes what it guards.
        let mut forbidden: Vec<String> = ["@gmail", "@outlook"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Ok(extra) = std::env::var("RTK_EVAL_DENYLIST") {
            forbidden.extend(
                extra
                    .split(',')
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty()),
            );
        }
        if let Ok(local) = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/corpus/denylist.local"
        )) {
            forbidden.extend(
                local
                    .lines()
                    .map(|l| l.trim().to_lowercase())
                    .filter(|l| !l.is_empty()),
            );
        }
        for forbidden in forbidden.iter().map(|s| s.as_str()) {
            assert!(
                !lower.contains(forbidden),
                "identity leak: corpus contains {forbidden:?}"
            );
        }
        // Email-like token scan: alnum '@' alnum with a dot later in the tail
        // (git diff hunk headers `@@` and lone '@' are fine).
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'@' && i > 0 && i + 1 < bytes.len() {
                let prev = bytes[i - 1] as char;
                let next = bytes[i + 1] as char;
                if prev.is_ascii_alphanumeric() && next.is_ascii_alphanumeric() {
                    let tail: String = text[i + 1..]
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '-')
                        .collect();
                    assert!(
                        !tail.contains('.'),
                        "email-like token in corpus around byte {i}: ...@{tail}"
                    );
                }
            }
        }
    }

    /// Structure gate: every line parses, every trace has tool payloads, and
    /// the corpus is big enough to mean something.
    #[test]
    fn corpus_parses_and_has_tool_payloads() {
        let text = corpus_text();
        let mut n = 0usize;
        let mut with_compressible_payload = 0usize;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let trace: Trace = serde_json::from_str(line).expect("every corpus line parses");
            assert!(!trace.id.is_empty());
            assert!(!trace.category.is_empty());
            assert!(
                trace.source == "real-command" || trace.source == "synthetic-generator",
                "trace {} has unknown source {:?}",
                trace.id,
                trace.source
            );
            let tool_units: Vec<String> = trace
                .messages
                .iter()
                .filter(|m| m.role == "tool")
                .flat_map(|m| text_units(&m.content))
                .collect();
            assert!(
                !tool_units.is_empty(),
                "trace {} has no tool text",
                trace.id
            );
            if tool_units.iter().any(|t| t.len() >= 256) {
                with_compressible_payload += 1;
            }
            n += 1;
        }
        assert!(n >= 15, "corpus too small to be meaningful: {n} traces");
        assert!(
            with_compressible_payload * 10 >= n * 8,
            "fewer than 80% of traces carry a >=256B tool payload ({with_compressible_payload}/{n})"
        );
    }
}
