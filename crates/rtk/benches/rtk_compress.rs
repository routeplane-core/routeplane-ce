//! ADR-085 perf artifact: a runnable p50/p99 microbench for [`routeplane_rtk::compress`]
//! — the per-`tool_result` cost the synchronous chat-completions path pays when
//! `Feature::TokenCompression` is active. It exercises the common filters
//! (git-diff, build-output, and the generic smart-truncate fallback) at realistic
//! sizes.
//!
//! Mirrors `crates/ledger`'s convention: this is a RUNNABLE MEASUREMENT, not an
//! asserted gate — no p99 threshold is asserted in any test (criterion prints the
//! median/percentiles). Wiring a p99 ceiling into CI as an automated release gate
//! is a tracked follow-up. Run:
//!
//! ```text
//! cargo bench -p routeplane-rtk
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use routeplane_rtk::compress;

/// A `git diff` with `n` unchanged context lines and one real change — the
/// GitDiff filter collapses the context.
fn git_diff(n: usize) -> String {
    let mut s = String::from(
        "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1,200 +1,5 @@\n",
    );
    for i in 0..n {
        s.push_str(&format!(" context line {i}\n"));
    }
    s.push_str("+the one new line\n");
    s
}

/// A cargo build log: `n` `Compiling …` lines plus a trailing error — the
/// BuildOutput filter collapses the compile spam, keeping the error.
fn build_output(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!("   Compiling crate_{i} v1.0.0\n"));
    }
    s.push_str("error[E0308]: mismatched types\n  --> src/main.rs:5:10\n");
    s
}

/// Generic verbose output (`n` numbered lines) — the smart-truncate fallback
/// keeps the head/tail and omits the middle.
fn generic_log(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!("log line number {i} doing some work\n"));
    }
    s
}

fn bench_compress(c: &mut Criterion) {
    let cases = [
        ("git_diff_200ctx", git_diff(200)),
        ("build_output_400", build_output(400)),
        ("generic_log_800", generic_log(800)),
    ];
    for (name, payload) in &cases {
        c.bench_function(&format!("compress/{name}"), |b| {
            b.iter(|| compress(black_box(payload)))
        });
    }
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
