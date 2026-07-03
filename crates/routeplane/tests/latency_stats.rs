//! Gateway-overhead latency stats (plan Phase 0 — the perf-stats deliverable).
//!
//! Drives the real `chat_completions` handler against an IN-PROCESS stub
//! provider (no network), so the measurement is pure gateway + handler
//! overhead. Prints P50/P90/P99/mean and gates against a regression ceiling.
//!
//! NOTE on the ceiling: integration tests run in a DEBUG build, which is much
//! slower than the release binary the <5 ms P99 SLO (ADR-001/023) applies to.
//! So this gate is a debug-tolerant regression guard; the strict release <5 ms
//! gate is a separate criterion bench (follow-up). Run with --nocapture to see
//! the numbers:
//!     cargo test -p routeplane --test latency_stats -- --nocapture

mod common;

use common::{build_stub_state, drive_buffered, percentile};
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gateway_overhead_stats() {
    let state = build_stub_state();

    // Warmup (let allocators / caches settle).
    for _ in 0..200 {
        drive_buffered(&state).await;
    }

    let iters = 2000usize;
    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        drive_buffered(&state).await;
        samples.push(t.elapsed());
    }

    let mean = samples.iter().sum::<Duration>() / (samples.len() as u32);
    let p50 = percentile(&mut samples, 50.0);
    let p90 = percentile(&mut samples, 90.0);
    let p99 = percentile(&mut samples, 99.0);
    let max = *samples.last().unwrap();

    println!(
        "gateway-overhead (debug build, in-process stub, n={iters}): \
         mean={:.3}ms p50={:.3}ms p90={:.3}ms p99={:.3}ms max={:.3}ms",
        mean.as_secs_f64() * 1e3,
        p50.as_secs_f64() * 1e3,
        p90.as_secs_f64() * 1e3,
        p99.as_secs_f64() * 1e3,
        max.as_secs_f64() * 1e3,
    );

    // Debug-tolerant regression ceiling. The strict release <5 ms P99 SLO is a
    // separate criterion bench; here we just catch gross regressions.
    let ceiling = Duration::from_millis(25);
    assert!(
        p99 < ceiling,
        "gateway-overhead P99 {:.3}ms exceeded debug ceiling {:?} — investigate a hot-path regression",
        p99.as_secs_f64() * 1e3,
        ceiling
    );
}
