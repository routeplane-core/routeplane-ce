//! Exact-cache tenant-admission performance artifact.
//!
//! Measures the real immutable tenant-map lookup, cache-entry construction,
//! tenant-local bounded lane `try_send`, drop counter, and ready-tenant signal
//! for 1/8/64 configured tenants. Both an admitted write and a saturated-lane
//! drop are covered. A deterministic percentile sampler writes the closed JSON
//! artifact and fails when any p99 exceeds `[cache.admission]`.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use routeplane_cache::bench_support::AdmissionHarness;

const TENANT_COUNTS: [usize; 3] = [1, 8, 64];
const PERCENTILE_SAMPLES: usize = 20_000;
const PERF_BUDGETS: &str = include_str!("../../../.github/perf-budgets.toml");

fn admission_ceiling_ns() -> u64 {
    let mut in_section = false;
    for raw_line in PERF_BUDGETS.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.starts_with('[') {
            in_section = line == "[cache.admission]";
            continue;
        }
        if !in_section {
            continue;
        }
        let Some(value) = line.strip_prefix("ceiling_us =") else {
            continue;
        };
        return value
            .trim()
            .parse::<u64>()
            .expect("cache admission ceiling_us must be an integer")
            * 1_000;
    }
    panic!("missing [cache.admission] ceiling_us in .github/perf-budgets.toml");
}

fn percentiles(mut samples: Vec<u64>) -> (u64, u64) {
    samples.sort_unstable();
    let p50 = samples[samples.len() * 50 / 100];
    let p99 = samples[samples.len() * 99 / 100];
    (p50, p99)
}

fn measure_admitted(tenant_count: usize) -> (u64, u64) {
    let harness = AdmissionHarness::new(tenant_count, 16);
    let handle = harness.handle();
    let tenant_index = tenant_count - 1;
    let mut samples = Vec::with_capacity(PERCENTILE_SAMPLES);
    for _ in 0..PERCENTILE_SAMPLES {
        harness.release_tenant(tenant_index);
        let started = Instant::now();
        assert!(handle.record(black_box(tenant_index)));
        samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    percentiles(samples)
}

fn measure_saturated_drop(tenant_count: usize) -> (u64, u64) {
    let mut harness = AdmissionHarness::new_paused(tenant_count, 1);
    let handle = harness.handle();
    let tenant_index = tenant_count - 1;
    harness.saturate_lane(tenant_index);
    let mut samples = Vec::with_capacity(PERCENTILE_SAMPLES);
    for _ in 0..PERCENTILE_SAMPLES {
        let started = Instant::now();
        assert!(!handle.record(black_box(tenant_index)));
        samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    assert_eq!(
        harness.dropped_total(),
        u64::try_from(PERCENTILE_SAMPLES).expect("sample count must fit in u64")
    );
    percentiles(samples)
}

fn artifact_path() -> PathBuf {
    std::env::var_os("RP_CACHE_PERF_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let target = std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("target"));
            target
                .join("criterion")
                .join("cache-admission-metrics.json")
        })
}

fn measure_and_enforce_percentiles() {
    let ceiling_ns = admission_ceiling_ns();
    let mut artifact = serde_json::Map::new();
    let mut failures = Vec::new();
    for tenant_count in TENANT_COUNTS {
        for (path, (p50_ns, p99_ns)) in [
            ("admitted", measure_admitted(tenant_count)),
            ("saturated_drop", measure_saturated_drop(tenant_count)),
        ] {
            let name = format!("cache_admission_{path}_{tenant_count}_tenants");
            let within_budget = p99_ns <= ceiling_ns;
            artifact.insert(
                name.clone(),
                serde_json::json!({
                    "p50_ns": p50_ns,
                    "p99_ns": p99_ns,
                    "ceiling_ns": ceiling_ns,
                    "within_budget": within_budget,
                }),
            );
            if !within_budget {
                failures.push(format!("{name}: p99={p99_ns}ns > {ceiling_ns}ns"));
            }
        }
    }

    let path = artifact_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create cache perf artifact directory");
    }
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&artifact).expect("serialize cache perf artifact"),
    )
    .expect("write cache perf artifact");
    assert!(
        failures.is_empty(),
        "cache admission performance budget failed: {}",
        failures.join("; ")
    );
}

fn bench_admission(c: &mut Criterion) {
    measure_and_enforce_percentiles();
    let mut group = c.benchmark_group("cache_admission");
    for tenant_count in TENANT_COUNTS {
        let admitted = AdmissionHarness::new(tenant_count, 16);
        let admitted_handle = admitted.handle();
        let tenant_index = tenant_count - 1;
        group.bench_with_input(
            BenchmarkId::new("admitted", tenant_count),
            &tenant_count,
            |b, _| {
                b.iter_batched(
                    || admitted.release_tenant(tenant_index),
                    |()| assert!(admitted_handle.record(black_box(tenant_index))),
                    BatchSize::PerIteration,
                );
            },
        );

        let mut dropped = AdmissionHarness::new_paused(tenant_count, 1);
        let dropped_handle = dropped.handle();
        dropped.saturate_lane(tenant_index);
        group.bench_with_input(
            BenchmarkId::new("saturated_drop", tenant_count),
            &tenant_count,
            |b, _| {
                b.iter_batched(
                    || (),
                    |()| assert!(!dropped_handle.record(black_box(tenant_index))),
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_admission);
criterion_main!(benches);
