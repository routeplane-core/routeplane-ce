//! Tenant-owned observability admission performance artifact.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use routeplane::observability::{bench_support::AdmissionHarness, UsageEvent};

const TENANT_COUNTS: [usize; 3] = [1, 8, 64];
const PERCENTILE_SAMPLES: usize = 20_000;
const PERF_BUDGETS: &str = include_str!("../../../.github/perf-budgets.toml");

fn event() -> UsageEvent {
    UsageEvent::success(
        "display_only".into(),
        "bench_key".into(),
        "openai".into(),
        "gpt-4o-mini".into(),
        16,
        8,
        24,
        None,
        false,
    )
}

fn admission_ceiling_ns() -> u64 {
    let mut in_section = false;
    for raw_line in PERF_BUDGETS.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.starts_with('[') {
            in_section = line == "[observability.admission]";
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
            .expect("observability admission ceiling_us must be an integer")
            * 1_000;
    }
    panic!("missing [observability.admission] ceiling_us in .github/perf-budgets.toml");
}

fn percentiles(mut samples: Vec<u64>) -> (u64, u64) {
    samples.sort_unstable();
    let p50 = samples[samples.len() * 50 / 100];
    let p99 = samples[samples.len() * 99 / 100];
    (p50, p99)
}

fn measure_admitted(tenant_count: usize) -> (u64, u64) {
    let runtime = tokio::runtime::Runtime::new().expect("benchmark runtime");
    let _guard = runtime.enter();
    let harness = AdmissionHarness::new(tenant_count);
    let tenant_index = tenant_count - 1;
    let mut samples = Vec::with_capacity(PERCENTILE_SAMPLES);
    for _ in 0..PERCENTILE_SAMPLES {
        harness.release_tenant(tenant_index);
        let started = Instant::now();
        assert!(harness.record(black_box(tenant_index), event()));
        samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    percentiles(samples)
}

fn measure_full_share_drop(tenant_count: usize) -> (u64, u64) {
    let runtime = tokio::runtime::Runtime::new().expect("benchmark runtime");
    let _guard = runtime.enter();
    let harness = AdmissionHarness::new(tenant_count);
    let tenant_index = tenant_count - 1;
    harness.saturate_tenant(tenant_index);
    let before = harness.dropped_capacity();
    let mut samples = Vec::with_capacity(PERCENTILE_SAMPLES);
    for _ in 0..PERCENTILE_SAMPLES {
        let started = Instant::now();
        assert!(!harness.record(black_box(tenant_index), event()));
        samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    assert_eq!(
        harness.dropped_capacity() - before,
        u64::try_from(PERCENTILE_SAMPLES).expect("sample count fits u64")
    );
    percentiles(samples)
}

fn artifact_path() -> PathBuf {
    std::env::var_os("RP_OBSERVABILITY_PERF_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let target = std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("target"));
            target
                .join("criterion")
                .join("observability-admission-metrics.json")
        })
}

fn measure_and_enforce_percentiles() {
    let ceiling_ns = admission_ceiling_ns();
    let mut artifact = serde_json::Map::new();
    let mut failures = Vec::new();
    for tenant_count in TENANT_COUNTS {
        for (path, (p50_ns, p99_ns)) in [
            ("admitted", measure_admitted(tenant_count)),
            ("full_share_drop", measure_full_share_drop(tenant_count)),
        ] {
            let name = format!("observability_admission_{path}_{tenant_count}_tenants");
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
        std::fs::create_dir_all(parent).expect("create observability perf artifact directory");
    }
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&artifact).expect("serialize observability perf artifact"),
    )
    .expect("write observability perf artifact");
    assert!(
        failures.is_empty(),
        "observability admission performance budget failed: {}",
        failures.join("; ")
    );
}

fn main() {
    measure_and_enforce_percentiles();
}
