# Gateway perf harness — measured results (release run)

> Produced by the pinned procedure in [README.md](README.md) — 15 s discarded warmup,
> 60 s measurement, 32 and 256 connections, 3 repeats per leg, median run reported
> (selected by requests/sec). Raw oha output for every rep is committed under
> [`results/2026-07-04-azure-d8s-v5/`](results/2026-07-04-azure-d8s-v5/). Do not edit
> the numbers by hand — re-run the harness.

**Benched commit:** `55a49cb` (= the `v0.1.0-rc.1` image) · **Load generator:** oha 1.14.0
· **Run date:** 2026-07-04

## Hardware disclosure

```text
CPU:            Intel Xeon Platinum 8370C @ 2.80 GHz (Ice Lake), 8 vCPU (4 cores x 2 threads)
                Azure Standard_D8s_v5 (dedicated vCPU, not burstable), ephemeral VM created for this run
Memory:         32 GiB
OS/kernel:      Ubuntu 24.04.4 LTS, kernel 6.17.0-1018-azure (dedicated cloud VM)
Rust toolchain: rustc 1.88.0 (6b00bc388 2025-06-23); gateway built with cargo build --release
Topology:       all three processes on one host over loopback: yes (oha + gateway + mock-upstream)
Isolation:      machine otherwise idle: yes; governor: N/A (Azure exposes no cpufreq control)
Gateway config: one virtual key; self_hosted -> mock-upstream; RTK off, no capability overrides
                (default CE feature set); default logging (routeplane=info) writing per-request
                lines to a file — that cost is INCLUDED in the gateway-leg numbers
```

## Results (median run of 3; all latencies in ms)

### 32 connections (moderate concurrency)

| Leg | Requests/sec | p50 | p90 | p99 | Success |
|---|---:|---:|---:|---:|---:|
| (A) floor: oha → mock | 173,210 | 0.15 | 0.32 | 0.67 | 100% |
| (B) gateway: oha → gateway → mock | 24,172 | 1.3 | 1.8 | 2.3 | 100% |
| **B − A (gateway overhead)** | — | **+1.16** | **+1.48** | **+1.63** | — |

Rep spread (requests/sec): floor 171,155 / 173,210 / 176,922 · gateway 24,002 / 24,172 / 24,643.

### 256 connections (saturating)

| Leg | Requests/sec | p50 | p90 | p99 | Success |
|---|---:|---:|---:|---:|---:|
| (A) floor: oha → mock | 258,976 | 0.83 | 1.68 | 3.59 | 100% |
| (B) gateway: oha → gateway → mock | 24,045 | 10.3 | 15.8 | 21.0 | 100% |
| **B − A (at saturation)** | — | +9.5 | +14.1 | +17.4 | — |

Rep spread (requests/sec): floor 257,734 / 258,976 / 261,090 · gateway 23,221 / 24,045 / 24,480.

## Footprint (same host, same run)

| Metric | Value |
|---|---|
| Gateway RSS, idle (post-boot, pre-load) | 24.5 MiB |
| Gateway RSS under load (median / max across 151 samples) | 88.3 / 89.6 MiB |
| Gateway CPU during the gateway legs (median / max) | 565% / 577% of 800% (8 vCPU) |
| Release binary size | 13.6 MiB |
| Container image, compressed (GHCR, v0.1.0-rc.1, all layers) | 36.0 MiB |

## Read the numbers honestly

- **The citable per-request overhead is the 32-connection row: ~1.2 ms p50 / ~1.6 ms p99
  added** over a zero-work upstream, with default per-request logging on.
- **The gateway saturates at ~24k req/s on this host at both connection counts** (it is
  CPU-bound; oha and the mock share the same 8 vCPU). The 256-connection deltas are
  therefore **queueing at the throughput ceiling**, not per-request compute: by Little's
  law 256 in-flight ÷ 24k req/s ≈ 10.6 ms — which is the measured p50. Every server does
  this at saturation; we publish the row because the methodology pins it.
- oha is closed-loop, so the two legs run at different throughputs by construction; B − A
  compares latency shapes, not equal-rate latencies.
- **Outliers:** each gateway 256-conn measurement contains a handful of ~1.06–1.10 s
  requests (32 of 1.45 M in the worst rep, ≤0.003%), consistent with SYN retransmission
  (1 s kernel retry) on the fresh connection ramp at measure start; p99.99 stays ≤ 30 ms.
- The floor leg measures oha + loopback + the deliberately zero-work mock; any mock cost
  is subtracted by construction.
- Colocated topology means the loadgen steals CPU from the gateway; on a dedicated
  loadgen host the saturation ceiling would be higher. Directionally conservative.
- Per-request logging wrote ~6.8 GB during the three gateway legs; that I/O is part of
  the measured overhead because it is the shipped default.

## Reproduce

```bash
# on a quiet host (this run: Azure Standard_D8s_v5, Ubuntu 24.04)
git clone https://github.com/routeplane-core/routeplane-ce && cd routeplane-ce
cargo build --release -p routeplane
(cd benchmarks && cargo build --release -p mock-upstream && cargo install oha --locked)
# start mock (:9100) and gateway (:8080, SELF_HOSTED_BASE_URL=http://127.0.0.1:9100), then:
cd benchmarks/perf
./run.sh floor      # x3
./run.sh gateway rp_YOUR_KEY   # x3
```
