# Gateway perf harness — pinned methodology

Reproducible latency/throughput measurement of the Routeplane gateway's
**added overhead**. This directory ships the runnable harness and the frozen
methodology; **no numbers are published from a contended machine** — the
official numbers ship with the release run on quiet hardware (see
"Publishing numbers" below).

## What IS and IS NOT measured

- **Measured**: the gateway's own cost — auth lookup, guardrails, residency
  classification, routing/health bookkeeping, request/response translation,
  and HTTP hop — expressed as the delta between (B) and (A) below.
- **NOT measured**: model inference. The upstream is `mock-upstream`
  (`perf/mock-upstream/`), a deliberately zero-work axum binary that consumes
  the request body and returns a static, plausible chat completion with a
  `usage` object. Any latency it adds is measured by leg (A) and subtracted
  by construction.
- **NOT measured (yet)**: streaming (SSE) legs, and RTK-on vs RTK-off deltas
  under load. Both are follow-ups; the buffered path is the baseline.

Two legs, same load shape:

- **(A) floor**: load generator → `mock-upstream` directly.
- **(B) through-gateway**: load generator → gateway → `mock-upstream`
  (gateway `self_hosted` provider pointed at the mock).

Report `B − A` per percentile (p50/p90/p99) and the RPS both legs sustain.

## Load generator

[`oha`](https://crates.io/crates/oha) — installable without sudo:

```bash
cargo install oha --locked
```

`run.sh` refuses to start if `oha` is absent. wrk/vegeta are fine
alternatives but the committed invocation and any published numbers use oha
so runs are comparable.

## Pinned run parameters (do not tune per-run)

| Parameter | Value | Why |
|---|---|---|
| Warmup | 15 s at the measurement connection count, discarded | JIT-free but caches/pools warm |
| Duration | 60 s per leg | long enough for stable percentiles |
| Connections | 32 and 256 (two rows) | moderate + saturating concurrency |
| Payload | `payload.json` (committed) | fixed ~1 KB chat request, non-streaming |
| Repeats | 3 per leg, report the median run | flags noisy-neighbor runs |

## Hardware disclosure template (fill in for every published run)

```text
CPU:            (model, cores/threads, base/boost clocks)
Memory:         (size, speed)
OS/kernel:      (e.g. Ubuntu 24.04, 6.8.x; note if WSL2/VM — VM numbers are
                 directional only, publish from bare metal or a dedicated VM)
Rust toolchain: (rustc -V; gateway built with cargo build --release)
Topology:       all three processes on one host over loopback (yes/no)
Isolation:      machine otherwise idle (yes/no); governor = performance
Gateway config: features active on the test key (RTK on/off, guardrails, …)
```

## Running

```bash
# 1. Mock upstream (terminal 1)
cd benchmarks && cargo run --release -p mock-upstream            # :9100

# 2. Gateway (terminal 2, from the repo root) — point self_hosted at the mock
export SELF_HOSTED_BASE_URL=http://127.0.0.1:9100
# keys.json: copy configs/keys.example.json, set an rp_ key and add
#   "self_hosted": "sk-local-perf"        to its provider_keys
# (the literal is never validated by the mock; the entry just has to resolve).
# To measure the RTK-on leg, also give the key
#   "capability_overrides": ["token_compression"]
cargo run --release -p routeplane                                 # :8080

# 3. Load (terminal 3)
cd benchmarks/perf
./run.sh floor                       # leg A: oha -> mock directly
./run.sh gateway rp_YOUR_TEST_KEY    # leg B: oha -> gateway -> mock
```

Requests go to `POST /v1/chat/completions` with
`x-routeplane-api-key: <key>` and `x-routeplane-provider: self_hosted`.

## Publishing numbers

This harness intentionally ships **without** results: the machine this
skeleton was authored on was running concurrent build pipelines, and numbers
measured under contention are noise. The release-run procedure is: quiet
dedicated host → fill the hardware template → 3×60 s per leg per connection
count → commit `RESULTS.md` here alongside the raw oha output. Until that
file exists in this directory, no Routeplane perf number is citable.
