# Routeplane benchmarks

Runnable, reproduce-it-yourself measurement harnesses backing Routeplane's
published performance and token-savings claims. Honesty rule: **the collateral
adjusts to the measured numbers, never the reverse** — if a harness run comes
out below a published claim, the claim changes.

This is a **standalone Cargo workspace**, deliberately excluded from the
product workspace so bench-only dependencies (tokenizers, load-test plumbing)
never enter the product `Cargo.lock` or SBOM. Everything below runs from this
directory; nothing here builds the product workspace (the only product code
compiled is the leaf crate under test, `../crates/rtk`).

| Harness | What it measures | Status |
|---|---|---|
| [`rtk-eval/`](rtk-eval/) | Real input-token reduction of RTK tool_result compression over a committed coding-agent trace corpus, with a real tokenizer | **Measured** — see [`rtk-eval/RESULTS.md`](rtk-eval/RESULTS.md) |
| [`perf/`](perf/) | Gateway added latency/throughput overhead vs a zero-work mock upstream | **Measured** — see [`perf/RESULTS.md`](perf/RESULTS.md) ([methodology](perf/README.md)) |

## Quick start

```bash
cd benchmarks
just eval          # regenerate rtk-eval/RESULTS.md from the committed corpus
just check         # fmt + clippy + tests (includes the corpus hygiene gates)
just mock          # start the perf mock upstream
```

(No `just`? The recipes are one-liner `cargo`/shell commands — see `justfile`.)

## Design notes

- `rtk-eval` replays the corpus through `routeplane_rtk::compress` **exactly
  as the gateway hot path applies it** (`tool`-role messages only, same
  fail-safes), then counts tokens before/after with tiktoken
  (o200k_base + cl100k_base) — the gateway itself reports byte savings, so the
  token claim needs a real tokenizer, which lives here and not in the product.
- The corpus is committed, frozen, identity-clean and licensing-clean
  (first-party command output only); provenance and the regeneration
  procedure are in [`rtk-eval/corpus/README.md`](rtk-eval/corpus/README.md).
  CI-enforced tests re-validate corpus hygiene on every run.
- `perf/` publishes methodology before numbers: pinned oha invocation,
  hardware disclosure template, and an explicit statement of what is and is
  not measured.
