# Routeplane CE vs. the alternatives

> Maintained by the Routeplane team. Every fact below was last verified against each
> project's public repository and documentation on **2026-07-04**. These projects ship
> fast and this page will drift — if anything here is wrong or stale,
> [open an issue](https://github.com/routeplane-core/routeplane-ce/issues) and we'll
> correct it. We would rather fix a cell than win an argument.

The rules this page follows:

- **Community Edition only.** We compare this repo (Apache-2.0) — features that live in
  our commercial edition count as *absent*, exactly as we count a competitor's
  hosted-only features.
- **Concessions first.** Every competitor section leads with what they do better than
  us, by name.
- **No side-by-side benchmark numbers.** Each vendor measures overhead differently; we
  link everyone's published numbers and methodology instead of pretending they are
  comparable. See [the performance note](#a-note-on-performance-comparisons).

## TL;DR — pick by what you actually need

- **You want the broadest provider catalog, a mature admin UI, and the largest
  community** → [LiteLLM](https://github.com/BerriAI/litellm). It is the de facto OSS
  standard, especially in Python shops.
- **You want an MCP gateway, semantic caching, and a built-in web UI in the OSS
  product** → [Bifrost](https://github.com/maximhq/bifrost). Fast-moving, strong
  developer ergonomics.
- **You want an experimentation / optimization flywheel (A/B testing, fine-tuning
  recipes, evals) around your gateway** → [TensorZero](https://github.com/tensorzero/tensorzero)
  is the most ambitious design here — but note its repository was archived on
  2026-06-12 (no further updates or security patches as of this writing).
- **You are already on the Portkey hosted platform** →
  [Portkey Gateway](https://github.com/Portkey-AI/gateway) integrates with it natively.
- **You want a tiny single static binary with no external dependencies, measured
  (worst-case, reproducible) overhead, and deterministic tool-output token compression
  for coding agents** → that is Routeplane CE's corner, and this repo is the argument.

## The field at a glance (verified 2026-07-04)

| Project | Language / shape | License | Activity (as of 2026-07-04) |
|---------|------------------|---------|------------------------------|
| **Routeplane CE** | Rust · single static binary, no DB/Redis | Apache-2.0 | Active (launched July 2026 — we are the new one here) |
| [LiteLLM](https://github.com/BerriAI/litellm) | Python (FastAPI) · proxy + Postgres + Redis for full features | MIT core + commercial `enterprise/` dir | Active, daily commits |
| [Bifrost](https://github.com/maximhq/bifrost) | Go · binary + web UI, SQLite by default | Apache-2.0 | Active, daily commits |
| [Portkey Gateway](https://github.com/Portkey-AI/gateway) | TypeScript/Node | MIT | No commits since 2026-05-25, following Portkey's acquisition by Palo Alto Networks (closed late May 2026); the README announces a Gateway 2.0 pre-release branch that merges enterprise features into open source — worth watching |
| [TensorZero](https://github.com/tensorzero/tensorzero) | Rust · gateway + ClickHouse (+ Postgres) | Apache-2.0 | **Repository archived 2026-06-12** — read-only, no updates or security patches |

## Capability table

Cells describe the **self-hosted OSS offering only**, from public docs as of 2026-07-04.
"Hosted" means the capability exists but requires the vendor's paid/hosted platform.
Corrections welcome — file an issue.

| Capability | Routeplane CE | LiteLLM | Bifrost | Portkey Gateway | TensorZero |
|------------|:-:|:-:|:-:|:-:|:-:|
| Providers (each project's own count) | 15 adapters + any OpenAI-compatible local server | [100+](https://docs.litellm.ai/docs/providers) | [23+](https://github.com/maximhq/bifrost) | [250+ LLMs](https://github.com/Portkey-AI/gateway) | 19 native |
| OpenAI-compatible inbound API | ✅ (+ Anthropic-style `/v1/messages`) | ✅ | ✅ | ✅ | ✅ |
| Runs with zero external dependencies (no DB / no Redis) | ✅ | ➖ (Postgres + Redis for the full proxy) | ◐ (SQLite default) | ✅ (stateless Node) | ➖ (ClickHouse required) |
| Fallback chains + circuit breaking | ✅ | ✅ | ✅ | ✅ | ◐ (fallbacks + retries; no circuit breaker documented) |
| Cost- and latency-aware routing strategies | ✅ | ✅ | ✅ | ◐ (config-based conditional routing) | ✅ |
| Multi-account key pools with cooldown failover | ✅ | ✅ | ✅ | ✅ | ◐ (not documented as a first-class feature) |
| Tool-output token compression (RTK-class), measured + reproducible | ✅ ([measured](benchmarks/rtk-eval/RESULTS.md)) | ➖ | ➖ | ➖ | ➖ |
| Exact-match response cache | ✅ | ✅ | ✅ | Hosted | ✅ |
| Semantic cache | ➖ (our Enterprise) | ✅ | ✅ | Hosted (paid tier) | ➖ |
| MCP gateway / agent tooling | ➖ (our Enterprise) | ✅ | ✅ (first-class, incl. code mode) | ◐ (adjacent tooling; management surface enterprise-leaning) | ➖ |
| Web dashboard / admin UI | ➖ (API + logs + metrics) | ✅ | ✅ | Hosted | ✅ (UI over ClickHouse) |
| Budgets / spend limits | ✅ basic (per key/tenant/model) | ✅ deep (key/user/team, DB-backed) | ✅ hierarchical | Hosted | ◐ (per-key rate limits) |
| Guardrails / PII handling | ◐ basic masking | ◐ framework + partner integrations | ◐ (enterprise-gated) | ◐ (deterministic checks in OSS; LLM checks hosted) | ➖ |
| Experimentation / evals / fine-tuning loop | ➖ | ◐ | ◐ (via Maxim platform) | Hosted | ✅ (its real moat) |
| Gateway-overhead benchmark published with committed harness + raw output | ✅ ([RESULTS.md](benchmarks/perf/RESULTS.md)) | ◐ (docs publish load-test figures) | ✅ (self-published) | ➖ | ✅ (self-published) |
| No telemetry / no phone-home | ✅ (grep the source) | ◐ (a `telemetry = True` default flag exists in its source; disableable) | not audited by us | not audited by us | not audited by us |
| Community size & production track record | ➖ brand-new (be honest: this is our weakest row) | ✅ largest by far | ✅ growing fast | ✅ large | ◐ (archived) |

Legend: ✅ ships in OSS · ◐ partial / caveat in the cell · ➖ absent. Where we were not
certain of a competitor cell we chose the reading more favorable to them; if we still got
one wrong, tell us and we'll fix it.

## What each alternative does better than Routeplane CE

### LiteLLM

- **Provider breadth nobody matches** — 100+ providers/models against our 15 adapters.
  If you need an exotic provider today, LiteLLM almost certainly has it and we may not.
- **The largest community and ecosystem in the category**, with years of production use
  at large companies. We launched in July 2026; they have battle scars we don't.
- **A full management platform in OSS**: admin web UI, database-backed virtual keys,
  teams, per-user budgets, semantic caching, and 20+ observability integrations.
  CE gives you a config file, an API, and Prometheus metrics.
- **MCP gateway support in the open-source product.** Ours is in the commercial edition.
- **Python extensibility** — custom hooks/plugins in the language your ML team already
  writes.

Trade-off you accept: a Python service with Postgres + Redis behind it versus a single
static binary, and gateway overhead you should measure on your own load (see the
performance note).

### Bifrost

- **MCP support in OSS is first-class** (multiple transports, agent + code modes) — the
  best in this table. Again: ours is enterprise-gated.
- **Semantic caching in OSS** across multiple vector backends.
- **Built-in web UI** and a genuinely excellent zero-config start (`npx @maximhq/bifrost`).
  Our quickstart is three commands and a `.env`; theirs is one.
- **Hierarchical budget governance** (key → team → customer) in OSS; CE's limits are
  flat per key/tenant/model.
- **Self-published performance work** with real methodology behind it.

Trade-off you accept: a Go runtime (GC — mitigated, not eliminated) and a heavier
service; we bet on Rust's deterministic tails and a ~14 MiB binary instead. Both
projects are single-vendor-backed — evaluate bus factor for either.

### Portkey Gateway

- **Very wide model coverage** (its README says 250+ LLMs / 1,600+ models) and years of
  production traffic ("10B+ tokens/day" per its README).
- **A mature hosted platform** — logs, analytics, prompt studio, guardrail partners — if
  a managed control plane is what you want (that's a different product category than CE).
- **Deterministic guardrail checks run in the OSS gateway** with a well-documented
  config DSL (conditional routing, canary weights).
- **Gateway 2.0 (pre-release)** promises to merge enterprise features into open source —
  if that lands, this column improves materially.

Trade-off you accept: the OSS repo has been quiet since 2026-05-25 (post-acquisition;
as of this writing), and the observability/caching story assumes the hosted platform.

### TensorZero

- **The optimization flywheel is genuinely unique**: built-in A/B testing with bandits,
  fine-tuning recipes, dynamic in-context learning, evals — no other project in this
  table has it, including us. It is an LLMOps platform, not just a gateway.
- **An embedded in-process gateway mode** (no HTTP hop at all) that nothing else here
  offers.
- **Rust, like us, with strong self-published performance numbers** (<1ms p99 claimed).
- **ClickHouse-grade analytics depth** far beyond CE's in-memory ring buffer.

The hard caveat: the repository was **archived on 2026-06-12** (as of this writing) —
no commits, no issue responses, no security patches. Impressive engineering; we note the
archival because production users need patches, not out of glee. If it un-archives,
re-read this section.

## What Routeplane CE bets on

Stated as bets, not victories — you should check each one against your workload:

1. **One small static binary, no external dependencies.** No Postgres, no Redis, no
   ClickHouse, no Node runtime: 13.6 MiB binary, 36 MiB compressed image, 24.5 MiB idle
   RSS ([measured](benchmarks/perf/RESULTS.md)).
2. **Deterministic tool-output compression (RTK) built into the gateway.** Coding
   agents re-send tool output every turn; RTK strips the redundant parts with 11
   deterministic filters — ~76% tool-message token reduction on our committed corpus,
   56% per-trace median, ~31% on a mixed session
   ([methodology + per-trace breakdown](benchmarks/rtk-eval/RESULTS.md)). Nothing else
   in this table does this in the gateway.
3. **Worst-case-honest performance publishing.** Our overhead numbers were measured at
   saturation with default logging on, raw load-generator output committed
   ([RESULTS.md](benchmarks/perf/RESULTS.md)) — and we don't quote competitor numbers
   next to ours, because methodologies differ.
4. **A lock-free hot path in Rust** — circuit breakers and latency tracking are
   atomics, no mutexes, no GC. Predictable tails are the point.
5. **Verifiable, not trusted:** cosign-signed images, SPDX SBOM on every release,
   public CI, zero telemetry ([verify what you run](README.md#verify-what-you-run)).
6. **An upgrade path that doesn't rewrite clients.** When you need residency-locked
   routing, signed audit evidence, or an MCP security gateway, the same wire surface
   upgrades to [Enterprise](https://routeplane.ai) — and
   [the covenant](README.md#community-edition-vs-enterprise) guarantees nothing in CE
   is ever paywalled retroactively.

If none of those bets matter to you, one of the projects above is probably the better
choice — genuinely.

## A note on performance comparisons

Every gateway in this table publishes its own overhead numbers: ours
([+1.16 ms p50 / +1.63 ms p99 at saturation, methodology and raw output committed](benchmarks/perf/RESULTS.md)),
Bifrost's ([README/benchmarks](https://github.com/maximhq/bifrost)), TensorZero's
([<1ms p99 claim](https://github.com/tensorzero/tensorzero)), Portkey's
([<1ms claim in its README](https://github.com/Portkey-AI/gateway)), LiteLLM's
([published load tests](https://docs.litellm.ai/docs/proxy/load_test)). **These numbers
are not comparable to each other** — different hardware, different load profiles,
different definitions of "overhead", different observability settings during the run.

We have not yet run an identical-conditions side-by-side (same box, same mock upstream,
same load profile). Our harness is built for exactly that and is one command per
gateway; when we publish a side-by-side, it will be with pinned hardware and the raw
output committed, and this page will link it. Until then: run
[`benchmarks/perf`](benchmarks/perf/) against your own candidates on your own load —
that beats any vendor table, including this one.

## Projects not covered here

Different category, not lesser projects: **OpenRouter** (hosted meta-aggregator — CE
fronts it as a provider), **Cloudflare AI Gateway** (edge/hosted), **Helicone**
(observability-first), **Kong / Apache APISIX / Envoy AI Gateway / agentgateway**
(API-management and Kubernetes-native platforms — if you already run their stack,
start there), and **vLLM / Ollama** (inference servers, not gateways — CE routes *to*
them via the `self_hosted` adapter).
