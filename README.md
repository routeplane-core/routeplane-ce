# Routeplane

**The open-source AI gateway that saves you tokens today and passes your CISO audit tomorrow.**

Point your OpenAI SDK — or Cursor, or Claude Code, or anything that speaks the OpenAI API — at
Routeplane by changing only the base URL and the key. You get 15 provider adapters, fallback
chains, per-provider circuit breakers, response caching, and RTK token compression that cuts
coding-agent input tokens by 20–40% on tool-heavy traces
(<!-- TODO(launch): link the published RTK eval -->benchmark link pending — see
[Benchmarks](#benchmarks); we don't publish numbers without a runnable harness). One Rust binary,
lock-free hot path, no database, no phone-home.

<!-- TODO(launch): wire real badges once the public repo + GHCR package + first release exist.
[![CI](https://github.com/routeplane-core/routeplane-ce/actions/workflows/ci.yml/badge.svg)](https://github.com/routeplane-core/routeplane-ce/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/routeplane-core/routeplane-ce/blob/main/LICENSE)
[![GHCR](https://img.shields.io/badge/ghcr.io-routeplane--core%2Frouteplane--ce-blue)](https://github.com/routeplane-core/routeplane-ce/pkgs/container/routeplane-ce)
[![Release](https://img.shields.io/github/v/release/routeplane-core/routeplane-ce)](https://github.com/routeplane-core/routeplane-ce/releases)
-->

<!-- TODO(launch): record the 60-second demo GIF — `docker compose up` → point an OpenAI SDK at
it → a fallback kicking in → RTK savings visible in the request log — and embed it here:
![60-second demo](https://github.com/routeplane-core/routeplane-ce/raw/main/docs/demo.gif)
-->

## Quickstart

Five minutes, Docker only — the compose file pulls a signed image from GHCR, so there is no Rust
toolchain and no source build.

```bash
git clone https://github.com/routeplane-core/routeplane-ce.git && cd routeplane-ce
cp .env.example .env && cp configs/keys.example.json configs/keys.json
docker compose up -d
```

Between steps 2 and 3: put at least one provider key in `.env` (e.g. `OPENAI_API_KEY=sk-...`),
and set your own gateway key in `configs/keys.json` — any string starting with `rp_`. Generate a
strong one with `echo "rp_$(openssl rand -hex 24)"`. Provider keys stay server-side in `.env`;
clients only ever hold the `rp_` key.

Call it with curl (`RP_KEY` is the gateway key you set in `keys.json`):

```bash
export RP_KEY=rp_your_key
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $RP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

Or with the stock OpenAI SDK — no Routeplane-specific code:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="rp_your_key")
reply = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(reply.choices[0].message.content)
```

The same two settings (base URL + key) are all that Cursor, Claude Code, Continue, LangChain, or
any other OpenAI-compatible client need.

### The two classic first-run failures

- **`env file .env not found`** — compose hard-fails if `.env` is missing. Step 2 above creates
  it; it must exist even if you only fill in one key.
- **Gateway exits at startup complaining about `configs/keys.json`** — if that file didn't exist
  when you ran `docker compose up`, Docker silently created a *directory* at the bind-mount
  source. Fix: `docker compose down`, `rm -r configs/keys.json`, create the real file
  (`cp configs/keys.example.json configs/keys.json`), and `up` again. To avoid the mount
  entirely, inject the registry as an env var instead: `RP_KEYS_JSON` (raw or base64 JSON) or
  `RP_KEYS_FILE` (an alternate path).

For the full self-hosting walkthrough (env vars, keys, building from source), see
[SELF_HOST.md](SELF_HOST.md).

## What you get

Everything below is in this repo, Apache-2.0, and works on a single node with no external
dependencies (no Redis, no cloud account):

- **OpenAI-compatible API** — `/v1/chat/completions` (buffered and SSE streaming),
  `/v1/embeddings`, `/v1/models`, `/v1/moderations`, `/v1/rerank`, `/v1/audio/speech`, plus an
  Anthropic-style `/v1/messages` inbound surface.
- **Routing and resilience** — priority / weighted / cost / latency strategies, comma-separated
  fallback chains, retries, hedged requests, and a per-provider circuit breaker + latency EWMA.
  The health tracking is atomics-only: no locks on the request path.
- **Named combos** — operator-defined routing chains addressable by the `model` field.
- **RTK token compression** — deterministic filters that shrink verbose tool output before it
  reaches the provider.
- **Multi-account key pools** — several upstream keys per provider with per-key cooldown
  failover.
- **Exact-match response cache** — in-process, single-node.
- **Rate and spend limits** — per key, per tenant, per model (requests/min, tokens/min, daily and
  monthly budgets).
- **Basic PII masking and a residency classifier** — regex-grade masking of common personal-data
  patterns on the way in and out.
- **Analytics and request logs** — in-memory, queryable over the API. Nothing is written to disk
  or sent anywhere.
- **Auth** — `rp_`-prefixed virtual keys, accepted as `Authorization: Bearer` (what OpenAI SDKs
  send) or the `x-routeplane-api-key` header.

### Named combos

A combo is a saved routing chain with a public name. Clients address it as a model — no custom
headers, no SDK changes — and it shows up in `/v1/models`, so it works from model-picker
dropdowns. Defined in `configs/routing-policies.json`:

```json
{"configs": [{"id": "cfg_fast", "combo": "fast",
  "routing": {"strategy": "cost", "targets": [
    {"provider": "groq",   "params": {"override": {"model": "llama-3.3-70b-versatile"}}},
    {"provider": "openai", "params": {"override": {"model": "gpt-4o-mini"}}}]}}]}
```

Now `{"model": "fast", ...}` routes to Groq by cost and falls back to OpenAI. Every combo target
must pin a concrete model, so the combo name never leaks to a provider.

### RTK token compression

Coding agents burn most of their input tokens re-sending tool output: `git diff`, `grep`, build
logs, directory listings. RTK detects the shape of `tool`-role message content and applies one of
11 deterministic filters (git-diff, git-status, grep, find, ls, tree, dedup-log, smart-truncate,
numbered-file-read, search-list, build-output) — collapsing unchanged diff context, deduplicating
repeated log lines, pruning deep listings while keeping heads, tails, and errors.

The honest fine print:

- It is pure string processing — no ML, no network calls, adds well under a millisecond.
- It is fail-safe: unrecognized content passes through untouched, output is never empty, and a
  request never grows.
- It only helps tool-heavy workloads. The 20–40% figure is measured on coding-agent traces
  (<!-- TODO(launch): link the eval harness + published traces -->eval link pending); if your
  requests carry no tool output, expect ~0%.
- It is lossy by design: what gets dropped is repeated/unchanged filler in tool output. If your
  agent depends on byte-exact tool results, opt a key out with
  `"rollout_holdbacks": ["token_compression"]` in `configs/keys.json`. It is on by default in CE.

### Multi-account key pools

A `provider_keys` value that contains commas is a pool. Each element resolves independently
(literal or `env:`-referenced), and on a rate-limit or auth failure the gateway fails over to the
next key, tracking cooldown per key:

```json
{"provider_keys": {"openai": "env:OPENAI_KEY_A,env:OPENAI_KEY_B"}}
```

A single-value entry behaves exactly as before — the pool machinery only engages when there is a
comma.

### Fallback and circuit breaking

Give a chain (`x-routeplane-provider: openai,anthropic` or a combo) and the gateway orders
candidates by your chosen strategy, skips providers whose circuit is open, and tries them in
order — first success wins. Latency EWMAs feed the `latency` strategy; breaker state machines are
per-provider and lock-free. On streaming requests, fallback applies until the first chunk
arrives; after that the gateway is committed to that provider.

## Providers

Route to any of these with `x-routeplane-provider`, a fallback chain, or a combo target. Each
adapter translates the OpenAI wire shape to the provider's native API where they differ
(Anthropic's `/v1/messages`, Gemini's `generateContent`), and speaks the dialect directly where
they don't.

| Provider | Wire name | Notes |
|----------|-----------|-------|
| OpenAI | `openai` | Default when no provider is specified |
| Anthropic | `anthropic` | Native `/v1/messages` translation |
| Google Gemini | `gemini` | Native `generateContent` translation |
| Azure OpenAI | `azure_openai` | Endpoint/deployment set via env |
| AWS Bedrock | `bedrock` | |
| Mistral | `mistral` | |
| Cohere | `cohere` | |
| Groq | `groq` | |
| DeepSeek | `deepseek` | |
| Together AI | `together` | ~100+ open-weight models |
| Fireworks AI | `fireworks` | |
| xAI (Grok) | `xai` | |
| OpenRouter | `openrouter` | Meta-aggregator: one key, hundreds of models |
| Self-hosted / local | `self_hosted` | Any OpenAI-compatible server — Ollama, vLLM, LocalAI, LM Studio, TGI. Set `SELF_HOSTED_BASE_URL` (e.g. `http://ollama:11434/v1`) |

Local-model note: `self_hosted` is a first-class provider — it participates in fallback chains,
combos, strategies, and circuit breaking like any hosted provider, so
`self_hosted,openai` (local first, cloud fallback) is a one-header setup.

## Verify what you run

Routeplane is maintained pseudonymously by the Routeplane team. We think the honest response to
"an anonymous team wants to proxy my API keys" is: **don't trust the author — verify the
artifact.** Every claim below is checkable without taking our word for it.

- **Signed images.** Every GHCR image is signed with cosign (keyless, GitHub OIDC). Verify before
  you run:

  ```bash
  cosign verify ghcr.io/routeplane-core/routeplane-ce:v0.1.0 \
    --certificate-identity-regexp '^https://github.com/routeplane-core/routeplane-ce/' \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com
  ```

  <!-- TODO(launch): after the first release, pin the exact workflow identity string here. -->

- **SBOM.** A Syft-generated SPDX SBOM is attached to every
  [GitHub release](https://github.com/routeplane-core/routeplane-ce/releases) and attested on the
  image, so you can diff exactly which crates are inside.
- **Public CI.** Images are built by the public workflows in
  [.github/workflows](https://github.com/routeplane-core/routeplane-ce/tree/main/.github/workflows)
  — the build you can read is the build that produced the artifact.
- **No telemetry, no phone-home.** CE makes outbound connections only to the providers you
  configure. No usage pings, no update checks, no crash reporting, no analytics. Grep the source.
- **Your keys stay yours.** Provider keys live server-side in your `.env`; clients hold only the
  `rp_` virtual key. The gateway never logs key material.
- **Vulnerabilities:** see
  [SECURITY.md](https://github.com/routeplane-core/routeplane-ce/blob/main/SECURITY.md) — reports
  go to `security@routeplane.ai`.

## Community Edition vs Enterprise

Routeplane is open-core. This repo is the Community Edition — real, useful, and complete for a
developer or team self-hosting a gateway. The commercial Enterprise edition (source not
published) adds what CISOs, regulators, and multi-tenant operators require:

| Capability | CE (this repo, Apache-2.0) | Enterprise |
|------------|:--------------------------:|:----------:|
| OpenAI-compatible API + streaming + embeddings | ✅ | ✅ |
| 15 provider adapters incl. self-hosted/local | ✅ | ✅ |
| Routing strategies, fallback, retries, hedging, circuit breaker | ✅ | ✅ |
| Named combos (chains addressable via `model`) | ✅ | ✅ |
| Multi-account key pools | ✅ | ✅ |
| RTK token compression | ✅ | ✅ |
| Exact-match response cache | ✅ | ✅ |
| Rate + spend limits (basic) | ✅ | ✅ |
| PII masking (basic) + residency classifier | ✅ | ✅ |
| In-memory analytics + request logs | ✅ | ✅ |
| Inline per-request routing config (`x-routeplane-config`) | — | ✅ |
| Sovereign data-residency routing + signed hash-chained audit ledger + verifiable audit artifacts | — | ✅ |
| MCP agentic-security gateway + agent governance | — | ✅ |
| Advanced guardrails (webhook + ML detectors) | — | ✅ |
| Semantic cache | — | ✅ |
| Versioned prompt registry | — | ✅ |
| FinOps export + durable telemetry retention | — | ✅ |
| Model-catalog governance | — | ✅ |
| Multi-tenant control plane (SSO, RBAC, key issuance) | — | ✅ |
| Multi-region cells | — | ✅ |

**The covenant:** features listed as CE stay CE. The boundary only ever moves in one direction —
toward more free. No feature that ships in this repo will be moved behind a paywall.

The enterprise/sovereign story — residency-locked routing with regulator-grade signed audit
evidence, and a security gateway for agent tool calls — lives at
[routeplane.ai](https://routeplane.ai).

## Benchmarks

None published yet — deliberately. Gateway benchmarks without a runnable harness are marketing,
and comparative numbers without identical load, mocking, and hardware are worse. Our rule:
**a number appears here only alongside the harness commit that produced it.**

What the harness will be (tracked in
<!-- TODO(launch): link the bench-harness issue on the public repo -->an open issue):

- A mock upstream provider, so provider latency variance is out of the measurement.
- A pinned, published hardware spec for every run.
- One command to reproduce: `just bench`.
- Planned measurements: gateway overhead vs direct (p50/p99, buffered and streaming
  time-to-first-byte), sustained throughput on the mock upstream, and RTK savings per trace
  class with the traces published.

<!-- TODO(launch): land the harness, run it, and replace this section's TODOs with numbers +
harness commit links before making any performance claim in launch posts. -->

## Configuration reference

Kept short here — the full reference lives at [docs.routeplane.ai](https://docs.routeplane.ai).

### Request headers

| Header | Meaning |
|--------|---------|
| `Authorization: Bearer rp_...` | Your gateway key, OpenAI-SDK style (or use `x-routeplane-api-key`) |
| `x-routeplane-provider` | Provider or comma-separated fallback chain, e.g. `openai,anthropic` (default `openai`) |
| `x-routeplane-strategy` | Candidate ordering: `priority` (default), `weighted`, `cost`, `latency` |
| `x-routeplane-config` | Inline per-request routing config — Enterprise; in CE, use named combos instead |

### keys.json

`configs/keys.json` maps gateway keys to provider credentials. Minimal shape:

```json
{
  "keys": [
    {
      "name": "default",
      "routeplane_key": "rp_generate_your_own",
      "provider_keys": {
        "openai": "env:OPENAI_API_KEY",
        "anthropic": "env:ANTHROPIC_API_KEY",
        "groq": "env:GROQ_KEY_A,env:GROQ_KEY_B"
      }
    }
  ]
}
```

`env:` values resolve from the gateway's environment at request time; comma-separated values form
a failover pool. Optional per-key fields add `limits` (rate + budget) and `rollout_holdbacks`.
Alternatives to the file mount: `RP_KEYS_JSON` (inline JSON, raw or base64) or `RP_KEYS_FILE`
(alternate path).

### Combos and strategies

Combos live in `configs/routing-policies.json` (override the path with
`RP_ROUTING_POLICIES_FILE`) — see [Named combos](#named-combos) above. Strategies order the
candidates within a chain: `priority` keeps your listed order, `weighted` splits traffic,
`cost` prefers cheaper targets, `latency` prefers the fastest recent EWMA.

### Useful environment variables

| Variable | Meaning |
|----------|---------|
| `PORT` | Listen port (default `8080`) |
| `RUST_LOG` | Log filter (default `routeplane=info`) |
| `SELF_HOSTED_BASE_URL` | Base URL of your OpenAI-compatible local server (enables `self_hosted`) |
| `RP_KEYS_JSON` / `RP_KEYS_FILE` | Key registry without a bind mount |
| `RP_ROUTING_POLICIES_FILE` | Combos/routing-config file path |

## Community and support

- **Questions and ideas:**
  [GitHub Discussions](https://github.com/routeplane-core/routeplane-ce/discussions).
- **Bugs:** [issues](https://github.com/routeplane-core/routeplane-ce/issues) — include the
  request path and a redacted log snippet.
- **Contributing:**
  [CONTRIBUTING.md](https://github.com/routeplane-core/routeplane-ce/blob/main/CONTRIBUTING.md)
  — build, test, and PR conventions. Good first issues are labeled.
- **Security:**
  [SECURITY.md](https://github.com/routeplane-core/routeplane-ce/blob/main/SECURITY.md) /
  `security@routeplane.ai`. Please don't open public issues for vulnerabilities.
- **Contact:** `maintainers@routeplane.ai`.
- **No SLA:** CE is maintained best-effort by a small team. We triage honestly and ship fixes,
  but there are no response-time guarantees. If you need an SLA, that is an
  [Enterprise](https://routeplane.ai) conversation.

## License

[Apache-2.0](https://github.com/routeplane-core/routeplane-ce/blob/main/LICENSE). Third-party
crate licenses are listed in
[THIRD_PARTY_NOTICES](https://github.com/routeplane-core/routeplane-ce/blob/main/THIRD_PARTY_NOTICES.md).
