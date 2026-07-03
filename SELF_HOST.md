# Self-hosting Routeplane (Community Edition)

Run the OpenAI-compatible AI gateway on your own box with one command. No Azure,
no database, no cloud account — a single Rust binary in front of 15 LLM providers.

Requires Docker with Compose **>= 2.24**.

## Quickstart

```bash
# 1. Provider secrets
cp .env.example .env
#    edit .env → add OPENAI_API_KEY (and/or ANTHROPIC_API_KEY, GEMINI_API_KEY)

# 2. Your gateway virtual key(s)
cp configs/keys.example.json configs/keys.json
#    edit configs/keys.json → replace the rp_..._REPLACE_ME keys with your own

# 3. Up — pulls the prebuilt CE image (ghcr.io/routeplane-core/routeplane-ce)
docker compose up
```

The gateway serves on **http://localhost:8080**.

Prefer building from source? `docker compose up --build` compiles the CE stage
of the Dockerfile locally (a few minutes on first build); subsequent `up` is
instant.

The published image is `linux/amd64`. On Apple Silicon, Docker Desktop runs it
under Rosetta emulation — functional, just slower; use `--build` for a native
arm64 binary.

## Call it

Point any OpenAI SDK at the gateway — zero code change beyond the base URL. Use
your `rp_` key (from `configs/keys.json`) as the `x-routeplane-api-key` header, or
as an `Authorization: Bearer` token (what OpenAI SDKs send automatically).

```bash
export ROUTEPLANE_KEY="rp_..."   # your rp_ key from configs/keys.json

curl http://localhost:8080/v1/chat/completions \
  -H "x-routeplane-api-key: $ROUTEPLANE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}'
```

```python
import os
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key=os.environ["ROUTEPLANE_KEY"])
client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"hi"}])
```

- `GET /v1/models` lists the providers and your named **combos**.
- Address a combo (an ordered fallback chain) by putting its name in `model`.
- `GET /healthz` is an unauthenticated liveness probe.

## What's in CE

OpenAI-compatible API + streaming · 15 providers (incl. Ollama/vLLM via the
OpenAI-compatible adapter) · routing strategies + circuit-breaker + fallback +
combos · RTK tool-output token compression · exact response cache · rate/spend
limits · usage analytics + logs · single-tenant `rp_` / Bearer auth.

Enterprise-only (not in CE): sovereign data-residency routing, the hash-chained
audit ledger, the MCP agentic-security gateway, semantic cache, and the
multi-tenant control plane (SSO/RBAC, key issuance, entitlements).

## Notes

- Single-node uses in-process rate-limits + cache — no Redis required.
- Provider keys live only in `.env` (server-side); never commit `.env` or
  `configs/keys.json`.
- The CE binary is distributed under Apache-2.0; the image ships `LICENSE` and
  `THIRD_PARTY_NOTICES.md` under `/usr/local/share/doc/routeplane/`.
