# The console demo GIF, reproducibly

Everything in [`../console-demo.gif`](../console-demo.gif) is a real headless-browser session
against this repo's gateway serving its bundled console, with a local Ollama model
(`qwen2.5:0.5b`) behind a **custom provider added at runtime through the UI** — no cloud
keys, no mocks, no edited frames. The haiku is whatever the model streams that day.

The beats, in order:

1. First boot — the console's sign-in screen; create the operator account (open signup,
   self-host bootstrap).
2. **Provider Integrations → Add provider**: `ollama-local`, base URL
   `http://localhost:11434`, one model — the toast confirms it's usable immediately,
   no restart.
3. **Playground**: pick the model the provider just added, stream a completion through the
   live gateway.
4. **API Keys**: reveal the gateway `rp_` key the session authorizes as.
5. **Usage & Analytics**: the request shows up in the in-memory analytics ring.

Recreate it:

```bash
# 1. Build both halves (from the repo root)
cargo build --release
(cd dashboard && npm ci && npm run build)

# 2. Serve a local model (any OpenAI-compatible server works)
ollama pull qwen2.5:0.5b

# 3. Boot the gateway + console with clean demo state
docs/demo-console/run.sh

# 4. Record (Playwright drives a headless Chromium and captures video)
npm i playwright && npx playwright install chromium
node docs/demo-console/walkthrough.js       # writes video/<hash>.webm, ~57s

# 5. Convert to the README GIF (two-pass palette, 960px, 10fps)
ffmpeg -i video/*.webm -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" docs/console-demo.gif
```

Notes for re-recording:

- `run.sh` resets console accounts + custom providers each run, so the recording always
  starts from first boot. The `rp_ce_demo_…` key it registers is a throwaway that appears
  on screen by design — keep it obviously fake.
- The walkthrough injects a fake cursor (a blue dot following `mousemove`) because captured
  video doesn't render the OS pointer; without it the GIF is hard to follow.
- The SPA preserves scroll position across routes — the script resets `window.scrollTo(0)`
  at each beat so every page starts framed from the top.
- Loopback custom providers require `RP_CUSTOM_PROVIDER_ALLOW_PRIVATE=on` (set by
  `run.sh`); link-local/metadata addresses are refused regardless.
- Override the defaults with `DEMO_BASE`, `DEMO_OLLAMA`, `DEMO_MODEL` env vars on
  `walkthrough.js`.
