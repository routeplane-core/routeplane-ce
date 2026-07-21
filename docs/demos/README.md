# Developer-workflow demo recordings

The four GIFs embedded in the main README's **SDKs, CLI & MCP Server** section are real
sessions against a real gateway. Everything needed to reproduce or re-record them lives
in this directory.

| GIF | Tape / source | Shows |
|-----|---------------|-------|
| `docs/cli-demo.gif` | `cli-demo.tape` | `rp init` → streamed `rp chat` with the provider/model/token footer → `rp models list` |
| `docs/python-sdk-demo.gif` | `python-demo.tape` + `demo.py` | `create_with_meta()` returning the completion plus gateway metadata |
| `docs/ts-sdk-demo.gif` | `ts-demo.tape` + `demo.mjs` | `@routeplane/sdk/core` streaming token-by-token over SSE |
| `docs/mcp-demo.gif` | `mcp-demo.tape` + `mcp-demo.mcp.json` | Claude Code calling `get_status` and `chat_completion` through the MCP server |

## Re-recording

1. Install [vhs](https://github.com/charmbracelet/vhs) (plus `ttyd` and `ffmpeg`).
2. Start a local gateway with one provider key (any OpenAI-compatible provider works;
   the recordings used Groq's `llama-3.1-8b-instant` for its speed):

   ```bash
   cat > keys.json <<'EOF'
   {"keys":[{"name":"local demo","routeplane_key":"rp_local_demo_2f8a1c","tenant_id":"t_local_demo","tier":"standard","provider_keys":{"groq":"env:GROQ_API_KEY"}}]}
   EOF
   docker run -d -p 8080:8080 \
     -e GROQ_API_KEY=<your-groq-key> \
     -e RP_KEYS_JSON=$(base64 -w0 keys.json) \
     ghcr.io/routeplane-core/routeplane-ce:latest
   ```

   (`podman run` works identically if you prefer a daemonless runtime.)
3. Rehearse the commands in the tape once by hand so caches are warm and timings match,
   then render: `vhs <name>.tape`.

The `rp_local_demo_2f8a1c` key shown on screen is a throwaway that only exists in the
local `keys.json` above — never put a real provider key anywhere a recording can see it.
