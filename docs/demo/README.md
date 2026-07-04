# The demo GIF, reproducibly

Everything in [`../demo.gif`](../demo.gif) is a real terminal session against this repo's
`docker compose up -d` and a local Ollama model (`qwen2.5:0.5b`) — no cloud keys, no mocks,
no edited waits. Token counts come from the model server's own `usage` field.

Recreate it:

```bash
# from the repo root, with the compose stack up and Ollama serving qwen2.5:0.5b
export SELF_HOSTED_BASE_URL=http://<your-ollama-host>:11434   # in .env, then docker compose up -d
python3 docs/demo/mk-payload.py            # builds rtk-demo.json (a ~15 KB tool_result turn)
docs/demo/ask.sh rp_your_key               # RTK on (CE default) -> small prompt_tokens
docs/demo/ask.sh rp_your_no_rtk_key        # a key with {"rollout_holdbacks":["token_compression"]}
```

The two keys live in `configs/keys.json`; the no-RTK key just adds
`"rollout_holdbacks": ["token_compression"]`. Numbers vary with the file you read and the
model's tokenizer — that's the point: measure your own.
