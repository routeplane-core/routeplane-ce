#!/usr/bin/env bash
# Boot the CE gateway with the bundled console for the demo recording — from a
# source checkout, repo-relative. Assumes:
#   * `cargo build --release` has produced target/release/routeplane
#   * `npm run build` has produced dashboard/dist
#   * Ollama is serving on :11434 (any OpenAI-compatible server works)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

WORK=${DEMO_WORKDIR:-/tmp/rp-console-demo}
mkdir -p "$WORK/configs"
# A throwaway single-key registry — the value below appears on screen in the
# recording, so keep it obviously fake.
cat > "$WORK/configs/keys.json" <<'EOF'
{
  "keys": [
    {
      "name": "CE Demo Key",
      "routeplane_key": "rp_ce_demo_2f8a1c9d4e6b0357",
      "provider_keys": {}
    }
  ]
}
EOF

# Clean state every run: the recording starts from first-boot (no accounts, no
# custom providers).
rm -f "$WORK/configs/console-accounts.json" "$WORK/configs/providers.json"

BIN="$PWD/target/release/routeplane"
CONSOLE_DIR="$PWD/dashboard/dist"
cd "$WORK"
env -i HOME="$HOME" PATH="$PATH" \
  PORT=8080 \
  RP_CONSOLE_DIR="$CONSOLE_DIR" \
  RP_CONSOLE_SESSION_SECRET=ce-console-demo-secret-0123456789abcdef \
  RP_CONSOLE_ACCOUNTS_FILE=configs/console-accounts.json \
  RP_CUSTOM_PROVIDER_ALLOW_PRIVATE=on \
  RUST_LOG=routeplane=info \
  "$BIN" > gateway.log 2>&1 &
sleep 2
curl -s -o /dev/null -w "healthz=%{http_code}\n" http://localhost:8080/healthz
echo "console: http://localhost:8080  (log: $WORK/gateway.log — stop with: pkill -x routeplane)"
