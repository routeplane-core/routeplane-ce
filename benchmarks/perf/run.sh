#!/usr/bin/env bash
# Pinned oha invocation for the gateway perf harness — see README.md for the
# full methodology. Two legs:
#   ./run.sh floor                      # leg A: oha -> mock-upstream directly
#   ./run.sh gateway <rp_api_key>       # leg B: oha -> gateway -> mock-upstream
#
# Parameters are PINNED (README "do not tune per-run"): 15s discarded warmup,
# 60s measurement, connection counts 32 and 256, committed ~1KB payload.
# Env overrides (for smoke-testing the harness only, never for published
# numbers): MOCK_URL, GATEWAY_URL, DURATION, WARMUP, CONNS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$HERE/payload.json"

MOCK_URL="${MOCK_URL:-http://127.0.0.1:9100}"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8080}"
DURATION="${DURATION:-60s}"
WARMUP="${WARMUP:-15s}"
CONNS="${CONNS:-32 256}"

command -v oha >/dev/null 2>&1 || {
  echo "oha not found — install with: cargo install oha --locked" >&2
  exit 1
}

leg="${1:-}"
case "$leg" in
  floor)
    url="$MOCK_URL/v1/chat/completions"
    headers=()
    ;;
  gateway)
    key="${2:-}"
    [ -n "$key" ] || { echo "usage: ./run.sh gateway <rp_api_key>" >&2; exit 1; }
    url="$GATEWAY_URL/v1/chat/completions"
    headers=(-H "x-routeplane-api-key: $key" -H "x-routeplane-provider: self_hosted")
    ;;
  *)
    echo "usage: ./run.sh floor | ./run.sh gateway <rp_api_key>" >&2
    exit 1
    ;;
esac

echo "== leg: $leg  url: $url  duration: $DURATION  warmup: $WARMUP =="
for c in $CONNS; do
  echo "-- warmup ($WARMUP, $c conns, results discarded) --"
  oha --no-tui -z "$WARMUP" -c "$c" -m POST \
    -H 'Content-Type: application/json' "${headers[@]}" \
    -D "$PAYLOAD" "$url" >/dev/null
  echo "-- measure ($DURATION, $c conns) --"
  oha --no-tui -z "$DURATION" -c "$c" -m POST \
    -H 'Content-Type: application/json' "${headers[@]}" \
    -D "$PAYLOAD" "$url"
done

echo
echo "NOTE: run each leg 3x on a quiet machine and report the median run;"
echo "published numbers require the hardware disclosure in README.md."
