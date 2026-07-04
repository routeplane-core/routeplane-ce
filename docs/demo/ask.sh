#!/usr/bin/env bash
# ask.sh <gateway-key> — send rtk-demo.json (a coding-agent turn with a big
# tool_result) through the gateway to my local Ollama; print the token usage.
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $1" \
  -H "x-routeplane-provider: self_hosted" \
  -H "Content-Type: application/json" \
  -d @rtk-demo.json | jq .usage
