#!/usr/bin/env python3
"""Build rtk-demo.json: one coding-agent turn whose tool_result is a real
`cat -n` file read — the shape RTK's read-numbered filter compresses."""
import json, subprocess, sys

target = sys.argv[1] if len(sys.argv) > 1 else "crates/routeplane/src/proxy.rs"
numbered = subprocess.run(["cat", "-n", target], capture_output=True, text=True).stdout
lines = numbered.splitlines(True)[:200]

json.dump({
    "model": "qwen2.5:0.5b",
    "messages": [
        {"role": "user", "content": f"Read {target} and tell me what it does."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": target})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "".join(lines)},
        {"role": "user", "content": "In one short sentence: what is this file?"},
    ],
    "max_tokens": 40,
}, open("rtk-demo.json", "w"))
print("wrote rtk-demo.json")
