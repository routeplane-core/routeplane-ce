Output mcp-demo.gif

Require claude
Require python3

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 760
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set Framerate 24

# --- enter the demo project quietly ---
Hide
Type "cd ~/rp-gifs/mcp/demo-project && clear"
Enter
Sleep 800ms
Show
Sleep 500ms

# --- 1) The entire integration is an 8-line MCP config ---
Type "cat .mcp.json"
Enter
Sleep 3.5s

# --- 2) Point Claude Code at the running gateway through the Routeplane MCP server ---
Type `claude -p "First check gateway status, then answer 'what is an LLM API gateway?' in one sentence via provider groq / model llama-3.1-8b-instant, and report the provider and token count." --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__routeplane__*" --output-format stream-json --verbose | python3 render.py`
Enter
Sleep 30s

Sleep 1.5s
