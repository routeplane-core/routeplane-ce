# Routeplane CLI demo — records cli-demo.gif
Output cli-demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 680
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Padding 24
Set Framerate 24
Set WindowBar Colorful

# --- hidden setup: clean prompt + ensure rp is on PATH ---
Hide
Type `export PATH="$(dirname "$(command -v node)"):$PATH"; export HOME=/home/dev`
Enter
Type "export PS1='$ '"
Enter
Type "clear"
Enter
Sleep 800ms
Show

# --- beat 1: init (interactive setup) ---
Type "rp init"
Sleep 500ms
Enter
Sleep 1200ms
Type "http://localhost:8080"
Sleep 400ms
Enter
Sleep 900ms
Type "rp_local_demo_2f8a1c"
Sleep 400ms
Enter
Sleep 900ms
Type "local"
Sleep 400ms
Enter
Sleep 2000ms

# --- beat 2: chat (real streaming + branded footer) ---
Type `rp chat "In one sentence, what is the Digital Personal Data Protection Act of India?" --provider groq --model llama-3.1-8b-instant`
Sleep 600ms
Enter
Sleep 4200ms

# --- beat 3: multi-provider model catalog ---
Type "rp models list --provider groq"
Sleep 600ms
Enter
Sleep 4000ms

Sleep 1200ms
