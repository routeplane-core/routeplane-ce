Output python-demo.gif

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 680
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Padding 24

# --- hidden setup: venv + clean prompt ---
Hide
Type "cd ~/rp-gifs/python && source .venv/bin/activate && export PS1='$ ' && clear"
Enter
Sleep 500ms
Show

# --- show the SDK code ---
Type "cat demo.py"
Sleep 400ms
Enter
Sleep 4s

# --- run it against the live gateway ---
Type "python demo.py"
Sleep 400ms
Enter
Sleep 5s

Sleep 1500ms
