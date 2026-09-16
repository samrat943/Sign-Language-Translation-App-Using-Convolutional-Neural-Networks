import os
import sys

# Redirect entry point to main.py
main_script = os.path.join(os.path.dirname(__file__), "main.py")
with open(main_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, main_script, "exec"), globals())
