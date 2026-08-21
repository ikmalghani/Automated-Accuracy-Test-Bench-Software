#!/usr/bin/env bash
# Launch AATBS as a program (double-click or ./run.sh). The GUI window pops up.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail() {
  local msg="$1"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="AATBS" --text="$msg" --width=420 || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "AATBS" "$msg" || true
  else
    printf '%s\n' "$msg" >&2
  fi
  exit 1
}

# File-manager launches skip ~/.bashrc, so load pyenv here.
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [[ -x "$PYENV_ROOT/bin/pyenv" ]]; then
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  if pyenv versions --bare 2>/dev/null | grep -qx 'AATBS_env'; then
    export PYENV_VERSION=AATBS_env
  fi
fi

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  fail "Python was not found. Install Python 3 or pyenv environment AATBS_env."
fi

if ! "$PYTHON" -c "import PIL, matplotlib, tkinter" >/dev/null 2>&1; then
  if [[ -f "$ROOT/requirements.txt" ]]; then
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt" \
      || fail "Could not install dependencies from requirements.txt."
  else
    fail "Missing Python packages (Pillow, matplotlib, tkinter)."
  fi
fi

# Start the GUI in a new session, then exit so no terminal stays open.
# File managers that wrap scripts in a terminal will close that window immediately.
if command -v setsid >/dev/null 2>&1; then
  setsid --fork "$PYTHON" "$ROOT/main.py" "$@" >/dev/null 2>&1
else
  nohup "$PYTHON" "$ROOT/main.py" "$@" >/dev/null 2>&1 &
  disown || true
fi
exit 0
