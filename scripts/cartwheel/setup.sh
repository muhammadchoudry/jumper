#!/bin/bash
# One-time setup for training on a Mac (or Linux) CPU: clone or update the
# cartwheel branch into ~/Projects/jumper and build its venv.
#   bash scripts/cartwheel/setup.sh            (or curl it and run from anywhere)
# Re-runnable: skips the clone and venv when they already exist.
set -euo pipefail
DIR="${JUMPER_DIR:-$HOME/Projects/jumper}"
mkdir -p "$(dirname "$DIR")"
if [ ! -d "$DIR/.git" ]; then
  git clone --branch cartwheel https://github.com/muhammadchoudry/jumper.git "$DIR"
fi
cd "$DIR"
git fetch -q origin cartwheel && git checkout -q cartwheel && git pull -q --ff-only
PY=""
for v in python3.13 python3.12 python3.11 python3.10; do
  if command -v "$v" >/dev/null 2>&1; then PY="$v"; break; fi
done
[ -n "$PY" ] || { echo "needs Python 3.10-3.13 (e.g. brew install python@3.12)"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q -U pip
python -m pip install -q -e ".[dev]"
python scripts/train.py --list | grep cartwheel
echo "ready: $DIR (source: $(git log --oneline -1))"
