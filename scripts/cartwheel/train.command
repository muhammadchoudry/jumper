#!/bin/bash
# Double-click launcher for a Mac: copy to ~/Projects, set RUN, chmod +x, open it.
# Starts the named run from scripts/cartwheel/runs.py (python scripts/cartwheel/runs.py list).
RUN="${RUN:-stunt_long}"
cd ~/Projects/jumper || exit 1
git fetch -q origin cartwheel && git checkout -q cartwheel && git pull -q --ff-only
. .venv/bin/activate
python scripts/cartwheel/runs.py start "$RUN"
