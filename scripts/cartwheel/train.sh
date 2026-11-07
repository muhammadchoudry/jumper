#!/bin/bash
# Start one training run in the background, the way every cartwheel run was trained
# (Mac mini M-series CPU, native backend, 1024 envs).
#
#   bash scripts/cartwheel/train.sh TASK [CHECKPOINT] [ROUNDS=3000] [ENVS=1024]
#   bash scripts/cartwheel/train.sh jumper.cartwheel_hill tasks/jumper/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt
#
# From the repo root with its venv. With a CHECKPOINT the run continues that
# policy (round numbers continue from it). Log: $LOG_DIR/train_TASK.log
# (default ~/Projects/jumper-train-logs); checkpoints every 50 rounds in
# logs/jumper/TASK/<date>/. scripts/cartwheel/runs.py has every run's arguments.
set -uo pipefail
TASK="${1:?task id, e.g. jumper.cartwheel_hill}"
CKPT="${2:-}"
ROUNDS="${3:-3000}"
ENVS="${4:-1024}"
LOG_DIR="${LOG_DIR:-$HOME/Projects/jumper-train-logs}"
mkdir -p "$LOG_DIR"
if pgrep -f "python scripts/train" > /dev/null; then
  echo "another training run is going; stop it first"; exit 1
fi
[ -d .venv ] && . .venv/bin/activate
ARGS=(--task "$TASK" --backend native --device cpu --num_envs "$ENVS" --max-iterations "$ROUNDS"
      --headless --no-tensorboard)
if [ -n "$CKPT" ]; then
  [ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
  ARGS+=(--checkpoint "$CKPT")
fi
WAKE=(); command -v caffeinate > /dev/null && WAKE=(caffeinate -i)      # keep a Mac awake
echo "source: $(git log --oneline -1)"
nohup "${WAKE[@]}" python scripts/train.py "${ARGS[@]}" > "$LOG_DIR/train_$TASK.log" 2>&1 &
echo $! > "$LOG_DIR/train.pid"
echo "started $TASK (pid $(cat "$LOG_DIR/train.pid")); log: $LOG_DIR/train_$TASK.log"
