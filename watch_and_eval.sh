#!/usr/bin/env bash
# Evaluate each arm on the composition grid as soon as its final checkpoint
# appears, instead of waiting for the whole chain.
#
# Concurrency with training is safe at this batch size: measured 10.7 GB free
# against a 6.5 GB training arm, and a single-cell probe reproduced the
# training-time validation exactly (27.75 vs 27.74).
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=${OUT:-results/composition}
CSV=${CSV:-$OUT/grid_rolling.csv}
EPISODES=${EPISODES:-128}
BATCH=${BATCH:-8}
mkdir -p "$OUT/.evaluated"

while true; do
  for dir in "$OUT"/train_*/; do
    [ -d "$dir" ] || continue
    arm=$(basename "$dir"); arm=${arm#train_}
    marker="$OUT/.evaluated/$arm"
    [ -f "$marker" ] && continue
    # Only once the final epoch checkpoint exists, so every arm is compared at
    # the same point in training and never on a best-by-score checkpoint.
    ls "$dir"*/epoch-10.pt >/dev/null 2>&1 || continue
    echo "=== evaluating $arm ==="
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$PY" evaluate_all.py \
      --results_csv "$CSV" --only_arms "$arm" \
      --test_batch_size "$BATCH" >> "$OUT/eval_rolling.log" 2>&1
    touch "$marker"
    echo "=== $arm done ==="
  done
  # Sentinel file, not pgrep: a pattern naming a script also matches any shell
  # whose command line mentions it, including a sibling waiter's.
  [ -f "$OUT/chain.done" ] && { echo "chain finished; last sweep done"; break; }
  sleep 300
done
