#!/usr/bin/env bash
# Final high-resolution sweep, once every arm has trained.
#
# The rolling evaluator runs 128 instances per cell so it can share the GPU with
# training; these are the publishable numbers at the benchmark's own 1000, plus
# the unseen-resource probe. Waits on the chain's sentinel file -- never pgrep.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=results/composition

# Gate on the EXTENSIONS, not the training queue. Two reasons, both bugs found
# by inspection rather than by failure:
#  1. run_remaining writes chain.done/fixed.done and exits, leaving a gap with no
#     training process. This sweep would pass wait_for_training in that gap and
#     then run 1000-instance evaluations concurrently with the extension arms --
#     wait_for_training only sees train.py, so extend.sh would start underneath.
#  2. It would grid the epoch-10 checkpoints and finish before the extended
#     epoch-40 ones exist, so the headline arms would be scored at the short
#     budget.
until [ -f "$OUT/extend.done" ]; do sleep 120; done
"$PY" wait_for_training.py

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo "=== final composition grid (1000 instances/cell) ==="
"$PY" evaluate_all.py --results_csv "$OUT/grid_final.csv" --test_batch_size 16 \
  > "$OUT/eval_final.log" 2>&1
echo "=== unseen-resource probe (1000 instances/cell) ==="
"$PY" evaluate_all.py --results_csv "$OUT/grid_final_probe.csv" --test_batch_size 16 \
  --probe_draft > "$OUT/eval_final_probe.log" 2>&1
touch "$OUT/sweep.done"
echo "=== final sweep complete ==="
