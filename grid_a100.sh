#!/usr/bin/env bash
# Grid the A100 interface+slack checkpoint against the published CaR reference.
# Waits for the timing benchmark: a concurrent eval is what faulted the GPU
# earlier today and cost nine hours.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
until [ -f results/speed/bench.done ]; do sleep 60; done
"$PY" wait_for_training.py
echo "=== gridding a100_interface_slack0.1 at 1000 instances/cell ==="
"$PY" evaluate_all.py --only_arms a100_interface_slack0.1 \
  --results_csv results/composition/grid_a100.csv \
  --test_episodes 1000 --test_batch_size 16 \
  > results/composition/eval_a100.log 2>&1
touch results/composition/grid_a100.done
echo "=== done ==="
