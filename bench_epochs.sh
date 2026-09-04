#!/usr/bin/env bash
# Two epochs each, on an idle GPU, so epoch 2 is measured without startup cost.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
for SPEC in "attr:attr" "interface:interface"; do
  TAG=${SPEC%%:*}; REPR=${SPEC##*:}
  "$PY" train.py --problem VRPBLTW --problem_size 50 \
    --epochs 2 --train_episodes 2560 --train_batch_size 16 --accumulation_steps 8 \
    --improve_steps 5 --validation_improve_steps 20 --soft_constrained True \
    --validation_batch_size 32 --val_episodes 64 --validation_interval 99 \
    --model_save_interval 99 --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$REPR" --slack_weight 0.0 --seed 1234 \
    --empty_cache_per_batch True --wandb_logger False --tb_logger False \
    --log_dir "results/speed/$TAG" > "results/speed/$TAG.log" 2>&1
  echo "=== $TAG done ==="
done
touch results/speed/bench.done
