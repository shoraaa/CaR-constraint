#!/usr/bin/env bash
# The right-hand column of the 2x2: the same two arms with PRISM's admissibility
# supervision switched on.  Written as a separate file rather than a flag on
# run_composition_study.sh because that script is running -- bash reads a script
# lazily, so editing one mid-run can corrupt the block it has not read yet.
set -uo pipefail

PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
SIZE=${SIZE:-50}
EPOCHS=${EPOCHS:-10}
EPISODES=${EPISODES:-2560}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-8}
SEED=${SEED:-1234}
SLACK=${SLACK:-1.0}
OUT=${OUT:-results/composition}
ARMS=${ARMS:-"attr interface"}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

for ARM in $ARMS; do
  TAG="${ARM}_slack${SLACK}"
  echo "=== training arm: $TAG ==="
  "$PY" train.py \
    --problem VRPBLTW --problem_size "$SIZE" \
    --epochs "$EPOCHS" --train_episodes "$EPISODES" \
    --train_batch_size "$BATCH" --accumulation_steps "$ACCUM" \
    --improve_steps 5 --validation_improve_steps 20 \
    --validation_batch_size 32 --val_episodes 64 \
    --validation_interval 2 --model_save_interval 5 \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$ARM" --slack_weight "$SLACK" --seed "$SEED" \
    --empty_cache_per_batch True \
    --wandb_logger True --tb_logger False \
    --log_dir "$OUT/train_$TAG" 2>&1 | tee "$OUT/train_$TAG.log"
done
