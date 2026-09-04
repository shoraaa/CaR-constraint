#!/usr/bin/env bash
# Paired, matched-budget comparison of CaR against CaR + the consequence interface.
#
# Both arms train on the full VRPBLTW composition with identical data, seed,
# budget and hyperparameters; they differ only in how the active constraints are
# described to the construction decoder.  Neither arm ever sees a held-out
# composition during training or model selection.
set -uo pipefail   # not -e: one arm failing must not abort the rest

PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
SIZE=${SIZE:-50}
EPOCHS=${EPOCHS:-10}
EPISODES=${EPISODES:-2560}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-8}
SEED=${SEED:-1234}
OUT=${OUT:-results/composition}
ARMS=${ARMS:-"attr interface"}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

for ARM in $ARMS; do
  echo "=== training arm: $ARM ==="
  "$PY" train.py \
    --problem VRPBLTW --problem_size "$SIZE" \
    --epochs "$EPOCHS" --train_episodes "$EPISODES" \
    --train_batch_size "$BATCH" --accumulation_steps "$ACCUM" \
    --improve_steps 5 --validation_improve_steps 20 \
    --validation_batch_size 32 --val_episodes 64 \
    --validation_interval ${VAL_EVERY:-2} --model_save_interval ${SAVE_EVERY:-5} \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$ARM" --seed "$SEED" --empty_cache_per_batch True \
    --wandb_logger True --tb_logger False \
    --log_dir "$OUT/train_$ARM" 2>&1 | tee "$OUT/train_$ARM.log"
done
