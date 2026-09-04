#!/usr/bin/env bash
# Matched POMO-host pair: does the interface still help without CaR's
# refinement stage?  Construction only, no k-opt, no diversity loss.
#
# Soft masking on purpose.  Under hard masks the mask enforces feasibility and
# the constraint representation cannot matter; soft is CaR's own POMO* regime
# and matches the construct-and-refine runs these are compared against.
#
# Every other flag is copied verbatim from the car-host matched pair
# (results/converge/attr_seed1234 and results/matched/interface_seed1234), so
# the host is the only difference across studies and the representation is the
# only difference within this one.
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for ARM in attr interface; do
  TAG="pomo_soft_${ARM}"
  echo "=== $(date -u) :: $TAG ==="
  python3 train.py \
    --problem VRPBLTW --problem_size 50 \
    --epochs 50 --train_episodes 20000 \
    --train_batch_size 32 --accumulation_steps 4 \
    --improve_steps 0 --validation_improve_steps 0 \
    --pomo_start True --diversity_loss False --soft_constrained True \
    --validation_batch_size 64 --val_episodes 128 \
    --validation_interval 5 --model_save_interval 5 \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$ARM" --slack_weight 0.0 \
    --couple_rows True --seed 1234 \
    --wandb_logger False \
    --log_dir "results/pomo/train_${TAG}"
done
echo "=== $(date -u) :: both POMO arms done ==="
