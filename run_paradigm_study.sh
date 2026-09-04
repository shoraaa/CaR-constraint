#!/usr/bin/env bash
# Does the constraint representation port across single-task paradigms?
#
# Two hosts, same problem, same instances, same budget, same seed:
#   car   -- construct-and-refine (the published framework)
#   pomo  -- multi-start autoregressive construction, no refinement, no
#            diversity loss.  This is CaR's own POMO* setting: relaxed masking,
#            which is the regime where a constraint representation can matter at
#            all, since under hard masks the mask does the work.
# Within each host the only thing that varies is --constraint_repr (and
# --slack_weight, if set).
set -uo pipefail

PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
HOST=${HOST:-pomo}
# MASK=hard reproduces standard POMO: strict feasibility masking during
# construction, so every rollout is feasible and the arms differ only in
# objective.  MASK=soft is CaR's POMO* / VRPBLTW setting: masking relaxed,
# feasibility carried by the policy and the penalty, which is the regime a
# constraint representation can actually affect.
MASK=${MASK:-soft}
SIZE=${SIZE:-50}
EPOCHS=${EPOCHS:-10}
EPISODES=${EPISODES:-2560}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-8}
SEED=${SEED:-1234}
SLACK=${SLACK:-0.0}
TAGSUFFIX=${TAGSUFFIX:-}
COUPLE=${COUPLE:-True}
# NODEREPR=rows swaps CaR's per-problem node feature tuple for the pooled row
# summary and pins the query context's width, so no parameter shape is selected
# by problem name any more. The widths must match evaluate_all's ROWS_*.
NODEREPR=${NODEREPR:-named}
NODEREPRDIM=${NODEREPRDIM:-4}
CONTEXTDIM=${CONTEXTDIM:-4}
OUT=${OUT:-results/composition}
ARMS=${ARMS:-"attr interface"}

case "$HOST" in
  car)  HOST_ARGS=(--improve_steps 5 --validation_improve_steps 20) ;;
  pomo) HOST_ARGS=(--improve_steps 0 --validation_improve_steps 0
                   --pomo_start True --diversity_loss False) ;;
  *) echo "unknown host: $HOST" >&2; exit 2 ;;
esac

# `--soft_constrained False` alone is NOT fully masked: the *upper* capacity
# bound (the backhaul rule, load - demand > 1) is only masked when
# --backhaul_mask is hard, which defaults to soft.  Without both flags a
# "masked" POMO still violates capacity on ~7.5 nodes per solution.
case "$MASK" in
  full) HOST_ARGS+=(--soft_constrained False --backhaul_mask hard) ;;
  soft) HOST_ARGS+=(--soft_constrained True) ;;
  *) echo "unknown mask: $MASK (use full or soft)" >&2; exit 2 ;;
esac

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

for ARM in $ARMS; do
  TAG="${HOST}_${MASK}_${ARM}"
  # Optional suffix so a re-run under corrected code does not collide with the
  # arm it supersedes; both stay on disk and can be compared.
  if [ "$SLACK" != "0.0" ]; then TAG="${TAG}_slack${SLACK}"; fi
  # The tag has to record every flag evaluate_all recovers by name.
  if [ "$COUPLE" != "True" ]; then TAG="${TAG}_nocouple"; fi
  if [ "$NODEREPR" = "rows" ]; then TAG="${TAG}_rows"; fi
  TAG="${TAG}${TAGSUFFIX:-}"
  echo "=== host: $HOST  mask: $MASK  arm: $ARM  slack: $SLACK ==="
  "$PY" train.py \
    --problem VRPBLTW --problem_size "$SIZE" \
    --epochs "$EPOCHS" --train_episodes "$EPISODES" \
    --train_batch_size "$BATCH" --accumulation_steps "$ACCUM" \
    "${HOST_ARGS[@]}" \
    --validation_batch_size 32 --val_episodes 64 \
    --validation_interval 2 --model_save_interval 5 \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$ARM" --slack_weight "$SLACK" --seed "$SEED" \
    --couple_rows "$COUPLE" \
    --node_repr "$NODEREPR" --node_repr_dim "$NODEREPRDIM" \
    --consequence_context_dim "$([ "$NODEREPR" = "rows" ] && echo "$CONTEXTDIM" || echo 0)" \
    --empty_cache_per_batch True \
    --wandb_logger True --tb_logger False \
    --log_dir "$OUT/train_$TAG" 2>&1 | tee "$OUT/train_$TAG.log"
done
