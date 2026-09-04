#!/usr/bin/env bash
# One serial pipeline: train every owed arm, evaluate, extend the headline pair,
# then sweep at full resolution.
#
# Serial on purpose.  The previous arrangement ran the 128-instance grids
# concurrently with training on the strength of a free-memory measurement; on
# 2026-09-04 an evaluation of the VRPBLTW cell raised
#   Memory access fault by GPU node-1 ... Reason: Page not present
# which wedged the card.  The trainer did not crash -- it spun at 99% GPU with
# no forward progress for nine hours until it was killed.  Nothing shares this
# GPU any more, and the stages below are sequential statements in one script
# rather than separate processes gated on sentinel files, which is also how the
# two ordering races in extend.sh/final_sweep.sh became possible.
#
# Every `interface` arm here trains against the bent margin/state coordinates
# (VRPBLTWEnv._soft_bound).  The pre-fix arms are kept under
# results/composition/prefix_clamp for the before/after, not deleted.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=results/composition
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

finished () { ls "$OUT/train_$1"/*/epoch-10.pt >/dev/null 2>&1; }

run () {
  local tag="$1"; shift
  local attempt
  for attempt in 1 2 3; do
    if finished "$tag"; then echo "=== $tag already complete ==="; return 0; fi
    echo "=== $tag (attempt $attempt) ==="
    env "$@" EPOCHS=10 EPISODES=2560 BATCH=16 ACCUM=8 ./run_paradigm_study.sh
    finished "$tag" && { echo "=== $tag complete ==="; return 0; }
    echo "=== $tag produced no checkpoint; retrying ==="; sleep 30
  done
  echo "=== $tag FAILED after 3 attempts ==="
}

# --- stage 1: train ---------------------------------------------------------
# Headline pair first, so a later failure cannot cost the arms the claim rests
# on.  car_soft_attr is already at epoch 10 and is untouched by the fix (it
# reads none of the consequence coordinates), so it is not retrained.
run car_soft_interface_bend          HOST=car  MASK=soft ARMS=interface            TAGSUFFIX=_bend
run car_soft_interface_slack0.1_bend HOST=car  MASK=soft ARMS=interface SLACK=0.1  TAGSUFFIX=_bend
# attr+slack does read the margin -- it regresses it as the admissibility
# target -- so it belongs on this side of the fix too.
run car_soft_attr_slack0.1           HOST=car  MASK=soft ARMS=attr      SLACK=0.1
run pomo_soft_interface              HOST=pomo MASK=soft ARMS=interface
run pomo_soft_attr                   HOST=pomo MASK=soft ARMS=attr
run pomo_full_interface              HOST=pomo MASK=full ARMS=interface
run pomo_full_attr                   HOST=pomo MASK=full ARMS=attr
echo "=== stage 1: all arms trained ==="

# --- stage 2: grid every arm at epoch 10 ------------------------------------
echo "=== stage 2: composition grid, 128 instances/cell ==="
"$PY" evaluate_all.py --results_csv "$OUT/grid_ep10.csv" --test_batch_size 8 \
  > "$OUT/eval_ep10.log" 2>&1
echo "=== stage 2 done ==="

# --- stage 3: extend the headline pair --------------------------------------
# Both must reach the SAME epoch or the comparison is a budget difference.
TARGET=${TARGET:-40}
for SPEC in "car_soft_interface_bend:interface:0.0" "car_soft_attr:attr:0.0"; do
  TAG=${SPEC%%:*}; REST=${SPEC#*:}; REPR=${REST%%:*}; SLACK=${REST##*:}
  CKPT=$(ls -v "$OUT/train_$TAG"/*/epoch-*.pt 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { echo "=== $TAG: no checkpoint, skipping ==="; continue; }
  DONE=$(basename "$CKPT" .pt); DONE=${DONE#epoch-}
  [ "$DONE" -ge "$TARGET" ] && { echo "=== $TAG already at epoch $DONE ==="; continue; }
  echo "=== stage 3: extending $TAG from epoch $DONE to $TARGET ==="
  "$PY" train.py --problem VRPBLTW --problem_size 50 \
    --epochs "$TARGET" --train_episodes 2560 \
    --train_batch_size 16 --accumulation_steps 8 \
    --improve_steps 5 --validation_improve_steps 20 \
    --validation_batch_size 32 --val_episodes 64 \
    --validation_interval 2 --model_save_interval 5 \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$REPR" --slack_weight "$SLACK" --slack_stride 5 \
    --couple_rows True --seed 1234 --empty_cache_per_batch True \
    --checkpoint "$CKPT" --load_optimizer True \
    --wandb_logger False --tb_logger False \
    --log_dir "$OUT/train_$TAG" 2>&1 | tee -a "$OUT/train_$TAG.log"
  echo "=== $TAG extended ==="
done

# --- stage 4: publishable resolution ----------------------------------------
echo "=== stage 4: final grid, 1000 instances/cell ==="
"$PY" evaluate_all.py --results_csv "$OUT/grid_final.csv" --test_batch_size 16 \
  > "$OUT/eval_final.log" 2>&1
echo "=== stage 4: unseen-resource probe ==="
"$PY" evaluate_all.py --results_csv "$OUT/grid_final_probe.csv" --test_batch_size 16 \
  --probe_draft > "$OUT/eval_final_probe.log" 2>&1
touch "$OUT/pipeline.done"
echo "=== pipeline complete ==="
