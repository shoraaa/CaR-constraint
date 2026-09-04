#!/usr/bin/env bash
# Resume finished arms to a longer budget.
#
# CaR resumes properly: Trainer.py:213 sets start_epoch = 1 + checkpoint['epoch']
# and restores optimizer + scheduler, so `--epochs 40` on a checkpoint at 10
# continues 11..40 rather than restarting. Resuming writes into a new timestamped
# subdirectory under the same arm dir; evaluate_all globs **/epoch-*.pt and takes
# the highest epoch, so it still finds the right checkpoint.
#
# Extends the pair that matters locally: the interface arm with the corrected
# token, and the code-matched baseline it is read against. Both must reach the
# SAME epoch or the comparison is a budget difference.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=results/composition
TARGET=${TARGET:-40}
ARMS=${ARMS:-"car_soft_interface_fixed:interface:0.0 car_soft_attr:attr:0.0"}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for the whole queue, not just the current arm: run_remaining.sh starts
# the next arm within seconds, so waiting only on "is training running" fires in
# the gap between two arms and puts two trainers on one GPU.
until [ -f "$OUT/chain.done" ]; do sleep 120; done
"$PY" wait_for_training.py

for SPEC in $ARMS; do
  TAG=${SPEC%%:*}; REST=${SPEC#*:}; REPR=${REST%%:*}; SLACK=${REST##*:}
  CKPT=$(ls -v "$OUT/train_$TAG"/*/epoch-*.pt 2>/dev/null | tail -1)
  if [ -z "$CKPT" ]; then echo "=== $TAG: no checkpoint, skipping ==="; continue; fi
  DONE=$(basename "$CKPT" .pt); DONE=${DONE#epoch-}
  if [ "$DONE" -ge "$TARGET" ]; then echo "=== $TAG already at epoch $DONE ==="; continue; fi
  echo "=== extending $TAG from epoch $DONE to $TARGET ==="
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

touch "$OUT/extend.done"
echo "=== extensions complete ==="
