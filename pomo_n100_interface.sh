#!/usr/bin/env bash
# Train the n=100 consequence/interface POMO on the 40 GiB A100.  CaR's
# default effective batch is 128.  The fastest path uses physical batch 128;
# fallbacks preserve the same effective batch through gradient accumulation.
set -uo pipefail
cd "$(dirname "$0")"

OUT=results/pomo_n100/train_pomo_soft_interface_compact
RUN_LOG=pomo_n100_interface_compact.log
mkdir -p "$OUT"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

latest_checkpoint() {
  find "$OUT" -type f -name 'epoch-*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

run_at_batch() {
  local batch="$1"
  local accumulation="$2"
  local checkpoint="$3"
  local resume_args=()
  if [[ -n "$checkpoint" ]]; then
    resume_args=(--checkpoint "$checkpoint" --load_optimizer True)
  fi
  echo "=== $(date -Is) :: batch=$batch accumulation=$accumulation effective_batch=$((batch * accumulation)) checkpoint=${checkpoint:-from_scratch} ==="
  python3 -u train.py \
    --problem VRPBLTW --problem_size 100 \
    --epochs 1000 --train_episodes 20000 \
    --train_batch_size "$batch" --accumulation_steps "$accumulation" \
    --improve_steps 0 --validation_improve_steps 0 \
    --pomo_start True --diversity_loss False --soft_constrained True \
    --validation_batch_size 64 --val_episodes 128 \
    --validation_interval 5 --model_save_interval 5 \
    --val_dataset vrpbltw100_uniform.pkl \
    --constraint_repr interface --node_repr named \
    --consequence_context_dim 0 --consequence_compact True \
    --slack_weight 0.0 --couple_rows True --seed 1234 \
    --wandb_logger False --log_dir "$OUT" \
    "${resume_args[@]}"
}

for spec in 128:1 64:2 32:4; do
  batch=${spec%%:*}
  accumulation=${spec##*:}
  checkpoint=$(latest_checkpoint)
  run_at_batch "$batch" "$accumulation" "$checkpoint" >> "$RUN_LOG" 2>&1
  status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "=== $(date -Is) :: training complete at batch=$batch accumulation=$accumulation ===" >> "$RUN_LOG"
    exit 0
  fi
  echo "=== $(date -Is) :: batch=$batch exited status=$status; falling back ===" >> "$RUN_LOG"
done

echo "=== $(date -Is) :: all batch sizes failed ===" >> "$RUN_LOG"
exit 1
