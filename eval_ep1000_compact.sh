#!/usr/bin/env bash
# Re-evaluate the n=50 epoch-1000 consequence POMO checkpoint with the
# compact live-candidate path, recording phase times and peak GPU memory.
set -uo pipefail
cd "$(dirname "$0")"

OUT=results/composition/ep1000_compact
CKPT=results/pomo_eval_ep1000/train_pomo_soft_interface/epoch-1000.pt
mkdir -p "$OUT"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

run_case() {
  local size="$1"
  local batch="$2"
  local tag="n${size}_bs${batch}"
  local log="$OUT/$tag.log"
  local csv="$OUT/$tag.csv"
  local samples="$OUT/$tag.gpu.csv"
  local summary="$OUT/$tag.summary"
  rm -f "$log" "$csv" "$samples" "$summary"

  local start_ns
  start_ns=$(date +%s%N)
  python3 -u test.py \
    --problem VRPBLTW --problem_size "$size" \
    --checkpoint "$CKPT" --disable_preset_args \
    --constraint_repr interface --node_repr named \
    --consequence_context_dim 0 --consequence_compact True \
    --slack_weight 0.0 --couple_rows True \
    --test_episodes 1000 --test_batch_size "$batch" \
    --improve_steps 0 --validation_improve_steps 0 \
    --pomo_start True --diversity_loss False --soft_constrained True \
    --active_constraints backhaul route_limit time_window \
    --results_csv "$csv" --arm_tag pomo_soft_interface_ep1000_compact \
    --variant_tag VRPBLTW --wandb_logger False > "$log" 2>&1 &
  local eval_pid=$!

  while kill -0 "$eval_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$samples" 2>/dev/null || true
    sleep 0.1
  done
  local status=0
  wait "$eval_pid" || status=$?
  local end_ns
  end_ns=$(date +%s%N)

  local wall_ms=$(( (end_ns - start_ns) / 1000000 ))
  local peak_mib
  peak_mib=$(awk -F, 'BEGIN {m=0} {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m}' "$samples")
  local construction_s
  construction_s=$(awk '/val construction time:/{s+=$NF} END {printf "%.6f",s}' "$log")
  local masked_s
  masked_s=$(awk '/val reconstruction time \[w. mask\]:/{s+=$NF} END {printf "%.6f",s}' "$log")
  local eval_s
  eval_s=$(awk '/Evaluation finished within/{v=$NF; sub(/s$/,"",v)} END {print v}' "$log")
  {
    echo "problem_size=$size"
    echo "episodes=1000"
    echo "batch=$batch"
    echo "consequence_compact=True"
    echo "status=$status"
    echo "construction_seconds=$construction_s"
    echo "masked_seconds=$masked_s"
    echo "evaluation_seconds=$eval_s"
    echo "process_wall_ms=$wall_ms"
    echo "peak_gpu_mib=$peak_mib"
  } | tee "$summary"
  return "$status"
}

run_case 50 256 || exit $?
run_case 100 64 || exit $?
