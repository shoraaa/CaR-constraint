#!/usr/bin/env bash
# End-to-end n=100 consequence training benchmark.  The GPU must be idle.
set -uo pipefail
cd "$(dirname "$0")"

OUT=results/speed/compact_n100
EPISODES=${EPISODES:-640}
BATCH=${BATCH:-32}
ACCUM=${ACCUM:-4}
REPR=${REPR:-interface}
mkdir -p "$OUT"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

run_case() {
  local tag="$1"
  local compact="$2"
  local log="$OUT/$tag.log"
  local samples="$OUT/$tag.gpu.csv"
  local summary="$OUT/$tag.summary"

  : > "$samples"
  local start_ns
  start_ns=$(date +%s%N)
  python3 -u train.py \
    --problem VRPBLTW --problem_size 100 \
    --epochs 1 --train_episodes "$EPISODES" \
    --train_batch_size "$BATCH" --accumulation_steps "$ACCUM" \
    --improve_steps 0 --validation_improve_steps 0 \
    --pomo_start True --diversity_loss False --soft_constrained True \
    --validation_batch_size 1 --val_episodes 1 \
    --validation_interval 99 --model_save_interval 99 \
    --val_dataset vrpbltw100_uniform.pkl \
    --constraint_repr "$REPR" --node_repr named \
    --consequence_context_dim 0 --consequence_compact "$compact" \
    --slack_weight 0.0 --couple_rows True --seed 1234 \
    --wandb_logger False --tb_logger False \
    --log_dir "$OUT/$tag" > "$log" 2>&1 &
  local train_pid=$!

  while kill -0 "$train_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$samples" 2>/dev/null || true
    sleep 0.2
  done
  local status=0
  wait "$train_pid" || status=$?
  local end_ns
  end_ns=$(date +%s%N)

  local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  local peak_mib
  peak_mib=$(awk -F, 'BEGIN {m=0} {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m}' "$samples")
  local mean_util
  mean_util=$(awk -F, 'BEGIN {s=0;n=0} {gsub(/ /,"",$3); if ($3+0>0) {s+=$3;n++}} END {if(n) printf "%.1f",s/n; else print 0}' "$samples")
  {
    echo "tag=$tag"
    echo "consequence_compact=$compact"
    echo "constraint_repr=$REPR"
    echo "episodes=$EPISODES"
    echo "batch=$BATCH"
    echo "accumulation_steps=$ACCUM"
    echo "status=$status"
    echo "wall_ms=$elapsed_ms"
    echo "peak_gpu_mib=$peak_mib"
    echo "mean_active_gpu_util_pct=$mean_util"
  } | tee "$summary"
  return "$status"
}

# Override CASES to reverse or repeat the order, for example:
# CASES="compact2:True dense2:False" ./bench_compact_n100.sh
CASES=${CASES:-"dense:False compact:True"}
for spec in $CASES; do
  tag=${spec%%:*}
  compact=${spec##*:}
  run_case "$tag" "$compact" || exit $?
done
