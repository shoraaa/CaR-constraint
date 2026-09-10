#!/usr/bin/env bash
# Short end-to-end ETA/memory benchmark for n=100 CaR + consequence interface.
set -uo pipefail
cd "$(dirname "$0")"

BATCH=${BATCH:-8}
ACCUM=${ACCUM:-16}
EPISODES=${EPISODES:-512}
TAG=${TAG:-bs${BATCH}_acc${ACCUM}_ep${EPISODES}}
OUT=results/speed/car_interface_n100
LOG="$OUT/$TAG.log"
SAMPLES="$OUT/$TAG.gpu.csv"
SUMMARY="$OUT/$TAG.summary"
mkdir -p "$OUT"
rm -f "$LOG" "$SAMPLES" "$SUMMARY"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_ns=$(date +%s%N)
python3 -u train.py \
  --problem VRPBLTW --problem_size 100 \
  --epochs 1 --train_episodes "$EPISODES" \
  --train_batch_size "$BATCH" --accumulation_steps "$ACCUM" \
  --improve_steps 5 --validation_improve_steps 20 \
  --pomo_start False --soft_constrained True \
  --validation_batch_size 1 --val_episodes 1 \
  --validation_interval 99 --model_save_interval 99 \
  --val_dataset vrpbltw100_uniform.pkl \
  --constraint_repr interface --node_repr named \
  --consequence_context_dim 0 --consequence_compact True \
  --slack_weight 0.0 --couple_rows True --seed 1234 \
  --wandb_logger False --tb_logger False \
  --log_dir "$OUT/$TAG" > "$LOG" 2>&1 &
pid=$!

while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits >> "$SAMPLES" 2>/dev/null || true
  sleep 0.1
done
status=0
wait "$pid" || status=$?
end_ns=$(date +%s%N)

wall_ms=$(( (end_ns - start_ns) / 1000000 ))
peak_mib=$(awk -F, 'BEGIN {m=0} {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m}' "$SAMPLES")
trainer_minutes=$(sed -n 's/.*Elapsed\[\([0-9.]*\)m\].*/\1/p' "$LOG" | tail -1)
{
  echo "batch=$BATCH"
  echo "accumulation_steps=$ACCUM"
  echo "effective_batch=$((BATCH * ACCUM))"
  echo "episodes=$EPISODES"
  echo "status=$status"
  echo "trainer_minutes=$trainer_minutes"
  echo "wall_ms=$wall_ms"
  echo "peak_gpu_mib=$peak_mib"
} | tee "$SUMMARY"
exit "$status"
