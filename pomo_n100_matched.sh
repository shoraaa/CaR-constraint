#!/usr/bin/env bash
# Matched n=100 POMO training.  A live two-arm memory probe at batch 8 used
# 7.95 GiB total, so the original physical/effective batch of 32/128 fits the
# 40 GiB A100 while both arms share it.
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

run_arm() {
  local arm="$1"
  python3 train.py \
    --problem VRPBLTW --problem_size 100 \
    --epochs 1000 --train_episodes 20000 \
    --train_batch_size 32 --accumulation_steps 4 \
    --improve_steps 0 --validation_improve_steps 0 \
    --pomo_start True --diversity_loss False --soft_constrained True \
    --validation_batch_size 16 --val_episodes 128 \
    --validation_interval 5 --model_save_interval 5 \
    --val_dataset vrpbltw100_uniform.pkl \
    --constraint_repr "$arm" --slack_weight 0.0 \
    --couple_rows True --seed 1234 \
    --wandb_logger False \
    --log_dir "results/pomo_n100/train_pomo_soft_${arm}"
}

mkdir -p results/pomo_n100
run_arm attr > pomo_n100_attr.log 2>&1 &
attr_pid=$!
run_arm interface > pomo_n100_interface.log 2>&1 &
interface_pid=$!
echo "attr pid=$attr_pid; interface pid=$interface_pid"

attr_status=0
interface_status=0
wait "$attr_pid" || attr_status=$?
wait "$interface_pid" || interface_status=$?
echo "attr exit=$attr_status; interface exit=$interface_status"
test "$attr_status" -eq 0 -a "$interface_status" -eq 0
