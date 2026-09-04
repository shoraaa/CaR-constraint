#!/usr/bin/env bash
# Resume the matched attr control from its salvaged epoch-15 weights, then run
# the POMO pair.
#
# The epoch-15 checkpoint carries model weights and an injected epoch counter
# only: the optimizer state was stripped when the file was copied off this
# machine and the original was lost.  `_restore_training_state` sets
# start_epoch=16 and scheduler.last_epoch=14 from the counter, so the learning
# rate schedule is correct, but Adam's moment estimates restart.  The original
# run's validation curve is known (ep20 34.31%, ep25 31.62%, ep30 28.89%,
# ep35 26.78%, ep40 25.93%), so the cost of the reset is measurable rather than
# assumed -- compare the resumed curve against those numbers.
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== $(date -u) :: attr control, resuming from epoch 15 ==="
python3 train.py \
  --problem VRPBLTW --problem_size 50 --epochs 50 --train_episodes 20000 \
  --train_batch_size 32 --accumulation_steps 4 \
  --improve_steps 5 --validation_improve_steps 20 \
  --validation_batch_size 64 --val_episodes 128 \
  --validation_interval 5 --model_save_interval 5 \
  --val_dataset vrpbltw50_uniform.pkl \
  --constraint_repr attr --slack_weight 0.0 --slack_stride 5 --seed 1234 \
  --checkpoint attr_ep15_resume.pt --load_optimizer True \
  --wandb_logger False --log_dir results/rerun/train_attr

echo "=== $(date -u) :: attr done, starting POMO pair ==="
./pomo_matched.sh
echo "=== $(date -u) :: all done ==="
