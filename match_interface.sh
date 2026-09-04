#!/usr/bin/env bash
# The arm that does not exist yet: pure interface, no slack loss.
#
# Its matched control is the `attr` run on 132.145.182.17
# (results/converge/attr_seed1234). Every flag below is identical to that job
# except --constraint_repr; --slack_weight is 0.0 on both, so this pair isolates
# the constraint representation and nothing else.
#
# 50 epochs to match the control exactly. --epochs does not enter the LR
# schedule (--milestones defaults to [4501], absolute) and nothing in Trainer.py
# reads it except the loop bound and the ETA, so the trajectory is the same as a
# 25-epoch run and stopping early costs nothing -- checkpoints land every 5.
#
# Bound and compaction are stated rather than left implicit: `clamp` is PRISM's
# own operation and measured to be all this ever needs, and compaction is a
# 0.93x regression on an A100 (1.50x on the RX 6800), so it stays off here.
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=${EPOCHS:-50}
SEED=${SEED:-1234}
python3 train.py --problem VRPBLTW --problem_size 50 \
  --epochs "$EPOCHS" --train_episodes 20000 \
  --train_batch_size 32 --accumulation_steps 4 \
  --improve_steps 5 --validation_improve_steps 20 \
  --validation_batch_size 64 --val_episodes 128 \
  --validation_interval 5 --model_save_interval 5 \
  --val_dataset vrpbltw50_uniform.pkl \
  --constraint_repr interface --slack_weight 0.0 \
  --consequence_bound clamp --consequence_compact False \
  --couple_rows True --seed "$SEED" --wandb_logger False \
  --log_dir "results/matched/interface_seed${SEED}"
echo "=== pure interface run complete ==="
touch results/matched/interface.done
