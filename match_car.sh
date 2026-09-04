#!/usr/bin/env bash
# Matched baseline: original CaR at exactly the interface run's budget.
#
# Every flag below is copied from the interface run on 132.145.182.17
# (results/converge/interface_seed1234), changing only --constraint_repr and
# --slack_weight. --epochs is 25 rather than 50 because that is where the
# interface checkpoint we are comparing against stopped; the LR schedule is
# unaffected (--milestones defaults to [4501], an absolute epoch, and nothing
# in Trainer.py reads `epochs` except the loop bound and the ETA display), so
# 25 epochs here is trajectory-identical to the first 25 of a 50-epoch run.
#
# `attr` touches none of the interface code paths -- consequence_interface
# resolves False, ConsequenceValuation is never constructed, and Wq_last falls
# through to CaR's own per-problem switch -- so this is stock CaR despite the
# tree carrying the interface work.
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=${EPOCHS:-25}
SEED=${SEED:-1234}
python3 train.py --problem VRPBLTW --problem_size 50 \
  --epochs "$EPOCHS" --train_episodes 20000 \
  --train_batch_size 32 --accumulation_steps 4 \
  --improve_steps 5 --validation_improve_steps 20 \
  --validation_batch_size 64 --val_episodes 128 \
  --validation_interval 5 --model_save_interval 5 \
  --val_dataset vrpbltw50_uniform.pkl \
  --constraint_repr attr --slack_weight 0.0 --slack_stride 5 \
  --seed "$SEED" --wandb_logger False \
  --log_dir "results/matched/attr_seed${SEED}"
echo "=== matched CaR baseline complete ==="
touch results/matched/attr.done
