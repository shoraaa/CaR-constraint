#!/usr/bin/env bash
# Reprioritised queue: the row-encoder arms first.
#
# `--node_repr rows` moves the interface out of the decoder and into the
# representation, which is where PRISM keeps it (net.py augment_nodes ->
# ResourcePool). Paired with --consequence_context_dim it also removes the last
# problem-name lookup from the parameter shapes (Wq_last), so the model is
# formulation-independent the way PRISM's is -- see
# tests/test_formulation_independence.py, which asserts CVRP/VRPTW/VRPBLTW all
# produce identical shapes and that the published baseline does not.
#
# This supersedes pipeline.sh, whose sequencer was killed between arms; the arm
# it had in flight was left running and is waited on below rather than
# restarted. Strictly serial: nothing shares this GPU (a concurrent eval raised
# a Memory access fault on 2026-09-04 and wedged a trainer for nine hours).
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

# Let the in-flight arm from pipeline.sh finish before touching the GPU.
"$PY" wait_for_training.py

# --- stage 1: the row encoder, against its own named-encoder counterpart -----
# car_soft_interface_bend (named encoder, same seed, same budget) is the control
# for the first of these; it is the only difference between the two arms.
run car_soft_interface_rows          HOST=car MASK=soft ARMS=interface           NODEREPR=rows
run car_soft_interface_slack0.1_rows HOST=car MASK=soft ARMS=interface SLACK=0.1 NODEREPR=rows
# --- stage 2: the arms still owed from the earlier queue ---------------------
run car_soft_interface_slack0.1_bend HOST=car  MASK=soft ARMS=interface SLACK=0.1 TAGSUFFIX=_bend
run car_soft_attr_slack0.1           HOST=car  MASK=soft ARMS=attr      SLACK=0.1
run pomo_soft_interface              HOST=pomo MASK=soft ARMS=interface
run pomo_soft_attr                   HOST=pomo MASK=soft ARMS=attr
run pomo_full_interface              HOST=pomo MASK=full ARMS=interface
run pomo_full_attr                   HOST=pomo MASK=full ARMS=attr
echo "=== stage 2: all arms trained ==="

# --- stage 3: grid every arm at its epoch 10 --------------------------------
"$PY" evaluate_all.py --results_csv "$OUT/grid_ep10.csv" --test_batch_size 8 \
  > "$OUT/eval_ep10.log" 2>&1
echo "=== stage 3: grid done ==="

# --- stage 4: extend the headline pair --------------------------------------
TARGET=${TARGET:-40}
for SPEC in "car_soft_interface_rows:interface:0.0:rows" \
            "car_soft_interface_bend:interface:0.0:named" \
            "car_soft_attr:attr:0.0:named"; do
  TAG=${SPEC%%:*}; R1=${SPEC#*:}; REPR=${R1%%:*}; R2=${R1#*:}; SLACK=${R2%%:*}; NR=${R2##*:}
  CKPT=$(ls -v "$OUT/train_$TAG"/*/epoch-*.pt 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { echo "=== $TAG: no checkpoint, skipping ==="; continue; }
  DONE=$(basename "$CKPT" .pt); DONE=${DONE#epoch-}
  [ "$DONE" -ge "$TARGET" ] && { echo "=== $TAG already at epoch $DONE ==="; continue; }
  echo "=== stage 4: extending $TAG from epoch $DONE to $TARGET ==="
  "$PY" train.py --problem VRPBLTW --problem_size 50 \
    --epochs "$TARGET" --train_episodes 2560 \
    --train_batch_size 16 --accumulation_steps 8 \
    --improve_steps 5 --validation_improve_steps 20 \
    --validation_batch_size 32 --val_episodes 64 \
    --validation_interval 2 --model_save_interval 5 \
    --val_dataset vrpbltw50_uniform.pkl \
    --constraint_repr "$REPR" --slack_weight "$SLACK" --slack_stride 5 \
    --node_repr "$NR" --node_repr_dim 4 \
    --consequence_context_dim "$([ "$NR" = "rows" ] && echo 4 || echo 0)" \
    --couple_rows True --seed 1234 --empty_cache_per_batch True \
    --checkpoint "$CKPT" --load_optimizer True \
    --wandb_logger False --tb_logger False \
    --log_dir "$OUT/train_$TAG" 2>&1 | tee -a "$OUT/train_$TAG.log"
done

# --- stage 5: publishable resolution ----------------------------------------
"$PY" evaluate_all.py --results_csv "$OUT/grid_final.csv" --test_batch_size 16 \
  > "$OUT/eval_final.log" 2>&1
"$PY" evaluate_all.py --results_csv "$OUT/grid_final_probe.csv" --test_batch_size 16 \
  --probe_draft > "$OUT/eval_final_probe.log" 2>&1
touch "$OUT/pipeline.done"
echo "=== pipeline complete ==="
