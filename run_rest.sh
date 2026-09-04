#!/usr/bin/env bash
# Remaining arms, most important first.
#
# Each arm is retried: this card throws transient "Memory access fault by GPU
# node-1" aborts that kill a run outright (one eval cell hit it earlier and
# passed on rerun, and it killed interface+slack 11 minutes in). A run counts
# as finished only when its final checkpoint exists, so a crashed arm is
# retried rather than silently skipped.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=results/composition

finished () {  # $1 = log-dir tag
  ls "$OUT/train_$1"/*/epoch-10.pt >/dev/null 2>&1
}

run () {  # $1 = expected tag, rest = env assignments
  local tag="$1"; shift
  local attempt
  for attempt in 1 2 3; do
    if finished "$tag"; then echo "=== $tag already complete ==="; return 0; fi
    echo "=== $tag (attempt $attempt) ==="
    env "$@" EPOCHS=10 EPISODES=2560 BATCH=16 ACCUM=8 ./run_paradigm_study.sh
    if finished "$tag"; then echo "=== $tag complete ==="; return 0; fi
    echo "=== $tag did not produce a checkpoint; retrying ==="
    sleep 30
  done
  echo "=== $tag FAILED after 3 attempts ==="
}

"$PY" wait_for_training.py

run car_soft_interface_slack0.1 HOST=car MASK=soft ARMS=interface SLACK=0.1
run car_soft_attr               HOST=car MASK=soft ARMS=attr
run car_soft_attr_slack0.1      HOST=car MASK=soft ARMS=attr SLACK=0.1
run pomo_full_attr              HOST=pomo MASK=full ARMS=attr
run pomo_full_interface         HOST=pomo MASK=full ARMS=interface
run pomo_soft_attr              HOST=pomo MASK=soft ARMS=attr
run pomo_soft_interface         HOST=pomo MASK=soft ARMS=interface

touch "$OUT/chain.done"
echo "=== chain complete ==="
