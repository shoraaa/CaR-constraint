#!/usr/bin/env bash
# Everything still owed, fixed-token arms first.
#
# Only the two already-trained `interface` arms used the leaky token; `attr` and
# `attr + slack` feed the head the named live-state vector (no margin), and the
# POMO arms have not trained yet, so they pick up the corrected code as-is.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/shora/Research/PRISM/.venv/bin/python}
OUT=results/composition

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

"$PY" wait_for_training.py   # let the in-flight attr arm finish

run car_soft_interface_slack0.1_fixed HOST=car MASK=soft ARMS=interface SLACK=0.1 TAGSUFFIX=_fixed
run car_soft_interface_fixed          HOST=car MASK=soft ARMS=interface           TAGSUFFIX=_fixed
run car_soft_attr                     HOST=car MASK=soft ARMS=attr
run car_soft_attr_slack0.1            HOST=car MASK=soft ARMS=attr SLACK=0.1
run pomo_full_attr                    HOST=pomo MASK=full ARMS=attr
run pomo_full_interface               HOST=pomo MASK=full ARMS=interface
run pomo_soft_attr                    HOST=pomo MASK=soft ARMS=attr
run pomo_soft_interface               HOST=pomo MASK=soft ARMS=interface

touch "$OUT/chain.done" "$OUT/fixed.done"
echo "=== all arms complete ==="
