#!/usr/bin/env bash
# Re-run the two interface arms with the corrected token.
#
# The pooled token carried the MARGIN, which (a) diverges from
# Eq. (app-resource-token), which pools pressure, and (b) fed the admissibility
# head the mean and max of the value it regresses, through the decoder context.
# The `interface` arm is affected by (a) only; `interface + slack` by both, so
# its result is not valid evidence about the loss.
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

until [ -f "$OUT/chain.done" ]; do sleep 120; done
"$PY" wait_for_training.py

run car_soft_interface_slack0.1_fixed HOST=car MASK=soft ARMS=interface SLACK=0.1 TAGSUFFIX=_fixed
run car_soft_interface_fixed          HOST=car MASK=soft ARMS=interface TAGSUFFIX=_fixed

touch "$OUT/fixed.done"
echo "=== fixed-token re-runs complete ==="
