#!/usr/bin/env bash
# One sequential chain, so no two runs can ever share the GPU.
#
# The previous arrangement used two independent waiters keyed on "is the study
# script running", which raced: between the two POMO invocations neither script
# exists for a moment, and the attr re-run could have started concurrently.
set -uo pipefail
cd "$(dirname "$0")"

run () {
  echo "=== $* ==="
  env "$@" EPOCHS=10 EPISODES=2560 BATCH=16 ACCUM=8 ./run_paradigm_study.sh
}

# No pgrep-based waiting here.  Two waiters keyed on `pgrep -f <script name>`
# deadlocked: each waiter's own command line contains the other's pattern, so
# each matched the other and neither ever exited.  Anything sequencing against
# this chain watches for the sentinel file written at the end.
rm -f results/composition/chain.done

# CaR host first: the arms most likely to show signal at this budget, plus the
# baseline they are read against.
run HOST=car MASK=soft ARMS=interface SLACK=0.1   # ours: interface + slack loss (ratio to the RL loss)
run HOST=car MASK=soft ARMS=attr                  # code-matched baseline
run HOST=car MASK=soft ARMS=attr SLACK=0.1        # control: does the loss alone do it?

# Then the second paradigm, both masking regimes.
run HOST=pomo MASK=full ARMS="attr interface"     # pure POMO vs POMO + ours
run HOST=pomo MASK=soft ARMS="attr interface"     # POMO* vs POMO* + ours

touch results/composition/chain.done
echo "=== chain complete ==="
