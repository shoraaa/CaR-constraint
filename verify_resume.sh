#!/usr/bin/env bash
# verify_resume.sh -- confirm that a resumed run actually loaded its checkpoint.
#
#   ./verify_resume.sh                 check every live job on this machine
#   ./verify_resume.sh results/car1000/train_attr    check one log_dir
#
# If --checkpoint is silently ignored the run restarts from scratch and numbers
# its epochs from 1.  A resumed run writes to a NEW timestamped directory, so
# checking "the highest epoch-*.pt" does NOT catch this: a from-scratch
# epoch-5.pt hides behind the old directory's epoch-205.pt under a numeric sort.
# The TensorBoard step numbers in the newest run directory do catch it, and they
# need one epoch rather than five.
#
#   steps 206, 207, 208 ... -> the checkpoint loaded
#   steps 1, 2, 3 ...       -> it did not; stop the job and fix the resume
cd "$(dirname "$(readlink -f "$0")")" || exit 1

dirs="$*"
if [ -z "$dirs" ]; then
  for pid in $(pgrep -f "trai[n].py" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    dirs="$dirs $(echo "$cmd" | grep -o -- '--log_dir [^ ]*' | head -1 | awk '{print $2}')"
  done
fi
[ -n "$dirs" ] && [ -n "${dirs// /}" ] || { echo "no live training jobs and no log_dir given"; exit 0; }

for d in $dirs; do
  newest=$(ls -td "$d"/*/ 2>/dev/null | head -1)
  if [ -z "$newest" ]; then echo "$d: no run directory"; continue; fi
  ev=$(ls -t "$newest"/events* 2>/dev/null | head -1)
  if [ -z "$ev" ]; then echo "$d: no events file in $newest"; continue; fi
  printf '%s\n  run dir: %s\n  ' "$d" "$newest"
  python3 - "$ev" 2>/dev/null <<'PY' || echo "could not read events (job may not have finished its first epoch yet)"
import sys, warnings
warnings.filterwarnings("ignore")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
a = EventAccumulator(sys.argv[1]); a.Reload()
tags = a.Tags()["scalars"]
if not tags:
    print("no scalars yet -- first epoch not finished"); raise SystemExit(0)
steps = [e.step for e in a.Scalars(tags[0])]
print("steps:", steps[:12], "..." if len(steps) > 12 else "")
print(" ", "RESUMED OK" if steps and steps[0] > 1 else "!! RESTARTED FROM SCRATCH -- checkpoint was NOT loaded")
PY
done
