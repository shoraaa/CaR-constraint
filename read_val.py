#!/usr/bin/env python3
"""Print `epoch masked constr infeas improve` for a training run's log_dir.

Read straight from the TensorBoard event files, not from the nohup log.  Python
block-buffers stdout at 8 KB when redirected and this trainer prints little per
epoch, so a log can sit hours behind the run; the event files are flushed every
epoch.  Tags verified against the log lines they mirror (epoch 145 of the CaR
interface arm: val_gap 12.2829 = "[Construction] AUG_Gap", val_gap_rc_masked
11.7671 = "[w. mask] AUG_Gap", improve_val_gap 11.9588 = "[Improvement]").

Scalars are merged across every timestamped run directory under the log_dir, so
the curve spans resumes instead of restarting at each move.
"""
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "masked":  "val/val_gap_rc_masked",          # the comparable objective
    "constr":  "val/val_gap",                    # soft construction, pre-mask
    "infeas":  "val/val_sol_infsb_rate",         # soft-construction infeasibility
    "improve": "val/improve_val_gap",            # CaR refinement pass (POMO: absent)
}

out = {}
for run_dir in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    for f in sorted(glob.glob(os.path.join(run_dir, "events*"))):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue
        available = set(acc.Tags()["scalars"])
        for key, tag in TAGS.items():
            if tag in available:
                for e in acc.Scalars(tag):
                    out.setdefault(e.step, {})[key] = e.value

for ep in sorted(out):
    row = out[ep]
    if "masked" not in row and "constr" not in row:
        continue
    print(ep, *(f"{row[k]:.4f}" if k in row else "-"
                for k in ("masked", "constr", "infeas", "improve")))
