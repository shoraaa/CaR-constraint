"""Validation curves for every arm, side by side.

The per-arm training logs each carry the masked re-construction score at every
validation epoch; this pulls them into one table so arms can be compared at
matched epochs rather than by eyeballing separate logs.
"""

import glob
import os
import re
import sys

# Anchor on the arrow: "AUG_Score" is a substring of "NO_AUG_Score", so an
# unanchored match silently reports the un-augmented number instead.
PATTERN = re.compile(r"\[w\. mask\].*?--> AUG_Score: ([0-9.]+)")


# The construction policy's own feasibility, which is what the admissibility
# supervision actually targets -- the objective above is measured on an
# already-feasible masked re-construction, so the slack loss should not be
# expected to move it.
INFEASIBLE = re.compile(r"\[Construction\].*?Infeasible rate: ([0-9.]+)%")


def infeasible_curve(path):
    values = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            if "Val Score" not in line or "[Construction]" not in line:
                continue
            found = INFEASIBLE.search(line)
            if found:
                values.append(float(found.group(1)))
    return values


def curve(path):
    scores = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            if "[w. mask]" not in line:
                continue
            found = PATTERN.search(line)
            if found:
                scores.append(float(found.group(1)))
    return scores


def main():
    logs = sorted(glob.glob("results/composition/train_*.log"))
    if len(sys.argv) > 1:
        logs = [p for p in logs if any(a in p for a in sys.argv[1:])]
    curves = {}
    for path in logs:
        name = os.path.basename(path)[len("train_"):-len(".log")]
        values = curve(path)
        if values:
            curves[name] = values
    if not curves:
        print("no validation points yet")
        return
    width = max(len(n) for n in curves) + 2
    epochs = max(len(v) for v in curves.values())
    header = "arm".ljust(width) + "".join(f"{'v' + str(i + 1):>9}" for i in range(epochs))
    print(header)
    print("-" * len(header))
    for name, values in sorted(curves.items()):
        print(name.ljust(width) + "".join(f"{v:>9.2f}" for v in values))
    print("\nv1..vN are successive validation points (every 2 epochs), masked "
          "re-construction objective; lower is better.")

    infeasible = {}
    for path in logs:
        name = os.path.basename(path)[len("train_"):-len(".log")]
        values = infeasible_curve(path)
        if values:
            infeasible[name] = values
    if infeasible:
        print("\nconstruction policy's own infeasibility (%), lower is better")
        print("-" * len(header))
        for name, values in sorted(infeasible.items()):
            print(name.ljust(width) + "".join(f"{v:>9.2f}" for v in values))


if __name__ == "__main__":
    main()
