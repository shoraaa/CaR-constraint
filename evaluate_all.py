"""Sweep every trained arm over the composition grid.

Each arm is evaluated under the host and masking regime it was trained in --
a model trained with hard masks and one trained without are not comparable on
the objective, so the grid is read within a (host, mask) group, never across.
The arm's flags are recovered from its log directory name, which
`run_paradigm_study.sh` and `run_composition_study.sh` encode.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# The `_rows` arm's declared widths. They live here because the directory name
# is the only record of how an arm was trained, and these two numbers have to
# match training exactly or the checkpoint will not load.
ROWS_NODE_DIM = 4
ROWS_CONTEXT_DIM = 4


def arm_flags(name):
    """Recover the training configuration from a log-directory name.

    Returns (direct, extra): arguments eval_composition understands itself, and
    raw arguments it forwards to every test.py call.
    """
    if "interface_nomargin" in name:
        representation = "interface_nomargin"
    elif "interface" in name:
        representation = "interface"
    else:
        representation = "attr"
    slack = re.search(r"_slack([0-9.]+)", name)

    if "pomo" in name:
        direct = ["--improve_steps", "0", "--validation_improve_steps", "0"]
        extra = ["--pomo_start", "True", "--diversity_loss", "False"]
        extra += (["--soft_constrained", "False", "--backhaul_mask", "hard"]
                  if "_full_" in name else ["--soft_constrained", "True"])
    else:
        direct = ["--improve_steps", "5", "--validation_improve_steps", "20"]
        extra = []
    direct += ["--constraint_repr", representation,
               "--slack_weight", slack.group(1) if slack else "0.0"]
    # The coupler modules are constructed either way, so a mismatch here would
    # not fail strict loading -- it would silently evaluate a coupled model
    # uncoupled.  Recover it from the name rather than trusting the default.
    direct += ["--couple_rows", "False" if "nocouple" in name else "True"]
    # A `_rows` arm replaces the encoder's per-problem feature tuple with the
    # pooled row summary and pins the query context's width. Both change
    # parameter SHAPES, so recovering them wrong fails loudly at load time
    # rather than silently evaluating a different model.
    if "_rows" in name:
        extra = extra + ["--node_repr", "rows",
                         "--node_repr_dim", str(ROWS_NODE_DIM),
                         "--consequence_context_dim", str(ROWS_CONTEXT_DIM)]
    return direct, extra


def newest_checkpoint(directory):
    """The last epoch checkpoint, never a best-by-validation one.

    Model selection must not see a held-out composition, and the shipped "best"
    checkpoints are selected on the training composition's score, so the final
    epoch is the honest choice and is identical across arms.
    """
    candidates = sorted(directory.glob("**/epoch-*.pt"),
                        key=lambda path: int(re.findall(r"epoch-(\d+)", path.name)[0]))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/composition")
    parser.add_argument("--results_csv", default="results/composition/grid.csv")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--backhaul_absent", default="zero",
                        choices=["zero", "abs"],
                        help="forwarded to eval_composition")
    # The published-CaR reference grid was measured at 1000; anything compared
    # against it has to use the same count.
    parser.add_argument("--test_episodes", type=int, default=128)
    parser.add_argument("--only_arms", nargs="*", default=None)
    parser.add_argument("--probe_draft", action="store_true",
                        help="append the unseen draft-limit row to every cell")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the commands without running them")
    args = parser.parse_args()

    for directory in sorted((ROOT / args.out).glob("train_*")):
        if not directory.is_dir():
            continue
        arm = directory.name[len("train_"):]
        if args.only_arms and arm not in args.only_arms:
            continue
        checkpoint = newest_checkpoint(directory)
        if checkpoint is None:
            print(">>> {}: no epoch checkpoint, skipping".format(arm), flush=True)
            continue
        direct, extra = arm_flags(arm)
        command = [sys.executable, "eval_composition.py",
                   "--checkpoint", str(checkpoint),
                   "--results_csv", args.results_csv,
                   "--arm_tag", arm,
                   "--test_batch_size", str(args.test_batch_size),
                   "--test_episodes", str(args.test_episodes),
                   "--backhaul_absent", args.backhaul_absent]
        command += direct
        if args.probe_draft:
            # eval_composition takes --probe before --extra, which is REMAINDER.
            command += ["--probe", "draft_limit"]
        if extra:
            # --extra is argparse.REMAINDER, so it has to come last.
            command += ["--extra"] + extra
        print("\n>>> {} :: {}".format(arm, " ".join(command)), flush=True)
        if args.dry_run:
            continue
        environment = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
        result = subprocess.run(command, env=environment, cwd=str(ROOT))
        if result.returncode != 0:
            print(">>> {} FAILED ({})".format(arm, result.returncode), flush=True)


if __name__ == "__main__":
    main()
