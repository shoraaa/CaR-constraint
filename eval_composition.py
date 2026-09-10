"""Evaluate one checkpoint across the constraint compositions of VRPBLTW.

A model is trained on the full composition (backhaul + duration limit + time
windows) and evaluated, without retraining, on every subset of those three
requirements.  Each subset runs on the *same* instances with the dropped rows
set so that they cannot bind, so the grid is paired on geometry: every arm
answers about the identical instances under a different active set.

READ THE GRID DOWN A CELL, NOT ACROSS CELLS.  A subset is not a relaxation of
its supersets here, because two of the three rows are not pure bound-widenings:

  * `backhaul` is not a restriction at all.  `VRPBLTWEnv.step` applies
    `self.load -= demand` with no branch on the row, and a backhaul customer
    has negative demand, so visiting one *returns* delivery capacity (capped
    at the vehicle capacity).  Backhaul customers are refills.  On the 128
    test instances the minimum route count is ceil(L-B) = 4.17 with the row
    and ceil(L) = 5.45 without it, so the backhaul cell is ~1.32x easier and
    scores shorter -- CVRP above VRPB, which looks like an impossible
    inversion and is not one.  `--backhaul_absent` chooses how the row is
    dropped (`zero` keeps total delivery load equal to the backhaul cell's
    linehaul load; the original `abs` turns the pickups into deliveries and
    adds 26% load on top), but neither setting makes the axis monotone.
  * dropping `time_window` widens the windows to the depot horizon *and*
    zeros the service times, since service time is part of the temporal model
    the row introduces.  That is cheaper in duration, which matters wherever
    `route_limit` is also active.

Arm-against-arm comparison within one cell is unaffected by any of this, since
every arm is evaluated on the same instances under the same active set.

Usage:
    python eval_composition.py --checkpoint path/to/checkpoint.pt \
        --constraint_repr interface --results_csv results/composition/run.csv
"""

import argparse
import itertools
import os
import subprocess
import sys

ROWS = ("backhaul", "route_limit", "time_window")
PROBE_SUFFIX = {"draft_limit": "DL"}
NAMES = {
    (): "CVRP",
    ("backhaul",): "VRPB",
    ("route_limit",): "VRPL",
    ("time_window",): "VRPTW",
    ("backhaul", "route_limit"): "VRPBL",
    ("backhaul", "time_window"): "VRPBTW",
    ("route_limit", "time_window"): "VRPLTW",
    ("backhaul", "route_limit", "time_window"): "VRPBLTW",
}


def compositions(probe=()):
    """The 8 subsets of the trained rows, optionally with a probe row appended.

    A probe row is one no training composition contains, so `VRPBLTW+DL` asks
    the model about every requirement it was trained on plus one it has never
    seen -- the addition case, as opposed to the removal case the bare subsets
    test.
    """
    suffix = "".join("+" + PROBE_SUFFIX.get(row, row) for row in probe)
    for size in range(len(ROWS) + 1):
        for active in itertools.combinations(ROWS, size):
            yield active + tuple(probe), NAMES[active] + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--constraint_repr", default="attr")
    # The checkpoint is loaded with strict=True, so the arm flags that decide
    # which submodules exist have to match the ones the run was trained with.
    parser.add_argument("--slack_weight", type=float, default=0.0)
    parser.add_argument("--couple_rows", default="True")
    parser.add_argument("--arm_tag", default=None,
                        help="name for this arm in the results CSV; defaults to "
                             "the constraint_repr")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--backhaul_absent", default="zero",
                        choices=["zero", "abs"],
                        help="how the backhaul row is dropped; see the module "
                             "docstring. `abs` reproduces grids measured "
                             "before the setting existed.")
    parser.add_argument("--fill_missing_free", action="store_true",
                        help="publish all trained rows and fill each row outside "
                             "the effective subset with a nonbinding value")
    parser.add_argument("--problem_size", type=int, default=50)
    parser.add_argument("--test_episodes", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--improve_steps", type=int, default=5)
    parser.add_argument("--validation_improve_steps", type=int, default=20)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these variant names")
    parser.add_argument("--probe", nargs="*", default=[],
                        choices=sorted(PROBE_SUFFIX),
                        help="rows absent from every training composition, "
                             "appended to each cell of the grid")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="raw arguments appended to every test.py call; use "
                             "this to reproduce the host and masking regime the "
                             "checkpoint was trained under")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    for active, name in compositions(tuple(args.probe)):
        if args.only and name not in args.only:
            continue
        trained_active = tuple(row for row in active if row in ROWS)
        missing = tuple(row for row in ROWS if row not in trained_active)
        published = ROWS + tuple(args.probe) if args.fill_missing_free else active
        command = [
            args.python, "test.py",
            "--problem", "VRPBLTW",
            "--problem_size", str(args.problem_size),
            "--checkpoint", args.checkpoint,
            "--disable_preset_args",          # applies the published VRPBLTW presets
            "--constraint_repr", args.constraint_repr,
            "--slack_weight", str(args.slack_weight),
            "--couple_rows", str(args.couple_rows),
            "--test_episodes", str(args.test_episodes),
            "--test_batch_size", str(args.test_batch_size),
            "--improve_steps", str(args.improve_steps),
            "--validation_improve_steps", str(args.validation_improve_steps),
            "--results_csv", args.results_csv,
            "--variant_tag", name,
            "--backhaul_absent", args.backhaul_absent,
        ]
        if args.arm_tag:
            command += ["--arm_tag", args.arm_tag]
        command += [
            "--wandb_logger", "False",
        ]
        command += list(args.extra)
        if args.fill_missing_free:
            command += ["--filled_free_constraints"] + list(missing)
        command += ["--active_constraints"] + list(published)
        # Only the full composition has a reference solution shipped with the
        # benchmark; the subsets are compared arm against arm on the objective.
        if name != "VRPBLTW":
            command += ["--val_opt_path", "/nonexistent", "--test_opt_path", "/nonexistent"]
        print("\n>>> {} :: {}".format(name, " ".join(command)), flush=True)
        environment = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
        result = subprocess.run(command, env=environment)
        if result.returncode != 0:
            print(">>> {} FAILED with code {}".format(name, result.returncode), flush=True)


if __name__ == "__main__":
    main()
