"""Render the composition grid CSVs into one arm-against-arm table."""

import argparse
import csv
import math
from collections import defaultdict

BASE_ORDER = ["CVRP", "VRPB", "VRPL", "VRPTW", "VRPBL", "VRPBTW", "VRPLTW", "VRPBLTW"]


def order_for(arms):
    """Base compositions first, then any probe variants the CSVs contain."""
    seen = {variant for rows in arms.values() for variant in rows}
    extra = sorted(seen - set(BASE_ORDER))
    return [name for name in BASE_ORDER if name in seen] + extra
TRAINED = "VRPBLTW"


def load(paths):
    rows = defaultdict(dict)
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                arm = row.get("arm") or row["constraint_repr"]
                rows[arm][row["variant"]] = row
    return rows


def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--metric", default="masked_aug",
                        help="masked_aug (feasible re-construction) or improve_aug")
    parser.add_argument("--infeasible", default="construct_sol_infeasible")
    parser.add_argument("--baseline", default="attr",
                        help="arm every other arm's delta is measured against")
    args = parser.parse_args()

    arms = load(args.csv)
    # The baseline first, so the deltas read left to right.
    names = ([args.baseline] if args.baseline in arms else []) + [
        name for name in arms if name != args.baseline]
    header = "{:<9}".format("variant")
    for name in names:
        header += "  {:>14} {:>8}".format(name[:14], "infsb%")
    print(header)
    print("-" * len(header))

    deltas = defaultdict(list)
    for variant in order_for(arms):
        line = "{:<9}".format(variant + ("*" if variant == TRAINED else ""))
        reference = float("nan")
        for index, name in enumerate(names):
            row = arms[name].get(variant)
            if row is None:
                line += "  {:>14} {:>8}".format("-", "-")
                continue
            objective = number(row, args.metric)
            infeasible = number(row, args.infeasible)
            if index == 0:
                reference = objective
                line += "  {:>14.4f} {:>7.2f}%".format(objective, infeasible)
                continue
            cell = "{:.4f}".format(objective)
            if not math.isnan(reference) and reference != 0 and not math.isnan(objective):
                delta = 100.0 * (objective - reference) / reference
                deltas[name].append((variant, delta))
                cell += " ({:+.1f}%)".format(delta)
            line += "  {:>14} {:>7.2f}%".format(cell, infeasible)
        print(line)

    if deltas:
        print("-" * len(header))
        for name, entries in deltas.items():
            held = [value for variant, value in entries if variant != TRAINED]
            if held:
                print("{:<20} mean delta vs {} on held-out compositions: {:+.2f}%".format(
                    name, names[0], sum(held) / len(held)))
        print("* the composition every arm was trained on; delta is vs " + names[0])


if __name__ == "__main__":
    main()
