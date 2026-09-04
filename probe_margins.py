"""What does a *trained* policy actually see on each constraint row?

The saturation that `VRPBLTWEnv._soft_bound` repairs was first measured under a
nearest-neighbour rollout, which is only a proxy: a policy that routes badly
runs the clock up faster than a trained one and would overstate the problem.
This runs the real evaluation path -- test.py, unmodified, via runpy -- with a
hook on `_build_consequence`, so the numbers come from whatever checkpoint is
passed on the command line.

It records the *raw*, pre-bend coordinates, so one run reports both what the old
clip did and what the bend does with the same values.

    PY probe_margins.py --checkpoint <ckpt> --constraint_repr interface \
        --test_episodes 128 --test_batch_size 8 [any other test.py flag]
"""

import atexit
import os
import runpy
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.VRPBLTWEnv import VRPBLTWEnv  # noqa: E402

ROWS = ["capacity", "duration", "time_window", "draft_limit"]
STATS = {}
_original = VRPBLTWEnv._build_consequence


def _record(name, margin, state, live):
    if live is not None:
        margin, state = margin[live], state[live]
    bucket = STATS.setdefault(name, {"margin": [], "state": []})
    # subsample: a full sweep is ~10^8 values and the quantiles converge long
    # before that
    for key, value in (("margin", margin), ("state", state)):
        flat = value.detach().flatten().float().cpu()
        if flat.numel() > 200000:
            flat = flat[torch.randperm(flat.numel())[:200000]]
        bucket[key].append(flat)


def _hooked(self, demand_list, route_limit, route_post, arrival_time,
            depot_return_time, reach_time, tw_start_all, tw_end_all,
            draft_post=None, draft_limit=None):
    eps = 1.0e-6
    live = None
    mask = getattr(self, "ninf_mask", None)
    if mask is not None and mask.shape == demand_list.shape:
        live = mask == 0

    cap_post = self.load[:, :, None] - demand_list
    _record("capacity", torch.minimum(cap_post, 1.0 - cap_post),
            self.load[:, :, None].expand_as(demand_list), live)

    if self.has_route_limit:
        limit = route_limit.clamp(min=eps)
        _record("duration", (route_limit - route_post) / limit,
                self.length[:, :, None].expand_as(demand_list) / limit, live)

    if self.has_time_window:
        horizon = max(self.depot_end, eps)
        _record("time_window",
                torch.minimum(tw_end_all - arrival_time,
                              self.depot_end - depot_return_time) / horizon,
                self.current_time[:, :, None].expand_as(demand_list) / horizon,
                live)

    if self.has_draft_limit and draft_post is not None:
        _record("draft_limit", draft_limit - draft_post,
                self.served[:, :, None].expand_as(demand_list), live)

    return _original(self, demand_list, route_limit, route_post, arrival_time,
                     depot_return_time, reach_time, tw_start_all, tw_end_all,
                     draft_post, draft_limit)


def _report():
    if not STATS:
        print("\n[probe] no consequence rows were built -- is this an "
              "`interface` checkpoint?", file=sys.stderr)
        return
    # Defined here rather than imported from the env, so the probe measures a
    # checkpoint against the code it actually trained under -- pointing it at a
    # pre-fix tree must not silently evaluate the policy under the fix.
    def bend(x):
        magnitude = x.abs()
        return torch.where(magnitude <= 1.0, x,
                           torch.sign(x) * (2.0 - 1.0 / magnitude.clamp(min=1.0)))

    print("\n" + "=" * 78)
    print("MARGIN / STATE AS THE TRAINED POLICY SEES THEM (live candidates only)")
    print("=" * 78)
    header = ("{:<13}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}"
              .format("row", "n(M)", "clip@-1", "bend_min", "p01", "std_clip",
                      "std_bend"))
    print(header)
    for name in ROWS:
        if name not in STATS:
            continue
        raw = torch.cat(STATS[name]["margin"])
        clipped = raw.clamp(-1.0, 1.0)
        bent = bend(raw)
        v = raw.numpy()
        print("{:<13}{:>9.1f}{:>8.1f}%{:>9.3f}{:>9.2f}{:>9.3f}{:>9.3f}".format(
            name, raw.numel() / 1e6,
            100.0 * float((raw <= -1.0).float().mean()),
            float(bent.min()), float(np.percentile(v, 1)),
            float(clipped.std()), float(bent.std())))
    print("\nclip@-1 = share of live candidates the OLD clamp collapsed onto a "
          "single value.\nstd_bend > std_clip is the signal the bend restores.")
    print("-" * 78)
    print("{:<13}{:>12}{:>12}{:>12}".format("row", "state p99", "state max",
                                            "state>1"))
    for name in ROWS:
        if name not in STATS:
            continue
        s = torch.cat(STATS[name]["state"]).numpy()
        print("{:<13}{:>12.2f}{:>12.2f}{:>11.1f}%".format(
            name, float(np.percentile(s, 99)), float(s.max()),
            100.0 * float(np.mean(s > 1.0))))
    print("=" * 78)


if __name__ == "__main__":
    VRPBLTWEnv._build_consequence = _hooked
    atexit.register(_report)
    sys.argv = ["test.py"] + sys.argv[1:]
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "test.py"), run_name="__main__")
