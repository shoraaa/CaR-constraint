"""With its rows all active, the environment must be upstream CaR, exactly.

The transfer claim is that the consequence interface helps CaR -- not that a
modified CaR helps.  That only holds if the additions are inert on the default
path, so this pins the default path against the vendored commit rather than
against a description of it: the upstream file is recovered from git and run
side by side.

`git show` is the source of truth here on purpose.  A vendored copy of the
original would drift with the working tree; the commit cannot.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_composition_grid import _problems, DATASET  # noqa: E402

# The commit that vendored CaR in, before any of our changes.
UPSTREAM = "76530456"
RELATIVE = "baselines/CaR-constraint/envs/VRPBLTWEnv.py"


def _upstream_environment_class():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    source = subprocess.run(["git", "show", "{}:{}".format(UPSTREAM, RELATIVE)],
                            cwd=root, capture_output=True, text=True, check=True).stdout
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(source)
    handle.close()
    spec = importlib.util.spec_from_file_location("upstream_vrpbltw", handle.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(handle.name)
    return module.VRPBLTWEnv


def _rollout(environment_class):
    """A deterministic rollout: always the lowest-index admissible node.

    No policy, no sampling, no seed dependence -- the trace is a function of
    the environment alone, so any behavioural difference has to show up.
    """
    environment = environment_class(problem_size=50, pomo_size=40,
                                    device=torch.device("cpu"))
    environment.load_problems(batch_size=8, rollout_size=40, problems=_problems(8))
    environment.reset()
    environment.pre_step()
    trace, done = [], False
    while not done:
        mask = environment.ninf_mask
        if mask is None:
            mask = torch.zeros(8, 40, 51)
        choice = (mask > float("-inf")).float().argmax(dim=-1)
        _, reward, done, infeasible = environment.step(choice, soft_constrained=False,
                                                       backhaul_mask="hard")
        trace.append((environment.load.clone(), environment.length.clone(),
                      environment.current_time.clone(), environment.ninf_mask.clone()))
    return trace, reward, infeasible, environment.selected_node_list.clone()


pytestmark = pytest.mark.skipif(not DATASET.exists(),
                                reason="shipped VRPBLTW instances not present")


def test_the_default_path_is_bit_identical_to_vendored_car():
    ours = _rollout(__import__("envs.VRPBLTWEnv", fromlist=["VRPBLTWEnv"]).VRPBLTWEnv)
    theirs = _rollout(_upstream_environment_class())

    our_trace, our_reward, our_infeasible, our_nodes = ours
    their_trace, their_reward, their_infeasible, their_nodes = theirs

    assert len(our_trace) == len(their_trace)
    assert torch.equal(our_nodes, their_nodes)
    assert torch.equal(our_reward, their_reward)
    assert torch.equal(our_infeasible, their_infeasible)
    for step, (mine, yours) in enumerate(zip(our_trace, their_trace)):
        for name, a, b in zip(("load", "length", "current_time", "ninf_mask"),
                              mine, yours):
            assert torch.equal(a, b), "{} diverged at step {}".format(name, step)


def test_the_additions_are_reachable_at_all():
    """Guard against the test above passing because nothing was ever added."""
    from envs.VRPBLTWEnv import VRPBLTWEnv

    environment = VRPBLTWEnv(problem_size=50, pomo_size=40,
                             device=torch.device("cpu"))
    assert environment.active_constraints == ("backhaul", "route_limit", "time_window")
    assert environment.backhaul_absent == "zero"
    assert hasattr(environment, "_consequence_row")
