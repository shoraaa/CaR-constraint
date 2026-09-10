"""What the composition grid's backhaul axis actually varies.

The grid looks like a subset lattice, so it invites being read across cells --
CVRP against VRPB.  It must not be, and these tests pin the two reasons.

First, the backhaul row is not a restriction: a backhaul customer has negative
demand and `step` applies `load -= demand`, so visiting one returns delivery
capacity.  The backhaul cell is the *easier* one, and no way of dropping the
row changes that.  Second, the row has no bound to widen, so dropping it is
done in the demand vector, and the original way of doing that (`abs`) turned
the pickups into deliveries and added 26% load on top of the first effect.
"""

import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.VRPBLTWEnv import VRPBLTWEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "CVRPBLTW" / "vrpbltw50_uniform.pkl"
ALL_ROWS = ("backhaul", "route_limit", "time_window")
NO_BACKHAUL = ("route_limit", "time_window")

pytestmark = pytest.mark.skipif(not DATASET.exists(),
                                reason="shipped VRPBLTW instances not present")


def _problems(count=16):
    """The shipped test instances, in `load_problems`' tuple layout."""
    raw = pickle.load(open(DATASET, "rb"))[:count]
    capacity = torch.tensor([float(r[3]) for r in raw]).view(-1, 1)
    return (torch.tensor([r[0] for r in raw], dtype=torch.float),
            torch.tensor([r[1] for r in raw], dtype=torch.float),
            torch.tensor([r[2] for r in raw], dtype=torch.float) / capacity,
            torch.tensor([float(r[4]) for r in raw]).view(-1, 1),
            torch.tensor([r[5] for r in raw], dtype=torch.float),
            torch.tensor([r[6] for r in raw], dtype=torch.float),
            torch.tensor([r[7] for r in raw], dtype=torch.float))


def _environment(active, absent="zero", count=16, filled_free=()):
    environment = VRPBLTWEnv(problem_size=50, pomo_size=40,
                             device=torch.device("cpu"),
                             active_constraints=active,
                             backhaul_absent=absent,
                             filled_free_constraints=filled_free)
    environment.load_problems(batch_size=count, rollout_size=40,
                              problems=_problems(count))
    return environment


def _demands(active, absent="zero"):
    return _environment(active, absent).depot_node_demand[:, 1:]


def test_dropping_backhaul_does_not_add_delivery_load():
    """Under the default the two cells differ in one thing, not two.

    Route count is bounded below by total delivery load, so leaving it equal
    to the backhaul cell's linehaul load keeps the capacity replenishment the
    only difference between the cells.  It does not make the axis monotone --
    see `test_a_backhaul_customer_returns_delivery_capacity`.
    """
    with_row = _demands(ALL_ROWS).clamp_min(0.0).sum(dim=1)
    without = _demands(NO_BACKHAUL).clamp_min(0.0).sum(dim=1)
    assert torch.allclose(without, with_row, atol=1e-6)


def test_dropping_backhaul_keeps_every_pickup_as_a_node_to_visit():
    """Capacity-free, not deleted: the customers still have to be served."""
    pickups = _demands(ALL_ROWS) < 0.0
    without = _demands(NO_BACKHAUL)
    assert pickups.any()
    assert (without[pickups] == 0.0).all()
    assert without.shape == pickups.shape


def test_the_abs_setting_adds_delivery_load_on_top():
    """A guard on the old behaviour, so the default cannot regress to it.

    Without this the first test passes trivially for anyone who assumes the
    two settings are equivalent; they are not, and this records the direction.
    """
    with_row = _demands(ALL_ROWS).clamp_min(0.0).sum(dim=1)
    as_abs = _demands(NO_BACKHAUL, absent="abs").clamp_min(0.0).sum(dim=1)
    assert (as_abs > with_row + 1e-6).all()


def test_dropping_backhaul_leaves_the_rollout_starts_paired():
    """POMO starts come from the positive-demand nodes (`load_problems`).

    Under `zero` the backhaul-free cell starts from the same customers as the
    backhaul cell, so the paired cells explore the same rollouts; under `abs`
    the pickups become eligible starts and the pairing is lost.
    """
    paired = _environment(NO_BACKHAUL).START_NODE
    assert torch.equal(paired, _environment(ALL_ROWS).START_NODE)
    assert not torch.equal(_environment(NO_BACKHAUL, "abs").START_NODE, paired)


def test_a_backhaul_customer_returns_delivery_capacity():
    """The reason the backhaul axis is not a relaxation axis.

    `step` applies `self.load -= demand` with no branch on the row, so a
    pickup adds to the remaining delivery capacity rather than consuming it.
    That makes the backhaul-active cell strictly easier than the same
    instances without the row, which is why it scores shorter -- the apparent
    CVRP-above-VRPB inversion in the grid.
    """
    environment = _environment(ALL_ROWS)
    environment.reset()
    demand = environment.depot_node_demand
    linehaul = int((demand[0] > 0).nonzero()[0])
    backhaul = int((demand[0] < 0).nonzero()[0])

    environment.step(torch.full((16, 40), linehaul), soft_constrained=True)
    after_delivery = environment.load[0, 0].item()
    environment.step(torch.full((16, 40), backhaul), soft_constrained=True)
    after_pickup = environment.load[0, 0].item()

    assert after_delivery < 1.0
    assert after_pickup > after_delivery
    assert after_pickup == pytest.approx(after_delivery - demand[0, backhaul].item())


def test_filled_free_rows_stay_published_with_nonbinding_values():
    environment = _environment(ALL_ROWS, filled_free=ALL_ROWS)

    assert environment.has_backhaul
    assert environment.has_route_limit
    assert environment.has_time_window
    assert (environment.depot_node_demand[:, 1:] >= 0.0).all()
    assert (environment.route_limit == environment.NO_ROUTE_LIMIT).all()
    assert (environment.depot_node_tw_start == 0.0).all()
    assert (environment.depot_node_tw_end == environment.depot_end).all()
    assert (environment.depot_node_service_time == 0.0).all()


def test_the_backhaul_cell_needs_fewer_routes_than_the_cell_without_it():
    """The replenishment, carried through to the quantity that sets cost.

    Total delivery load bounds route count from below.  With the row that
    bound is ceil(L - B) because pickups hand capacity back; without it the
    pickups are inert and the bound is ceil(L).
    """
    served = _demands(ALL_ROWS)
    linehaul = served.clamp_min(0.0).sum(dim=1)
    pickup = served.clamp_max(0.0).abs().sum(dim=1)
    without = _demands(NO_BACKHAUL).clamp_min(0.0).sum(dim=1)

    with_row = torch.clamp_min(linehaul - pickup, 0.0).ceil()
    assert (with_row < without.ceil()).all()
