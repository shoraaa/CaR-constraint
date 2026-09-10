"""Properties the consequence valuation has to have for the transfer claim.

These are cheap and CPU-only.  They check the structure the composition study
depends on: that rows carry no identity, that dropping a requirement is the same
as never having declared it, and that the pooled magnitude does not drift with
how many rows happen to be active.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.consequence import (  # noqa: E402
    ACTIVE_INDEX,
    CONSEQUENCE_DIM,
    MARGIN_INDEX,
    POST_INDEX,
    ConsequenceValuation,
)


def _consequence(batch=2, pomo=3, nodes=5, rows=3, active=None, seed=0):
    torch.manual_seed(seed)
    value = torch.rand(batch, pomo, nodes, rows, CONSEQUENCE_DIM)
    # The live state coordinate is candidate-independent by construction.
    value[..., 1] = value[:, :, :1, :, 1].expand_as(value[..., 1])
    flags = torch.ones(rows) if active is None else torch.tensor(active).float()
    value[..., ACTIVE_INDEX] = flags.view(1, 1, 1, rows).expand_as(value[..., ACTIVE_INDEX])
    return value * value[..., ACTIVE_INDEX].unsqueeze(-1)


def _model(seed=0, **kwargs):
    torch.manual_seed(seed)
    return ConsequenceValuation(context_dim=3, hidden_dim=8, **kwargs).eval()


def test_rows_are_permutation_invariant():
    """Reordering the active rows must not change the valuation."""
    model = _model()
    value = _consequence()
    permuted = value[:, :, :, [2, 0, 1], :]
    with torch.no_grad():
        assert torch.allclose(model(value), model(permuted), atol=1e-5)
        assert torch.allclose(
            model.query_context(value), model.query_context(permuted), atol=1e-5)


def test_an_inactive_row_is_the_same_as_no_row():
    """Dropping a requirement must equal never having declared it."""
    model = _model()
    two_of_three = _consequence(active=[1, 1, 0])
    only_two = two_of_three[:, :, :, :2, :]
    with torch.no_grad():
        assert torch.allclose(model(two_of_three), model(only_two), atol=1e-5)
        assert torch.allclose(
            model.query_context(two_of_three),
            model.query_context(only_two), atol=1e-5)


def test_pooled_magnitude_does_not_drift_with_row_count():
    """A one-row composition must not be systematically quieter than a three-row
    one purely because fewer terms entered the sum."""
    model = _model()
    with torch.no_grad():
        wide = model(_consequence(rows=3, seed=1)).abs().mean()
        narrow = model(_consequence(rows=1, seed=1)).abs().mean()
    # Within a factor of two; an unnormalized sum drifts by ~sqrt(3) plus the
    # extra rows' contributions and fails this comfortably.
    assert 0.5 < float(wide / narrow.clamp_min(1e-6)) < 2.0


def test_empty_active_set_reduces_to_zero():
    """No active row means no valuation, not a learned constant."""
    model = _model()
    with torch.no_grad():
        value = model(_consequence(active=[0, 0, 0]))
        context = model.query_context(_consequence(active=[0, 0, 0]))
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)
    assert torch.allclose(context, torch.zeros_like(context), atol=1e-6)


def test_the_coupler_changes_the_valuation():
    """The state coupling must actually do something, so its ablation means
    something."""
    coupled = _model(couple_rows=True)
    uncoupled = _model(couple_rows=False)
    value = _consequence()
    with torch.no_grad():
        assert not torch.allclose(coupled(value), uncoupled(value), atol=1e-4)


def test_the_token_reads_candidate_pressure():
    """Rows must be distinguishable exactly when they price the candidate set
    differently, so the token has to respond to pressure -- the consumption a
    candidate causes, post minus state.  (It must NOT respond to the margin;
    that is pinned separately.)"""
    model = _model()
    value = _consequence()
    heavier = value.clone()
    heavier[..., POST_INDEX] = heavier[..., POST_INDEX] * 0.1
    with torch.no_grad():
        assert not torch.allclose(
            model.query_context(value), model.query_context(heavier), atol=1e-4)


def test_the_token_leaks_no_margin_to_the_admissibility_head():
    """The head regresses the per-candidate margin from the decoder context,
    and that context is built from these tokens.  If the token carried margin
    information the head could read its own target, which is exactly the
    representational demand the supervision exists to create."""
    model = _model()
    value = _consequence()
    altered = value.clone()
    # Change ONLY the margin coordinate, leaving state/post/event untouched.
    altered[..., MARGIN_INDEX] = -altered[..., MARGIN_INDEX] * 0.5 + 0.25
    with torch.no_grad():
        assert torch.allclose(
            model.query_context(value), model.query_context(altered), atol=1e-6)


def test_the_valuation_still_reads_the_margin():
    """The per-candidate valuation must keep reading the margin even though the
    pooled token does not -- otherwise the margin ablation is vacuous."""
    model = _model()
    value = _consequence()
    altered = value.clone()
    altered[..., MARGIN_INDEX] = -altered[..., MARGIN_INDEX] * 0.5 + 0.25
    with torch.no_grad():
        assert not torch.allclose(model(value), model(altered), atol=1e-5)


def test_evaluate_matches_the_separate_calls():
    """Sharing the row tokens between the context and the valuation must be a
    pure refactor -- if it were not, arms trained before and after would not be
    comparable."""
    model = _model()
    value = _consequence()
    mask = torch.ones(*value.shape[:3])
    with torch.no_grad():
        context, bias = model.evaluate(value, mask)
        assert torch.allclose(context, model.query_context(value, mask), atol=1e-6)
        assert torch.allclose(bias, model(value, mask), atol=1e-6)


def test_the_candidate_mask_changes_the_token():
    """Pooling is over the candidates under consideration, so masking some out
    must change the token."""
    model = _model()
    value = _consequence()
    full = torch.ones(*value.shape[:3])
    partial = full.clone()
    partial[:, :, 2:] = 0.0
    with torch.no_grad():
        assert not torch.allclose(
            model.query_context(value, full),
            model.query_context(value, partial), atol=1e-4)


def test_an_unseen_row_needs_no_new_parameters():
    """A composition may add a row the model never trained with.

    Rows are pooled by shared weights with no identity, so the same parameters
    must accept a fourth row without a shape change -- that is what makes an
    unseen-resource probe possible at all.
    """
    model = _model()
    trained = _consequence(rows=3)
    probed = _consequence(rows=4)
    with torch.no_grad():
        three = model(trained)
        four = model(probed)
    assert three.shape == four.shape
    assert torch.isfinite(four).all()
    # And the extra row must actually be read, not silently dropped.
    assert not torch.allclose(three, four[:, :, :], atol=1e-6)


def test_the_margin_ablation_changes_the_valuation():
    """`interface_nomargin` must differ from `interface` on the same input."""
    full = _model(use_margin=True)
    blind = _model(use_margin=False)
    value = _consequence()
    with torch.no_grad():
        assert not torch.allclose(full(value), blind(value), atol=1e-4)


def test_soft_bound_is_the_identity_on_prisms_range():
    """In range the bend must change nothing, so a row that already respects its
    scale -- and every row under PRISM's own masked construction -- is
    untouched."""
    from envs.VRPBLTWEnv import VRPBLTWEnv

    inside = torch.linspace(-1.0, 1.0, 401)
    assert torch.allclose(VRPBLTWEnv._soft_bound(inside), inside, atol=0.0)


def test_soft_bound_is_monotone_bounded_and_smooth():
    """Out of range it must stay strictly order-preserving -- that is the whole
    point, the clip it replaces collapsed 43% of live time-window candidates
    onto one value -- and it must stay bounded and C1 at the join."""
    from envs.VRPBLTWEnv import VRPBLTWEnv

    x = torch.linspace(-40.0, 40.0, 20001, dtype=torch.float64)
    y = VRPBLTWEnv._soft_bound(x)
    assert (y.diff() > 0).all(), "not strictly increasing"
    assert y.abs().max() < 2.0, "left the bounded image"

    # derivative 1 on both sides of the join, so there is no kink at |x| = 1
    step = 1e-6
    for point in (1.0, -1.0):
        here = torch.tensor([point - step, point + step], dtype=torch.float64)
        slope = VRPBLTWEnv._soft_bound(here).diff() / (2 * step)
        assert abs(slope.item() - 1.0) < 1e-3, (point, slope)


def test_soft_bound_keeps_far_violations_rankable():
    """The failure this repairs: two candidates 1 and 3.2 horizons late used to
    both report exactly -1."""
    from envs.VRPBLTWEnv import VRPBLTWEnv

    late = VRPBLTWEnv._soft_bound(torch.tensor([-1.0, -3.2, -6.0]))
    assert late[0] > late[1] > late[2]
    assert (late[0] - late[1]).abs() > 0.5


def test_soft_bound_preserves_the_legality_sign():
    """A monotone bend through zero must not reclassify anything."""
    from envs.VRPBLTWEnv import VRPBLTWEnv

    x = torch.linspace(-9.0, 9.0, 4001)
    assert torch.equal(VRPBLTWEnv._soft_bound(x) >= 0.0, x >= 0.0)


def test_compaction_does_not_change_any_live_candidate():
    """The speed-up must be invisible in the numbers.

    Compaction restricts the per-candidate work to live candidates. Every live
    candidate must get exactly the value the dense path would have given it,
    and the query context -- which pools over candidates -- must be unchanged
    too, or the two paths are different models.
    """
    torch.manual_seed(11)
    # compact=True explicitly so this unit test cannot depend on CLI defaults.
    model = _model(compact=True)
    model.eval()
    assert model.compact, "test must exercise the compacted path"
    consequence = _consequence()
    mask = torch.rand(consequence.shape[:3]) > 0.55
    mask[..., 0] = True  # never leave a decision with nothing to choose

    with torch.no_grad():
        compact_context, compact_value = model.evaluate(consequence, mask)

        tokens, state, active = model._row_tokens(consequence, mask)
        reference_context = model._reduce(tokens, active, model.context)
        multipliers, _ = model._multipliers_from(tokens, state, active)
        reference_value = model._value(consequence, multipliers)

    assert torch.allclose(compact_context, reference_context, atol=1e-5), (
        (compact_context - reference_context).abs().max())
    live = mask
    assert torch.allclose(compact_value[live], reference_value[live], atol=1e-5), (
        (compact_value[live] - reference_value[live]).abs().max())


def test_compaction_preserves_the_training_signal():
    """Masked candidates had zero policy gradient and may be omitted safely."""
    dense = _model(compact=False)
    dense.train()
    compact = _model(compact=True)
    compact.load_state_dict(dense.state_dict())
    compact.train()
    consequence = _consequence(seed=17)
    mask = torch.rand(consequence.shape[:3]) > 0.4
    mask[..., 0] = True

    dense_context, dense_value = dense.evaluate(consequence, mask)
    compact_context, compact_value = compact.evaluate(consequence, mask)
    dense_loss = dense_context.square().sum() + dense_value[mask].square().sum()
    compact_loss = (compact_context.square().sum()
                    + compact_value[mask].square().sum())
    dense_loss.backward()
    compact_loss.backward()

    assert torch.equal(dense_context, compact_context)
    assert torch.equal(dense_value[mask], compact_value[mask])
    assert torch.equal(dense_loss, compact_loss)
    for dense_parameter, compact_parameter in zip(dense.parameters(),
                                                   compact.parameters()):
        assert torch.allclose(dense_parameter.grad, compact_parameter.grad,
                              atol=1e-5, rtol=1e-6)


def test_compaction_handles_a_full_candidate_set():
    """The flat path must also be equivalent before candidates are removed."""
    model = _model(compact=True)
    consequence = _consequence()
    full = torch.ones(consequence.shape[:3], dtype=torch.bool)
    with torch.no_grad():
        compact_context, compact_value = model.evaluate(consequence, full)
        tokens, state, active = model._row_tokens(consequence, full)
        reference_context = model._reduce(tokens, active, model.context)
        multipliers, _ = model._multipliers_from(tokens, state, active)
        reference_value = model._value(consequence, multipliers)
    assert torch.equal(compact_context, reference_context)
    assert torch.allclose(compact_value, reference_value, atol=1e-5)


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            test()
            print("PASS", name)
        except AssertionError:
            failures += 1
            print("FAIL", name)
    sys.exit(1 if failures else 0)
