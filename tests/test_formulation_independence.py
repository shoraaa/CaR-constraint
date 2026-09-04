"""The parameter layout must stop depending on which formulation is being run.

CaR selects widths from tables of problem names: the encoder's node embedding
(`SINGLEModel.py`, `Linear(3/4/5 + feature_plus, ...)`) and the decoder's query
projection (`Wq_last`, `embedding_dim + 1/2/3/4`).  That is what makes it a
single-task architecture -- one parameter set cannot span formulations even
when the underlying requirements are the same.

PRISM has no such table: the encoder appends a fixed-width pooled summary over
whatever rows are active (`net.py`, `augment_nodes` -> `ResourcePool`), so its
shapes are the same for every problem it can express.  These tests assert the
port reaches the same property, and -- just as importantly -- that it does NOT
hold for the published baseline, so the test would catch the flags silently
failing to take effect.
"""

import os
import sys

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SINGLEModel import SINGLE_Decoder, SINGLE_Encoder  # noqa: E402

# Problems whose named layouts genuinely differ: 3, 5 and 5 encoder columns and
# 1, 2 and 3 query columns respectively.
PROBLEMS = ["CVRP", "VRPTW", "VRPBLTW"]

BASE = dict(
    embedding_dim=128, encoder_layer_num=2, supplement_feature_dim=17,
    impr_encoder_start_idx=0, improve_steps=0, qkv_dim=16, head_num=8,
    ff_hidden_dim=512, logit_clipping=10.0, pip_decoder=False,
    extra_feature=False, norm="instance", norm_loc="norm_last",
    use_fast_attention=False, pairwise_merge=[], which_feature=None,
    succ_attention_bias=1.0, n2s_decoder=False, unified_encoder=True,
    unified_decoder=True, with_RNN=False, dual_decoder=False,
    aspect_num=False, with_explore_stat_feature=False, k_max=False,
    rm_num=False, gumbel=False,
)


def _shapes(problem, **overrides):
    params = dict(BASE, problem=problem, **overrides)
    encoder = SINGLE_Encoder(**params)
    decoder = SINGLE_Decoder(**params)
    return {
        "embedding_node": encoder.embedding_node.in_features,
        "embedding_depot": encoder.embedding_depot.in_features,
        "Wq_last": decoder.Wq_last.in_features,
    }


def test_the_published_layout_depends_on_the_formulation():
    """Guard on the baseline: if this ever passes, the test below is vacuous."""
    seen = {problem: _shapes(problem) for problem in PROBLEMS}
    assert len({tuple(sorted(v.items())) for v in seen.values()}) == len(PROBLEMS), seen


def test_rows_and_a_declared_context_remove_every_problem_lookup():
    """Encoder and decoder both, since fixing only one leaves a lookup behind."""
    overrides = dict(constraint_repr="interface", node_repr="rows",
                     node_repr_dim=4, consequence_context_dim=4)
    seen = {problem: _shapes(problem, **overrides) for problem in PROBLEMS}
    distinct = {tuple(sorted(v.items())) for v in seen.values()}
    assert len(distinct) == 1, seen


def test_rows_alone_still_leaves_the_query_projection_keyed_by_problem():
    """`--node_repr rows` fixes the encoder only; the decoder needs its own flag.

    This is the gap that made the interface arms non-generalist even with the
    row encoder: `Wq_last` was still `embedding_dim + ATTR_WIDTH[problem]`.
    """
    seen = {problem: _shapes(problem, constraint_repr="interface",
                             node_repr="rows", node_repr_dim=4)
            for problem in PROBLEMS}
    encoder_widths = {v["embedding_node"] for v in seen.values()}
    query_widths = {v["Wq_last"] for v in seen.values()}
    assert len(encoder_widths) == 1, seen
    assert len(query_widths) == len(PROBLEMS), seen


def test_the_declared_context_is_off_by_default():
    """Zero must keep the published table, or old checkpoints stop loading."""
    for problem in PROBLEMS:
        assert (_shapes(problem, constraint_repr="interface")["Wq_last"]
                == _shapes(problem)["Wq_last"])


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            test()
            print("PASS", name)
        except AssertionError as error:
            failures += 1
            print("FAIL", name, error)
    sys.exit(1 if failures else 0)
