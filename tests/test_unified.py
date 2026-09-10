"""CPU smoke tests for the 110-variant CaR-POMO path."""

import os
import sys

import torch


CAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URS_ROOT = os.path.abspath(os.path.join(CAR_ROOT, '..', 'URS'))
for path in (CAR_ROOT, URS_ROOT):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, CAR_ROOT)
sys.path.insert(0, URS_ROOT)

from env.UniVRPEnv import UniVRPEnv  # noqa: E402
from env.mask.mask_registry import mask_registry  # noqa: E402
from problem.ProblemRep import get_problem_representations  # noqa: E402
from problem.ProblemSet import ProblemSet  # noqa: E402
from unified_model import UnifiedCaRPOMO  # noqa: E402


def _params(eval_type='greedy'):
    return dict(
        embedding_dim=32, encoder_layer_num=1, ff_hidden_dim=64,
        logit_clipping=10, demand_max1=True, eval_type=eval_type,
        head_num=4, qkv_dim=8, constraint_repr='interface',
        consequence_context_dim=1, consequence_hidden_dim=8,
        consequence_norm='none', consequence_compact=True,
        consequence_compact_rows=True, consequence_checkpoint=False,
        couple_rows=True, slack_weight=0.0, slack_stride=5)


def test_every_published_variant_has_a_urs_mask_path():
    variants = ProblemSet.get(name='all_evaluated_list')
    assert len(variants) == 110
    assert set(variants) == set(get_problem_representations())
    assert all(callable(mask_registry.build_combined_mask(name))
               for name in variants)


def test_one_policy_rolls_out_distinct_variant_families():
    model = UnifiedCaRPOMO(**_params()).eval()
    representations = get_problem_representations()
    for name in ('tsp', 'atsp', 'cvrp', 'cvrpbltw', 'pdtsp', 'aopdcvrp'):
        env = UniVRPEnv(consequence_interface=True,
                        consequence_compact_rows=True)
        env.load_problems(
            batch_size=1, problem_name=name, problem_size=8, pomo_size=8,
            device=torch.device('cpu'),
            capacity=50 if 'vrp' in name else None)
        reset, _, _ = env.reset()
        representation = torch.tensor(representations[name],
                                      dtype=torch.float32)
        model.pre_forward(reset, name, representation)
        state, reward, done = env.pre_step()
        mask = mask_registry.build_combined_mask(name)
        while not done:
            selected, _ = model(state, env.get_local_feature())
            state, reward, done = env.step(selected, mask_fn=mask)
        assert torch.isfinite(reward).all()

