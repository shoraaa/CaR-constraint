"""The second POMO path executes families, not named URS masks."""

import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CAR = os.path.join(ROOT, 'baselines', 'CaR-constraint')
URS = os.path.join(ROOT, 'baselines', 'URS')
for path in (CAR, URS, ROOT):
    if path in sys.path:
        sys.path.remove(path)
for path in (CAR, URS, ROOT):
    sys.path.insert(0, path)

from problem_data import (BENCHMARK_VARIANTS, decoder_problem,
                          generate_evrp_data, generate_extended_problem,
                          generated_problem)  # noqa: E402
from semantic_env import SemanticPOMOEnv  # noqa: E402
from semantic_pomo import make_problems  # noqa: E402
from unified_model import UnifiedCaRPOMO  # noqa: E402


def _model():
    return UnifiedCaRPOMO(
        embedding_dim=32, encoder_layer_num=1, ff_hidden_dim=64,
        logit_clipping=10, demand_max1=True, eval_type='greedy',
        head_num=4, qkv_dim=8, constraint_repr='interface',
        node_repr='rows', node_repr_dim=5, consequence_context_dim=1,
        consequence_hidden_dim=8, consequence_norm='none',
        consequence_compact=True, consequence_compact_rows=True,
        consequence_checkpoint=False, couple_rows=True, slack_weight=0.0,
        slack_stride=5).eval()


def _rollout(problem):
    env = SemanticPOMOEnv([problem], pomo_size=3)
    reset, _, _ = env.reset()
    model = _model()
    model.pre_forward(reset, problem['name'], torch.zeros(13))
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state, env.get_local_feature())
        state, reward, done = env.step(selected)
    return env, reward


def test_unseen_accumulator_and_draft_variants_roll_out_feasibly():
    data = generate_evrp_data(6, 1, seed=17)
    evrp = decoder_problem('evrp', data)
    draft = generate_extended_problem('cvrpdl', 6)
    for problem in (evrp, draft):
        env, reward = _rollout(problem)
        assert torch.isfinite(reward).all()
        assert all(
            env.decoders[0].evaluate(route)['feasible']
            for route in env.routes[0])


def test_precedence_uses_the_same_family_level_refresh():
    problem = generated_problem('pdtsp', 6)
    env, reward = _rollout(problem)
    assert torch.isfinite(reward).all()
    assert {row.operator for row in env.decoders[0].programs} == {'precedence'}


def test_resource_name_does_not_select_masking_code():
    problem = generate_extended_problem('cvrpdl', 6)
    problem['resources'][0]['name'] = 'previously_unseen_row_name'
    env, reward = _rollout(problem)
    assert torch.isfinite(reward).all()
    assert env.decoders[0].programs[-1].name == 'previously_unseen_row_name'


def test_semantic_path_compiles_all_110_benchmark_variants():
    assert len(BENCHMARK_VARIANTS) == 110
    for index, name in enumerate(BENCHMARK_VARIANTS):
        problems = make_problems(name, 6, 1, 1000 + index)
        reset, _, _ = SemanticPOMOEnv(problems, pomo_size=2).reset()
        assert reset.node_resource_features.shape[0] == 1
