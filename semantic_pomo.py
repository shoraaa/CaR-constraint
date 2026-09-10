#!/usr/bin/env python3
"""Train/evaluate POMO with PRISM's family-level construction semantics.

This is intentionally separate from ``unified.py``.  It does not import the
URS environment or mask registry; URS defaults are merely used as experiment
defaults.  The same executable accumulator/precedence interpreter handles both
the 110 benchmark variants and undeclared-at-training variants such as EVRP and
VRPDL.
"""

import argparse
import json
import os
import random
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
URS_ROOT = os.path.abspath(os.path.join(HERE, '..', 'URS'))
for path in (HERE, URS_ROOT, ROOT):
    if path in sys.path:
        sys.path.remove(path)
for path in (HERE, URS_ROOT, ROOT):
    sys.path.insert(0, path)

from problem_data import (decoder_problem, generate_aevrp_data,
                          generate_evrp_data, generate_evrpl_data,
                          generate_evrptw_data, generate_extended_problem,
                          generate_vrpdb_data, generate_vrpdbtw_data,
                          generated_problem)  # noqa: E402
from semantic_env import SemanticPOMOEnv  # noqa: E402
from unified_model import UnifiedCaRPOMO  # noqa: E402


URS_TRAIN_VARIANTS = (
    'atsp', 'acvrp', 'tsp', 'op', 'pctsp', 'cvrp', 'cvrpb', 'cvrptw',
    'ocvrp', 'ocvrptw', 'pdtsp')


def _slice(data, index):
    return {
        key: (value[index:index + 1]
              if torch.is_tensor(value) and value.ndim else value)
        for key, value in data.items()
    }


def _vrpdl_data(name, size, count, seed):
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand(count, size + 1, 2, generator=generator)
    demand = torch.randint(1, 10, (count, size + 1), generator=generator).float() / 50
    demand[:, 0] = 0
    limit = demand + (1.0 - demand) * torch.rand(
        count, size + 1, generator=generator)
    limit[:, 0] = 1.0
    data = {'xy': xy, 'demand': demand, 'draft_limit': limit}
    if name.endswith('tw'):
        radius = torch.linalg.vector_norm(xy - xy[:, :1], dim=-1)
        horizon = 3.2
        data['tw_start'] = torch.zeros(count, size + 1)
        data['tw_end'] = torch.cat((
            torch.full((count, 1), horizon),
            horizon - radius[:, 1:] - 0.2), dim=1)
        data['service_time'] = torch.cat((
            torch.zeros(count, 1), torch.full((count, size), 0.2)), dim=1)
    return data


def make_problems(name, size, count, seed):
    """Generate instances, then lower them through PRISM's schema compiler."""
    name = name.lower()
    special = {
        'evrp': generate_evrp_data,
        'evrptw': generate_evrptw_data,
        'aevrp': generate_aevrp_data,
        'evrpl': generate_evrpl_data,
        'vrpdb': generate_vrpdb_data,
        'vrpdbtw': generate_vrpdbtw_data,
    }
    if name in special:
        data = special[name](size, count, seed=seed)
        return [decoder_problem(name, _slice(data, i)) for i in range(count)]
    if name in ('tspdl', 'vrpdl', 'cvrpdl', 'vrpdltw', 'cvrpdltw'):
        data = _vrpdl_data(name, size, count, seed)
        return [decoder_problem(name, _slice(data, i)) for i in range(count)]
    try:
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        return [generated_problem(name, size) for _ in range(count)]
    except (NotImplementedError, ValueError):
        torch.manual_seed(seed)
        return [generate_extended_problem(name, size) for _ in range(count)]
    finally:
        torch.random.set_rng_state(state)


def model_params(args, eval_type):
    return dict(
        embedding_dim=args.embedding_dim,
        encoder_layer_num=args.encoder_layer_num,
        ff_hidden_dim=args.ff_hidden_dim,
        logit_clipping=args.logit_clipping,
        demand_max1=True,
        eval_type=eval_type,
        head_num=args.head_num,
        qkv_dim=args.qkv_dim,
        constraint_repr='interface',
        node_repr='rows',
        node_repr_dim=args.node_repr_dim,
        consequence_context_dim=args.consequence_context_dim,
        consequence_hidden_dim=args.consequence_hidden_dim,
        consequence_norm=args.consequence_norm,
        consequence_compact=True,
        consequence_compact_rows=True,
        consequence_checkpoint=not args.no_consequence_checkpoint,
        couple_rows=not args.no_couple_rows,
        slack_weight=0.0,
        slack_stride=5)


def rollout(model, problems, pomo_size, device):
    env = SemanticPOMOEnv(problems, pomo_size=pomo_size, device=device)
    reset, _, _ = env.reset()
    model.pre_forward(reset, problems[0].get('name', 'schema'),
                      torch.zeros(13, device=device))
    state, reward, done = env.pre_step()
    probabilities = []
    while not done:
        selected, probability = model(state, env.get_local_feature())
        state, reward, done = env.step(selected)
        if probability is not None:
            probabilities.append(probability)
    return reward, probabilities, env.routes


def train(args):
    device = torch.device('cpu' if args.cuda < 0 or not torch.cuda.is_available()
                          else 'cuda:%d' % args.cuda)
    random.seed(args.seed); torch.manual_seed(args.seed)
    model = UnifiedCaRPOMO(**model_params(args, 'sampling')).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer_lr,
                                  weight_decay=args.weight_decay)
    variants = [value.strip() for value in args.train_variants.split(',')
                if value.strip()]
    history = []
    for epoch in range(1, args.training_epochs + 1):
        for batch in range(args.batches_per_epoch):
            variant = random.choice(variants)
            problems = make_problems(
                variant, args.problem_size, args.batch_size,
                args.seed + epoch * args.batches_per_epoch + batch)
            model.train(); model.set_decoder_type('sampling')
            reward, probabilities, _ = rollout(
                model, problems, args.pomo_size, device)
            advantage = reward - reward.mean(dim=1, keepdim=True)
            log_probability = torch.stack(probabilities, dim=2).log().sum(dim=2)
            loss = (-advantage * log_probability).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            history.append(float(loss.detach()))
        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_params': model_params(args, 'greedy'),
            'train_variants': variants}
        os.makedirs(args.output, exist_ok=True)
        torch.save(checkpoint, os.path.join(args.output,
                                           'checkpoint_%d.pt' % epoch))
        print('epoch=%d loss=%.6f' % (epoch, sum(history[-args.batches_per_epoch:])
                                     / args.batches_per_epoch))


def solve(args):
    device = torch.device('cpu' if args.cuda < 0 or not torch.cuda.is_available()
                          else 'cuda:%d' % args.cuda)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    params = checkpoint.get('model_params', model_params(args, 'greedy'))
    params['eval_type'] = 'greedy'
    model = UnifiedCaRPOMO(**params).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    problems = make_problems(args.variant, args.problem_size,
                             args.episodes, args.seed)
    with torch.inference_mode():
        reward, _, routes = rollout(model, problems, args.pomo_size, device)
    best_reward, best = reward.max(dim=1)
    records = []
    for row, index in enumerate(best.tolist()):
        decoder = SemanticPOMOEnv([problems[row]], pomo_size=1).decoders[0]
        solution = decoder.evaluate(routes[row][index])
        records.append({'variant': args.variant, 'instance': row,
                        'objective': solution['objective'],
                        'feasible': solution['feasible'],
                        'route': routes[row][index]})
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'solve'))
    parser.add_argument('--variant', default='evrp')
    parser.add_argument('--train_variants', default=','.join(URS_TRAIN_VARIANTS))
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--pomo_size', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--training_epochs', type=int, default=500)
    parser.add_argument('--batches_per_epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--output', default='./result_semantic_pomo')
    parser.add_argument('--checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--encoder_layer_num', type=int, default=12)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--logit_clipping', type=float, default=50)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--node_repr_dim', type=int, default=5)
    parser.add_argument('--consequence_context_dim', type=int, default=1)
    parser.add_argument('--consequence_hidden_dim', type=int, default=16)
    parser.add_argument('--consequence_norm', choices=('layer', 'none'),
                        default='layer')
    parser.add_argument('--no_consequence_checkpoint', action='store_true')
    parser.add_argument('--no_couple_rows', action='store_true')
    args = parser.parse_args()
    if args.command == 'solve' and not args.checkpoint:
        parser.error('solve requires --checkpoint')
    (train if args.command == 'train' else solve)(args)


if __name__ == '__main__':
    main()

