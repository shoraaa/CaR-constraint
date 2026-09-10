#!/usr/bin/env python3
"""Train or solve with one CaR-POMO model across URS's 110 variants.

Examples:
  python unified.py train --cuda 0
  python unified.py solve --model_load result_train/.../best_checkpoint.pt \
      --problem_set all_evaluated_list --test_scale_list 100 \
      --test_episodes 1000
"""

import argparse
import logging
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
URS_ROOT = os.path.abspath(os.path.join(HERE, '..', 'URS'))
# URS uses top-level imports (env.*, model.*, problem.*), so its root must
# precede CaR's unrelated ``utils.py`` module. ``unified_model`` exists only in
# the CaR directory and is still found on the next search entry.
if URS_ROOT in sys.path:
    sys.path.remove(URS_ROOT)
sys.path.insert(0, URS_ROOT)

import pytz  # noqa: E402
import torch  # noqa: E402

from args import (build_consequence_params, build_resume_params,
                  obtain_all_hyperparameters)  # noqa: E402
from problem.ProblemSet import ProblemSet  # noqa: E402
from unified_model import UnifiedCaRPOMO  # noqa: E402
from utils.utils import (copy_all_src, create_logger, get_result_folder,
                         print_startup, seed_everything)  # noqa: E402


def _problem_list(value):
    value = value.strip().lower()
    if value in ('tsplib', 'cvrplib'):
        return [value.replace('lib', '')], value, True
    if value.endswith('_list'):
        return ProblemSet.get(name=value), value, False
    selected = [item.strip() for item in value.split(',') if item.strip()]
    unknown = sorted(set(selected) - set(ProblemSet.get_all_problem_names()))
    if not selected or unknown:
        raise ValueError('invalid --problem_set: %s' % value)
    return selected, '_'.join(selected), False


def _model_params(args):
    params = {
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'ff_hidden_dim': args.ff_hidden_dim,
        'logit_clipping': args.logit_clipping,
        'demand_max1': not args.no_demand_max1,
        'eval_type': args.eval_type,
        'head_num': args.head_num,
        'qkv_dim': args.qkv_dim,
    }
    consequence_model, _ = build_consequence_params(args)
    params.update(consequence_model)
    return params


def _env_params(args):
    _, consequence_env = build_consequence_params(args)
    return consequence_env


def train(args):
    train_problems = ProblemSet.get(name='train_problem_list')
    if args.add_training_problems:
        train_problems = sorted(dict.fromkeys(
            train_problems + args.add_training_problems), key=len)
    validation = sorted(dict.fromkeys(
        train_problems + ProblemSet.get(name=args.validation_problem_set)),
        key=len)

    env_params = {
        'problem_size': args.problem_size,
        'capacity': args.capacity,
        'data_dir': args.data_dir,
        **_env_params(args),
    }
    model_params = _model_params(args)
    optimizer_params = {
        'optimizer_type': args.optimizer_type,
        'optimizer': {'lr': args.optimizer_lr,
                      'weight_decay': args.weight_decay},
        'lr_decay_epoch': args.lr_decay_epoch,
    }
    trainer_params = {
        'use_cuda': args.cuda >= 0 and torch.cuda.is_available(),
        'cuda_device_num': args.cuda,
        'epochs': args.training_epochs,
        'batches_per_epoch': args.batches_per_epoch,
        'batch_size': args.batch_size,
        'validation_scale': args.validation_scale,
        'validation_episodes': args.validation_episodes,
        'validation_batch_size': args.validation_batch_size,
        'train_problem_list': train_problems,
        'validation_problem_list': validation,
        'validation_generated': args.validation_generated,
        'validation_seed': args.validation_seed,
        'logging': {
            'model_save_interval': args.model_save_interval,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_score.json'},
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'},
        },
        'model_load': build_resume_params(args),
    }

    now = datetime.now(pytz.timezone('Asia/Shanghai'))
    logger_params = {'log_file': {
        'desc': '_car_pomo_unified', 'filename': 'run.log',
        'filepath': './result_train/{}/{}{{desc}}'.format(
            now.strftime('%Y-%m-%d'), now.strftime('%Y%m%d_%H%M%S'))}}
    seed_everything(args.seed)
    create_logger(**logger_params)
    print_startup(args=args, logger=logging.getLogger('root'),
                  result_folder=get_result_folder(),
                  log_filename='run.log', phase='training')

    from Trainer import Trainer
    trainer = Trainer(env_params, model_params, optimizer_params,
                      trainer_params, model_class=UnifiedCaRPOMO)
    copy_all_src(trainer.result_folder)
    trainer.run()


def solve(args):
    problems, log_flag, benchmark = _problem_list(args.problem_set)
    env_params = {
        'test_problem_list': problems,
        'test_scale_list': args.test_scale_list,
        'data_dir': args.data_dir,
        'scale_range_lib': args.scale_range_lib,
        **_env_params(args),
    }
    tester_params = {
        'use_cuda': args.cuda >= 0 and torch.cuda.is_available(),
        'cuda_device_num': args.cuda,
        'model_load': args.model_load,
        'test_episodes': args.test_episodes,
        'augmentation_enable': not args.disable_aug,
        'aug_factor': args.aug_factor,
        'test_batch_size_small': args.test_batch_size_small,
        'test_batch_size_large': args.test_batch_size_large,
        'summary_problems_per_row': args.summary_problems_per_row,
        'detailed_log': args.detailed_log,
    }
    if len(args.test_episodes) != len(args.test_scale_list):
        raise ValueError('--test_episodes must match --test_scale_list')

    now = datetime.now(pytz.timezone('Asia/Shanghai'))
    logger_params = {'log_file': {
        'desc': '_car_pomo_unified_' + log_flag, 'filename': 'run.log',
        'filepath': './result_test/{}/{}{{desc}}'.format(
            now.strftime('%Y-%m-%d'), now.strftime('%Y%m%d_%H%M%S'))}}
    seed_everything(args.seed)
    create_logger(**logger_params)
    print_startup(args=args, logger=logging.getLogger('root'),
                  result_folder=get_result_folder(),
                  log_filename='run.log', phase='testing')

    if benchmark:
        from Tester_Bench import Tester
    else:
        from Tester import Tester
    tester = Tester(env_params, _model_params(args), tester_params,
                    model_class=UnifiedCaRPOMO)
    copy_all_src(tester.result_folder)
    tester.run()


def main():
    parser = argparse.ArgumentParser(
        description='Unified CaR-POMO baseline on the URS 110-variant suite')
    parser.add_argument('command', choices=('train', 'solve'))
    obtain_all_hyperparameters(parser)
    parser.add_argument('--head_num', type=int, default=8,
                        help='heads in the CaR POMO attention blocks')
    parser.add_argument('--qkv_dim', type=int, default=16,
                        help='per-head width in the CaR POMO decoder')
    # This path is the consequence-conditioned baseline.  Flags remain exposed
    # for the nested no-margin/coupling ablations, but the identity-bearing URS
    # attribute defaults are inappropriate here.
    parser.set_defaults(constraint_repr='interface', node_repr='rows')
    args = parser.parse_args()
    if args.constraint_repr == 'attr' or args.node_repr != 'rows':
        parser.error('unified CaR-POMO requires --constraint_repr interface '
                     '(or interface_nomargin) and --node_repr rows')
    (train if args.command == 'train' else solve)(args)


if __name__ == '__main__':
    main()
