import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface_training import ConsequenceInterfaceTraining


def _trainer(representation="interface", physical_batch=43):
    return SimpleNamespace(
        model_params={
            "constraint_repr": representation,
            "improvement_only": False,
        },
        trainer_params={
            "interface_physical_batch": physical_batch,
            "train_batch_size": 32,
            "accumulation_steps": 4,
            "baseline": "group",
            "bonus_for_construction": False,
            "extra_bonus": False,
            "uncertainty_weight": False,
            "neighborhood_search": False,
            "reward_gating": False,
            "subgradient": False,
            "improve_start_when_dummy_ok": False,
        },
        has_improve_steps=True,
        has_pip_decoder=False,
        args=SimpleNamespace(multiple_gpu=False),
    )


def test_interface_plan_preserves_the_effective_batch():
    trainer = _trainer()
    plan = ConsequenceInterfaceTraining.microbatch_plan(trainer, remaining=256)
    assert plan == [43, 43, 42]
    assert sum(plan) == 32 * 4


def test_old_paths_do_not_get_a_microbatch_plan():
    assert ConsequenceInterfaceTraining.microbatch_plan(
        _trainer(representation="attr"), remaining=256) is None
    assert ConsequenceInterfaceTraining.microbatch_plan(
        _trainer(physical_batch=0), remaining=256) is None


def test_incomplete_epoch_tail_uses_car_original_loop():
    assert ConsequenceInterfaceTraining.microbatch_plan(
        _trainer(), remaining=127) is None


def test_node_representation_is_not_a_precondition():
    trainer = _trainer()
    trainer.model_params["node_repr"] = "named"
    assert ConsequenceInterfaceTraining(trainer).supported
