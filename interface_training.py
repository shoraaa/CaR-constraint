"""Training schedule used only by the consequence-interface CaR arm.

The upstream Trainer deliberately keeps its original construction/refinement
schedule.  This module owns the narrow optimization needed by the added
interface path: consume the large construction graph before refinement, then
rebuild the encoder and imitate only instances whose refinement improved the
route.  Rows omitted from imitation had a zero coefficient in CaR's original
objective, so this changes neither the loss nor which examples contribute a
gradient.
"""

from __future__ import annotations

import math

import torch


class ConsequenceInterfaceTraining:
    """Per-batch controller for the optimized consequence-interface path."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.active = False
        self.supported = self._is_supported()

    def _is_supported(self):
        trainer = self.trainer
        model = trainer.model_params
        recipe = trainer.trainer_params
        return (
            int(recipe.get("interface_physical_batch", 0)) > 0
            and not model["improvement_only"]
            and model.get("constraint_repr", "attr") != "attr"
            and trainer.has_improve_steps
            and recipe["baseline"] == "group"
            and not recipe["bonus_for_construction"]
            and not recipe["extra_bonus"]
            and not recipe["uncertainty_weight"]
            and not recipe["neighborhood_search"]
            and not recipe["reward_gating"]
            and not recipe["subgradient"]
            and not recipe["improve_start_when_dummy_ok"]
            and not trainer.has_pip_decoder
            and not trainer.args.multiple_gpu
        )

    @classmethod
    def microbatch_plan(cls, trainer, remaining):
        """Coalesce a full logical batch into fewer, larger microbatches.

        For the reported 32 x 4 recipe, a cap of 43 produces [43, 43, 42].
        Every sample still receives weight 1/128 and the optimizer still steps
        once per 128 instances.  An incomplete tail uses CaR's original loop.
        """
        controller = cls(trainer)
        if not controller.supported:
            return None
        logical = (trainer.trainer_params["train_batch_size"]
                   * trainer.trainer_params["accumulation_steps"])
        cap = int(trainer.trainer_params["interface_physical_batch"])
        if cap <= trainer.trainer_params["train_batch_size"] or remaining < logical:
            return None
        count = math.ceil(logical / cap)
        if count >= trainer.trainer_params["accumulation_steps"]:
            return None
        size, extra = divmod(logical, count)
        return [size + (step < extra) for step in range(count)]

    @staticmethod
    def set_microbatch(trainer, batch_size, logical_size, step, steps):
        trainer._interface_microbatch = {
            "weight": batch_size / logical_size,
            "first": step == 0,
            "last": step == steps - 1,
        }

    @staticmethod
    def clear_microbatch(trainer):
        trainer.__dict__.pop("_interface_microbatch", None)

    def _loss_scale(self):
        plan = getattr(self.trainer, "_interface_microbatch", None)
        if plan is not None:
            return plan["weight"]
        return 1.0 / self.trainer.trainer_params["accumulation_steps"]

    def _first_microbatch(self, accumulation_step):
        plan = getattr(self.trainer, "_interface_microbatch", None)
        return plan["first"] if plan is not None else accumulation_step == 0

    def _last_microbatch(self, accumulation_step):
        plan = getattr(self.trainer, "_interface_microbatch", None)
        if plan is not None:
            return plan["last"]
        return accumulation_step == self.trainer.trainer_params["accumulation_steps"] - 1

    def activate(self, refinement_will_run):
        self.active = self.supported and refinement_will_run
        return self.active

    def backward_construction(self, loss, accumulation_step):
        """Consume construction before CaR allocates its refinement graph."""
        trainer = self.trainer
        if self._first_microbatch(accumulation_step):
            trainer.model.zero_grad()
            trainer.optimizer.zero_grad()
        loss = loss * self._loss_scale()
        if trainer.trainer_params["amp_training"]:
            loss = trainer.scaler.scale(loss)
        loss.backward()

    @staticmethod
    def _select(data, index):
        if isinstance(data, torch.Tensor):
            return data.index_select(0, index.to(data.device))
        return tuple(item.index_select(0, index.to(item.device))
                     for item in data)

    def imitation_loss(self, data, env, best_solution, is_improved,
                       batch_size):
        """Run the unchanged imitation objective on its non-zero rows only."""
        trainer = self.trainer
        index = is_improved.nonzero(as_tuple=False).squeeze(-1).to(
            trainer.device)
        if index.numel() == 0:
            return torch.zeros((), device=trainer.device)

        if index.numel() < batch_size:
            data = self._select(data, index)
            best_solution = best_solution.index_select(
                0, index.to(best_solution.device))

        imitation_batch = best_solution.size(0)
        env.load_problems(
            imitation_batch, rollout_size=1, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()

        # The original encoder graph was consumed with construction.  Encoding
        # only live imitation rows recovers the exact dependencies needed by
        # the late loss without replaying zero-gradient instances.
        trainer._get_model().pre_forward(reset_state, None)
        state, _, _ = env.pre_step()
        probability_steps = []
        for step in range(best_solution.size(-1)):
            with torch.amp.autocast(
                    device_type="cuda",
                    enabled=trainer.trainer_params["amp_training"]):
                _, probability, _ = trainer.model(
                    state,
                    pomo=trainer.env_params["pomo_start"],
                    selected=best_solution[:, :, step],
                    candidate_feature=(env.node_tw_end
                                       if trainer.args.problem == "TSPTW"
                                       else None))
            if trainer.has_pip_decoder:
                probability, _ = probability
            probability_steps.append(probability)
            state, _, _, _ = env.step(
                best_solution[:, :, step].to(trainer.device),
                out_reward=trainer.trainer_params["out_reward"],
                soft_constrained=trainer.trainer_params["soft_constrained"],
                backhaul_mask=trainer.trainer_params["backhaul_mask"],
                penalty_normalize=trainer.trainer_params["penalty_normalize"],
                generate_PI_mask=trainer.trainer_params["generate_PI_mask"],
                use_predicted_PI_mask=False,
                pip_step=trainer.trainer_params["pip_step"])

        values = torch.stack(probability_steps, dim=2).mean(-1).mean(-1)
        if values.numel() < batch_size:
            full_values = values.new_zeros(batch_size)
            values = full_values.scatter(0, index, values)
        return -values.mean()

    def add_construction_loss(self, downstream_loss, construction_loss):
        """Construction was already differentiated on the optimized path."""
        if self.active:
            return downstream_loss
        return construction_loss + downstream_loss

    def backward_remaining(self, loss, accumulation_step):
        """Backpropagate refinement/imitation and perform the scheduled step."""
        trainer = self.trainer
        loss = loss * self._loss_scale()
        if trainer.trainer_params["amp_training"]:
            trainer.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self._last_microbatch(accumulation_step):
            if trainer.trainer_params["amp_training"]:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                trainer.optimizer.step()
