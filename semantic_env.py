"""POMO environment backed only by PRISM's executable resource algebra.

Unlike ``URS/env/UniVRPEnv.py``, this environment does not select masks from a
variant name.  Every candidate is probed through ``decoder.Decoder`` and is
admitted iff the structural rule and every declared resource row admit it.
Consequently an unseen accumulator (battery, draft, driver time) or relation
uses the same construction path as a trained capacity or precedence row.

This reference implementation prioritizes semantic fidelity.  It vectorizes
the neural tensors but executes the readable Python interpreter per rollout;
it is suitable for zero-shot evaluation and small training probes.  A future
GPU executor can replace ``_refresh`` without changing the model contract.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from decoder import Decoder, Event
from decoder_features import _node_term, _phase_total
from program import is_relational


CONSEQUENCE_DIM = 6


@dataclass
class SemanticStepState:
    batch_size: int
    pomo_size: int
    depot_num: int
    selected_count: int = 0
    current_node: torch.Tensor | None = None
    ninf_mask: torch.Tensor | None = None
    consequence: torch.Tensor | None = None
    finished: torch.Tensor | None = None
    start_nodes: torch.Tensor | None = None
    pomo_start_nodes: torch.Tensor | None = None


class SemanticPOMOEnv:
    """A name-free POMO facade over a homogeneous batch of PRISM problems."""

    def __init__(self, problems, pomo_size=None, device=None):
        if not problems:
            raise ValueError('SemanticPOMOEnv needs at least one problem')
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.decoders = [Decoder(problem) for problem in problems]
        node_counts = {decoder.node_count for decoder in self.decoders}
        resource_counts = {len(decoder.programs) for decoder in self.decoders}
        depot_counts = {decoder.depot_count for decoder in self.decoders}
        if len(node_counts) != 1 or len(resource_counts) != 1 \
                or len(depot_counts) != 1:
            raise ValueError('a semantic batch must share node/resource/depot counts')
        self.batch_size = len(self.decoders)
        self.node_count = node_counts.pop()
        self.resource_count = resource_counts.pop()
        self.depot_num = depot_counts.pop()
        self.customer_count = self.node_count - self.depot_num
        default_pomo = self.node_count if not self.depot_num else self.customer_count
        self.pomo_size = int(default_pomo if pomo_size is None else pomo_size)
        self.pomo_size = max(1, min(self.pomo_size, default_pomo))
        self.states = None
        self.routes = None
        self.state = SemanticStepState(
            self.batch_size, self.pomo_size, self.depot_num)

    @staticmethod
    def _node_resource_rows(decoder):
        """PRISM's anonymous six-coordinate static row representation."""
        result = np.zeros((decoder.node_count, len(decoder.programs), 7),
                          dtype=np.float32)
        for resource, program in enumerate(decoder.programs):
            scale = max(float(program.scale), 1e-6)
            top_class = (int(np.max(program.node_class))
                         if program.node_class is not None else 0)
            for node in range(decoder.node_count):
                slot = result[node, resource]
                slot[0] = 1.0
                increment = _node_term(decoder, program, node)
                slot[1] = np.clip(max(increment, 0.0) / scale, 0, 1)
                slot[2] = np.clip(max(-increment, 0.0) / scale, 0, 1)
                for term in program.terms:
                    if term.operation == 'join':
                        slot[3] = np.clip(
                            decoder._term_value(term, node, node, 0.0) / scale,
                            0, 1)
                        break
                upper = program.bound.upper_at(node)
                lower = program.bound.lower_at(node)
                bound = upper if np.isfinite(upper) else lower
                slot[4] = (np.clip(bound / scale, 0, 1)
                           if np.isfinite(bound) else 1.0)
                after = _phase_total(decoder, program, 'after_bound', node, node)
                slot[5] = np.clip(after / scale, 0, 1) if after else 0.0
                if program.node_class is not None:
                    slot[6] = (float(program.node_class[node]) + 1) / (top_class + 1)
                elif is_relational(program) and program.relation in ('pairwise', 'dag'):
                    opens = bool(program.successors and program.successors[node])
                    required = (bool(program.predecessor is not None
                                     and program.predecessor[node] >= 0)
                                if program.relation == 'pairwise'
                                else bool(program.predecessors
                                          and program.predecessors[node]))
                    slot[6] = ((2 if opens and required else 1 if opens else 3) / 3
                               if opens or required else 0.0)
        return result

    def reset(self):
        positions, node_types, node_rows, objectives, distances = [], [], [], [], []
        for decoder in self.decoders:
            coordinates = decoder.problem.get('coordinates')
            if coordinates is not None:
                coordinates = np.asarray(coordinates, dtype=np.float32)
                if coordinates.ndim == 3:
                    coordinates = coordinates[0]
            position = np.zeros((self.node_count, 3), dtype=np.float32)
            if coordinates is None:
                position[:, 0] = np.linspace(0.0, 1.0, self.node_count,
                                             dtype=np.float32)
            else:
                minimum = coordinates.min(axis=0)
                span = max(float(np.ptp(coordinates, axis=0).max()), 1e-6)
                position[:, 1:] = np.clip((coordinates - minimum) / span, 0, 1)
            kind = np.zeros((self.node_count, 5), dtype=np.float32)
            kind[:self.depot_num, 0] = 1.0
            kind[self.depot_num:, 1] = 1.0
            kind[:, 2] = decoder.visit_values > 0
            kind[:, 3] = decoder.omission_values > 0
            rows = self._node_resource_rows(decoder)
            objective = decoder.visit_values - decoder.omission_values
            scale = max(float(np.max(np.abs(objective))), 1e-6)
            positions.append(position); node_types.append(kind); node_rows.append(rows)
            objectives.append((objective / scale).astype(np.float32))
            distances.append(np.asarray(decoder.distance, dtype=np.float32))

        self.states = [[None] * self.pomo_size for _ in self.decoders]
        self.routes = [[[] for _ in range(self.pomo_size)] for _ in self.decoders]
        self.state.selected_count = 0
        self.state.current_node = torch.zeros(
            self.batch_size, self.pomo_size, dtype=torch.long, device=self.device)
        self.state.finished = torch.zeros(
            self.batch_size, self.pomo_size, dtype=torch.bool, device=self.device)
        starts = torch.empty_like(self.state.current_node)
        for batch, decoder in enumerate(self.decoders):
            count = decoder.depot_count or decoder.node_count
            starts[batch] = torch.arange(count, device=self.device).repeat(
                (self.pomo_size + count - 1) // count)[:self.pomo_size]
        self.state.start_nodes = starts
        self.state.pomo_start_nodes = None
        self.state.ninf_mask = torch.zeros(
            self.batch_size, self.pomo_size, self.node_count, device=self.device)
        self.state.consequence = None
        reset = SimpleNamespace(
            problems=torch.zeros(self.batch_size, self.node_count, 9,
                                 device=self.device),
            position_features=torch.as_tensor(np.stack(positions), device=self.device),
            node_type_features=torch.as_tensor(np.stack(node_types), device=self.device),
            node_resource_features=torch.as_tensor(np.stack(node_rows), device=self.device),
            objective_coefficient=torch.as_tensor(np.stack(objectives), device=self.device),
            dist=torch.as_tensor(np.stack(distances), device=self.device))
        return reset, None, False

    def pre_step(self):
        return self.state, None, False

    def get_local_feature(self):
        if self.state.selected_count == 0:
            return None
        distance = torch.as_tensor(
            np.stack([decoder.distance for decoder in self.decoders]),
            dtype=torch.float32, device=self.device)
        batch = torch.arange(self.batch_size, device=self.device)[:, None]
        return distance[batch, self.state.current_node]

    def _refresh(self):
        mask = np.full((self.batch_size, self.pomo_size, self.node_count),
                       -np.inf, dtype=np.float32)
        rows = np.zeros((self.batch_size, self.pomo_size, self.node_count,
                         self.resource_count, CONSEQUENCE_DIM), np.float32)
        pomo_starts = np.zeros((self.batch_size, self.pomo_size), np.int64)
        for batch, decoder in enumerate(self.decoders):
            for rollout, state in enumerate(self.states[batch]):
                if self.state.finished[batch, rollout]:
                    mask[batch, rollout, state.current] = 0.0
                    continue
                legal = []
                for destination in range(self.node_count):
                    action = decoder.probe(state, destination, construction=True)
                    # Resource execution is meaningful even for a structurally
                    # unavailable node, and is what the policy representation
                    # promises to publish.
                    outcomes = action.resources
                    if not outcomes:
                        outcomes, _ = decoder._probe_resources(
                            state, destination, construction=True)
                    for resource, outcome in enumerate(outcomes):
                        program = decoder.programs[resource]
                        live = decoder._normalize_state(
                            program, state.resources[resource].value,
                            state.current)
                        rows[batch, rollout, destination, resource] = (
                            1.0, live, outcome.normalized_next_state,
                            outcome.normalized_signed_margin,
                            float(outcome.admissible),
                            float(outcome.events != Event.NONE))
                    if action.admissible:
                        mask[batch, rollout, destination] = 0.0
                        legal.append(destination)
                if not legal:
                    raise RuntimeError('semantic construction has no legal action')
                pomo_starts[batch, rollout] = legal[rollout % len(legal)]
        self.state.ninf_mask = torch.as_tensor(mask, device=self.device)
        self.state.consequence = torch.as_tensor(rows, device=self.device)
        if self.state.selected_count == 1:
            self.state.pomo_start_nodes = torch.as_tensor(
                pomo_starts, device=self.device)

    def step(self, selected):
        chosen = selected.detach().cpu().numpy()
        if self.state.selected_count == 0:
            for batch, decoder in enumerate(self.decoders):
                for rollout in range(self.pomo_size):
                    node = int(chosen[batch, rollout])
                    self.states[batch][rollout] = decoder.initial_state(node)
                    self.routes[batch][rollout] = [node]
        else:
            for batch, decoder in enumerate(self.decoders):
                for rollout, state in enumerate(self.states[batch]):
                    if self.state.finished[batch, rollout]:
                        continue
                    node = int(chosen[batch, rollout])
                    outcome = decoder.probe(state, node, construction=True)
                    if not outcome.admissible:
                        raise ValueError(outcome.error)
                    decoder.commit(state, node, outcome)
                    self.routes[batch][rollout].append(node)
                    self.state.finished[batch, rollout] = decoder.complete(state)
        self.state.selected_count += 1
        self.state.current_node = selected.clone()
        done = bool(self.state.finished.all())
        if not done:
            self._refresh()
            return self.state, None, False

        reward = torch.empty(self.batch_size, self.pomo_size, device=self.device)
        for batch, decoder in enumerate(self.decoders):
            for rollout, route in enumerate(self.routes[batch]):
                solution = decoder.evaluate(route)
                if not solution['feasible']:
                    raise RuntimeError(solution.get('error', 'infeasible completed route'))
                # POMO maximizes reward. Decoder objectives are reported in
                # their natural direction.
                sign = 1.0 if solution['direction'] == 'maximize' else -1.0
                reward[batch, rollout] = sign * float(solution['objective'])
        return self.state, reward, True
