"""CaR-POMO construction policy on URS's unified 110-variant environment.

This module deliberately owns only the policy.  Instance generation, POMO
starts, rewards, datasets and feasibility masks remain URS's, so comparisons
against URS change the architecture rather than the experimental protocol.
Constraint information is read exclusively through the already-published
consequence rows; problem names are used only for rollout conventions and for
non-constraint node types (depot, pickup and delivery).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.consequence import ROW_COUNT, ROW_TOKENS
from model.Model_LIB import (get_encoding, select_next_node,
                             unified_node_attribute_construction,
                             unified_node_position_construction)
from model.consequence import (AdmissibilityHead, ConsequenceValuation,
                               NodeRowPool)
from problem.ProblemSet import ProblemSet


class _EncoderLayer(nn.Module):
    """The standard attention/FFN block used by CaR's POMO encoder."""

    def __init__(self, embedding_dim, head_num, ff_hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, head_num, bias=False, batch_first=True)
        self.norm1 = nn.InstanceNorm1d(embedding_dim, affine=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim), nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim))
        self.norm2 = nn.InstanceNorm1d(embedding_dim, affine=True)

    @staticmethod
    def _normalize(norm, value):
        return norm(value.transpose(1, 2)).transpose(1, 2)

    def forward(self, value):
        attended, _ = self.attention(value, value, value, need_weights=False)
        value = self._normalize(self.norm1, value + attended)
        return self._normalize(self.norm2, value + self.ff(value))


class _Encoder(nn.Module):
    def __init__(self, embedding_dim, head_num, ff_hidden_dim, layer_num):
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderLayer(embedding_dim, head_num, ff_hidden_dim)
            for _ in range(layer_num)
        ])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class _Decoder(nn.Module):
    """CaR's multi-head POMO decoder with a directed-distance channel."""

    def __init__(self, embedding_dim, head_num, qkv_dim, context_dim,
                 logit_clipping):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.embedding_dim = embedding_dim
        self.logit_clipping = logit_clipping
        self.query = nn.Linear(embedding_dim + context_dim,
                               head_num * qkv_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        # CaR's coordinate encoder exposes geometry implicitly.  ATSP has no
        # coordinates, so its directed current-to-candidate costs must remain
        # visible to a unified policy.  A single shared scalar adds that signal
        # without conditioning parameters on a variant identity.
        self.distance_weight = nn.Parameter(torch.tensor(1.0))
        self.k = None
        self.v = None
        self.single_head_key = None

    def assign(self, _problem_representation):
        """Compatibility hook for URS's trainer; this policy has no hypernet."""

    def set_kv(self, encoded_nodes):
        batch, nodes, _ = encoded_nodes.shape
        shape = (batch, nodes, self.head_num, self.qkv_dim)
        self.k = self.key(encoded_nodes).view(shape).transpose(1, 2)
        self.v = self.value(encoded_nodes).view(shape).transpose(1, 2)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last, context, cur_dist, ninf_mask,
                logit_bias=None):
        query = self.query(torch.cat((encoded_last, context), dim=-1))
        batch, pomo, _ = query.shape
        query = query.view(batch, pomo, self.head_num,
                           self.qkv_dim).transpose(1, 2)
        score = torch.matmul(query, self.k.transpose(2, 3))
        score = score / math.sqrt(self.qkv_dim)
        score = score + ninf_mask[:, None, :, :]
        weights = torch.softmax(score, dim=-1)
        heads = torch.matmul(weights, self.v).transpose(1, 2)
        attended = self.combine(heads.reshape(batch, pomo, -1))
        logits = torch.matmul(attended, self.single_head_key)
        logits = logits / math.sqrt(self.embedding_dim)

        distance = cur_dist / cur_dist.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = logits - F.softplus(self.distance_weight) * distance
        logits = self.logit_clipping * torch.tanh(logits)
        if logit_bias is not None:
            logits = logits + logit_bias
        return torch.softmax(logits + ninf_mask, dim=-1)


class UnifiedCaRPOMO(nn.Module):
    """One consequence-conditioned CaR-POMO policy for all URS variants."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = int(model_params.get('head_num', 8))
        qkv_dim = int(model_params.get('qkv_dim', embedding_dim // head_num))
        context_dim = int(model_params.get('consequence_context_dim', 1))
        hidden_dim = int(model_params.get('consequence_hidden_dim', 16))
        node_repr_dim = int(model_params.get('node_repr_dim', 5))
        norm = model_params.get('consequence_norm', 'layer')

        self.position_embedding = nn.Linear(3, embedding_dim, bias=False)
        self.resource_embedding = nn.Linear(node_repr_dim + 1, embedding_dim,
                                            bias=False)
        self.node_type_embedding = nn.Linear(5, embedding_dim, bias=False)
        self.node_rows = NodeRowPool(output_dim=node_repr_dim,
                                     hidden_dim=hidden_dim, norm=norm)
        self.encoder = _Encoder(
            embedding_dim, head_num, model_params['ff_hidden_dim'],
            model_params['encoder_layer_num'])
        self.decoder = _Decoder(embedding_dim, head_num, qkv_dim, context_dim,
                                model_params['logit_clipping'])
        self.consequence = ConsequenceValuation(
            context_dim=context_dim, hidden_dim=hidden_dim,
            use_margin=model_params.get('constraint_repr') != 'interface_nomargin',
            logit_clipping=model_params['logit_clipping'],
            couple_rows=model_params.get('couple_rows', True), norm=norm,
            compact=model_params.get('consequence_compact', True),
            gradient_checkpoint=model_params.get('consequence_checkpoint', True))

        self.slack_weight = float(model_params.get('slack_weight', 0.0))
        self.slack_stride = max(1, int(model_params.get('slack_stride', 5)))
        if self.slack_weight > 0:
            self.admissibility = AdmissibilityHead(
                embedding_dim, context_dim, ROW_COUNT, hidden_dim,
                read_consequence=True,
                gradient_checkpoint=model_params.get(
                    'consequence_checkpoint', True))
            self.admissibility.row_states_fn = self.consequence.row_states

        self.problem_name = None
        self.encoded_nodes = None
        self._active_row_selector = slice(None)
        self._reset_slack()

    def _reset_slack(self):
        self.slack_loss_sum = None
        self.slack_loss_steps = 0
        self.slack_steps_seen = 0

    def set_decoder_type(self, decoder_type):
        self.model_params['eval_type'] = decoder_type

    def _active_rows(self, rows):
        # Training keeps the stable public layout to avoid a device sync.  At
        # evaluation, compact the inactive slots when the environment did not.
        return rows if self.training else rows[..., self._active_row_selector, :]

    def pre_forward(self, reset_state, problem_name, _problem_representation):
        if reset_state.node_resource_features is None:
            raise ValueError(
                "UnifiedCaRPOMO requires the consequence interface; use "
                "--constraint_repr interface --node_repr rows")
        self.problem_name = problem_name
        self._reset_slack()
        if self.model_params.get('consequence_compact_rows', True):
            self._active_row_selector = slice(None)
        else:
            slots = [i for i, token in enumerate(ROW_TOKENS)
                     if token in problem_name]
            if not slots:
                self._active_row_selector = slice(0, 0)
            elif slots == list(range(slots[0], slots[-1] + 1)):
                self._active_row_selector = slice(slots[0], slots[-1] + 1)
            else:
                self._active_row_selector = torch.tensor(
                    slots, device=reset_state.problems.device)

        # A PRISM semantic environment supplies these directly and never asks
        # the policy to parse a variant name.  URS-backed runs retain their
        # released feature construction as the compatibility fallback.
        positions = getattr(reset_state, 'position_features', None)
        named = getattr(reset_state, 'node_type_features', None)
        semantic_static = positions is not None and named is not None
        if positions is None or named is None:
            positions = unified_node_position_construction(
                reset_state.problems, problem_name)
            named = unified_node_attribute_construction(
                reset_state.problems, problem_name,
                demand_max1=self.model_params.get('demand_max1', True))
        pooled = self.node_rows(
            self._active_rows(reset_state.node_resource_features))
        resources = torch.cat(
            (pooled, reset_state.objective_coefficient.unsqueeze(-1)), dim=-1)
        initial = (self.position_embedding(positions)
                   + self.resource_embedding(resources)
                   + self.node_type_embedding(
                       named if semantic_static else named[..., 6:]))
        self.encoded_nodes = self.encoder(initial)
        self.decoder.set_kv(self.encoded_nodes)

    def _accumulate_admissibility(self, state, encoded_last, context):
        if self.slack_weight <= 0 or not self.training:
            return
        self.slack_steps_seen += 1
        if (self.slack_steps_seen - 1) % self.slack_stride:
            return
        consequence = self._active_rows(state.consequence)
        prediction = self.admissibility(
            self.encoded_nodes, torch.cat((encoded_last, context), dim=-1),
            consequence)
        loss = AdmissibilityHead.loss(
            prediction, consequence, state.ninf_mask == 0)
        self.slack_loss_sum = (loss if self.slack_loss_sum is None
                               else self.slack_loss_sum + loss)
        self.slack_loss_steps += 1

    def admissibility_loss(self):
        if self.slack_loss_sum is None:
            return None
        return self.slack_loss_sum / self.slack_loss_steps

    def forward(self, state, cur_dist):
        batch, pomo = state.batch_size, state.pomo_size
        device = self.encoded_nodes.device
        if state.selected_count == 0:
            declared_starts = getattr(state, 'start_nodes', None)
            if declared_starts is not None:
                selected = declared_starts
            elif self.problem_name in ('tsp', 'atsp'):
                selected = torch.arange(pomo, device=device)[None, :]
                selected = selected.expand(batch, pomo)
            elif 'md' in self.problem_name:
                selected = torch.arange(state.depot_num, device=device)
                selected = selected.repeat_interleave(pomo // state.depot_num)
                selected = selected[None, :].expand(batch, pomo)
            else:
                selected = torch.zeros(batch, pomo, dtype=torch.long,
                                       device=device)
            return selected, torch.ones(batch, pomo, device=device)

        if state.selected_count == 1 and pomo > 1 \
                and self.problem_name not in ('tsp', 'atsp'):
            declared_pomo = getattr(state, 'pomo_start_nodes', None)
            if declared_pomo is not None:
                selected = declared_pomo
            elif self.problem_name in ProblemSet.get(included='b', excluded='bp'):
                selected = state.START_NODE
            elif 'md' in self.problem_name:
                customers = cur_dist.shape[-1] - state.depot_num
                selected = torch.arange(state.depot_num,
                                        state.depot_num + customers,
                                        device=device)
                selected = selected.repeat(
                    (pomo + customers - 1) // customers)[:pomo]
                selected = selected[None, :].expand(batch, pomo)
            else:
                selected = torch.arange(1, pomo + 1,
                                        device=device)[None, :]
                selected = selected.expand(batch, pomo)
            return selected, torch.ones(batch, pomo, device=device)

        encoded_last = get_encoding(self.encoded_nodes, state.current_node)
        consequence = self._active_rows(state.consequence)
        context, logit_bias = self.consequence.evaluate(
            consequence, state.ninf_mask == 0)
        self._accumulate_admissibility(state, encoded_last, context)
        probs = self.decoder(encoded_last, context, cur_dist, state.ninf_mask,
                             logit_bias)
        return select_next_node(probs, self.model_params['eval_type'])


# The short name is convenient for code that expects a conventional Model.
Model = UnifiedCaRPOMO
