"""Valuation of routing decisions from the consequences the active rows assign them.

The decoder in ``SINGLEModel`` reads its constraint context from
``_get_extra_features``, a switch over the problem name that emits the live
state of each named constraint (``load``, ``current_time``, ``length``).  That
representation is addressed by constraint identity, carries only the live state,
is not intrinsically scaled, and is shared by every candidate at a step -- the
only per-candidate constraint signal reaching the logits is the binary mask.

This module replaces it with the interface the environment now publishes: for
each candidate and each active row, the row's presence, its live state, the
post-state the candidate would produce, the signed distance to the declared
bound on the row's own scale, the legality that margin implies, and the
transition events the candidate fires.  Rows are valued by one shared network
and pooled without identity, so removing a requirement removes a row from the
sum and leaves the learned valuation untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# `net.py:_unit_scale` layer-normalizes each projection before summing the
# attribute and descriptor streams, so the two are comparable.  Here there is
# only one stream (the descriptor is constant under program-blind), and the
# consequence coordinates are already normalized by construction -- each by its
# own row's published scale, then passed through `VRPBLTWEnv._soft_bound`, which
# is the identity on [-1, 1] and bends monotonically to (-2, 2) beyond it rather
# than clipping (see that method for why a clip is wrong under soft
# construction).  Measured on an A100 at
# (128, 40, 51, 3): the layer_norm is 0.31 s of the row MLP's 0.42 s per 51
# decoding steps, because it reduces over 16 elements across 783k rows and sits
# on autocast's fp32 list, so AMP does not help it.  `--consequence_norm none`
# drops it; `layer` keeps the ported behaviour and stays the default so arms
# trained either side of this flag are not silently mixed.
_NORMALIZERS = {
    "layer": lambda x: F.layer_norm(x, (x.shape[-1],)),
    "none": lambda x: x,
}


def _unit_scale(projected, kind="layer"):
    return _NORMALIZERS[kind](projected)

CONSEQUENCE_DIM = 6
ACTIVE_INDEX = 0
STATE_INDEX = 1
POST_INDEX = 2
MARGIN_INDEX = 3
TOKEN_DIM = 4
NODE_ROW_DIM = 7
VALID_INDEX = 4
EVENT_INDEX = 5


class ConsequenceValuation(nn.Module):
    """Shared, identity-free valuation over the consequence interface.

    ``use_margin=False`` is the nested control: the same pathway with the same
    parameter count, reading everything except the signed admissibility margin,
    so a gain can be attributed to the margin rather than to the extra capacity
    a per-candidate pathway adds.
    """

    def __init__(self, context_dim, hidden_dim=16, use_margin=True, logit_clipping=10.0,
                 couple_rows=True, norm="layer"):
        super().__init__()
        self.norm = norm
        self.use_margin = use_margin
        self.logit_clipping = logit_clipping
        self.couple_rows = couple_rows
        # phi: shared across rows, applied under a unit-scaled projection, as
        # ResourcePool does.  No row carries an identity, so the per-row states
        # are permutation-equivariant.
        self.row_proj = nn.Linear(CONSEQUENCE_DIM, hidden_dim)
        self.row = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        # The row field and its multiplier, following PRISM's split: the field
        # is read per candidate from that row's consequence, the multiplier per
        # decision from the row's own token coupled to every other active row's
        # live state.  A plain sum of independent row fields cannot represent
        # "capacity binds differently when the clock is tight"; the coupling is
        # what carries the interaction, and it drops a summand rather than
        # changing a parameter shape when a row leaves the active set.
        # rho: the post-pool network.  It has to be nonlinear -- a linear rho
        # would make the valuation a plain sum of independent per-row terms,
        # which is the additive-over-rows model the coupling exists to escape.
        self.value = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1))
        self.coupler_query = nn.Linear(hidden_dim, hidden_dim)
        self.coupler_key = nn.Linear(hidden_dim, hidden_dim)
        self.coupler_bias = nn.Linear(hidden_dim, 1)
        # The query-side context keeps the width the named switch used, so the
        # two arms differ in what the query says, not in how much it can say.
        # The token follows Eq. (app-resource-token): an active flag plus the
        # mean and max pressure the row places on the candidates currently under
        # consideration.  Two rows are therefore distinguishable exactly when
        # they price the current candidate set differently, which is the
        # representational commitment being tested.  PRISM's token also carries
        # the declared objective's coefficients and scale; CaR minimizes tour
        # length in every one of these compositions, so those coordinates would
        # be constant and are omitted rather than passed as dead inputs.
        self.state_proj = nn.Linear(TOKEN_DIM, hidden_dim)
        self.state_row = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.context = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim))

    @staticmethod
    def _reduce(hidden, active, head):
        """DeepSets reduction: normalized masked sum, then cardinality, then rho.

        `hidden` is (..., rows, units) and `active` is (..., rows), with any
        number of leading batch axes.  An empty active set reduces to zero
        rather than to rho(0), which would be a learned constant standing in
        for "no requirements at all".
        """
        count = active.sum(dim=-1, keepdim=True)
        pooled = hidden.sum(dim=-2) / count.clamp_min(1.0).sqrt()
        cardinality = count / (1.0 + count)
        summary = head(torch.cat((pooled, cardinality), dim=-1))
        return summary.masked_fill(count <= 0, 0.0)

    def _read(self, consequence):
        if self.use_margin:
            return consequence
        blind = consequence.clone()
        blind[..., MARGIN_INDEX] = 0.0
        return blind

    def _row_tokens(self, consequence, candidate_mask=None):
        """Per-row token, live state and active flag.

        The token is candidate-*pooled*: the mean and max **pressure** the row
        places on the candidates under consideration, where pressure is the
        consumption a candidate causes, `post - state`.

        Pressure, not margin, for two independent reasons.  It is what
        Eq. (app-resource-token) pools -- "the normalized pressure the
        constraint places on candidate e" -- and the margin is the signed slack
        to the bound, a different quantity.  And the admissibility head consumes
        this token through the decoder context, so pooling the margin here hands
        that head the mean and max of the very value it is asked to regress,
        which collapses the representational demand the supervision exists to
        create.

        Returns (tokens [B,P,R,H], state [B,P,R], active [B,P,R]).
        """
        read = self._read(consequence)
        rows = read[:, :, 0, :, :]
        active = rows[..., ACTIVE_INDEX]
        state = rows[..., STATE_INDEX]
        pressure = read[..., POST_INDEX] - read[..., STATE_INDEX]
        # shape: (batch, pomo, nodes, rows)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(pressure[..., 0])
        weights = candidate_mask.to(pressure.dtype).unsqueeze(-1)
        present = weights.sum(dim=2)
        count = present.clamp_min(1.0)
        mean_pressure = (pressure * weights).sum(dim=2) / count
        # A sentinel below any real pressure, zeroed where no candidate remains
        # so an empty set reports nothing rather than the sentinel itself.
        max_pressure = pressure.masked_fill(weights <= 0, -1.0e4).amax(dim=2)
        max_pressure = torch.where(present > 0, max_pressure.clamp(-1.0, 1.0),
                                   torch.zeros_like(max_pressure))
        token = torch.stack((active, state, mean_pressure, max_pressure), dim=-1)
        tokens = self.state_row(_unit_scale(self.state_proj(token), self.norm)) * active.unsqueeze(-1)
        return tokens, state, active

    def _multipliers_from(self, tokens, state, active):
        """One multiplier per active row, coupled to the other active rows.

        lambda_r = 2 * sigmoid( sum_s w(z_r, z_s) * x_s + b(z_r) ), so a row's
        weight depends on what else is active and on the state those rows are
        in.  2*sigmoid(0) == 1, so an untrained or uncoupled model leaves every
        multiplier at one and the valuation reduces to the plain row sum.
        """
        if not self.couple_rows:
            return torch.ones_like(active), active
        query = self.coupler_query(tokens)
        key = self.coupler_key(tokens) * active.unsqueeze(-1)
        scale = query.shape[-1] ** 0.5
        weights = torch.matmul(query, key.transpose(-1, -2)) / scale
        # shape: (batch, pomo, rows, rows)
        coupling = torch.matmul(weights, (state * active).unsqueeze(-1)).squeeze(-1)
        bias = self.coupler_bias(tokens).squeeze(-1)
        return 2.0 * torch.sigmoid(coupling + bias), active

    @staticmethod
    def _compact(consequence, candidate_mask):
        """Restrict the per-candidate work to the candidates still in play.

        The decoder adds this valuation to the content logits and only then
        applies its own mask -- `score_masked = score_clipped + ninf_mask` in
        SINGLEModel -- so whatever is computed for a masked candidate is
        discarded. Computing it anyway is the largest single cost in the arm:
        the live set shrinks by one every decoding step, and the valuation's
        forward+backward measures 1.64x faster at 75% of the width and 2.16x at
        25%.

        Padded slots point at candidates that are masked out downstream, so the
        garbage computed for them is harmless; positions outside the gather
        receive a zero bias, which is the same neutral value an inactive row
        contributes anywhere else.

        Returns (compact, compact_mask, index), or None when there is nothing
        to gain.
        """
        if candidate_mask is None:
            return None
        counts = candidate_mask.sum(dim=-1)
        width = int(counts.max())
        if width >= candidate_mask.shape[-1] or width == 0:
            return None
        # stable so a rerun gathers the same order: the padded slots differ
        # between orderings and they are what the scatter writes garbage into.
        order = torch.argsort(candidate_mask.to(torch.uint8), dim=-1,
                              descending=True, stable=True)
        index = order[..., :width]
        rows, coordinates = consequence.shape[-2], consequence.shape[-1]
        gather = index[..., None, None].expand(-1, -1, -1, rows, coordinates)
        compact = consequence.gather(2, gather)
        positions = torch.arange(width, device=candidate_mask.device)
        return compact, positions < counts.unsqueeze(-1), index

    @staticmethod
    def _scatter(value, index, nodes):
        """Put the compacted per-candidate valuation back on the node axis."""
        restored = value.new_zeros(value.shape[0], value.shape[1], nodes)
        return restored.scatter(2, index, value)

    def evaluate(self, consequence, candidate_mask=None):
        """The decoder query context and the per-candidate valuation together.

        `forward` and `query_context` each rebuild the row tokens -- the same
        pooled reduction over every candidate plus a layer-norm and two linear
        layers -- so calling both per decoding step does that work twice and
        discards half of it.  This shares it.  The two are numerically identical
        to calling the pair, which `test_evaluate_matches_the_separate_calls`
        pins.
        """
        packed = self._compact(consequence, candidate_mask)
        read, mask, index = (packed if packed is not None
                             else (consequence, candidate_mask, None))
        tokens, state, active = self._row_tokens(read, mask)
        context = self._reduce(tokens, active, self.context)
        multipliers, _ = self._multipliers_from(tokens, state, active)
        value = self._value(read, multipliers)
        if index is not None:
            value = self._scatter(value, index, consequence.shape[2])
        return context, value

    def forward(self, consequence, candidate_mask=None):
        """Per-candidate additive valuation.

        consequence:    (batch, pomo, nodes, rows, CONSEQUENCE_DIM)
        candidate_mask: (batch, pomo, nodes), the candidates under consideration
        returns:        (batch, pomo, nodes)
        """
        packed = self._compact(consequence, candidate_mask)
        read, mask, index = (packed if packed is not None
                             else (consequence, candidate_mask, None))
        tokens, state, active = self._row_tokens(read, mask)
        multipliers, _ = self._multipliers_from(tokens, state, active)
        value = self._value(read, multipliers)
        if index is None:
            return value
        return self._scatter(value, index, consequence.shape[2])

    def _value(self, consequence, multipliers):
        read = self._read(consequence)
        active = read[..., ACTIVE_INDEX]
        hidden = self.row(_unit_scale(self.row_proj(read), self.norm)) * active.unsqueeze(-1)
        hidden = hidden * multipliers.unsqueeze(2).unsqueeze(-1)
        # Normalizing by the active count is what makes a three-row composition
        # and a one-row composition comparable: an unnormalized sum shrinks by
        # roughly sqrt(rows) when a requirement is dropped, which is a magnitude
        # shift on exactly the axis this study measures.
        value = self._reduce(hidden, active, self.value).squeeze(-1)
        # Bounded on the same scale as the clipped content logits, so the
        # valuation reorders the legal candidates without saturating them.
        return self.logit_clipping * torch.tanh(value)

    def query_context(self, consequence, candidate_mask=None):
        """Identity-free replacement for the named live-state vector.

        The live state is the same for every candidate, so it is read off the
        first one.  Returns (batch, pomo, context_dim).
        """
        tokens, _, active = self._row_tokens(consequence, candidate_mask)
        return self._reduce(tokens, active, self.context)


class AdmissibilityHead(nn.Module):
    """Read-out head supervised by the executed signed admissibility margin.

    PRISM's ``_slack_loss``: for every candidate and every active row, regress
    the signed distance by which taking that candidate would clear or breach the
    row's declared bound -- the number the interpreter already computed in order
    to decide legality, so nothing about the target is a feature choice.

    The head is deliberately *not* fed the consequence interface.  It reads only
    what the policy reads -- the encoder's node embeddings and the decoder's
    state context -- so fitting the margin places the demand on the shared
    representation rather than being copied from an input.  Its prediction never
    enters the logits; it shapes the representation the logits are read from.

    One departure from PRISM, forced by the host architecture: PRISM applies the
    head per constraint *token*, so the head itself carries no row identity.
    CaR has no constraint tokens, so the output layer is row-indexed.  The
    interface pathway stays identity-free; this head does not.
    """

    def __init__(self, embedding_dim, context_dim, row_count, hidden_dim=16):
        super().__init__()
        self.row_count = row_count
        self.node = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.context = nn.Linear(embedding_dim + context_dim, hidden_dim)
        self.margin = nn.Linear(hidden_dim, row_count)

    def forward(self, encoded_nodes, context):
        """encoded_nodes: (batch, nodes, embedding); context: (batch, pomo, dim).

        Returns the predicted margin per candidate and row,
        (batch, pomo, nodes, rows).
        """
        node = self.node(encoded_nodes).unsqueeze(1)
        # shape: (batch, 1, nodes, hidden)
        query = self.context(context).unsqueeze(2)
        # shape: (batch, pomo, 1, hidden)
        return self.margin(torch.relu(node + query))

    @staticmethod
    def loss(prediction, consequence, candidate_mask):
        """Masked mean squared error against the executed margin.

        Supervision is restricted to active rows and to candidates actually
        under consideration -- a row that is not part of this composition, and a
        node the policy cannot choose, have no margin worth predicting.
        """
        # A probe may switch on a row the head was never sized for; supervise
        # the rows it has and leave the rest alone rather than failing.
        rows = min(prediction.shape[-1], consequence.shape[-2])
        prediction = prediction[..., :rows]
        consequence = consequence[..., :rows, :]
        target = consequence[..., MARGIN_INDEX]
        active = consequence[..., ACTIVE_INDEX]
        mask = active * candidate_mask.unsqueeze(-1).to(active.dtype)
        error = (prediction - target) * mask
        return error.square().sum() / mask.sum().clamp_min(1.0)


class NodeRowPool(nn.Module):
    """Identity-free replacement for a per-formulation node feature layout.

    CaR's encoder reads `(xy, demand, tw_start, tw_end)` for VRPBLTW and a
    different tuple for every other problem, so one parameter set cannot span
    formulations -- and even within one superset, feeding a "neutral" value for
    an absent requirement still puts that value in a slot the model knows is a
    time window.  Here each row reports the same coordinates about a node, they
    are pooled by the same DeepSets reduction the valuation uses, and a row that
    is not part of this composition contributes nothing at all.
    """

    def __init__(self, output_dim, hidden_dim=16, norm="layer"):
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(NODE_ROW_DIM, hidden_dim)
        self.row = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, node_rows):
        """node_rows: (batch, nodes, rows, NODE_ROW_DIM) -> (batch, nodes, out)."""
        active = node_rows[..., ACTIVE_INDEX]
        hidden = self.row(_unit_scale(self.proj(node_rows), self.norm)) * active.unsqueeze(-1)
        return ConsequenceValuation._reduce(hidden, active, self.head)
