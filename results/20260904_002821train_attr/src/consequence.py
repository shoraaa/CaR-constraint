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


def _unit_scale(projected):
    """Layer-normalize a projection, as `net.py:_unit_scale` does."""
    return F.layer_norm(projected, (projected.shape[-1],))

CONSEQUENCE_DIM = 6
ACTIVE_INDEX = 0
STATE_INDEX = 1
POST_INDEX = 2
MARGIN_INDEX = 3
TOKEN_DIM = 4
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
                 couple_rows=True):
        super().__init__()
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

        The token is candidate-*pooled*, not candidate-independent: the mean and
        max margin the row reports over the candidates under consideration.

        Returns (tokens [B,P,R,H], state [B,P,R], active [B,P,R]).
        """
        read = self._read(consequence)
        rows = read[:, :, 0, :, :]
        active = rows[..., ACTIVE_INDEX]
        state = rows[..., STATE_INDEX]
        margin = read[..., MARGIN_INDEX]
        # shape: (batch, pomo, nodes, rows)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(margin[..., 0])
        weights = candidate_mask.to(margin.dtype).unsqueeze(-1)
        count = weights.sum(dim=2).clamp_min(1.0)
        mean_pressure = (margin * weights).sum(dim=2) / count
        max_pressure = margin.masked_fill(weights <= 0, -1.0).amax(dim=2)
        token = torch.stack((active, state, mean_pressure, max_pressure), dim=-1)
        tokens = self.state_row(_unit_scale(self.state_proj(token))) * active.unsqueeze(-1)
        return tokens, state, active

    def _multipliers(self, consequence, candidate_mask=None):
        """One multiplier per active row, coupled to the other active rows.

        lambda_r = 2 * sigmoid( sum_s w(z_r, z_s) * x_s + b(z_r) ), so a row's
        weight depends on what else is active and on the state those rows are
        in.  2*sigmoid(0) == 1, so an untrained or uncoupled model leaves every
        multiplier at one and the valuation reduces to the plain row sum.
        """
        tokens, state, active = self._row_tokens(consequence, candidate_mask)
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

    def forward(self, consequence, candidate_mask=None):
        """Per-candidate additive valuation.

        consequence:    (batch, pomo, nodes, rows, CONSEQUENCE_DIM)
        candidate_mask: (batch, pomo, nodes), the candidates under consideration
        returns:        (batch, pomo, nodes)
        """
        read = self._read(consequence)
        active = read[..., ACTIVE_INDEX]
        hidden = self.row(_unit_scale(self.row_proj(read))) * active.unsqueeze(-1)
        multipliers, _ = self._multipliers(consequence, candidate_mask)
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
        target = consequence[..., MARGIN_INDEX]
        active = consequence[..., ACTIVE_INDEX]
        mask = active * candidate_mask.unsqueeze(-1).to(active.dtype)
        error = (prediction - target) * mask
        return error.square().sum() / mask.sum().clamp_min(1.0)
