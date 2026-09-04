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

CONSEQUENCE_DIM = 6
ACTIVE_INDEX = 0
STATE_INDEX = 1
MARGIN_INDEX = 3


class ConsequenceValuation(nn.Module):
    """Shared, identity-free valuation over the consequence interface.

    ``use_margin=False`` is the nested control: the same pathway with the same
    parameter count, reading everything except the signed admissibility margin,
    so a gain can be attributed to the margin rather than to the extra capacity
    a per-candidate pathway adds.
    """

    def __init__(self, context_dim, hidden_dim=16, use_margin=True, logit_clipping=10.0):
        super().__init__()
        self.use_margin = use_margin
        self.logit_clipping = logit_clipping
        self.row = nn.Sequential(
            nn.Linear(CONSEQUENCE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value = nn.Linear(hidden_dim, 1)
        # The query-side context keeps the width the named switch used, so the
        # two arms differ in what the query says, not in how much it can say.
        self.state_row = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.context = nn.Linear(hidden_dim, context_dim)

    def _read(self, consequence):
        if self.use_margin:
            return consequence
        blind = consequence.clone()
        blind[..., MARGIN_INDEX] = 0.0
        return blind

    def forward(self, consequence):
        """Per-candidate additive valuation.

        consequence: (batch, pomo, nodes, rows, CONSEQUENCE_DIM)
        returns:     (batch, pomo, nodes)
        """
        read = self._read(consequence)
        hidden = self.row(read) * read[..., ACTIVE_INDEX:ACTIVE_INDEX + 1]
        pooled = hidden.sum(dim=-2)
        # Bounded on the same scale as the clipped content logits, so the
        # valuation reorders the legal candidates without saturating them.
        return self.logit_clipping * torch.tanh(self.value(pooled).squeeze(-1))

    def query_context(self, consequence):
        """Identity-free replacement for the named live-state vector.

        The live state is the same for every candidate, so it is read off the
        first one.  Returns (batch, pomo, context_dim).
        """
        rows = consequence[:, :, 0, :, :]
        # shape: (batch, pomo, rows, CONSEQUENCE_DIM)
        state = torch.stack((rows[..., ACTIVE_INDEX], rows[..., STATE_INDEX]), dim=-1)
        hidden = self.state_row(state) * rows[..., ACTIVE_INDEX:ACTIVE_INDEX + 1]
        return self.context(hidden.sum(dim=-2))
