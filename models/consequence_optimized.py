"""Low-memory execution of the added consequence valuation.

The model and state dict are unchanged.  The only custom operation fuses the
row layer-normalization with its following SiLU and saves one fp32 activation
instead of the two retained by eager autograd.  Its backward is analytic; it
does not replay the forward pass or checkpoint activations.
"""

import torch

from models.consequence import ACTIVE_INDEX, ConsequenceValuation, _unit_scale


class _LayerNormSiLU(torch.autograd.Function):
    """LayerNorm + SiLU with a single retained candidate-row activation."""

    @staticmethod
    def forward(ctx, projected):
        ctx.input_dtype = projected.dtype
        normalized, _, reciprocal_std = torch.native_layer_norm(
            projected.float(), (projected.shape[-1],), None, None, 1e-5)
        ctx.save_for_backward(normalized, reciprocal_std)
        return torch.nn.functional.silu(normalized)

    @staticmethod
    def backward(ctx, grad_output):
        normalized, reciprocal_std = ctx.saved_tensors
        grad_normalized = torch.ops.aten.silu_backward(
            grad_output, normalized)
        mean_grad = grad_normalized.mean(dim=-1, keepdim=True)
        mean_product = (grad_normalized * normalized).mean(
            dim=-1, keepdim=True)
        grad_projected = reciprocal_std * (
            grad_normalized - mean_grad - normalized * mean_product)
        return grad_projected.to(ctx.input_dtype)


class OptimizedConsequenceValuation(ConsequenceValuation):
    """State-dict-compatible valuation for the opt-in training path."""

    def _value_rows(self, consequence, multipliers):
        read = self._read(consequence)
        active = read[..., ACTIVE_INDEX]
        projected = self.row_proj(read)
        if self.norm == "layer":
            activated = _LayerNormSiLU.apply(projected)
            hidden = self.row[2](self.row[1](activated))
        else:
            hidden = self.row(_unit_scale(projected, self.norm))
        hidden = hidden.masked_fill(active.unsqueeze(-1) <= 0, 0.0)
        hidden = hidden * multipliers.to(active.dtype).unsqueeze(-1)
        value = self._reduce(hidden, active, self.value).squeeze(-1)
        return self.logit_clipping * torch.tanh(value)

