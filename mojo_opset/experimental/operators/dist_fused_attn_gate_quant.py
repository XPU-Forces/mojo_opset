from typing import Optional, Tuple

import torch
import torch.distributed as dist

from ...core.operator import MojoOperator
from .attention_gate import MojoFusedConcatAttnOutputGate

__all__ = ["MojoDistFusedConcatAttnGateQuant"]


class MojoDistFusedConcatAttnGateQuant(MojoOperator):
    """Distributed variant of :class:`MojoFusedConcatAttnGateQuant`.

    Same gate + smooth-quant pipeline as the base op, but the per-token amax
    used to derive the int8 scale is reduced across the channel slices held
    by every rank in ``tp_group`` via AllGather + max. The resulting scale
    is bit-identical on every rank — required when the int8 output is
    redistributed by a downstream A2A and the dequant step on the receiving
    side must use the same scale that produced each token's int8 row.

    Computation::

        gated  = sigmoid(hidden_states @ cat([full_gate_w, swa_gate_w]).T)
        gated *= cat([full_attn, swa_attn], dim=channel)        # bf16
        smoothed = gated.float() * inv_smooth_scale             # smooth-quant pre-scale
        local_amax = smoothed.abs().amax(dim=-1, keepdim=True)  # [T, 1] per-token, per-rank
        # AllGather local_amax across tp_group, then per-token max-reduce
        gathered      = all_gather(local_amax, group=tp_group)  # [tp, T, 1]
        unified_amax  = gathered.amax(dim=0)                    # [T, 1]
        unified_scale = (unified_amax / 127).clamp(min=1e-12)
        quant = round(smoothed / unified_scale).clamp(-128, 127).to(int8)
        return quant, unified_scale

    Token padding
    -------------
    For the AllGather + downstream A2A to split evenly across ``tp_size``
    ranks, ``T`` must be a multiple of ``tp_size``. The op pads internally
    along the token axis when ``tp_group`` is set; the caller is responsible
    for trimming the post-o_proj output back to the original ``T``.

    Construction
    ------------
    The caller can pass an externally-owned ``attn_gate`` (a built
    :class:`MojoFusedConcatAttnOutputGate`) so this op shares the gate sub-module
    with the parent module instead of allocating duplicates. Sub-Module
    sharing is safe under PyTorch's lazy-init: ``Module._apply`` only
    replaces entries in ``_parameters`` / ``_buffers``, not in ``_modules``.

    The ``inv_smooth_scale`` Parameter, in contrast, is owned by this op
    directly. Sharing it via plain attribute would break under lazy
    materialisation: ``Module._apply`` swaps the meta Parameter object for a
    new cuda one in the parent's ``_parameters`` dict, leaving any external
    Python reference dangling on the old (meta) object. State-dict path is
    therefore ``<parent>.<this op>.inv_smooth_scale``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads_full: int,
        num_heads_swa: int,
        head_dim: int,
        bias: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        tp_group: Optional[dist.ProcessGroup] = None,
        attn_gate: Optional[MojoFusedConcatAttnOutputGate] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8."
            )

        self.hidden_size = hidden_size
        self.num_heads_full = num_heads_full
        self.num_heads_swa = num_heads_swa
        self.head_dim = head_dim
        self.tp_group = tp_group
        self.quant_dtype = quant_dtype
        self.q_max = 127
        self.q_min = -128

        if attn_gate is not None:
            object.__setattr__(self, "attn_gate", attn_gate)
        else:
            AttnGateCls = MojoFusedConcatAttnOutputGate._registry.get("torch")
            self.attn_gate = AttnGateCls(
                hidden_size=hidden_size,
                num_heads_full=num_heads_full,
                num_heads_swa=num_heads_swa,
                head_dim=head_dim,
                bias=bias,
            )

        output_size = (num_heads_full + num_heads_swa) * head_dim
        self.inv_smooth_scale = torch.nn.Parameter(
            torch.empty(output_size, **self.tensor_factory_kwargs)
        )
        setattr(self.inv_smooth_scale, "force_dtype", torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_output: torch.Tensor,
        swa_attn_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:    [T, hidden_size] — gate input (pre-attn residual).
            full_attn_output: [T, N_full, D] or [T, N_full * D].
            swa_attn_output:  [T, N_swa, D] or [T, N_swa * D].

        Returns:
            Tuple of:
              * ``quant``: int8 ``[T_padded, (N_full + N_swa) * D]``.
              * ``unified_scale``: float32 ``[T_padded, 1]``, bit-identical on
                every rank in ``tp_group``.

            ``T_padded == T`` when ``tp_group`` is None or ``T %% tp_size == 0``;
            otherwise ``T_padded == T + ((-T) %% tp_size)``.
        """
        gated = self.attn_gate(hidden_states, full_attn_output, swa_attn_output)

        # Pad tokens to a multiple of tp_size so downstream AllGather/A2A can
        # split evenly. No-op when not distributed or already aligned.
        if (
            self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            tp_size = dist.get_world_size(group=self.tp_group)
            if tp_size > 1:
                pad = (-gated.shape[0]) % tp_size
                if pad > 0:
                    gated = torch.nn.functional.pad(gated, (0, 0, 0, pad))

        smoothed = gated.float() * self.inv_smooth_scale
        local_amax = smoothed.abs().amax(dim=-1, keepdim=True)

        if (
            self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=self.tp_group) > 1
        ):
            tp_size = dist.get_world_size(group=self.tp_group)
            # Flat-buffer all_gather form for cross-backend compatibility
            # (gloo's all_gather_into_tensor only accepts the concatenated layout).
            local_flat = local_amax.contiguous().view(-1)
            gathered_flat = torch.empty(
                tp_size * local_flat.numel(),
                dtype=local_amax.dtype,
                device=local_amax.device,
            )
            dist.all_gather_into_tensor(
                gathered_flat, local_flat, group=self.tp_group
            )
            unified_amax = gathered_flat.view(tp_size, *local_amax.shape).amax(dim=0)
        else:
            unified_amax = local_amax

        unified_scale = (unified_amax / self.q_max).clamp(min=1e-12)
        unified_scale = torch.where(unified_scale < 1e-6, 1.0, unified_scale)
        quant = torch.clamp(
            torch.round(smoothed / unified_scale),
            self.q_min,
            self.q_max,
        ).to(self.quant_dtype)
        return quant, unified_scale

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads_full={self.num_heads_full}, "
            f"num_heads_swa={self.num_heads_swa}, "
            f"head_dim={self.head_dim}, "
            f"tp_group={'set' if self.tp_group is not None else 'None'}"
        )
