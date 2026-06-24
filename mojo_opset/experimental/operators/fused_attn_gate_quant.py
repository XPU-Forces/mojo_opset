from typing import Tuple

import torch

from ...core.operator import MojoOperator
from ...core.operators.quantize import MojoDynamicQuant
from ..operators.attention_gate import MojoFusedAttnOutputGate, MojoFusedConcatAttnOutputGate

__all__ = ["MojoFusedAttnGateQuant", "MojoFusedConcatAttnGateQuant"]


class MojoFusedAttnGateQuant(MojoOperator):
    """Fused attention output gate + dynamic per-token quantization (single-path).

    Composes MojoFusedAttnOutputGate (gate sigmoid + broadcast multiply) and
    MojoDynamicQuant (smooth-scale + per-token int8 quantization) into a single
    operator to eliminate the intermediate BF16 materialization.

    Computation::

        gate = sigmoid(hidden_states @ gate_weight.T)
        gated = attn.view(T, N, D) * gate.unsqueeze(-1)
        gated_flat = gated.view(T, N * D)
        smooth = gated_flat * inv_smooth_scale  # per-channel smooth quant
        scale = amax(|smooth|, dim=-1) / 127
        output = clamp(round(smooth / scale), -128, 127)

    Returns (quant_output int8 [T, N * D], scale float32 [T, 1]).

    Sub-operations:
      - MojoFusedAttnOutputGate: gate computation
      - MojoDynamicQuant: smooth-scale dynamic quantization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        AttnGateCls = MojoFusedAttnOutputGate._registry.get("torch")
        DynamicQuantCls = MojoDynamicQuant._registry.get("torch")

        self.attn_gate = AttnGateCls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
        )
        output_size = num_heads * head_dim
        self.o_quantize = DynamicQuantCls(
            input_size=output_size,
            quant_dtype=quant_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [T, hidden_size] — gate input (pre-attn residual).
            attn_output:   [T, N, D] or [T, N * D] — attention output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quant_output: int8, shape [T, N * D]
                - scale: float32, shape [T, 1]
        """
        gated = self.attn_gate(hidden_states, attn_output)
        quant_output, scale = self.o_quantize(gated)
        return quant_output, scale

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"bias={self.attn_gate.gate_bias is not None}"
        )


class MojoFusedConcatAttnGateQuant(MojoOperator):
    """Fused attention output gate + dynamic per-token quantization (dual-path).

    Composes MojoFusedConcatAttnOutputGate (gate sigmoid + broadcast multiply) and
    MojoDynamicQuant (smooth-scale + per-token int8 quantization) into a single
    operator to eliminate the intermediate BF16 materialization.

    Computation::

        gate = sigmoid(hidden_states @ cat([full_gate_weight, swa_gate_weight]).T)
        gated = cat([full_attn, swa_attn], dim=1) * gate.unsqueeze(-1)
        gated_flat = gated.view(T, K)          # K = (N_full + N_swa) * head_dim
        smooth = gated_flat * inv_smooth_scale  # per-channel smooth quant
        scale = amax(|smooth|, dim=-1) / 127
        output = clamp(round(smooth / scale), -128, 127)

    Returns (quant_output int8 [T, K], scale float32 [T, 1]).

    Sub-operations:
      - MojoFusedConcatAttnOutputGate: gate computation
      - MojoDynamicQuant: smooth-scale dynamic quantization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads_full: int,
        num_heads_swa: int,
        head_dim: int,
        bias: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads_full = num_heads_full
        self.num_heads_swa = num_heads_swa
        self.head_dim = head_dim

        AttnGateCls = MojoFusedConcatAttnOutputGate._registry.get("torch")
        DynamicQuantCls = MojoDynamicQuant._registry.get("torch")

        self.attn_gate = AttnGateCls(
            hidden_size=hidden_size,
            num_heads_full=num_heads_full,
            num_heads_swa=num_heads_swa,
            head_dim=head_dim,
            bias=bias,
        )
        output_size = (num_heads_full + num_heads_swa) * head_dim
        self.o_quantize = DynamicQuantCls(
            input_size=output_size,
            quant_dtype=quant_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_output: torch.Tensor,
        swa_attn_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:    [T, hidden_size] — gate input (pre-attn residual).
            full_attn_output: [T, N_full, D] or [T, N_full * D] — full attention output.
            swa_attn_output:  [T, N_swa, D] or [T, N_swa * D] — SWA attention output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quant_output: int8, shape [T, (N_full + N_swa) * D]
                - scale: float32, shape [T, 1]
        """
        gated = self.attn_gate(hidden_states, full_attn_output, swa_attn_output)
        quant_output, scale = self.o_quantize(gated)
        return quant_output, scale

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads_full={self.num_heads_full}, "
            f"num_heads_swa={self.num_heads_swa}, "
            f"head_dim={self.head_dim}, "
            f"bias={self.attn_gate.full_gate_bias is not None}"
        )
