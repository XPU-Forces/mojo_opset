import torch
from torch import nn

from ixformer import functions as ixf_f
from mojo_opset.experimental import MojoFusedAttnGateQuant, MojoFusedConcatAttnGateQuant


class IxformerFusedAttnGateQuant(MojoFusedAttnGateQuant):
    """ilu backend for the single-path variant.

    The parent's sub-ops (``attn_gate``, ``o_quantize``) hold the parameters
    in their torch form; this subclass reads those parameters and runs the
    actual computation through ixformer kernels:
    ``mixed_type_linear`` (gate GEMM, bf16 × fp32 → fp32) → ``fused_sigmoid_mul``
    (sigmoid + per-head broadcast multiply) → ``dynamic_quant`` (per-token
    smooth-quant + int8 round). ixformer has no single-path
    ``fused_..._dynamic_smoothquant`` kernel and rejects 0-head / non-contig
    splits, so we run the two stages explicitly instead of reusing the
    dual-path fused kernel.
    """

    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cast_weight_and_bias_hook(module, incompatible_keys):
        module._cached_weight = module.attn_gate.gate_weight.data.float()
        if module.attn_gate.gate_bias is not None:
            module._cached_bias = module.attn_gate.gate_bias.data.float()

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(hidden_size, num_heads, head_dim, bias, quant_dtype, **kwargs)

        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.register_load_state_dict_post_hook(self._cast_weight_and_bias_hook)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        gate = ixf_f.mixed_type_linear(hidden_states, self._cached_weight)

        attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
        gated = ixf_f.fused_sigmoid_mul(attn_output, gate, self._cached_bias)

        gated_int8, gated_scale = ixf_f.dynamic_quant(
            gated.view(-1, self.num_heads * self.head_dim),
            self.o_quantize.inv_smooth_scale,
        )
        return gated_int8, gated_scale


class IxformerFusedConcatAttnGateQuant(MojoFusedConcatAttnGateQuant):
    """ilu backend for the dual-path variant.

    Runs the entire gate + smoothquant pipeline through
    ``fused_concat_sigmoid_mul_dynamic_smoothquant``. Reads the parent's
    torch sub-op parameters and caches a concatenated fp32 weight/bias copy
    on ``self`` for the kernel.
    """

    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cat_weight_and_bias_hook(module, incompatible_keys):
        module._cached_weight = torch.cat(
            [module.attn_gate.full_gate_weight.data, module.attn_gate.swa_gate_weight.data], dim=0
        ).float()
        if module.attn_gate.full_gate_bias is not None:
            module._cached_bias = torch.cat(
                [module.attn_gate.full_gate_bias.data, module.attn_gate.swa_gate_bias.data], dim=0
            ).float()

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
        super().__init__(hidden_size, num_heads_full, num_heads_swa, head_dim, bias, quant_dtype, **kwargs)

        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.register_load_state_dict_post_hook(self._cat_weight_and_bias_hook)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_output: torch.Tensor,
        swa_attn_output: torch.Tensor,
    ) -> torch.Tensor:
        gate = ixf_f.mixed_type_linear(hidden_states, self._cached_weight)

        full_attn_output = full_attn_output.view(-1, self.num_heads_full, self.head_dim)
        swa_attn_output = swa_attn_output.view(-1, self.num_heads_swa, self.head_dim)

        gated_int8, gated_scale = ixf_f.fused_concat_sigmoid_mul_dynamic_smoothquant(
            full_attn_output, swa_attn_output, gate, self._cached_bias, self.o_quantize.inv_smooth_scale
        )
        return gated_int8, gated_scale
