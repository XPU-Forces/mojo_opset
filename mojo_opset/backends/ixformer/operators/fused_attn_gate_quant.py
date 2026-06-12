import torch
from torch import nn

from ixformer import functions as ixf_f
from mojo_opset.experimental import MojoFusedAttnGateQuant

class IxformerFusedAttnGateQuant(MojoFusedAttnGateQuant):
    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cat_weight_and_bias_hook(module, incompatible_keys):
        module.attn_gate._cached_weight = torch.cat([module.attn_gate.full_gate_weight.data, module.attn_gate.swa_gate_weight.data], dim=0).float()
        if module.attn_gate.full_gate_bias is not None:
            module.attn_gate._cached_bias = torch.cat([module.attn_gate.full_gate_bias.data, module.attn_gate.swa_gate_bias.data], dim=0).float()

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

        self.register_load_state_dict_post_hook(self._cat_weight_and_bias_hook)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        full_attn_output: torch.Tensor, 
        swa_attn_output: torch.Tensor
    ) -> torch.Tensor:
        gate = ixf_f.mixed_type_linear(hidden_states, self.attn_gate._cached_weight)

        full_attn_output = full_attn_output.view(-1, self.num_heads_full, self.head_dim)
        swa_attn_output = swa_attn_output.view(-1, self.num_heads_swa, self.head_dim)

        gated_int8, gated_scale = ixf_f.fused_concat_sigmoid_mul_dynamic_smoothquant(full_attn_output, swa_attn_output, gate, self.attn_gate._cached_bias, self.o_quantize.inv_smooth_scale)

        return gated_int8, gated_scale

