import torch
from torch import nn

from ixformer import functions as ixf_f
from mojo_opset.experimental import MojoFusedAttnOutputGate

class IxformerFusedAttnOutputGate(MojoFusedAttnOutputGate):
    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cat_weight_and_bias_hook(module, incompatible_keys):
        module._cached_weight = torch.cat([module.full_gate_weight.data, module.swa_gate_weight.data], dim=0).float()
        if module.full_gate_bias is not None:
            module._cached_bias = torch.cat([module.full_gate_bias.data, module.swa_gate_bias.data], dim=0).float()

    def __init__(
        self,
        hidden_size: int,
        num_heads_full: int,
        num_heads_swa: int,
        head_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(hidden_size, num_heads_full, num_heads_swa, head_dim, bias, **kwargs)

        self.register_load_state_dict_post_hook(self._cat_weight_and_bias_hook)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        full_attn_output: torch.Tensor, 
        swa_attn_output: torch.Tensor
    ) -> torch.Tensor:
        gate = ixf_f.mixed_type_linear(hidden_states, self._cached_weight)

        full_attn_output = full_attn_output.view(-1, self.num_heads_full, self.head_dim)
        swa_attn_output = swa_attn_output.view(-1, self.num_heads_swa, self.head_dim)

        gated = ixf_f.fused_concat_sigmoid_mul(full_attn_output, swa_attn_output, gate, self._cached_bias)

        return gated.view(-1, self.num_heads_total * self.head_dim)

