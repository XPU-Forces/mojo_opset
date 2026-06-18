from typing import Optional

import torch
import torch.distributed as dist

from ixformer import functions as ixf_f
from ixformer.distributed import symmetric_memory as symm

from mojo_opset.experimental import MojoDistFusedConcatAttnGateQuant
from mojo_opset.experimental import MojoFusedConcatAttnOutputGate

class IxformerDistFusedConcatAttnGateQuant(MojoDistFusedConcatAttnGateQuant):
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
        tp_group: Optional[dist.ProcessGroup] = None,
        attn_gate: Optional[MojoFusedConcatAttnOutputGate] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, num_heads_full, num_heads_swa, head_dim, bias, quant_dtype, tp_group, attn_gate, **kwargs)

        self._workspace = None
        if (
            self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=self.tp_group) > 1
        ):
            self._enable_symm_mem(self.tp_group)
            world_size = dist.get_world_size(group=self.tp_group)
            workspace_bytes = ixf_f.fused_concat_sigmoid_mul_dynamic_smoothquant_workspace_bytes(
                5120, world_size, pad_to_group=True
            )
            device = torch.device("cuda", torch.cuda.current_device())
            self._workspace = symm.empty(
                workspace_bytes,
                dtype=torch.int8,
                device=device,
            )
            self._workspace.zero_()
            symm.rendezvous(self._workspace, self.tp_group)
            dist.barrier(group=self.tp_group)

        self.register_load_state_dict_post_hook(self._cat_weight_and_bias_hook)

    def _enable_symm_mem(self, pg):
        if symm.is_nvshmem_available():
            symm.set_backend("NVSHMEM")
        symm.enable_symm_mem_for_group(pg.group_name)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        full_attn_output: torch.Tensor, 
        swa_attn_output: torch.Tensor
    ) -> torch.Tensor:
        gate = ixf_f.mixed_type_linear(hidden_states, self.attn_gate._cached_weight)

        full_attn_output = full_attn_output.view(-1, self.num_heads_full, self.head_dim)
        swa_attn_output = swa_attn_output.view(-1, self.num_heads_swa, self.head_dim)

        gated_int8, gated_scale = ixf_f.fused_concat_sigmoid_mul_dynamic_smoothquant(full_attn_output, swa_attn_output, gate, self.attn_gate._cached_bias, self.inv_smooth_scale, workspace=self._workspace, group=self.tp_group, pad_to_group=True)

        return gated_int8, gated_scale

    def __del__(self):
        if getattr(self, "_workspace", None) is None:
            return

        try:
            group = getattr(self, "tp_group", None)
            group_name = getattr(group, "group_name", None)
            self._workspace = None
            symm.destroy(group_name=group_name)
        except Exception:
            pass