import torch
from mojo_opset.backends.ttx_kernels.src.ascend.group_gemm import ttx_m_grouped_gemm


from mojo_opset.core import MojoGroupLinear


class TTXGroupLinear(MojoGroupLinear, default_priority=0):
    def forward_std(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        return ttx_m_grouped_gemm(input, self.weight, group_list, self.trans_weight)