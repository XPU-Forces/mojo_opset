import torch

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.core import MojoGroupGemm
from torch.distributed.tensor import DTensor

class TTXGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2
        assert self.weight.dim() == 3

        M, K = input.shape

        assert input.stride(-1) == 1, "Please make sure input is K-major."

        if self.trans_weight:
            num_groups, N, BK = self.weight.shape
            strideBN, strideBK = self.weight.stride(1), self.weight.stride(2)
        else:
            num_groups, BK, N = self.weight.shape
            strideBK, strideBN = self.weight.stride(1), self.weight.stride(2)

        assert BK == K, "K of input should be equal to K of self.weight."

        C = input.new_empty(M, N)
        print(isinstance(self.weight, DTensor), flush=True)
        m_grouped_matmul(input, self.weight.to_local(), C, group_list, num_groups, M, N, K, strideBN, strideBK, self.trans_weight)

        return C
