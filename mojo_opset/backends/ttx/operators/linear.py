import torch

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.core import MojoGroupLinear


class TTXGroupLinear(MojoGroupLinear):
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
        assert num_groups == group_list.numel()

        C = input.new_empty(M, N)

        m_grouped_matmul(input, self.weight, C, group_list, num_groups, M, N, K, strideBN, strideBK, self.trans_weight)

        return C
