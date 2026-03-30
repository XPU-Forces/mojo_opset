import torch

from torch.distributed.tensor import DTensor

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.core import MojoGroupGemm


class TTXGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2
        assert self.weight.dim() == 3

        M, K = input.shape

        assert input.stride(-1) == 1, "Please make sure input is K-major (last dim contiguous)."

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        if not self.trans_weight:
            num_groups, BK, N = weight.shape
            strideBK, strideBN = weight.stride(1), weight.stride(2)
        else:
            num_groups, N, BK = weight.shape
            strideBN, strideBK = weight.stride(1), weight.stride(2)

        assert BK == K, "Input K must be equal to weight K."

        C = input.new_empty(M, N)

        if isinstance(input, DTensor):
            input = input.to_local()

        if isinstance(C, DTensor):
            C = C.to_local()

        m_grouped_matmul(input, weight, C, group_list, num_groups, M, N, K, strideBN, strideBK, not self.trans_weight)

        return C
