from tkinter import X
import torch

from mojo_opset.backends.ttx.kernels import matmul, m_grouped_matmul
from mojo_opset.core import MojoGroupLinear, MojoLinear


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

        m_grouped_matmul(
            input,
            self.weight,
            C,
            group_list,
            num_groups,
            M,
            N,
            K,
            strideBN,
            strideBK,
            self.trans_weight,
        )

        return C


class TTXLinear(MojoLinear):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        in_dim = self.weight.shape[1]
        if input.shape[-1] != in_dim:
            raise ValueError(f"input should have last dim {in_dim}, but got {input.shape[-1]}")
        if input.ndim not in (3, 4):
            raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")

        return matmul(input, self.weight, bias)
