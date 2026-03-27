from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.backends.ttx.kernels.npu.int8_gemm import int8_gemm_dequant_impl
from mojo_opset.backends.ttx.kernels.npu.int8_gemm import prepare_b
from mojo_opset.core import MojoGemmDequant
from mojo_opset.core import MojoGroupGemm


class TTXGemmDequant(MojoGemmDequant):
    """Triton INT8 GEMM + fused dequantization on Ascend NPU.

    Uses a hand-tuned Triton kernel with persistent scheduling,
    B-transposed layout, double-buffering, and heuristic tile selection.
    The kernel fuses int8 × int8 → int32, per-token × per-channel
    scale application, optional bias add, and output dtype cast into
    a single kernel epilogue — eliminating intermediate memory traffic.
    """

    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.trans_weight:
            weight = weight.t().contiguous()

        M, K = input.shape
        K_w, N = weight.shape

        bt = prepare_b(weight)

        if not input.is_contiguous():
            input = input.contiguous()

        return int8_gemm_dequant_impl(
            input, bt,
            input_scale.flatten().float(),
            weight_scale.flatten().float(),
            bias,
            M, N,
            self.output_dtype,
        )


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
        assert num_groups == group_list.numel()

        C = input.new_empty(M, N)

        m_grouped_matmul(input, self.weight, C, group_list, num_groups, M, N, K, strideBN, strideBK, self.trans_weight)

        return C
