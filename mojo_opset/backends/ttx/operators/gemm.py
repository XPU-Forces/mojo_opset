from typing import Optional

import torch

from torch.distributed.tensor import DTensor

from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.backends.ttx.kernels import int8_gemm_dequant
from mojo_opset.backends.ttx.kernels import prepare_b
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

        return int8_gemm_dequant(
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
