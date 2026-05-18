import torch

from torch.distributed.tensor import DTensor

from mojo_opset.backends.ttx.kernels import int8_gemm_dequant
from mojo_opset.backends.ttx.kernels import m_grouped_matmul
from mojo_opset.backends.ttx.kernels import prepare_b
from mojo_opset.backends.ttx.kernels import quant_batch_gemm_reduce_sum_impl
from mojo_opset.core import MojoQuantGemm
from mojo_opset.core import MojoGroupGemm
from mojo_opset.experimental import MojoQuantBatchGemmReduceSum


class TTXQuantGemm(MojoQuantGemm):
    """Triton INT8 GEMM + fused dequantization.

    Uses a Triton kernel with B-transposed layout and fused epilogue.
    The kernel fuses int8 x int8 -> int32, per-token x per-channel
    scale application, optional bias add, and output dtype cast into
    a single kernel epilogue -- eliminating intermediate memory traffic.
    """

    supported_platforms_list = ["npu", "ilu"]

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.trans_weight:
            weight = weight.t().contiguous()

        M, K = input.shape
        K_w, N = weight.shape

        bt = prepare_b(weight)

        if not input.is_contiguous():
            input = input.contiguous()

        return int8_gemm_dequant(
            input,
            bt,
            input_scale.flatten().float(),
            self.weight_scale.flatten().float(),
            None,  # no bias.
            M,
            N,
            self.output_dtype,
        )


class TTXGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        assert group_list.dtype == torch.int32
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


class TTXQuantBatchGemmReduceSum(MojoQuantBatchGemmReduceSum):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        input: torch.Tensor,
        x1_scale: torch.Tensor,
        x2_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight.contiguous()

        b, _, k = input.shape
        b_w, k_w, _ = weight.shape
        assert b == b_w, "input and weight must have same batch size"
        assert k == k_w, "K of input should be equal to K of weight"

        return quant_batch_gemm_reduce_sum_impl(input, weight, x1_scale, x2_scale)
