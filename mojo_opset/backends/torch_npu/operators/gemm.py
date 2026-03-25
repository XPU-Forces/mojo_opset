from typing import Optional

import torch
import torch_npu

from mojo_opset.core import MojoGemmDequant
from mojo_opset.core import MojoGroupLinear
from mojo_opset.core import MojoQuantGroupLinearReduceSum
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class TorchNpuGemmDequant(MojoGemmDequant):
    """NPU backend for fused int8 GEMM + dequantization via ``npu_quant_matmul``.

    Uses the NPU's native int8 GEMM kernel which performs true int8 → int32
    accumulation with hardware-fused scale application, yielding higher
    throughput than the float32-emulated core reference.
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

        return torch_npu.npu_quant_matmul(
            input,
            weight,
            weight_scale.flatten(),
            pertoken_scale=input_scale.flatten(),
            bias=bias,
            output_dtype=self.output_dtype,
        )


class TorchNpuGroupGemm(MojoGroupLinear):
    def forward(
        self,
        input: torch.Tensor,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        weight_list = [weight[g].contiguous() for g in range(num_groups)]
        group_list_values = [int(x) for x in group_list.cumsum(0).tolist()]
        outputs = torch_npu.npu_grouped_matmul(
            [input],
            weight_list,
            group_type=0,
            group_list=group_list_values,
        )
        return torch.cat(outputs, dim=0)


class TorchNpuQuantGroupLinearReduceSum(MojoQuantGroupLinearReduceSum):
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
            weight = self.weight

        if x2_scale.dtype != torch.bfloat16:
            x2_scale = x2_scale.to(torch.bfloat16)

        fmt = torch_npu.get_npu_format(weight)
        if fmt != 29:
            x2_nz = torch_npu.npu_format_cast(weight, 29)
            logger.info(f"Not support weight format {fmt}, cast to NZ format")
        else:
            x2_nz = weight

        return torch_npu.npu_quant_matmul_reduce_sum(input, x2_nz, x1_scale=x1_scale, x2_scale=x2_scale)
