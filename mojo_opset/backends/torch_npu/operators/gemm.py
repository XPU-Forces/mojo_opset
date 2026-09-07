import torch
import torch_npu

from mojo_opset.core import MojoBatchGemm
from mojo_opset.core import MojoGroupGemm
from mojo_opset.core import MojoQuantGemm
from mojo_opset.core import MojoQuantGroupGemm
from mojo_opset.experimental import MojoQuantBatchGemmReduceSum
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class TorchNpuQuantGemm(MojoQuantGemm):
    """NPU backend for fused int8 GEMM + dequantization via ``npu_quant_matmul``.

    Uses the NPU's native int8 GEMM kernel which performs true int8 → int32
    accumulation with hardware-fused scale application, yielding higher
    throughput than the float32-emulated core reference.
    """

    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.trans_weight:
            weight = weight.t().contiguous()

        kernel_output_dtype = self.output_dtype
        if self.weight_scale.dtype == torch.bfloat16 and self.output_dtype not in (torch.bfloat16, torch.int32):
            kernel_output_dtype = torch.bfloat16

        out = torch_npu.npu_quant_matmul(
            input,
            weight,
            self.weight_scale.flatten(),
            pertoken_scale=input_scale.flatten(),
            output_dtype=kernel_output_dtype,
        )
        if out.dtype != self.output_dtype:
            out = out.to(self.output_dtype)
        return out


class TorchNpuBatchGemm(MojoBatchGemm):
    """Transpose-BMM backend; CANN A2 BF16/FP16 requires K and N aligned to 128."""

    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [M, B, K] -> [B, M, K] for GEMM, then restore output[M, B, N].
        return torch_npu.npu_transpose_batchmatmul(
            input.contiguous(),
            self.weight.contiguous(),
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )


class TorchNpuQuantGroupGemm(MojoQuantGroupGemm):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        group_list: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Group along the token axis using counts and return one concatenated tensor.
        return torch_npu.npu_grouped_matmul(
            [input],
            [self.weight],
            scale=None if self.weight_scale is None else [self.weight_scale],
            per_token_scale=None if input_scale is None else [input_scale.reshape(-1)],
            group_list=group_list,
            split_item=3,
            output_dtype=self.output_dtype,
            group_type=0,
            group_list_type=1,
        )[0]


class TorchNpuGroupGemm(MojoGroupGemm):
    def forward(
        self,
        input: torch.Tensor,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        if input.dtype == torch.float32:
            raise NotImplementedError("NPU grouped matmul does not support float32")
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


class TorchNpuQuantBatchGemmReduceSum(MojoQuantBatchGemmReduceSum):
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
