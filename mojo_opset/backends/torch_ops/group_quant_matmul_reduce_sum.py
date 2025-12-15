import torch
from mojo_opset.core import MojoGroupQuantMatmulReduceSum
import torch_npu
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class TorchGroupQuantMatmulReduceSum(MojoGroupQuantMatmulReduceSum, default_priority=0):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor):
        fmt = torch_npu.get_npu_format(x2)
        if fmt != 29:
            x2_nz = torch_npu.npu_format_cast(x2, 29)
            logger.info(f"Not support x2 format {fmt}, cast to NZ format")
        else:
            x2_nz = x2

        result = torch_npu.npu_quant_matmul_reduce_sum(x1, x2_nz, x1_scale = x1_scale, x2_scale = x2_scale)
        return result
