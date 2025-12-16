import torch
import torch_npu
from mojo_opset.core import MojoGroupedMatmul


class TorchGroupedMatmul(MojoGroupedMatmul, default_priority=10):

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(
        self,
        input_tensor: torch.Tensor,
        other_tensor: torch.Tensor,
        group_info: list,
        trans_input: bool = False,
        trans_other: bool = False,
    ):

        return torch_npu.npu_grouped_matmul(
            input_tensor, other_tensor, group_info, trans_input, trans_other
        )