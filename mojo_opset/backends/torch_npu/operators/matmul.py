from typing import Any, List, Optional
import torch
import torch_npu
from mojo_opset.core import MojoGroupedMatmul
from mojo_opset.core import MojoGroupQuantMatmulReduceSum
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class TorchNpuGroupedMatmul(MojoGroupedMatmul, default_priority=10):

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__() 
    
    def forward(
        self,
        input_tensors: List[torch.Tensor],
        weights: List[torch.Tensor],
        group_list: Optional[list] = None,
        bias: Optional[List[torch.Tensor]] = None,
        scale: Optional[List[torch.Tensor]] = None,
        offset: Optional[List[torch.Tensor]] = None,
        antiquant_scale: Optional[List[torch.Tensor]] = None,
        antiquant_offset: Optional[List[torch.Tensor]] = None,
        per_token_scale: Optional[List[torch.Tensor]] = None,
        activation_input: Optional[List[torch.Tensor]] = None,
        activation_quant_scale: Optional[List[torch.Tensor]] = None,
        activation_quant_offset: Optional[List[torch.Tensor]] = None,
        split_item: int = 0,
        group_type: Optional[int] = None,
        group_list_type: int = 0,
        act_type: int = 0,
        output_dtype: Optional[torch.dtype] = None,
        tuning_config: Optional[Any] = None,
    ) -> List[torch.Tensor]:

        kernel_kwargs = {}
        if bias is not None:
            kernel_kwargs["bias"] = bias
        if scale is not None:
            kernel_kwargs["scale"] = scale
        if offset is not None:
            kernel_kwargs["offset"] = offset
        if antiquant_scale is not None:
            kernel_kwargs["antiquant_scale"] = antiquant_scale
        if antiquant_offset is not None:
            kernel_kwargs["antiquant_offset"] = antiquant_offset
        if per_token_scale is not None:
            kernel_kwargs["per_token_scale"] = per_token_scale
        if group_list is not None:
            kernel_kwargs["group_list"] = group_list
        if activation_input is not None:
            kernel_kwargs["activation_input"] = activation_input
        if activation_quant_scale is not None:
            kernel_kwargs["activation_quant_scale"] = activation_quant_scale
        if activation_quant_offset is not None:
            kernel_kwargs["activation_quant_offset"] = activation_quant_offset
        if split_item != 0:
            kernel_kwargs["split_item"] = split_item
        if group_type is None:
            group_type = -1
        kernel_kwargs["group_type"] = group_type
        if group_list_type != 0:
            kernel_kwargs["group_list_type"] = group_list_type
        if act_type != 0:
            kernel_kwargs["act_type"] = act_type
        if output_dtype is not None:
            kernel_kwargs["output_dtype"] = output_dtype
        if tuning_config is not None:
            kernel_kwargs["tuning_config"] = tuning_config

        return torch_npu.npu_grouped_matmul(
            input_tensors,
            weights,
            **kernel_kwargs
        )


class TorchNpuGroupQuantMatmulReduceSum(MojoGroupQuantMatmulReduceSum, default_priority=0):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__() 
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor):
        # NPU kernel requires x2_scale in BF16
        if x2_scale.dtype != torch.bfloat16:
            x2_scale = x2_scale.to(torch.bfloat16)

        fmt = torch_npu.get_npu_format(x2)
        if fmt != 29:
            x2_nz = torch_npu.npu_format_cast(x2, 29)
            logger.info(f"Not support x2 format {fmt}, cast to NZ format")
        else:
            x2_nz = x2

        result = torch_npu.npu_quant_matmul_reduce_sum(x1, x2_nz, x1_scale=x1_scale, x2_scale=x2_scale)
        return result