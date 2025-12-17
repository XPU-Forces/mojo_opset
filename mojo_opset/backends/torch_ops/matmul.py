from typing import Any, List, Optional
import torch
import torch_npu
from mojo_opset.core import MojoGroupedMatmul


class TorchGroupedMatmul(MojoGroupedMatmul, default_priority=10):

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(
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

    def forward_analysis(self, *args, **kwargs):

        pass