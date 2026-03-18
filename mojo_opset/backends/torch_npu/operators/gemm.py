from typing import Optional, List
import torch
import torch_npu


from mojo_opset.core import MojoQuantMatmul
from mojo_opset.core import MojoGroupGemm

class TorchNpuQuantMatmul(MojoQuantMatmul):
    supported_platforms_list = ["npu"]

    def forward(
        x1,
        x2,
        scale,
        offset: torch.Tensor = None,
        pre_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        output_dtype: int = None,
    ) -> torch.Tensor:

        if scale.dim() == 2:
            assert offset is not None, "offset must be provided when scale is 2D"
            assert pre_scale is not None, "pre_scale must be provided when scale is 2D"
        
        assert output_dtype in [torch.float16, torch.bfloat16, torch.int8, torch.int32], "output_dtype must be float16, bfloat16, int8 or int32"

        return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, pre_scale=pre_scale, bias=bias, output_dtype=output_dtype)



class TorchNpuGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu"]
    def __init__(
        self, 
        weight: torch.Tensor,
        trans_weight=False,
    ):
        super().__init__()
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(
        self,
        input: List[torch.Tensor],
        group_list=None,
        bias: torch.Tensor = None,
        scale=None,
        offset: torch.Tensor = None,
        antiquant_offset=None,
        per_token_scale=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=0,
        group_type=None,
        group_list_type=0,
        act_type=0,
        output_dtype=None,
        tuning_config=None,
    ) -> List[torch.Tensor]:
        """
        case1: multi inputs; multi weights; multi outputs
        case2: single input; multi weights; single output
        case3: single input; multi weights; multi outputs
        case4: multi inputs; multi weights; single output
        """
        
        
        return npu_grouped_matmul(
            input=input,
            weight=self.weight,
            bias=bias,
            scale=scale,
            offset=offset,
            antiquant_offset=antiquant_offset,
            per_token_scale=per_token_scale,
            group_list=group_list,
            activation_input=activation_input,
            activation_quant_scale=activation_quant_scale,
            activation_quant_offset=activation_quant_offset,
            split_item=split_item,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            output_dtype=output_dtype,
            tuning_config=tuning_config,
        )
