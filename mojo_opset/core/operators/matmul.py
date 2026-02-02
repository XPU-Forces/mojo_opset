from abc import abstractmethod
from typing import Any, List, Optional, Tuple
import torch
from ..operator import MojoOperator


class MojoGroupedMatmul(MojoOperator):
    """
    Performs the standard grouped matrix multiplication.
    This operator is designed to handle a list of matrices for batch processing.
    """

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__()

    def forward(
        self,
        input_tensors: List[torch.Tensor],
        other_tensors: List[torch.Tensor],
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
        group_type: Optional[int] = None,  # group_type is ignored in ref impl
        group_list_type: int = 0,
        act_type: int = 0,
        output_dtype: Optional[torch.dtype] = None,
        tuning_config: Optional[Any] = None,
    ) -> List[torch.Tensor]:

        outputs = []
        for i, (input_tensor, other_tensor) in enumerate(zip(input_tensors, other_tensors)):
            output = torch.matmul(input_tensor, other_tensor)

            if bias and i < len(bias) and bias[i] is not None:
                output += bias[i]

            outputs.append(output)
        return outputs


class MojoGroupQuantMatmulReduceSum(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__()

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor
    ) -> torch.Tensor:
        out = torch.bmm(x1.float(), x2.float()).to(torch.float32)
        out = x2_scale[None, None, :] * out
        out = x1_scale[:, :, None] * out

        b, m, k = x1.shape
        b, k, n = x2.shape
        out_1 = torch.zeros(m, n, dtype=torch.bfloat16, device=out.device)
        for i in range(b):
            out_1 += out[i, ...].to(torch.bfloat16)

        return out_1