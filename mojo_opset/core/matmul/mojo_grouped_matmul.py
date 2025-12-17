from abc import abstractmethod
from typing import Any, List, Optional, Tuple
import torch
from mojo_opset.core.mojo_operator import MojoOperator


class MojoGroupedMatmul(MojoOperator):
    """
    Performs the standard grouped matrix multiplication.
    This operator is designed to handle a list of matrices for batch processing.
    """

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    @abstractmethod
    def forward_std(
        self,
        input_tensors: List[torch.Tensor],
        weight: List[torch.Tensor],
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

        raise NotImplementedError

    def forward_ref(
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

    @abstractmethod
    def forward_analysis(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError