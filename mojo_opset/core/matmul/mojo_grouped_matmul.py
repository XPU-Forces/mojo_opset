from abc import abstractmethod
import torch
from mojo_opset.core.mojo_operator import MojoOperator


class MojoGroupedMatmul(MojoOperator):
    """
    Performs the standard (non-quantized) grouped matrix multiplication.

    Args:
    - input_tensor (torch.Tensor): The first input tensor.
    - other_tensor (torch.Tensor): The second input tensor.
    - group_info (list): Grouping information for the matmul operation.
    - trans_input (bool): Whether to transpose the first input.
    - trans_other (bool): Whether to transpose the second input.
    """

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    @abstractmethod
    def forward_std(
        self,
        input_tensor: torch.Tensor,
        other_tensor: torch.Tensor,
        group_info: list,
        trans_input: bool = False,
        trans_other: bool = False,
    ):

        raise NotImplementedError

    def forward_ref(
        self,
        input_tensor: torch.Tensor,
        other_tensor: torch.Tensor,
        group_info: list,
        trans_input: bool = False,
        trans_other: bool = False,
    ):
        """
        A reference implementation for grouped matrix multiplication.
        This can be used for verification.
        """
        if trans_input:
            input_tensor = input_tensor.transpose(-1, -2)
        if trans_other:
            other_tensor = other_tensor.transpose(-1, -2)

        outputs = []
        for i_group in group_info:
            # Assuming group_info contains slices for each group
            input_slice = input_tensor[i_group[0]]
            other_slice = other_tensor[i_group[1]]
            outputs.append(torch.matmul(input_slice, other_slice))
        return torch.cat(outputs, dim=0)