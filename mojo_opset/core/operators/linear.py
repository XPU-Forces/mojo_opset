from typing import Optional

import torch

from ..operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Common parameter definitions for Linear operator.

        Init parameters:
        - weight (torch.Tensor): Weight tensor, shape [in_dim, out_dim].
        - bias (Optional[torch.Tensor]): Bias tensor, shape aligned with output dimension; optional.
        """
        super().__init__()

        if weight.ndim not in (2,):
            raise ValueError(f"weight should be 2-D, but got {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias should be torch.Tensor or None")
            if weight.ndim == 2:
                # Standard PyTorch Linear weight shape is [out_features, in_features]
                out_dim = weight.shape[0]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError(f"bias should be 1-D with shape [out_dim={out_dim}], but got {tuple(bias.shape)}")
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Standard PyTorch Linear weight shape is [out_features, in_features]
        in_dim = self.weight.shape[1]
        if input.shape[-1] != in_dim:
            raise ValueError(f"input should have last dim {in_dim}, but got {input.shape[-1]}")
        if input.ndim not in (3, 4):
            raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
        return torch.nn.functional.linear(input, self.weight, self.bias)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight=False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        Grouped linear forward over variable-length segments.

        Splits the 2D input into contiguous groups defined by `group_list`,
        applies a per-group weight, and concatenates outputs.

        Args:
            input (torch.Tensor): 2D tensor of shape (N, Din); rows are grouped
                contiguously. Sum(group_list) must equal N.
            group_list (torch.Tensor): 1D tensor of length G with row counts per group.

        Returns:
            torch.Tensor: 2D tensor of shape (N, Dout), concatenated per-group outputs.

        Notes:
            - Expects `self.weight` of shape (G, Din, Dout). If `trans_weight` is True,
            weights are transposed from (G, Dout, Din) to (G, Din, Dout).
            - Each group's output is computed as `input_g @ weight_g`.
        """
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            self.weight = self.weight.transpose(1, 2).contiguous()

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = self.weight[g, :, :]
            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)


class MojoLinearAllReduce(MojoOperator):
    pass


class MojoAllGatherLinear(MojoOperator):
    pass


class MojoLinearAll2All(MojoOperator):
    pass


class MojoLinearReduceScatter(MojoOperator):
    pass
