from typing import Optional

import torch

from ..operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        input_layout: str = "NZ",
        weight: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for Linear operator.

        Init parameters:
        - input_layout (str): Input layout enumeration, values {"KN","NZ"}, default "NZ".
        - weight (torch.Tensor): Weight tensor, shape [in_dim, out_dim].
        - bias (Optional[torch.Tensor]): Bias tensor, shape aligned with output dimension; optional.
        - is_varlen (bool): When True, prioritize TND (per token) computation; when False, use BSND; default True.
        - op_name (str): Operator name placeholder.
        """
        super().__init__(op_name)

        if input_layout not in {"KN", "NZ"}:
            raise ValueError('input_layout should be {"KN","NZ"}')
        self.input_layout = input_layout

        if weight is None or not isinstance(weight, torch.Tensor):
            raise TypeError("weight should be torch.Tensor and not None")
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
        self.is_varlen = is_varlen

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Linear forward with optional varlen mode.

        Validates input shape depending on `is_varlen`, then applies a
        standard PyTorch linear transformation with weight shape
        [out_features, in_features].

        Args:
            input (torch.Tensor):
                - is_varlen=True expects shapes (T, N, D) or (T, D).
                - is_varlen=False expects shapes (B, N, S, D) or (B, S, D).

        Returns:
            torch.Tensor: Output tensor with the same rank as input; the
            last dimension equals `out_features`.

        Raises:
            ValueError: If last dimension != `in_features`, or the rank
            does not match the expected layout for the chosen mode.
        """
        # Standard PyTorch Linear weight shape is [out_features, in_features]
        in_dim = self.weight.shape[1]
        if input.shape[-1] != in_dim:
            raise ValueError(f"input should have last dim {in_dim}, but got {input.shape[-1]}")
        if self.is_varlen:
            if input.ndim not in (2, 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight, self.bias)
        else:
            if input.ndim not in (3, 4):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight, self.bias)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        trans_weight=False,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

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
