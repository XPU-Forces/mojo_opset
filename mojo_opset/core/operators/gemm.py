import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


MojoGemm = MojoLinear


class MojoGroupGemm(MojoOperator):
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


class MojoQuantGroupLinearReduceSum(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight: bool = False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        x1_scale: torch.Tensor,
        x2_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantized grouped linear with per-token scaling and batch reduction.

        Applies batched matmul on int8 inputs/weights in float32, scales by
        per-token `x1_scale` and per-output `x2_scale`, then reduces over batch.

        Args:
            input (torch.Tensor): 3D tensor of shape (B, M, K).
            x1_scale (torch.Tensor): Per-token scale of shape (B, M).
            x2_scale (torch.Tensor): Per-output scale of shape (N,), bfloat16 preferred.

        Returns:
            torch.Tensor: Reduced output of shape (M, N) in bfloat16.

        Notes:
            - Expects `self.weight` of shape (B, K, N) if `trans_weight` is False,
              otherwise (B, N, K) and transposed to (B, K, N).
            - The reduction sums outputs across batch dimension B.
        """
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        b, m, k = input.shape
        b_w, k_w, n = weight.shape
        assert b == b_w, "input and weight must have same batch size"
        assert k == k_w, "K of input should be equal to K of weight"

        out = torch.bmm(input.float(), weight.float()).to(torch.float32)
        out = x2_scale[None, None, :] * out
        out = x1_scale[:, :, None] * out

        reduced_out = torch.zeros(m, n, dtype=torch.bfloat16, device=out.device)
        for i in range(b):
            reduced_out += out[i, ...].to(torch.bfloat16)

        return reduced_out


class MojoGemmAllReduce(MojoOperator):
    pass


class MojoAllGatherGemm(MojoOperator):
    pass


class MojoAllGatherGemm(MojoOperator):
    pass


class MojoGemmAll2All(MojoOperator):
    pass


class MojoGemmReduceScatter(MojoOperator):
    pass
