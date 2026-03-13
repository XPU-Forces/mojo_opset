import torch
import torch.nn.functional as F

from ..operator import MojoOperator


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
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = weight[g, :, :]
            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        weight_dtype = self.weight.dtype if isinstance(self.weight, torch.Tensor) else None
        weight_device = self.weight.device if isinstance(self.weight, torch.Tensor) else None
        return (
            f"{weight_shape=}, {weight_dtype=}, {weight_device=}".replace("self.", "")
        )


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

        if x2_scale.dtype != torch.bfloat16:
            x2_scale = x2_scale.to(torch.bfloat16)

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
