from typing import Optional

import torch
from torch.distributed.tensor import DTensor

from ..operator import MojoOperator


class MojoGroupGemm(MojoOperator):
    def __init__(
        self,
        weight,
        trans_weight=False,
    ):
        super().__init__()
        self.weight = weight
        self.trans_weight = trans_weight

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

        if group_list.device.type != "cpu":
            group_list = group_list.to("cpu")

        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"

        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "weight group count must match group_list length"

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


class MojoGemmDequant(MojoOperator):
    """Fused int8 GEMM + dequantization.

    Performs int8 matrix multiplication with int32 accumulation, then applies
    per-token × per-channel scale factors to dequantize the result, adds an
    optional bias, and casts to ``output_dtype``.

    The reference uses float32 matmul to emulate int8 GEMM because
    ``torch.matmul`` does not natively support int8 → int32 accumulation.
    float32 is exact for all int8 partial sums at practical ``K`` dimensions.

    Computation:
        ``output = (input_i8 @ weight_i8) * input_scale * weight_scale + bias``
    """

    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
    ):
        """
        Args:
            output_dtype (torch.dtype): Target dtype for the dequantized output.
                Supported: ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
            trans_weight (bool): If True, the weight tensor is provided as
                ``(N, K)`` and will be transposed to ``(K, N)`` internally.
        """
        super().__init__()
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Quantised activation ``(M, K)`` in int8.
            weight (torch.Tensor): Quantised weight ``(K, N)`` in int8, or
                ``(N, K)`` when ``trans_weight=True``.
            input_scale (torch.Tensor): Per-token activation scale ``(M, 1)``
                or ``(M,)``.
            weight_scale (torch.Tensor): Per-channel weight scale ``(1, N)``
                or ``(N,)``.
            bias (Optional[torch.Tensor]): Optional bias ``(N,)`` in
                ``output_dtype``.

        Returns:
            torch.Tensor: Dequantized result ``(M, N)`` in ``output_dtype``.
        """
        if self.trans_weight:
            weight = weight.t()

        out = torch.matmul(input.float(), weight.float())

        if input_scale.dim() == 1:
            input_scale = input_scale.unsqueeze(-1)
        if weight_scale.dim() == 1:
            weight_scale = weight_scale.unsqueeze(0)

        out = out * input_scale.float() * weight_scale.float()

        if bias is not None:
            out = out + bias.float()

        return out.to(self.output_dtype)

    def extra_repr(self) -> str:
        return f"output_dtype={self.output_dtype}, trans_weight={self.trans_weight}"


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
