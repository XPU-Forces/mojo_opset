import torch

from typing import Union

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
            num_groups_w, n, bk = self.weight.shape
        else:
            num_groups_w, bk, n = self.weight.shape

        m, k = input.shape
        assert bk == k, "K of input should be equal to K of self.weight."
        assert num_groups_w == num_groups

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight
        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)
        out_list = []

        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            out_list.append(input[start:end, :] @ weight[g, :, :])
        return torch.cat(out_list, dim=0)

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        weight_dtype = self.weight.dtype if isinstance(self.weight, torch.Tensor) else None
        weight_device = self.weight.device if isinstance(self.weight, torch.Tensor) else None
        return f"{weight_shape=}, {weight_dtype=}, {weight_device=}".replace("self.", "")


class MojoGemmDequant(MojoOperator):
    """Fused int8 GEMM + dequantization.

    Performs int8 matrix multiplication with int32 accumulation, then applies
    per-token x per-channel scale factors to dequantize the result and casts
    to ``output_dtype``.

    The reference uses float32 matmul to emulate int8 GEMM because
    ``torch.matmul`` does not natively support int8 → int32 accumulation.
    float32 is exact for all int8 partial sums at practical ``K`` dimensions.

    Computation:
        ``output = (input_i8 @ weight_i8) * input_scale * weight_scale``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        weight_dtype: Union[str, torch.dtype] = torch.int8,
        **kwargs,
    ):
        """
        Args:
            in_features (int): Logical K dimension of the int8 weight.
            out_features (int): Logical N dimension of the int8 weight and
                weight scale.
            output_dtype (torch.dtype): Target dtype for the dequantized output.
                Supported: ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
            trans_weight (bool): If True, the weight tensor is provided as
                ``(N, K)`` and will be transposed to ``(K, N)`` internally.
                Otherwise it is stored directly as ``(K, N)``.
            **kwargs: Additional tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = (out_features, in_features) if trans_weight else (in_features, out_features)
        weight_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": quant_dtype}
        weight_scale_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": torch.bfloat16}
        self.quant_dtype = quant_dtype
        assert self.quant_dtype == torch.int8, f"GemmDequant only support int8 quantization yet, but get {quant_dtype=}"
        self.weight_dtype = weight_dtype
        assert self.weight_dtype == torch.int8, f"GemmDequant only support int8 weight yet, but get {weight_dtype=}"
        self.register_buffer(
            "weight",
            torch.empty(self.weight_shape, **weight_factory_kwargs),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, **weight_scale_factory_kwargs),
        )
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight

    def forward(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Quantised activation ``(M, K)`` in int8.
            input_scale (torch.Tensor): Runtime per-token activation scale ``(M,)`` or ``(M, 1)``.

        Returns:
            torch.Tensor: Dequantized result ``(M, N)`` in ``output_dtype``.
        """
        weight = self.weight

        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(f"input K {input.shape[-1]} must match weight K {self.in_features}.")
        if self.weight_scale.shape != (self.out_features,):
            raise ValueError(
                f"weight_scale shape {tuple(self.weight_scale.shape)} must match output dim {(self.out_features,)}."
            )

        if not self.trans_weight:
            weight = weight.mT
        out = torch.mul(input.int().unsqueeze(-2), weight.int()).float().sum(dim=-1)

        weight_scale = self.weight_scale
        if input_scale.dim() == 1:
            input_scale = input_scale.unsqueeze(-1)
        if weight_scale.dim() == 1:
            weight_scale = weight_scale.unsqueeze(0)

        out = out * input_scale.float() * weight_scale.float()

        return out.to(self.output_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"output_dtype={self.output_dtype}, trans_weight={self.trans_weight}"
            f"quant_dtype={self.quant_dtype}, weight_dtype={self.weight_dtype}"
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
