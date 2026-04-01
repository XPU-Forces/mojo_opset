from typing import Optional

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


def _expand_group_param(
    param: Optional[torch.Tensor],
    token_count: Optional[torch.Tensor],
    row_count: int,
) -> Optional[torch.Tensor]:
    if param is None:
        return None

    param_fp = param.float()
    if token_count is None:
        if param_fp.dim() == 1:
            return param_fp.unsqueeze(0).expand(row_count, -1)
        if param_fp.dim() == 2 and param_fp.size(0) == 1:
            return param_fp.expand(row_count, -1)
        return param_fp

    token_count_i64 = token_count.to(dtype=torch.int64, device=param.device)
    if param_fp.dim() == 1:
        return param_fp.unsqueeze(0).expand(row_count, -1)
    if param_fp.dim() == 2 and param_fp.size(0) == 1:
        return param_fp.expand(row_count, -1)
    if param_fp.dim() != 2 or param_fp.size(0) != token_count_i64.numel():
        raise ValueError(
            "Grouped tensor must be 2D with the first dimension equal to token_count length, "
            f"but got shape {tuple(param.shape)} and token_count length {token_count_i64.numel()}."
        )

    expanded = param_fp.repeat_interleave(token_count_i64, dim=0)
    if expanded.size(0) != row_count:
        raise ValueError(f"Expanded grouped tensor row count mismatch: expected {row_count}, got {expanded.size(0)}.")
    return expanded


def _apply_smooth_scale(
    input_fp: torch.Tensor,
    smooth_scale: Optional[torch.Tensor],
    token_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if smooth_scale is None:
        return input_fp

    if token_count is None:
        scale_fp = smooth_scale.float()
        while scale_fp.dim() < input_fp.dim():
            scale_fp = scale_fp.unsqueeze(0)
        return input_fp * scale_fp

    flat_input = input_fp.reshape(-1, input_fp.shape[-1])
    expanded_scale = _expand_group_param(smooth_scale, token_count, flat_input.size(0))
    return (flat_input * expanded_scale).reshape_as(input_fp)


class MojoQuant(MojoOperator):
    def __init__(
        self,
        quant_dtype: torch.dtype = torch.int8,
        symmetric: bool = True,
        group_size: int = -1,
    ):
        """
        Initialize quantization operator.

        Args:
            quant_dtype (torch.dtype, default=torch.int8): Target quantization dtype.
                Supported: torch.int8, torch.float8_e4m3fn.
            symmetric (bool, default=True): If True, use symmetric quantization (no zero_point).
            group_size (int, default=-1): Group size for per-group quantization.
                -1 means no grouping. Must divide the last dimension evenly when > 0.
        """
        super().__init__()
        self.quant_dtype = quant_dtype
        self.symmetric = symmetric
        self.group_size = group_size

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128 if symmetric else 0
        elif quant_dtype == torch.float8_e4m3fn:
            self.q_max = torch.finfo(torch.float8_e4m3fn).max
            self.q_min = -torch.finfo(torch.float8_e4m3fn).max
        else:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8 or torch.float8_e4m3fn"
            )

    def forward(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Quantize a floating-point tensor with a caller-supplied scale.

        Args:
            input (torch.Tensor): Input floating-point tensor of shape (..., K).
            scale (torch.Tensor): Pre-computed scale tensor. Shape must be
                broadcastable to ``input``.
                - per-token: shape (..., 1) or (...,) matching all but the last dim.
                - per-group: shape (..., K // group_size, 1) after the internal reshape.
                - per-tensor: scalar or shape (1,).
            zero_point (Optional[torch.Tensor]): Zero point tensor. Only used when
                ``symmetric=False``; must be broadcastable to ``input``. Required
                when ``symmetric=False``.

        Returns:
            torch.Tensor: Quantized tensor in ``self.quant_dtype``, same shape as ``input``.
        """
        if not self.symmetric and zero_point is None:
            raise ValueError("zero_point is required when symmetric=False")

        input_fp = input.float()

        if self.group_size > 0:
            orig_shape = input.shape
            assert input.shape[-1] % self.group_size == 0, (
                f"Last dim {input.shape[-1]} must be divisible by group_size {self.group_size}"
            )
            input_fp = input_fp.reshape(*input.shape[:-1], -1, self.group_size)

        if self.symmetric:
            output = torch.clamp(torch.round(input_fp / scale.float()), self.q_min, self.q_max)
        else:
            output = torch.clamp(
                torch.round(input_fp / scale.float()) + zero_point.float(),
                self.q_min,
                self.q_max,
            )

        if self.group_size > 0:
            output = output.reshape(orig_shape)

        return output.to(self.quant_dtype)

    def extra_repr(self) -> str:
        return (
            f"quant_dtype={self.quant_dtype}, symmetric={self.symmetric}, "
            f"group_size={self.group_size}, q_max={self.q_max}, q_min={self.q_min}"
        )


class MojoDynamicQuant(MojoOperator):
    """Per-token dynamic int8 quantization on the last dimension.

    Optionally applies a per-channel ``scale_tensor`` before quantizing.
    Returns ``(q_int8, quant_scale)``.
    """

    def forward(self, input_tensor: torch.Tensor, scale_tensor: Optional[torch.Tensor] = None):
        if scale_tensor is None:
            scaled_fp = input_tensor.to(torch.float64)
        else:
            scaled_fp = input_tensor.to(torch.float64) * scale_tensor.to(torch.float64)

        max_abs = scaled_fp.abs().amax(dim=-1).clamp(min=1e-10)
        quant_vals = 127.0 * (scaled_fp / max_abs.unsqueeze(-1))
        q = torch.trunc(quant_vals + 0.5 * torch.sign(quant_vals))
        return q.to(torch.int8), (max_abs / 127.0).to(input_tensor.dtype)


class MojoDequant(MojoOperator):
    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        symmetric: bool = True,
        group_size: int = -1,
    ):
        """
        Initialize dequantization operator.

        Args:
            output_dtype (torch.dtype, default=torch.bfloat16): Target output dtype
                after dequantization.
            symmetric (bool, default=True): Must match the MojoQuant that produced the
                quantized tensor. If True, dequantize as ``x * scale``; otherwise
                dequantize as ``(x - zero_point) * scale``.
            group_size (int, default=-1): Group size used during quantization.
                -1 means no grouping. Must match the MojoQuant group_size.
        """
        super().__init__()
        self.output_dtype = output_dtype
        self.symmetric = symmetric
        self.group_size = group_size

    def forward(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dequantize a quantized tensor back to floating point.

        Args:
            input (torch.Tensor): Quantized tensor (e.g., int8 or float8).
            scale (torch.Tensor): Scale tensor produced by MojoQuant.
            zero_point (Optional[torch.Tensor]): Zero point tensor, required when
                ``symmetric=False``.

        Returns:
            torch.Tensor: Dequantized tensor in ``self.output_dtype``.
        """
        input_fp = input.float()
        scale_fp = scale.float()
        zp_fp = zero_point.float() if zero_point is not None else None

        if self.group_size > 0:
            orig_shape = input.shape
            input_fp = input_fp.reshape(*input.shape[:-1], -1, self.group_size)
        else:
            while scale_fp.dim() < input_fp.dim():
                scale_fp = scale_fp.unsqueeze(-1)
            if zp_fp is not None:
                while zp_fp.dim() < input_fp.dim():
                    zp_fp = zp_fp.unsqueeze(-1)

        if self.symmetric:
            output = input_fp * scale_fp
        else:
            assert zp_fp is not None, "zero_point is required for asymmetric dequantization"
            output = (input_fp - zp_fp) * scale_fp

        if self.group_size > 0:
            output = output.reshape(orig_shape)

        return output.to(self.output_dtype)

    def extra_repr(self) -> str:
        return f"output_dtype={self.output_dtype}, symmetric={self.symmetric}, group_size={self.group_size}"


class MojoDynamicQuant(MojoOperator):
    def __init__(
        self,
        quant_dtype: torch.dtype = torch.int8,
        smooth_input: bool = False,
        moe_mode: bool = False,
        **kwargs,
    ):
        """
        Dynamic per-token symmetric quantization with optional smooth quant scaling.

        Args:
            quant_dtype (torch.dtype): Target quantized dtype. Currently only ``torch.int8`` is supported.
            smooth_input (bool): Whether a caller-provided ``smooth_scale`` is expected and applied before
                computing the dynamic per-token scale.
            moe_mode (bool): Whether ``token_count`` is interpreted as grouped token counts for MoE-style
                per-group smooth scaling.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.quant_dtype = quant_dtype
        self.smooth_input = smooth_input
        self.moe_mode = moe_mode

        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")

        self.q_max = 127
        self.q_min = -128

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input (torch.Tensor): Floating-point input of shape ``(*, K)``.
            smooth_scale (Optional[torch.Tensor]): Optional smooth-quant scale.
                - non-MoE: broadcastable to the last dimension, typically ``(K,)``.
                - MoE: ``(num_groups, K)`` paired with ``token_count``.
            token_count (Optional[torch.Tensor]): Optional grouped token counts for MoE mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Quantized int8 tensor with the same shape as ``input``.
                - Per-token dynamic scale of shape ``input.shape[:-1]``.
        """
        if self.smooth_input and smooth_scale is None:
            raise ValueError("smooth_scale is required when smooth_input=True")
        if self.moe_mode and token_count is None:
            raise ValueError("token_count is required when moe_mode=True")
        if token_count is not None and not self.moe_mode:
            raise ValueError("token_count is only supported when moe_mode=True")

        input_fp = _apply_smooth_scale(input.float(), smooth_scale, token_count)
        scale = input_fp.abs().amax(dim=-1).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(input_fp / scale.unsqueeze(-1)), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return f"quant_dtype={self.quant_dtype}, smooth_input={self.smooth_input}, moe_mode={self.moe_mode}"


class MojoDequantSwiGLUQuant(MojoOperator):
    def __init__(
        self,
        quant_dtype: torch.dtype = torch.int8,
        activate_left: bool = False,
        quant_mode: int = 1,
        **kwargs,
    ):
        """
        Fused dequantization + SwiGLU + dynamic quantization.

        This mirrors the common W8A8 MLP path where the FC1 output is dequantized, activated with SwiGLU,
        optionally smooth-scaled for FC2, and quantized again.

        Args:
            quant_dtype (torch.dtype): Target quantized dtype. Currently only ``torch.int8`` is supported.
            activate_left (bool): Whether SwiGLU applies SiLU on the left split instead of the right split.
            quant_mode (int): Quantization mode. Currently only dynamic quantization (``1``) is supported.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.quant_dtype = quant_dtype
        self.activate_left = activate_left
        self.quant_mode = quant_mode

        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")
        if quant_mode != 1:
            raise NotImplementedError("Only dynamic quant_mode=1 is currently supported.")

        self.q_max = 127
        self.q_min = -128

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        activation_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): Input tensor of shape ``(tokens, 2H)``.
            weight_scale (Optional[torch.Tensor]): Optional dequant scale, either ``(2H,)`` or grouped
                ``(num_groups, 2H)`` with ``token_count``.
            activation_scale (Optional[torch.Tensor]): Optional per-token activation scale of shape ``(tokens,)``.
            bias (Optional[torch.Tensor]): Optional bias, either ``(2H,)`` or grouped ``(num_groups, 2H)``.
            quant_scale (Optional[torch.Tensor]): Optional smooth scale applied before the re-quantization stage,
                either ``(H,)`` or grouped ``(num_groups, H)``.
            quant_offset (Optional[torch.Tensor]): Optional quant offset. Currently unsupported and must be ``None``.
            token_count (Optional[torch.Tensor]): Optional grouped token counts for grouped dequant/smooth-quant.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Quantized int8 output of shape ``(tokens, H)``.
                - Per-token dynamic scale of shape ``(tokens,)``.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D with shape (tokens, 2H), but got {tuple(x.shape)}")
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"x last dimension must be even for SwiGLU split, but got {x.shape[-1]}")
        if quant_offset is not None:
            raise NotImplementedError("quant_offset is not supported by the torch reference implementation.")

        token_num = x.shape[0]
        if token_count is not None:
            token_count_i64 = token_count.to(dtype=torch.int64, device=x.device)
            if token_count_i64.sum().item() != token_num:
                raise ValueError(
                    f"token_count sum must equal token number {token_num}, got {token_count_i64.sum().item()}."
                )

        x_fp = x.float()

        weight_scale_fp = _expand_group_param(weight_scale, token_count, token_num)
        if weight_scale_fp is not None:
            x_fp = x_fp * weight_scale_fp

        if activation_scale is not None:
            activation_scale_fp = activation_scale.float()
            if activation_scale_fp.dim() != 1 or activation_scale_fp.numel() != token_num:
                raise ValueError(
                    f"activation_scale must be 1D with {token_num} elements, got shape {tuple(activation_scale.shape)}."
                )
            x_fp = x_fp * activation_scale_fp.unsqueeze(-1)

        bias_fp = _expand_group_param(bias, token_count, token_num)
        if bias_fp is not None:
            x_fp = x_fp + bias_fp

        left, right = x_fp.chunk(2, dim=-1)
        if self.activate_left:
            out_fp = F.silu(left) * right
        else:
            out_fp = F.silu(right) * left

        quant_scale_fp = _expand_group_param(quant_scale, token_count, token_num)
        if quant_scale_fp is not None:
            out_fp = out_fp * quant_scale_fp

        scale = out_fp.abs().amax(dim=-1).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(out_fp / scale.unsqueeze(-1)), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return f"quant_dtype={self.quant_dtype}, activate_left={self.activate_left}, quant_mode={self.quant_mode}"
