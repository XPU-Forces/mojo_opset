from typing import Optional

import torch

from ..operator import MojoOperator


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
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8 or torch.float8_e4m3fn")

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
        return (
            f"output_dtype={self.output_dtype}, symmetric={self.symmetric}, "
            f"group_size={self.group_size}"
        )
