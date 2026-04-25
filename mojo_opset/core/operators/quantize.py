from typing import Optional

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoStaticQuant(MojoOperator):
    def __init__(
        self,
        input_size: int,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        """
        Quantize a floating-point tensor with a one-dimensional scale parameter.

        Args:
            input_size (int): Size of the 1-D scale vector. It is expected to match ``input.shape[-1]``.
            quant_dtype (torch.dtype, default=torch.int8): Target quantization dtype.
                Supported: torch.int8, torch.float8_e4m3fn.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.scale = torch.nn.Parameter(torch.empty(input_size, **self.tensor_factory_kwargs))
        self.quant_dtype = quant_dtype

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128
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
    ) -> torch.Tensor:
        """
        Quantize a floating-point tensor with a caller-supplied scale.

        Args:
            input (torch.Tensor): Input floating-point tensor of shape (..., K).
        Returns:
            torch.Tensor: Quantized tensor in ``self.quant_dtype``, same shape as ``input``.
        """
        input_fp = input.float()
        output = torch.clamp(torch.round(input_fp / self.scale.float()), self.q_min, self.q_max)
        return output.to(self.quant_dtype)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, quant_dtype={self.quant_dtype}, q_max={self.q_max}, q_min={self.q_min}"


class MojoDequant(MojoOperator):
    def __init__(
        self,
        input_size: int,
        output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """
        Initialize dequantization operator.

        Args:
            input_size (int): Size of the 1-D scale vector. It is expected to match ``input.shape[-1]``.
            output_dtype (torch.dtype, default=torch.bfloat16): Target output dtype
                after dequantization.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.scale = torch.nn.Parameter(torch.empty(input_size, **self.tensor_factory_kwargs))
        if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise NotImplementedError(
                f"Unsupported output_dtype: {output_dtype}, expected torch.float16, torch.bfloat16, or torch.float32."
            )
        self.output_dtype = output_dtype

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize a quantized tensor back to floating point.

        Args:
            input (torch.Tensor): Quantized tensor (e.g., int8 or float8).
        Returns:
            torch.Tensor: Dequantized tensor in ``self.output_dtype``.
        """
        input_fp = input.float()
        output = input_fp * self.scale.float()
        return output.to(self.output_dtype)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_dtype={self.output_dtype}"


class MojoDynamicQuant(MojoOperator):
    def __init__(
        self,
        input_size: Optional[int] = None,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        """
        Dynamic per-token symmetric quantization with optional smooth quant scaling.

        Args:
            input_size (Optional[int]): Size of the optional 1-D smooth scale vector.
            quant_dtype (torch.dtype): Target quantized dtype. Currently only ``torch.int8`` is supported.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        if input_size is None:
            self.register_parameter("smooth_scale", None)
        else:
            self.smooth_scale = torch.nn.Parameter(torch.empty(input_size, **self.tensor_factory_kwargs))
        self.quant_dtype = quant_dtype

        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")

        self.q_max = 127
        self.q_min = -128

    def forward(
        self,
        input: torch.Tensor,
    ):
        """
        Args:
            input (torch.Tensor): Floating-point input of shape ``(*, K)``.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Quantized int8 tensor with the same shape as ``input``.
                - Per-token dynamic scale of shape ``input.shape[:-1]``.
        """
        if input.dim() < 1:
            raise ValueError("input must have at least one dimension.")

        input_fp = input.float()
        if self.smooth_scale is not None:
            smooth_scale = self.smooth_scale.float()
            while smooth_scale.dim() < input_fp.dim():
                smooth_scale = smooth_scale.unsqueeze(0)
            input_fp = input_fp * smooth_scale
        scale = input_fp.abs().amax(dim=-1).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(input_fp / scale.unsqueeze(-1)), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, quant_dtype={self.quant_dtype}"


class MojoMoEDynamicQuant(MojoOperator):
    def __init__(
        self,
        expert_num: int,
        input_size: int,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        """
        MoE dynamic per-token int8 quantization with grouped smooth-quant scaling.

        Args:
            expert_num (int): Number of experts.
            input_size (int): Last dimension of the input tensor.
            quant_dtype (torch.dtype): Target quantized dtype. Currently only ``torch.int8`` is supported.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.expert_num = expert_num
        self.input_size = input_size
        self.smooth_scale = torch.nn.Parameter(torch.empty((expert_num, input_size), **self.tensor_factory_kwargs))
        self.quant_dtype = quant_dtype

        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")

        self.q_max = 127
        self.q_min = -128

    def forward(
        self,
        input: torch.Tensor,
        token_count: torch.Tensor,
    ):
        """
        Args:
            input (torch.Tensor): Floating-point input of shape ``(tokens, K)`` or ``(*, K)``.
            token_count (torch.Tensor): int32/int64 token count per group. Sum must equal flattened token count.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Quantized int8 tensor with the same shape as ``input``.
                - Per-token dynamic scale of shape ``input.shape[:-1]``.
        """
        if input.dim() < 2:
            raise ValueError(f"input must have at least 2 dimensions for MoE dynamic quant, got {input.dim()}.")
        if token_count.dim() != 1:
            raise ValueError(f"token_count must be 1D, got shape {tuple(token_count.shape)}.")
        if token_count.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_count must be int32 or int64, got {token_count.dtype}.")
        if torch.any(token_count < 0):
            raise ValueError("token_count must be non-negative.")
        row_count = input.reshape(-1, input.shape[-1]).size(0)
        if int(token_count.sum().item()) != row_count:
            raise ValueError(
                f"token_count sum must equal flattened row count {row_count}, got {token_count.sum().item()}."
            )
        expanded_scale = self.smooth_scale.float().repeat_interleave(token_count, dim=0)
        input_fp = input.float() * expanded_scale
        scale = input_fp.abs().amax(dim=-1).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(input_fp / scale.unsqueeze(-1)), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return f"expert_num={self.expert_num}, input_size={self.input_size}, quant_dtype={self.quant_dtype}"


class MojoDequantSwiGLUQuant(MojoOperator):
    def __init__(
        self,
        expert_num: int,
        hidden_size: int,
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
            expert_num (int): Number of experts.
            hidden_size (int): SwiGLU output hidden size. Input last dimension is expected to be ``2 * hidden_size``.
            quant_dtype (torch.dtype): Target quantized dtype. Currently only ``torch.int8`` is supported.
            activate_left (bool): Whether SwiGLU applies SiLU on the left split instead of the right split.
            quant_mode (int): Quantization mode. Currently only dynamic quantization (``1``) is supported.
            **kwargs: Tensor factory kwargs.
        """
        super().__init__(**kwargs)
        self.expert_num = expert_num
        self.hidden_size = hidden_size
        self.weight_scale = torch.nn.Parameter(torch.empty((expert_num, hidden_size * 2), **self.tensor_factory_kwargs))
        self.quant_scale = torch.nn.Parameter(torch.empty((expert_num, hidden_size), **self.tensor_factory_kwargs))
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
        activation_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): Input tensor of shape ``(tokens, 2H)``.
            activation_scale (Optional[torch.Tensor]): Optional runtime per-token activation scale of shape ``(tokens,)``.
            bias (Optional[torch.Tensor]): Optional bias, either ``(2H,)`` or grouped ``(num_groups, 2H)``.
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
            if token_count.sum().item() != token_num:
                raise ValueError(
                    f"token_count sum must equal token number {token_num}, got {token_count.sum().item()}."
                )

        x_fp = x.float()

        weight_scale = self.weight_scale.float()
        if token_count is not None:
            weight_scale = weight_scale.repeat_interleave(token_count, dim=0)
        x_fp = x_fp * weight_scale
        if activation_scale is not None:
            x_fp = x_fp * activation_scale.float().unsqueeze(-1)

        if bias is not None:
            bias_fp = bias.float()
            if token_count is not None and bias_fp.dim() == 2:
                bias_fp = bias_fp.repeat_interleave(token_count, dim=0)
            x_fp = x_fp + bias_fp

        left, right = x_fp.chunk(2, dim=-1)
        if self.activate_left:
            out_fp = F.silu(left) * right
        else:
            out_fp = F.silu(right) * left

        quant_scale = self.quant_scale.float()
        if token_count is not None:
            quant_scale = quant_scale.repeat_interleave(token_count, dim=0)
        out_fp = out_fp * quant_scale

        scale = out_fp.abs().amax(dim=-1).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(out_fp / scale.unsqueeze(-1)), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return (
            f"expert_num={self.expert_num}, hidden_size={self.hidden_size}, quant_dtype={self.quant_dtype}, "
            f"activate_left={self.activate_left}, quant_mode={self.quant_mode}"
        )
