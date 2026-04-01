from typing import Optional

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


def _apply_optional_smooth_scale(input_fp: torch.Tensor, smooth_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if smooth_scale is None:
        return input_fp

    scale_fp = smooth_scale.float()
    while scale_fp.dim() < input_fp.dim():
        scale_fp = scale_fp.unsqueeze(0)
    return input_fp * scale_fp


class MojoLayerNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        """
        Initialize LayerNorm patch parameters.

        Args:
            norm_size (int): Size of 1-D affine scale and shift vector.
            eps (float, default=1e-5): Epsilon added to the variance for numerical stability; must be > 0.
            elementwise_affine (bool, default=True): Whether to apply elementwise affine transform.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.
        """
        super().__init__(**kwargs)
        self.norm_size = norm_size
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None
        self.variance_epsilon = eps

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm over the last dimension of the input.

        Args:
            hidden_state (torch.Tensor): Input tensor whose last dimension is the hidden size
                (e.g., shape (B, T, D) or (..., D)). The normalization is performed across D.

        Returns:
            torch.Tensor: Tensor of the same shape and dtype as `hidden_state`, normalized
                over the last dimension.
        """
        return F.layer_norm(
            hidden_state,
            [hidden_state.shape[-1]],
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.variance_epsilon=}, {self.elementwise_affine=}".replace("self.", "")


class MojoRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Initialize RMSNorm patch parameters.

        Args:
            norm_size (int): Size of 1-D affine scale vector.
            eps (float, default=1e-5): Epsilon added for numerical stability; must be > 0.\
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.
        """
        super().__init__(**kwargs)
        self.norm_size = norm_size
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm over the last dimension of the input.

        Args:
            hidden_state (torch.Tensor): Input tensor whose last dimension is the hidden size
                (e.g., shape (B, T, D) or (..., D)). The normalization is performed across D.

        Returns:
            torch.Tensor: Tensor of the same shape and dtype as `hidden_state`, normalized
            over the last dimension.
        """
        return F.rms_norm(
            hidden_state,
            [hidden_state.shape[-1]],
            weight=self.weight,
            eps=self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.variance_epsilon=}".replace("self.", "")


class MojoGroupRMSNorm(MojoOperator):
    def __init__(self, num_groups, norm_size, eps, elementwise_affine=True, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.norm_size = norm_size
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty((num_groups, norm_size), **self.tensor_factory_kwargs))
        else:
            self.weight = None
        self.variance_epsilon = eps

    def forward(self, input_groups):
        # Note: input_groups is a list of tensors, each tensor has compatible shapes for norm
        output_groups = []
        for group_id in range(self.num_groups):
            output_groups.append(F.rms_norm(input_groups[group_id], (self.norm_size,), weight=self.weight[group_id], eps=self.variance_epsilon))
        return output_groups

    def extra_expr(self) -> str:
        return f"{self.num_groups=}, {self.norm_size=}, {self.variance_epsilon=} {self.elementwise_affine=}".replace("self.", "")

class MojoGroupLayerNorm(MojoOperator):
    def __init__(self, num_groups, norm_size, eps, elementwise_affine=True, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.norm_size = norm_size
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty((num_groups, norm_size), **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty((num_groups, norm_size), **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None
        self.variance_epsilon = eps

    def forward(self, input_groups):
        # Note: input_groups is a list of tensors, each tensor has compatible shapes for norm
        output_groups = []
        for group_id in range(self.num_groups):
            output_groups.append(F.layer_norm(input_groups[group_id], (self.norm_size,), weight=self.weight[group_id], bias=self.bias[group_id], eps=self.variance_epsilon))
        return output_groups

    def extra_repr(self) -> str:
        return f"{self.num_groups=}, {self.norm_size=}, {self.variance_epsilon=} {self.elementwise_affine=}".replace("self.", "")

class MojoRMSNormQuant(MojoOperator):
    """Fused RMSNorm + dynamic per-token quantization.

    Semantics::

        normed = rms_norm(hidden_state)
        scale  = amax(|normed|, dim=-1) / q_max
        output = clamp(round(normed / scale), q_min, q_max)

    Returns ``(quant_output, scale)``.
    """

    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        quant_dtype: torch.dtype = torch.int8,
        symmetric: bool = True,
        **kwargs,
    ):
        """
        Args:
            norm_size (int): Hidden dimension for RMSNorm.
            eps (float): Epsilon for RMSNorm stability.
            quant_dtype (torch.dtype): Target quantization dtype.
            symmetric (bool): Symmetric quantization flag.
            **kwargs: Tensor factory kwargs (device, dtype).
        """
        super().__init__(**kwargs)
        self.norm_size = norm_size
        self.variance_epsilon = eps
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.quant_dtype = quant_dtype
        self.symmetric = symmetric

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128 if symmetric else 0
        elif quant_dtype == torch.float8_e4m3fn:
            self.q_max = torch.finfo(torch.float8_e4m3fn).max
            self.q_min = -torch.finfo(torch.float8_e4m3fn).max
        else:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, "
                f"expected torch.int8 or torch.float8_e4m3fn"
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (torch.Tensor): ``(*, D)`` input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``quant_output`` in ``quant_dtype``, same shape as input.
                - ``scale`` of shape ``(*, 1)`` (per-token).
        """
        # Float32 RMSNorm matches fp32 parameters and test reference (see test_rmsnorm_quant).
        normed = F.rms_norm(
            hidden_state.float(),
            [hidden_state.shape[-1]],
            weight=self.weight,
            eps=self.variance_epsilon,
        )
        normed_fp = _apply_optional_smooth_scale(normed, smooth_scale)
        scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(normed_fp / scale), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return (
            f"norm_size={self.norm_size}, variance_epsilon={self.variance_epsilon}, "
            f"quant_dtype={self.quant_dtype}, symmetric={self.symmetric}"
        )


class MojoLayerNormQuant(MojoOperator):
    """Fused LayerNorm + dynamic per-token quantization.

    Semantics::

        normed = layer_norm(hidden_state)
        scale  = amax(|normed|, dim=-1) / q_max
        output = clamp(round(normed / scale), q_min, q_max)

    Returns ``(quant_output, scale)``.
    """

    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        quant_dtype: torch.dtype = torch.int8,
        symmetric: bool = True,
        **kwargs,
    ):
        """
        Args:
            norm_size (int): Hidden dimension for LayerNorm.
            eps (float): Epsilon for LayerNorm stability.
            elementwise_affine (bool): Whether to use learnable affine params.
            quant_dtype (torch.dtype): Target quantization dtype.
            symmetric (bool): Symmetric quantization flag.
            **kwargs: Tensor factory kwargs (device, dtype).
        """
        super().__init__(**kwargs)
        self.norm_size = norm_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        self.quant_dtype = quant_dtype
        self.symmetric = symmetric

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128 if symmetric else 0
        elif quant_dtype == torch.float8_e4m3fn:
            self.q_max = torch.finfo(torch.float8_e4m3fn).max
            self.q_min = -torch.finfo(torch.float8_e4m3fn).max
        else:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, "
                f"expected torch.int8 or torch.float8_e4m3fn"
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (torch.Tensor): ``(*, D)`` input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``quant_output`` in ``quant_dtype``, same shape as input.
                - ``scale`` of shape ``(*, 1)`` (per-token).
        """
        # Float32 LN avoids dtype mismatch between activations (fp16/bf16) and fp32 parameters on
        # strict backends; quantization still uses float32 normed values.
        normed = F.layer_norm(
            hidden_state.float(),
            [hidden_state.shape[-1]],
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )
        normed_fp = _apply_optional_smooth_scale(normed, smooth_scale)
        scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(normed_fp / scale), self.q_min, self.q_max)
        return output.to(self.quant_dtype), scale

    def extra_repr(self) -> str:
        return (
            f"norm_size={self.norm_size}, variance_epsilon={self.variance_epsilon}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"quant_dtype={self.quant_dtype}, symmetric={self.symmetric}"
        )


class MojoResidualAddRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "pre",
        **kwargs,
    ):
        """
        Initialize residual-add RMSNorm operator with position control.

        Args:
            norm_size (int): Size of  1-D affine scale of length D (hidden size).
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = rms_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = rms_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__(**kwargs)
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.norm_size = norm_size
        self.variance_epsilon = float(eps)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))

        self.norm_pos = norm_pos

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            hidden_state = F.rms_norm(
                residual,
                (residual.size(-1),),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            hidden_state = F.rms_norm(
                hidden_state,
                (hidden_state.size(-1),),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        return hidden_state, residual

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.variance_epsilon=}, {self.norm_pos=}".replace("self.", "")


class MojoResidualAddLayerNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "pre",
        **kwargs,
    ):
        """
        Initialize residual-add LayerNorm operator with position control.

        Args:
            norm_size (int): Size of 1-D affine scale and shift vector.
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = layer_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = layer_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__(**kwargs)
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.norm_size = norm_size
        self.variance_epsilon = float(eps)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.norm_pos = norm_pos
        self.affine = self.weight is not None and self.bias is not None

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Residual-add LayerNorm with configurable position ("pre"/"post").

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (..., D), normalized over the last dim D.
            residual (torch.Tensor): Residual tensor to add; must be provided and shape-compatible.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized `hidden_state` and updated `residual`.
        """
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            hidden_state = F.layer_norm(
                residual,
                [residual.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            hidden_state = F.layer_norm(
                hidden_state,
                [hidden_state.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        return hidden_state, residual

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.variance_epsilon=}, {self.norm_pos=}, {self.affine=}".replace("self.", "")


class MojoChannelRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        """
        Initialize channel-wise RMS-like normalization operator.
        
        Args:
            norm_size (int): Number of channels to normalize over.
            channel_first (bool, default=True): If True, treat input as channel-first (e.g., NCHW/NCTHW).
            images (bool, default=True): Controls broadcast shape of parameters:
                - True  -> parameters shaped as (C, 1, 1) for 2D/broadcast to 3D
                - False -> parameters shaped as (C, 1, 1, 1) for explicit time dimension
            bias (bool, default=False): Whether to include learnable bias.
            **kwargs: Additional tensor factory kwargs (device, dtype, etc.).
        """
        super().__init__(**kwargs)
        self.norm_size = norm_size
        self.images = images
        self.has_bias = bias
        b_dims = (1, 1) if images else (1, 1, 1)
        shape = (norm_size, *b_dims) if channel_first else (norm_size,)
        self.scale = norm_size**0.5
        self.weight = torch.nn.Parameter(torch.ones(shape, **self.tensor_factory_kwargs))
        self.bias = torch.nn.Parameter(torch.zeros(shape, **self.tensor_factory_kwargs)) if bias else None
        self.channel_first = channel_first

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise normalization by:
          1) L2-normalization along the channel axis (or last axis if channel_first=False)
          2) Scaling by sqrt(norm_size) to match RMS normalization semantics
          3) Applying affine transform with `weight` (and optional `bias`)
        
        Args:
            hidden_state (torch.Tensor): Input must include a channel dimension. Shapes must match constructor:
                - channel_first=True,  images=True  -> (N, C, H, W)
                - channel_first=True,  images=False -> (N, C, T, H, W)
                - channel_first=False, images=True  -> (N, H, W, C)
                - channel_first=False, images=False -> (N, T, H, W, C)
                Here, N is batch size, C is channels, and T/H/W are time/height/width. Normalization is applied along the channel dimension.
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape and dtype as `hidden_state`.
        """
        dim = 1 if self.channel_first else -1
        hidden_state = torch.nn.functional.normalize(hidden_state, dim=dim) * self.scale
        hidden_state = hidden_state * self.weight
        if self.bias is not None:
            hidden_state = hidden_state + self.bias
        return hidden_state

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.channel_first=}, {self.images=}, {self.has_bias=}, {self.scale=}".replace(
            "self.", ""
        )


class MojoResidualAddRMSNormQuant(MojoOperator):
    """Fused ResidualAdd + RMSNorm + dynamic per-token quantization.

    Semantics (``norm_pos="pre"``, the common case)::

        residual = hidden_state + residual
        normed   = rms_norm(residual)
        scale    = amax(|normed|, dim=-1) / q_max
        output   = clamp(round(normed / scale), q_min, q_max)

    Returns ``(quant_output, residual, scale)``.
    """

    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        norm_pos: str = "pre",
        quant_dtype: torch.dtype = torch.int8,
        symmetric: bool = True,
        **kwargs,
    ):
        """
        Args:
            norm_size (int): Hidden dimension for RMSNorm.
            eps (float): Epsilon for RMSNorm stability.
            norm_pos (str): ``"pre"`` or ``"post"`` residual-add placement.
            quant_dtype (torch.dtype): Target quantization dtype.
            symmetric (bool): Symmetric quantization flag.
            **kwargs: Tensor factory kwargs (device, dtype).
        """
        super().__init__(**kwargs)
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.norm_size = norm_size
        self.variance_epsilon = float(eps)
        self.norm_pos = norm_pos
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.quant_dtype = quant_dtype
        self.symmetric = symmetric

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128 if symmetric else 0
        elif quant_dtype == torch.float8_e4m3fn:
            self.q_max = torch.finfo(torch.float8_e4m3fn).max
            self.q_min = -torch.finfo(torch.float8_e4m3fn).max
        else:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, "
                f"expected torch.int8 or torch.float8_e4m3fn"
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (torch.Tensor): ``(*, D)`` input.
            residual (torch.Tensor): ``(*, D)`` residual tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ``quant_output`` in ``quant_dtype``.
                - Updated ``residual``.
                - ``scale`` of shape ``(*, 1)`` (per-token).
        """
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            normed = F.rms_norm(
                residual.float(),
                (residual.shape[-1],),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            normed = F.rms_norm(
                hidden_state.float(),
                (hidden_state.shape[-1],),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        normed_fp = _apply_optional_smooth_scale(normed, smooth_scale)
        scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(normed_fp / scale), self.q_min, self.q_max)
        return output.to(self.quant_dtype), residual, scale

    def extra_repr(self) -> str:
        return (
            f"norm_size={self.norm_size}, variance_epsilon={self.variance_epsilon}, "
            f"norm_pos={self.norm_pos!r}, quant_dtype={self.quant_dtype}, "
            f"symmetric={self.symmetric}"
        )


class MojoResidualAddLayerNormQuant(MojoOperator):
    """Fused ResidualAdd + LayerNorm + dynamic per-token quantization.

    Semantics (``norm_pos="pre"``, the common case)::

        residual = hidden_state + residual
        normed   = layer_norm(residual)
        scale    = amax(|normed|, dim=-1) / q_max
        output   = clamp(round(normed / scale), q_min, q_max)

    Returns ``(quant_output, residual, scale)``.
    """

    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        norm_pos: str = "pre",
        quant_dtype: torch.dtype = torch.int8,
        symmetric: bool = True,
        **kwargs,
    ):
        """
        Args:
            norm_size (int): Hidden dimension for LayerNorm.
            eps (float): Epsilon for LayerNorm stability.
            elementwise_affine (bool): Whether to use learnable affine params.
            norm_pos (str): ``"pre"`` or ``"post"`` residual-add placement.
            quant_dtype (torch.dtype): Target quantization dtype.
            symmetric (bool): Symmetric quantization flag.
            **kwargs: Tensor factory kwargs (device, dtype).
        """
        super().__init__(**kwargs)
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.norm_size = norm_size
        self.variance_epsilon = float(eps)
        self.norm_pos = norm_pos
        self.elementwise_affine = elementwise_affine
        self.quant_dtype = quant_dtype
        self.symmetric = symmetric

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None

        if quant_dtype == torch.int8:
            self.q_max = 127
            self.q_min = -128 if symmetric else 0
        elif quant_dtype == torch.float8_e4m3fn:
            self.q_max = torch.finfo(torch.float8_e4m3fn).max
            self.q_min = -torch.finfo(torch.float8_e4m3fn).max
        else:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, "
                f"expected torch.int8 or torch.float8_e4m3fn"
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (torch.Tensor): ``(*, D)`` input.
            residual (torch.Tensor): ``(*, D)`` residual tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ``quant_output`` in ``quant_dtype``.
                - Updated ``residual``.
                - ``scale`` of shape ``(*, 1)`` (per-token).
        """
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            normed = F.layer_norm(
                residual.float(),
                [residual.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            normed = F.layer_norm(
                hidden_state.float(),
                [hidden_state.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        normed_fp = _apply_optional_smooth_scale(normed, smooth_scale)
        scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / self.q_max
        output = torch.clamp(torch.round(normed_fp / scale), self.q_min, self.q_max)
        return output.to(self.quant_dtype), residual, scale

    def extra_repr(self) -> str:
        return (
            f"norm_size={self.norm_size}, variance_epsilon={self.variance_epsilon}, "
            f"elementwise_affine={self.elementwise_affine}, norm_pos={self.norm_pos!r}, "
            f"quant_dtype={self.quant_dtype}, symmetric={self.symmetric}"
        )


class MojoResidualAddNormCast(MojoOperator):
    pass
