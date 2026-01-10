from typing import Any
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoNorm(MojoOperator):
    def __init__(
        self,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        weight: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize normalization operator configuration.

        Args:
            eps (float): Small constant for numerical stability (>0).
            norm_type (str): Normalization type, one of {"rmsnorm", "layernorm"}.
            weight (Optional[torch.Tensor]): Optional 1-D affine weight (float dtype).
            beta (Optional[torch.Tensor]): Optional 1-D offset; valid only for LayerNorm.
            is_varlen (bool): If True, prefer token-first TND layout; else use BSND.
            op_name (str): Operator name metadata.
            layer_idx (int): Layer index metadata.

        Raises:
            ValueError: If `norm_type` is not supported or `beta` is provided with RMSNorm.

        Notes:
            Sets `self.affine` True only when both `weight` and `beta` are provided.
        """
        super().__init__(op_name, layer_idx)

        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type should be {"rmsnorm","layernorm"}')

        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm don't support beta.")

        self.eps = float(eps)
        self.weight = weight
        self.beta = beta
        self.norm_type = norm_type
        self.affine = weight is not None and beta is not None
        self.is_varlen = is_varlen

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LayerNorm or RMSNorm.

        Validates input layout depending on `is_varlen`:
        - is_varlen=True expects TND (2D or 3D) inputs.
        - is_varlen=False expects BSND (rank >= 3) inputs.

        Behavior:
        - layernorm: mean/variance over the last dimension, normalized with epsilon for stability,
          followed by optional affine `weight` (scale) and `beta` (shift).
        - rmsnorm: normalize by RMS over the last dimension (no mean subtraction); computation
          is performed in float32 for numerical stability, then optional affine `weight` is applied.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (..., D).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.

        Raises:
            ValueError: If `norm_type` is invalid or input shape/layout does not match `is_varlen`.
        """
        x = hidden_state
        eps = float(self.eps)
        if self.is_varlen:
            if x.ndim not in (2, 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
        else:
            if x.ndim < 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
        if self.norm_type == "layernorm":
            mu = x.mean(dim=-1, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + eps)
            if self.weight is not None:
                y = y * self.weight
            if self.beta is not None:
                y = y + self.beta
        elif self.norm_type == "rmsnorm":
            x_dtype = x.dtype
            x = x.to(torch.float32)
            y = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps)
            if self.weight is not None:
                y = y * self.weight
                y = y.to(x_dtype)
        else:
            raise ValueError("norm_type should be 'layernorm' or 'rmsnorm'")
        return y


class MojoNormQuant(MojoOperator):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.variance_epsilon = eps

        self.norm_type = norm_type
        assert self.norm_type in ["rmsnorm", "layernorm"]


class MojoResidualAddNorm(MojoOperator):
    def __init__(
        self,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        weight: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        norm_pos: str = "pre",
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize normalization operator (LayerNorm/RMSNorm) with position control.

        Args:
            eps (float, default=1e-5): Epsilon for numerical stability (>0).
            norm_type (str, default="rmsnorm"): One of {"rmsnorm", "layernorm"}.
            weight (Optional[torch.Tensor], default=None): Optional 1-D affine scale.
            beta (Optional[torch.Tensor], default=None): Optional 1-D affine shift; allowed only for LayerNorm.
            norm_pos (str, default="pre"): Apply norm "pre" (before sublayer) or "post" (after sublayer).
            is_varlen (bool, default=True): Prefer TND when True; else use BSND.
            op_name (str, default=""): Operator name metadata.
            layer_idx (int, default=0): Layer index metadata.

        Raises:
            ValueError: If `norm_type` unsupported, `beta` provided with RMSNorm, or `norm_pos` not in {"pre","post"}.

        Notes:
            `self.affine` is True only when both `weight` and `beta` are provided.
        """
        super().__init__(op_name, layer_idx)

        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type should be {"rmsnorm","layernorm"}')

        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm don't support beta.")

        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.eps = float(eps)
        self.weight = weight
        self.beta = beta
        self.norm_type = norm_type
        self.norm_pos = norm_pos
        self.affine = weight is not None and beta is not None
        self.is_varlen = is_varlen

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Fused normalization with configurable position ("pre"/"post") and residual handling.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (..., D).
            residual (torch.Tensor, optional): Residual to combine; defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized `hidden_state` and updated `residual`.

        Behavior:
            - layernorm: uses F.layer_norm with `eps`, optional `weight` (scale) and `beta` (shift).
            - rmsnorm: uses F.rms_norm with `eps` and optional `weight`.
            - norm_pos="pre": residual = hidden_state + residual (or hidden_state if None);
              hidden_state = norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual (if provided);
              hidden_state = norm(hidden_state); residual = hidden_state.

        Note:
            The method returns a tuple `(hidden_state, residual)` although the type annotation
            indicates `torch.Tensor`; callers should unpack both values.
        """

        def norm_func(hidden_state: torch.Tensor) -> Tuple[Any]:
            if self.norm_type == "layernorm":
                return F.layer_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.weight,
                    bias=self.beta,
                    eps=self.eps,
                )
            elif self.norm_type == "rmsnorm":
                return F.rms_norm(hidden_state, (hidden_state.size(-1),), weight=self.weight, eps=self.eps)

        if self.norm_pos == "pre":
            if residual is not None:
                residual = hidden_state + residual
            else:
                residual = hidden_state
            hidden_state = norm_func(residual)
        else:
            if residual is not None:
                hidden_state = hidden_state + residual
            hidden_state = norm_func(hidden_state)
            residual = hidden_state

        return hidden_state, residual


class MojoResidualAddNormQuant(MojoOperator):
    pass


class MojoResidualAddNormCast(MojoOperator):
    pass
