from typing import Any
from typing import Tuple

import torch
import torch.nn.functional as F

from mojo_opset.core import MojoNorm
from mojo_opset.core import MojoResidualAddNorm


class RefNorm(MojoNorm):
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
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


class RefResidualAddNorm(MojoResidualAddNorm):
    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
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
