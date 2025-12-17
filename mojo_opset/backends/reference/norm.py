import torch

from mojo_opset.core import LAST_PRIORITY
from mojo_opset.core import MojoNorm


class RefNorm(MojoNorm, default_priority=LAST_PRIORITY):
    def forward_std(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = hidden_state
        eps = float(self.epsilon)
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
            if self.gamma is not None:
                y = y * self.gamma
            if self.beta is not None:
                y = y + self.beta
        elif self.norm_type == "rmsnorm":
            rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
            y = x / rms.to(x.dtype)
            if self.gamma is not None:
                y = y * self.gamma
        else:
            raise ValueError("norm_type should be 'layernorm' or 'rmsnorm'")
        return y
