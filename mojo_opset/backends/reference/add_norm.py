from typing import Any
from typing import Tuple

import torch
import torch.nn.functional as F

from mojo_opset.core import LAST_PRIORITY
from mojo_opset.core import MojoResidualAddNorm


class RefResidualAddNorm(MojoResidualAddNorm, default_priority=LAST_PRIORITY):
    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        def norm_func(hidden_state: torch.Tensor) -> Tuple[Any]:
            if self.norm_type == "layernorm":
                return F.layer_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.gamma,
                    bias=self.beta,
                    eps=self.epsilon,
                )
            elif self.norm_type == "rmsnorm":
                return F.rms_norm(hidden_state, (hidden_state.size(-1),), weight=self.gamma, eps=self.epsilon)

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
