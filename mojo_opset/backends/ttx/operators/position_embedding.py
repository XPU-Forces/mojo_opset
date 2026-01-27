from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRoPE


class TTXRoPE(MojoRoPE):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert cu_seqlens is None, "cu_seqlens is not supported yet."
        return rope_fwd(q, k, cos, sin)
