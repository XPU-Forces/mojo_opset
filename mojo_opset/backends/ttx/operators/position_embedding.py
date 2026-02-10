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
        kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = True,
        rope_percentage: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_fwd(q, k, cos, sin, cu_seqlens, kv_lens, head_first, rope_percentage)
