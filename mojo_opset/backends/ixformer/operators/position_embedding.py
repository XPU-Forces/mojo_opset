from typing import Tuple

import torch

from ixformer import functions as ixf_f

from mojo_opset.core import MojoApplyRoPE


class IxformerApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ixf_f.apply_rope(q, k, cos, sin, head_first)
