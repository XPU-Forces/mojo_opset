from typing import List
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels.npu.mrope import mrope_fwd_impl
from mojo_opset.core.operators.mrope import MojoMRoPE


class TTXMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return mrope_fwd_impl(q, k, cos, sin, mrope_section, is_interleaved)
