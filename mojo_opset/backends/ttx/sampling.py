from typing import Any
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels.ascend.sampling import ttx_top_p_sampling
from mojo_opset.core import MojoTopPSampling


class TTXTopPSampling(MojoTopPSampling, default_priority=0):
    def forward_std(self, logits: torch.Tensor) -> Tuple[Any]:
        return ttx_top_p_sampling(
            logits=logits,
            top_p=self.top_p,
            filter_value=self.filter_value,
            min_tokens_to_keep=self.min_tokens_to_keep,
            rand_top_k=self.rand_top_k,
        )
