from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from mojo_opset.backends.ttx.kernels.ascend.sampling import ttx_apply_penalties
from mojo_opset.backends.ttx.kernels.ascend.sampling import ttx_top_p_sampling
from mojo_opset.core import MojoApplyPenalties
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


class TTXApplyPenalties(MojoApplyPenalties, default_priority=0):
    def forward_std(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        return ttx_apply_penalties(
            logits, token_freqs, frequency_penalties, presence_penalties, repetition_penalties, temps
        )
