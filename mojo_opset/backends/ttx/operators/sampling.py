from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from mojo_opset.backends.ttx.kernels import fused_penalties_temp
from mojo_opset.backends.ttx.kernels import join_prob_reject_sampling
from mojo_opset.backends.ttx.kernels import reject_sampling
from mojo_opset.backends.ttx.kernels import top_p_filter
from mojo_opset.backends.ttx.kernels import top_p_sampling
from mojo_opset.core import MojoApplyPenaltiesTempurate
from mojo_opset.core import MojoJoinProbRejectSampling
from mojo_opset.core import MojoRejectSampling
from mojo_opset.core import MojoTopPFilter
from mojo_opset.core import MojoTopPSampling


class TTXTopPSampling(MojoTopPSampling):
    supported_platforms_list = ["npu"]

    def forward(self, logits: torch.Tensor) -> Tuple[Any]:
        return top_p_sampling(
            logits=logits,
            top_p=self.top_p,
            filter_value=self.filter_value,
            min_tokens_to_keep=self.min_tokens_to_keep,
            rand_top_k=self.rand_top_k,
        )


class TTXTopPFilter(MojoTopPFilter):
    supported_platforms_list = ["npu"]

    def forward(self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int) -> Tuple[Any]:
        return top_p_filter(
            logits=logits,
            top_p=top_p,
            filter_value=self.filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
            rand_top_k=rand_top_k,
        )


class TTXRejectSampling(MojoRejectSampling):
    def forward(
        self,
        target_logits: torch.Tensor,  # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,  # [batch, spec_step]
        random_seed: int = None,
    ):
        return reject_sampling(
            target_logits,
            draft_tokens,
            draft_probs,
            random_seed,
        )


class TTXJoinProbRejectSampling(MojoJoinProbRejectSampling):
    def forward(
        self,
        target_logits: torch.Tensor,  # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,  # [batch, spec_step]
        random_seed: int = None,
    ):
        return join_prob_reject_sampling(
            target_logits,
            draft_tokens,
            draft_probs,
            random_seed,
        )


class TTXApplyPenaltiesTempurate(MojoApplyPenaltiesTempurate):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        if len(temps) == 0:
            temps = None
        return fused_penalties_temp(
            logits, token_freqs, frequency_penalties, presence_penalties, repetition_penalties, temps
        )
