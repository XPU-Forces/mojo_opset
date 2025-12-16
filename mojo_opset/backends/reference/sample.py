from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from mojo_opset.core import LAST_PRIORITY
from mojo_opset.core import MojoApplyPenaltiesTempurate
from mojo_opset.core import MojoTopPFilter
from mojo_opset.core import MojoTopPSampling


class RefTopPSampling(MojoTopPSampling, default_priority=LAST_PRIORITY):
    def forward_std(self, logits: torch.Tensor) -> Tuple[Any]:
        logits = logits.to(torch.float32)
        top_k = min(self.rand_top_k, logits.size(-1))
        sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)

        cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, self.filter_value)

        final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1)

        select_index = torch.multinomial(final_probs_dist, num_samples=1)

        next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
        next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)

        return next_probs, next_tokens


class RefTopPFilter(MojoTopPFilter, default_priority=LAST_PRIORITY):
    def forward_std(self, logits: torch.Tensor) -> Tuple[Any]:
        logits = logits.to(torch.float32)
        top_k = min(self.rand_top_k, logits.size(-1))
        sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)

        cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, self.filter_value)

        final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1)

        return final_probs_dist, sorted_topk_indices


class RefApplyPenaltiesTempurate(MojoApplyPenaltiesTempurate, default_priority=LAST_PRIORITY):
    def forward_std(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        dtype = logits.dtype
        logits = logits.to(torch.float32)

        for i, freq_token in enumerate(token_freqs):
            if freq_token is not None:
                device_freq_token = freq_token.to(logits.device, non_blocking=True)
                if frequency_penalties[i] != 0.0:
                    logits[i] -= frequency_penalties[i] * device_freq_token
                if presence_penalties[i] != 0.0:
                    logits[i] -= presence_penalties[i] * (device_freq_token > 0)
                if repetition_penalties[i] != 1.0:
                    conds = logits[i] * device_freq_token
                    logits[i] = torch.where(
                        conds < 0,
                        logits[i] * repetition_penalties[i],
                        torch.where(conds > 0, logits[i] / repetition_penalties[i], logits[i]),
                    )
            if temps is not None and temps[i] is not None:
                logits[i] /= temps[i]
        return logits.to(dtype)
