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
from mojo_opset.core import MojoRejectSampling
from mojo_opset.core import MojoMagicRejectSampling


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
    def forward_std(
        self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = logits.dtype
        logits = logits.to(torch.float32)
        top_k = min(rand_top_k, logits.size(-1))
        sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)

        cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, self.filter_value)

        final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1).to(dtype)

        return final_probs_dist, sorted_topk_indices

class RefRejectSampling(MojoRejectSampling, default_priority=LAST_PRIORITY):
    def forward_std(
        self, 
        target_logits: torch.Tensor, 
        draft_tokens: torch.Tensor, 
        draft_probs: torch.Tensor, 
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = target_logits.dtype
        bs = target_logits.shape[0]
        target_logits = target_logits.to(torch.float32)
        top_k = min(rand_top_k, target_logits.size(-1))

        # topk topp filter
        target_topk_logits, target_topk_indices = torch.topk(target_logits, top_k)

        cumulative_probs = target_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = target_topk_logits.masked_fill(sorted_indices_to_remove, filter_value)

        target_probs = torch.nn.functional.softmax(filtered_logits, dim=-1).to(dtype)

        target_probs = target_probs.view(bs, -1, target_probs.shape[-1])
        target_topk_indices = target_topk_indices.view(bs, -1, target_topk_indices.shape[-1])
        filter_target_probs = torch.zeros_like(target_logits)
        filter_target_probs.scatter_(-1, target_topk_indices, target_probs)

        # reject sampling
        target_token_probs = torch.gather(filter_target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        rand_vals = torch.rand(bs, 1, device=target_logits.device)
        reject_matrix = (target_token_probs / draft_probs) < rand_vals
        reject_matrix = torch.cat(
            [reject_matrix.int(), torch.ones((bs, 1), device=target_logits.device)], dim=1
        )
        accepted_len = torch.argmax(reject_matrix, dim=1)

        # gumbel sampling
        last_token_probs = filter_target_probs[range(bs), accepted_len]
        last_token = torch.multinomial(last_token_probs, num_samples=1)

        # generate total next token
        next_tokens = torch.cat(
            [draft_tokens, torch.zeros((bs, 1), dtype=torch.long, device=target_logits.device)], dim=-1
        )
        next_tokens.scatter_(-1, accepted_len.unsqueeze(1), last_token)

        return next_tokens, accepted_len.int()


class RefRejectMagicRejectSampling(MojoMagicRejectSampling, default_priority=LAST_PRIORITY):
    def forward_std(self, 
        target_logits: torch.Tensor, 
        draft_tokens: torch.Tensor, 
        draft_probs: torch.Tensor, 
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    )-> Tuple[torch.Tensor, torch.Tensor]:
        dtype = target_logits.dtype
        bs = target_logits.shape[0]
        target_logits = target_logits.to(torch.float32)
        top_k = min(rand_top_k, target_logits.size(-1))

        # topk topp filter
        target_topk_logits, target_topk_indices = torch.topk(target_logits, top_k)

        cumulative_probs = target_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = target_topk_logits.masked_fill(sorted_indices_to_remove, filter_value)

        target_probs = torch.nn.functional.softmax(filtered_logits, dim=-1).to(dtype)

        target_probs = target_probs.view(bs, -1, target_probs.shape[-1])
        target_topk_indices = target_topk_indices.view(bs, -1, target_topk_indices.shape[-1])
        filter_target_probs = torch.zeros_like(target_logits)
        filter_target_probs.scatter_(-1, target_topk_indices, target_probs)

        # reject sampling
        target_token_probs = torch.gather(filter_target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        ratios = torch.minimum(torch.ones_like(target_token_probs), target_token_probs / draft_probs)
        pi = torch.cumprod(ratios, dim=1)
        ratios = torch.rand_like(pi)
        _rand = torch.cumprod(ratios, dim=1)
        reject_matrix = pi < _rand
        reject_matrix = torch.cat(
            [torch.zeros((bs, 1), device=target_logits.device), reject_matrix.int()], dim=1
        )
        accepted_len = spec_step - reject_matrix.flip(dims=[1]).argmin(dim=1).int()

        # gumbel sampling
        last_token_probs = filter_target_probs[range(bs), accepted_len]
        last_token = torch.multinomial(last_token_probs, num_samples=1)

        # generate total next token
        next_tokens = torch.cat(
            [draft_tokens, torch.zeros((bs, 1), dtype=torch.long, device=target_logits.device)], dim=-1
        )
        next_tokens.scatter_(-1, accepted_len.unsqueeze(1), last_token)

        return next_tokens, accepted_len.int()


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
