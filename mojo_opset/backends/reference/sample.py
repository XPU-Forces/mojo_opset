from typing import Any
from typing import Tuple

import torch

from mojo_opset.core import MojoTopPFilter

class RefTopPFilter(MojoTopPFilter, default_priority=999, backend="reference"):
    def __init__(self, top_p = 0.75, filter_value = ..., min_tokens_to_keep = 1, rand_top_k = 1000, op_name = "", layer_idx = 0):
        super().__init__(top_p, filter_value, min_tokens_to_keep, rand_top_k, op_name, layer_idx)

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
