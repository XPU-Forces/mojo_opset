from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from ..operator import MojoOperator


class MojoTopKSampling(MojoOperator):
    pass


class MojoTopPSampling(MojoOperator):
    def __init__(
        self,
        top_p: float = 0.75,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        rand_top_k: int = 1000,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize nucleus (top-p) sampling configuration.

        Args:
            top_p (float, default=0.75): Cumulative probability threshold for filtering.
            filter_value (float, default=-inf): Logit value used to mask filtered tokens.
            min_tokens_to_keep (int, default=1): Minimum tokens retained regardless of `top_p`.
            rand_top_k (int, default=1000): Randomized upper cap for top-k fallback.
            op_name (str, default=""): Operator name metadata.
            layer_idx (int, default=0): Layer index metadata.

        Notes:
            Stores configuration only; actual sampling logic is implemented elsewhere.
        """
        super().__init__(op_name, layer_idx)

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.rand_top_k = rand_top_k

    def forward(self, logits: torch.Tensor) -> Tuple[Any]:
        """
        Perform nucleus (top-p) sampling over the last dimension of logits.

        Args:
            logits (torch.Tensor): Logits of shape (..., V) where the last dimension
                `V` is the vocabulary size. Cast to float32 for numerical stability.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(next_probs, next_tokens)` where each has
            shape (..., 1). `next_tokens` contains sampled token indices; `next_probs`
            contains the corresponding probabilities.

        Notes:
            - Caps top-k to `min(rand_top_k, V)` before applying nucleus filtering.
            - Ensures at least `min_tokens_to_keep` tokens remain unmasked.
            - Masks filtered logits with `filter_value` (typically `-inf`) before softmax.
            - Sampling uses `torch.multinomial` over the filtered probability distribution.
        """
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


class MojoTopPFilter(MojoOperator):
    def __init__(
        self,
        filter_value: float = -float("Inf"),
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize filtering configuration for sampling operators.

        Args:
            filter_value (float, default=-inf): Logit value used to mask filtered tokens.
            op_name (str, default=""): Operator name metadata.
            layer_idx (int, default=0): Layer index metadata.

        Notes:
            Stores configuration only; actual sampling/masking is handled in `forward`.
        """
        super().__init__(op_name, layer_idx)

        self.filter_value = filter_value

    def forward(
        self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute nucleus (top-p) sampling candidates from logits.

        Args:
            logits (torch.Tensor): Logits of shape (..., V), where V is the vocabulary size.
            top_p (float): Cumulative probability threshold for nucleus filtering.
            min_tokens_to_keep (int): Minimum tokens retained regardless of `top_p`.
            rand_top_k (int): Upper cap for the top-k candidates considered before filtering.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(final_probs_dist, sorted_topk_indices)` where
            both tensors have shape (..., K). `final_probs_dist` contains probabilities over
            the retained top-k tokens; `sorted_topk_indices` are the corresponding token indices.

        Notes:
            - Casts logits to float32 for numerical stability, then restores original dtype.
            - Caps K to `min(rand_top_k, V)` prior to nucleus filtering.
            - Ensures at least `min_tokens_to_keep` tokens remain by clearing masks on lowest positions.
            - Masks filtered logits with `self.filter_value` before softmax.
        """
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


class MojoRejectSampling(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(
        self,
        target_probs: torch.Tensor,  # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,  # [batch, spec_step]
        random_seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Speculative sampling acceptance step.

        Args:
            target_probs (torch.Tensor): Target model probabilities of shape (B, S+1, V).
            draft_tokens (torch.Tensor): Draft token ids of shape (B, S).
            draft_probs (torch.Tensor): Draft model probabilities of drafted tokens (B, S).
            random_seed (int, optional): Seed for reproducible randomness; defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_tokens (B, S+1): Draft tokens with an extra sentinel slot for fallback.
                - accepted_len (B,): Number of consecutive draft tokens accepted per batch.

        Notes:
            - Accepts draft token i if (target_prob_i / draft_prob_i) >= u, with u ~ U(0,1).
            - Appends a sentinel "always accept" at the end to trigger fallback when all are rejected.
            - The final fallback token content is not sampled here; caller should fill it externally.
        """
        device = target_probs.device
        batch_size, _, _ = target_probs.shape
        spec_step = draft_probs.shape[1]

        if random_seed is not None:
            torch.manual_seed(random_seed)

        rand_vals = torch.rand(batch_size, 1, device=device)
        target_probs = torch.gather(target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        reject_matrix = (target_probs / draft_probs) < rand_vals
        reject_matrix = torch.cat([reject_matrix.int(), torch.ones((batch_size, 1), device=device)], dim=1)
        accepted_len = torch.argmax(reject_matrix, dim=1)

        next_tokens = torch.empty((batch_size, spec_step + 1), device=device, dtype=torch.int32)
        next_tokens = torch.cat([draft_tokens, torch.zeros((batch_size, 1), dtype=torch.long, device=device)], dim=-1)

        return next_tokens, accepted_len


class MojoJoinProbRejectSampling(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        random_seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Speculative sampling acceptance via cumulative ratios.

        Args:
            target_probs (torch.Tensor): Target model probabilities of shape (B, S+1, V).
            draft_tokens (torch.Tensor): Draft token indices of shape (B, S).
            draft_probs (torch.Tensor): Draft model probabilities for drafted tokens, shape (B, S).
            random_seed (int, optional): Seed for reproducible randomness; defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_tokens (B, S+1): Draft tokens with an extra placeholder for fallback.
                - accepted_len (B,): Number of consecutive draft tokens accepted per batch.

        Notes:
            - Computes per-step ratios r_i = min(1, target_prob_i / draft_prob_i) and
              accepts prefix length by comparing cumprod(r_i) against cumprod of U(0,1).
            - Appends a sentinel token slot; caller should fill the final token if fallback is needed.
        """
        batch_size, _, _ = target_probs.shape
        spec_step = draft_probs.shape[1]

        # reject sampling
        target_token_probs = torch.gather(target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        ratios = torch.minimum(torch.ones_like(target_token_probs), target_token_probs / draft_probs)
        pi = torch.cumprod(ratios, dim=1)
        if random_seed is not None:
            torch.manual_seed(random_seed)

        ratios = torch.rand(batch_size, spec_step, device=target_probs.device)
        _rand = torch.cumprod(ratios, dim=1)

        reject_matrix = pi < _rand
        reject_matrix = torch.cat(
            [torch.zeros((batch_size, 1), device=target_probs.device), reject_matrix.int()], dim=1
        )
        accepted_len = spec_step - reject_matrix.flip(dims=[1]).argmin(dim=1).int()

        # generate total next token
        next_tokens = torch.cat(
            [draft_tokens, torch.zeros((batch_size, 1), dtype=torch.long, device=draft_probs.device)], dim=-1
        )
        return next_tokens, accepted_len.int()


class MojoApplyPenaltiesTempurate(MojoOperator):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        """
        Apply presence, frequency, repetition penalties and optional temperature per batch.

        Args:
            logits (torch.Tensor): Logits of shape (B, V); cast to float32 internally and
                restored to original dtype on return.
            token_freqs (List[Union[None, torch.Tensor]]): Per-batch frequency vectors; each
                entry is None or a 1-D tensor of shape (V,).
            presence_penalties (List[float]): Subtracts penalty when token frequency > 0.
            frequency_penalties (List[float]): Subtracts `penalty * frequency` from logits.
            repetition_penalties (List[float]): Scales logits element-wise based on sign of
                `logit * frequency`: multiply by penalty if < 0, divide if > 0, unchanged if == 0.
            temps (Optional[List[Optional[float]]], default=None): Per-batch temperatures;
                divides logits[i] by temps[i] when provided.

        Returns:
            torch.Tensor: Logits after penalties and temperature, cast back to original dtype.

        Notes:
            - All penalty lists must have length B (batch size).
            - Frequency tensors are moved to `logits.device` (non-blocking where supported).
        """
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
