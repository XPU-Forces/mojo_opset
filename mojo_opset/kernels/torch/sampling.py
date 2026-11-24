"""Original sampling reference formulas; randomness follows each public contract."""

from typing import Optional

import torch


def top_k_sampling_fwd(
    logits: torch.Tensor, top_k: int = 50, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1
):
    """Perform top-k sampling over the last dimension of logits."""
    logits = logits.to(torch.float32)
    top_k = min(top_k, logits.size(-1))
    top_k = max(top_k, min_tokens_to_keep)
    sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)
    final_probs_dist = torch.nn.functional.softmax(sorted_topk_logits, dim=-1)
    select_index = torch.multinomial(final_probs_dist, num_samples=1)
    next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
    next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)
    return (next_probs, next_tokens)


def top_p_sampling_fwd(
    logits: torch.Tensor,
    top_p: float = 0.75,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
):
    """Perform nucleus (top-p) sampling over the last dimension of logits."""
    logits = logits.to(torch.float32)
    top_k = min(rand_top_k, logits.size(-1))
    sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)
    cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, filter_value)
    final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1)
    select_index = torch.multinomial(final_probs_dist, num_samples=1)
    next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
    next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)
    return (next_probs, next_tokens)


def top_p_filter_fwd(
    logits: torch.Tensor,
    top_p: float = 0.75,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
):
    """Compute nucleus (top-p) sampling candidates from logits."""
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
    filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, filter_value)
    final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1).to(dtype)
    return (final_probs_dist, sorted_topk_indices)


def reject_sampling_fwd(
    target_probs: torch.Tensor, draft_tokens: torch.Tensor, draft_probs: torch.Tensor, random_seed: Optional[int] = None
):
    """Speculative sampling acceptance step."""
    device = target_probs.device
    batch_size, _, _ = target_probs.shape
    spec_step = draft_probs.shape[1]
    if random_seed is not None:
        torch.manual_seed(random_seed)
    rand_vals = torch.rand(batch_size, 1, device=device)
    target_probs = torch.gather(target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    reject_matrix = target_probs / draft_probs < rand_vals
    reject_matrix = torch.cat([reject_matrix.int(), torch.ones((batch_size, 1), device=device)], dim=1)
    accepted_len = torch.argmax(reject_matrix, dim=1)
    next_tokens = torch.empty((batch_size, spec_step + 1), device=device, dtype=torch.int32)
    next_tokens = torch.cat([draft_tokens, torch.zeros((batch_size, 1), dtype=torch.long, device=device)], dim=-1)
    return (next_tokens, accepted_len)


def join_prob_reject_sampling_fwd(
    target_probs: torch.Tensor, draft_tokens: torch.Tensor, draft_probs: torch.Tensor, random_seed: Optional[int] = None
):
    """Speculative sampling acceptance via cumulative ratios."""
    batch_size, _, _ = target_probs.shape
    spec_step = draft_probs.shape[1]
    target_token_probs = torch.gather(target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    ratios = torch.clamp(target_token_probs / draft_probs, 0, 1)
    pi = torch.cumprod(ratios, dim=1)
    if random_seed is not None:
        torch.manual_seed(random_seed)
    ratios = torch.rand(batch_size, spec_step, device=target_probs.device)
    _rand = torch.cumprod(ratios, dim=1)
    reject_matrix = pi < _rand
    reject_matrix = torch.cat([torch.zeros((batch_size, 1), device=target_probs.device), reject_matrix.int()], dim=1)
    accepted_len = spec_step - reject_matrix.flip(dims=[1]).argmin(dim=1).int()
    next_tokens = torch.cat(
        [draft_tokens, torch.zeros((batch_size, 1), dtype=torch.long, device=draft_probs.device)], dim=-1
    )
    return (next_tokens, accepted_len.int())


def apply_penalties_temperature_fwd(
    logits: torch.Tensor,
    token_freqs: list[Optional[torch.Tensor]],
    presence_penalties: list[float],
    frequency_penalties: list[float],
    repetition_penalties: list[float],
    temps: Optional[list[Optional[float]]] = None,
):
    """Apply presence, frequency, repetition penalties and optional temperature per batch."""
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
