"""Forward-only sampling. Token content after accepted_len is unspecified."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def top_k_sampling(
    logits: torch.Tensor,
    top_k: int = 50,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("top_k_sampling", logits)
    forward, _ = load_impl("top_k_sampling", implementation)
    return forward(logits, top_k, filter_value, min_tokens_to_keep)


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float = 0.75,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("top_p_sampling", logits)
    forward, _ = load_impl("top_p_sampling", implementation)
    return forward(logits, top_p, filter_value, min_tokens_to_keep, rand_top_k)


def top_p_filter(
    logits: torch.Tensor,
    top_p: float = 0.75,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("top_p_filter", logits)
    forward, _ = load_impl("top_p_filter", implementation)
    return forward(logits, top_p, filter_value, min_tokens_to_keep, rand_top_k)


def reject_sampling(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    random_seed: Optional[int] = None,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("reject_sampling", target_probs, draft_tokens, draft_probs)
    forward, _ = load_impl("reject_sampling", implementation)
    return forward(target_probs, draft_tokens, draft_probs, random_seed)


def join_prob_reject_sampling(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    random_seed: Optional[int] = None,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("join_prob_reject_sampling", target_probs, draft_tokens, draft_probs)
    forward, _ = load_impl("join_prob_reject_sampling", implementation)
    return forward(target_probs, draft_tokens, draft_probs, random_seed)


def apply_penalties_temperature(
    logits: torch.Tensor,
    token_freqs: list[Optional[torch.Tensor]],
    presence_penalties: list[float],
    frequency_penalties: list[float],
    repetition_penalties: list[float],
    temps: Optional[list[Optional[float]]] = None,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("apply_penalties_temperature", logits)
    forward, _ = load_impl("apply_penalties_temperature", implementation)
    return forward(logits, token_freqs, presence_penalties, frequency_penalties, repetition_penalties, temps)
