"""Stateful configuration for forward-only sampling functions."""

from typing import Optional

import torch

from mojo_opset import functions


class TopKSampling(torch.nn.Module):
    def __init__(self, top_k=50, filter_value=-float("inf"), min_tokens_to_keep=1, *, implementation=None):
        super().__init__()
        self.top_k, self.filter_value = top_k, filter_value
        self.min_tokens_to_keep, self.implementation = min_tokens_to_keep, implementation

    def forward(self, logits):
        return functions.top_k_sampling(
            logits, self.top_k, self.filter_value, self.min_tokens_to_keep, implementation=self.implementation
        )


class TopPSampling(torch.nn.Module):
    def __init__(
        self, top_p=0.75, filter_value=-float("inf"), min_tokens_to_keep=1, rand_top_k=1000, *, implementation=None
    ):
        super().__init__()
        self.top_p, self.filter_value = top_p, filter_value
        self.min_tokens_to_keep, self.rand_top_k = min_tokens_to_keep, rand_top_k
        self.implementation = implementation

    def forward(self, logits):
        return functions.top_p_sampling(
            logits,
            self.top_p,
            self.filter_value,
            self.min_tokens_to_keep,
            self.rand_top_k,
            implementation=self.implementation,
        )


class TopPFilter(torch.nn.Module):
    def __init__(self, filter_value=-float("inf"), *, implementation=None):
        super().__init__()
        self.filter_value, self.implementation = filter_value, implementation

    def forward(self, logits, top_p=0.75, min_tokens_to_keep=1, rand_top_k=1000):
        return functions.top_p_filter(
            logits, top_p, self.filter_value, min_tokens_to_keep, rand_top_k, implementation=self.implementation
        )


class RejectSampling(torch.nn.Module):
    def __init__(self, *, implementation: Optional[str] = None):
        super().__init__()
        self.implementation = implementation

    def forward(self, target_probs, draft_tokens, draft_probs, random_seed=None):
        return functions.reject_sampling(
            target_probs, draft_tokens, draft_probs, random_seed, implementation=self.implementation
        )


class JoinProbRejectSampling(torch.nn.Module):
    def __init__(self, *, implementation: Optional[str] = None):
        super().__init__()
        self.implementation = implementation

    def forward(self, target_probs, draft_tokens, draft_probs, random_seed=None):
        return functions.join_prob_reject_sampling(
            target_probs, draft_tokens, draft_probs, random_seed, implementation=self.implementation
        )


class ApplyPenaltiesTemperature(torch.nn.Module):
    def __init__(self, *, implementation: Optional[str] = None):
        super().__init__()
        self.implementation = implementation

    def forward(self, logits, token_freqs, presence_penalties, frequency_penalties, repetition_penalties, temps=None):
        return functions.apply_penalties_temperature(
            logits,
            token_freqs,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            temps,
            implementation=self.implementation,
        )
