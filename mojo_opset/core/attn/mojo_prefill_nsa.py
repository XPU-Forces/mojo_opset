import os
import torch
from torch import nn

from ..mojo_operator import MojoOperator


class MojoPrefillNSA(MojoOperator):
    """
    MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedPrefillNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
