import os
import torch

from ..mojo_operator import MojoOperator


class MojoDecodeNSA(MojoOperator):
    """
    MojoDecodeNSA operator.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedDecodeNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Decode.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
