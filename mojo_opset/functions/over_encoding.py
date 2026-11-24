"""Over-encoding index generation and NF4 embedding lookup."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def n_gram_prefill(
    input_ids: torch.Tensor,
    q_lens: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    vocab_size: int,
    *,
    implementation: Optional[str] = None,
):
    forward, _ = load_impl("n_gram_prefill", implementation)
    return forward(input_ids, q_lens, oe_history_inputs, oe_vocab_sizes, oe_vocab_offsets, n_grams, vocab_size)


def n_gram_decode(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    vocab_size: int,
    *,
    implementation: Optional[str] = None,
):
    forward, _ = load_impl("n_gram_decode", implementation)
    return forward(input_ids, oe_history_inputs, oe_vocab_sizes, oe_vocab_offsets, n_grams, vocab_size)


def embedding_nf4_dequant(
    input_ids: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    *,
    group_size: int = 1,
    codebook: Optional[torch.Tensor] = None,
    vocab_start_id: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    implementation: Optional[str] = None,
):
    _require_inference("embedding_nf4_dequant", qweight, scale, mean, codebook)
    forward, _ = load_impl("embedding_nf4_dequant", implementation)
    return forward(input_ids, qweight, scale, mean, group_size, codebook, vocab_start_id, output_dtype)


def over_encoding_decode(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    *,
    ori_vocab_size: int,
    group_size: int = 1,
    codebook: Optional[torch.Tensor] = None,
    vocab_start_id: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    implementation: Optional[str] = None,
):
    _require_inference("over_encoding_decode", qweight, scale, mean, codebook)
    forward, _ = load_impl("over_encoding_decode", implementation)
    return forward(
        input_ids,
        oe_history_inputs,
        oe_vocab_sizes,
        oe_vocab_offsets,
        n_grams,
        qweight,
        scale,
        mean,
        ori_vocab_size,
        group_size,
        codebook,
        vocab_start_id,
        output_dtype,
    )
