"""Combine a sublayer output with hyper-connection residual streams (mHC post)."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Update ``hc_mult`` residual streams with a finished sublayer's output.

    ``x`` is ``[num_tokens, hidden_size]`` (the sublayer output, BF16),
    ``residual`` is ``[num_tokens, hc_mult, hidden_size]`` (BF16),
    ``post_layer_mix`` is ``[num_tokens, hc_mult, 1]`` FP32 and
    ``comb_res_mix`` is ``[num_tokens, hc_mult, hc_mult]`` FP32 (the gates
    from the matching mHC pre block). Returns a contiguous
    ``[num_tokens, hc_mult, hidden_size]`` tensor with ``residual``'s dtype:
    ``out[t, j] = post[t, j] * x[t] + sum_i comb[t, i, j] * residual[t, i]``,
    computed in FP32 and rounded once (matching the vLLM ``mhc_post``
    reference). ``post_layer_mix`` may also be ``[num_tokens, hc_mult]``.
    Inference only; there is no backward.
    """

    _require_inference("mhc_post", x, residual, post_layer_mix, comb_res_mix)
    if residual.ndim != 3:
        raise ValueError(f"mhc_post residual must be [num_tokens, hc_mult, hidden_size], got {tuple(residual.shape)}")
    num_tokens, hc_mult, hidden_size = residual.shape
    if tuple(x.shape) != (num_tokens, hidden_size):
        raise ValueError(f"mhc_post x shape {tuple(x.shape)} must be {(num_tokens, hidden_size)}")
    if post_layer_mix.ndim == 2:
        if tuple(post_layer_mix.shape) != (num_tokens, hc_mult):
            raise ValueError(
                f"mhc_post post_layer_mix shape {tuple(post_layer_mix.shape)} must be {(num_tokens, hc_mult, 1)}"
            )
        post_layer_mix = post_layer_mix.unsqueeze(-1)
    elif tuple(post_layer_mix.shape) != (num_tokens, hc_mult, 1):
        raise ValueError(
            f"mhc_post post_layer_mix shape {tuple(post_layer_mix.shape)} must be {(num_tokens, hc_mult, 1)}"
        )
    if tuple(comb_res_mix.shape) != (num_tokens, hc_mult, hc_mult):
        raise ValueError(
            f"mhc_post comb_res_mix shape {tuple(comb_res_mix.shape)} must be {(num_tokens, hc_mult, hc_mult)}"
        )
    forward, _ = load_impl("mhc_post", implementation)
    return forward(x, residual, post_layer_mix, comb_res_mix)
