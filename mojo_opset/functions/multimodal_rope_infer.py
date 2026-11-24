"""Forward-only multimodal rotary embedding, with explicit in-place control."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def multimodal_rope_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    is_interleaved: bool = False,
    head_dim: Optional[int] = None,
    *,
    inplace: bool = False,
    implementation: Optional[str] = None,
):
    _require_inference("multimodal_rope_infer", q, k, cos, sin)
    q_out = q.contiguous() if inplace else q.clone(memory_format=torch.contiguous_format)
    k_out = k.contiguous() if inplace else k.clone(memory_format=torch.contiguous_format)
    forward, _ = load_impl("multimodal_rope_infer", implementation)
    forward(q_out, k_out, cos.contiguous(), sin.contiguous(), list(mrope_section), is_interleaved, head_dim)
    if inplace:
        if q_out is not q:
            q.copy_(q_out)
        if k_out is not k:
            k.copy_(k_out)
        return q, k
    return q_out, k_out
