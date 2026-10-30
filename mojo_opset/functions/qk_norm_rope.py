"""Fused QK RMSNorm + RoPE, the MiniMax-H3 inference contract.

Public entry point for the ``fused_qk_norm_rope`` family (vLLM-Omni
``vllm_omni/diffusion/layers/fused_qk_norm_rope.py``). One call returns both rotated Q and K;
the backend issues one launch per stream, matching upstream's kernel count.
"""

from typing import Optional
from typing import Tuple

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

# The kernel rounds the normalized value to BF16 before rotating it, so the activations have
# to be a float that has a BF16 representation. Upstream accepts the same three.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, op):
    for name, x in (("q", q), ("k", k)):
        if x.ndim != 3:
            raise ValueError(f"{name} must be [tokens, heads, head_dim], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if x.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}")
    if q.device != k.device:
        raise ValueError(f"q and k must share a device, got {q.device} and {k.device}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q and k must share the token count, got {q.shape[0]} and {k.shape[0]}")
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must share head_dim, got {q.shape[2]} and {k.shape[2]}"
        )
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must share a dtype, got {q.dtype} and {k.dtype}")

    head_dim = q.shape[2]
    for name, w in (("q_weight", q_weight), ("k_weight", k_weight)):
        if w.ndim != 1 or w.shape[0] != head_dim:
            raise ValueError(f"{name} must be [{head_dim}], got {tuple(w.shape)}")
        if not w.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if w.device != q.device:
            raise ValueError(f"{name} must be on the q/k device {q.device}, got {w.device}")

    if rope_table.ndim != 2 or rope_table.shape[0] != q.shape[0]:
        raise ValueError(
            f"rope_table must be [tokens, rotary_dim] with {q.shape[0]} tokens, "
            f"got {tuple(rope_table.shape)}"
        )
    if not rope_table.is_floating_point():
        raise ValueError("rope_table must be floating point")
    if rope_table.device != q.device:
        raise ValueError(
            f"rope_table must be on the q/k device {q.device}, got {rope_table.device}"
        )
    if rope_table.dtype not in (q.dtype, torch.float32):
        raise ValueError(
            f"rope_table must have the activation dtype ({q.dtype}) or float32, "
            f"got {rope_table.dtype}"
        )
    rotary_dim = rope_table.shape[1]
    if rotary_dim == 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim={rotary_dim} must be a positive even number")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim={rotary_dim} must not exceed head_dim={head_dim}")
    if not isinstance(eps, (int, float)):
        raise ValueError(f"eps must be a scalar, got {type(eps).__name__}")
    _require_inference(op, q, k, q_weight, k_weight, rope_table)


def qk_norm_rope_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    implementation: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head Q/K RMSNorm followed by half-split RoPE from a packed ``[cos | sin]`` table.

    ``q`` is ``[tokens, num_q_heads, head_dim]`` and ``k`` is ``[tokens, num_kv_heads,
    head_dim]``; both may be strided views of a packed QKV buffer, and their strides reach
    the kernel, so nothing calls ``.contiguous()`` on them. ``rope_table`` is
    ``[tokens, rotary_dim]`` storing ``cos(theta)`` in its first half and ``sin(theta)`` in
    the second, with ``theta`` of width ``rotary_dim // 2``. Dimensions beyond ``rotary_dim``
    pass through unrotated.

    ``q_weight``, ``k_weight`` and ``rope_table`` are read with unit stride by the kernel, so
    a non-contiguous view of either is copied here instead of being read at the wrong
    offsets. Upstream does the same and also accepts only the activation dtype or float32
    for the table, which is what this contract checks.

    Forward only, like ``rms_norm_infer``: the caller has to be inside ``torch.no_grad()``
    or an inference mode, because the norm weights are normally ``nn.Parameter``s. Only the
    half-split (non-interleaved) pairing is provided; the interleaved Boogu-Image variant
    stays on the caller's eager path.
    """
    _validate_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, "qk_norm_rope_infer")
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    forward, _ = load_impl("qk_norm_rope_infer", implementation)
    return forward(q, k, q_weight, k_weight, rope_table, float(eps))
