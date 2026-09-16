"""Forward-only lightning attention indexing."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl
from .normalization import layer_norm_infer
from .position_embedding import apply_rope_infer
from .quantization import dynamic_quant
from .rotate_activation import rotate_activation


def lightning_indexer(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("lightning_indexer", query, query_scale, key, key_scale)
    batch, tokens, heads, _ = query.shape
    key_tokens = key.shape[1]
    if query_scale.shape != (batch, tokens, heads):
        raise ValueError("query_scale must have shape [batch, query_tokens, heads]")
    if key_scale is None:
        key_scale = torch.ones(batch, key_tokens, device=query.device, dtype=torch.float32)
    elif key_scale.ndim == 1 and key_scale.shape[0] == key_tokens:
        key_scale = key_scale.float().unsqueeze(0).expand(batch, -1)
    elif key_scale.shape != (batch, key_tokens):
        raise ValueError("key_scale must have shape [key_tokens] or [batch, key_tokens]")
    forward, _ = load_impl("lightning_indexer", implementation)
    return forward(query.contiguous(), query_scale.contiguous(), key.contiguous(), key_scale)


def indexer(
    x,
    qr,
    wq_b,
    wk,
    k_norm_weight,
    k_norm_bias,
    weights_proj,
    k_cache,
    k_scale_cache,
    start_pos,
    freqs_cis,
    mask=None,
    *,
    n_heads=128,
    head_dim=128,
    topk=2048,
    norm_eps=1e-5,
    implementation=None,
):
    """Composite inference indexer; update the supplied key/scale caches and return top-k indices and scores."""
    _require_inference("indexer", x, qr, wq_b, wk, k_norm_weight, k_norm_bias, weights_proj)
    batch, tokens, _ = x.shape
    end_pos = start_pos + tokens
    q = torch.nn.functional.linear(qr, wq_b).view(batch, tokens, n_heads, head_dim)
    k = layer_norm_infer(
        torch.nn.functional.linear(x, wk), k_norm_weight, k_norm_bias, norm_eps, implementation=implementation
    ).unsqueeze(2)
    cos, sin = (torch.cat((part, part), dim=-1) for part in (freqs_cis.real, freqs_cis.imag))
    q, k = apply_rope_infer(q, k, cos, sin, head_first=False, keep_cos_sin_dtype=True, implementation=implementation)
    # RotateActivation had no optimized NPU provider in master; select its Torch formula explicitly.
    q = rotate_activation(q, implementation="torch_reference")
    k = rotate_activation(k.squeeze(2), implementation="torch_reference")
    q_quant, q_scale = dynamic_quant(q, implementation=implementation)
    k_quant, k_scale = dynamic_quant(k, implementation=implementation)
    q_scale = q_scale.squeeze(-1)
    if k_scale.ndim == 3:
        k_scale = k_scale.amax(dim=-1)
    k_cache[:batch, start_pos:end_pos] = k_quant
    k_scale_cache[:batch, start_pos:end_pos] = k_scale
    # Keep this projection in FP32 even after module.to(dtype=bf16/fp16).
    weights = torch.nn.functional.linear(x.float(), weights_proj.float()) * n_heads**-0.5
    weights = weights * q_scale * head_dim**-0.5
    scores = lightning_indexer(
        q_quant.contiguous(),
        weights.contiguous(),
        k_cache[:batch, :end_pos].contiguous(),
        k_scale_cache[:batch, :end_pos].contiguous(),
        implementation=implementation,
    )
    if mask is not None:
        scores = scores + mask
    return scores.topk(min(topk, end_pos), dim=-1).indices, scores
