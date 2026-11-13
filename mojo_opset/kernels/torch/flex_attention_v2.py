"""Striped FP32 reference for grouped FlexAttention, including prepared dropout."""

import torch

from mojo_opset.utils.attention_dropout import _dropout_keep

from .flex_attention import _block_mask_to_dense
from .flex_attention import flex_attention_bwd
from .flex_attention import flex_attention_fwd


def _dropout_attention(q, k, v, visible, keep, scale, drop_scale):
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~visible, -torch.inf)
    safe_scores = torch.where(visible.any(-1, keepdim=True), scores, 0)
    probabilities = safe_scores.softmax(-1).masked_fill(~visible, 0)
    # Preserve the undropped denominator; nominal drop scale is applied after PV.
    return ((probabilities * keep) @ v.float()) * drop_scale, scores.logsumexp(-1)


def flex_attention_v2_fwd(q, k, v, block_mask, scale=None, *, dropout=None):
    if dropout is None:
        return flex_attention_fwd(q, k, v, block_mask, scale)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    visible = _block_mask_to_dense(block_mask, q.shape[2], k.shape[2], q.device)
    saved = torch.empty((*q.shape[:3], v.shape[-1]), dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    for start in range(0, q.shape[2], 256):
        stop = min(start + 256, q.shape[2])
        saved[:, :, start:stop], lse[:, :, start:stop] = _dropout_attention(
            q[:, :, start:stop], k, v, visible[start:stop], _dropout_keep(dropout, start, stop), scale, dropout.scale
        )
    return saved.to(q.dtype), lse, saved


def flex_attention_v2_bwd(grad_output, q, k, v, output, lse, block_mask, scale=None, *, dropout=None):
    if dropout is None:
        return flex_attention_bwd(grad_output, q, k, v, output, lse, block_mask, scale)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    visible = _block_mask_to_dense(block_mask, q.shape[2], k.shape[2], q.device)
    dq, dk, dv = (torch.zeros_like(x, dtype=torch.float32) for x in (q, k, v))
    for start in range(0, q.shape[2], 256):
        stop = min(start + 256, q.shape[2])
        with torch.enable_grad():
            qg = q[:, :, start:stop].detach().float().requires_grad_(True)
            kg, vg = (x.detach().float().requires_grad_(True) for x in (k, v))
            out, _ = _dropout_attention(
                qg, kg, vg, visible[start:stop], _dropout_keep(dropout, start, stop), scale, dropout.scale
            )
        grads = torch.autograd.grad(out, (qg, kg, vg), grad_output[:, :, start:stop].float())
        dq[:, :, start:stop] = grads[0]
        dk += grads[1]
        dv += grads[2]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


flex_attention_v2_fwd.supports_dropout = True
flex_attention_v2_fwd.create_block_mask = flex_attention_fwd.create_block_mask
__all__ = ["flex_attention_v2_fwd", "flex_attention_v2_bwd"]
