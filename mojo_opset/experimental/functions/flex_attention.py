import math

from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def _flex_attention_torch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    group = Hq // Hkv if Hq >= Hkv else 1

    output = torch.empty_like(q)
    lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

    for z in range(Z):
        for hq in range(Hq):
            hkv = hq // group
            q_i = q[z, hq]
            k_i = k[z, hkv]
            v_i = v[z, hkv]

            scores = torch.matmul(q_i, k_i.t().float()).float() * sm_scale

            if block_mask is not None:
                dense_mask = getattr(block_mask, "dense_mask", None)
                if dense_mask is not None:
                    mask_2d = dense_mask[z, hq, :M, :N] if dense_mask.dim() >= 4 else dense_mask[0, 0, :M, :N]
                    scores = scores.masked_fill(~mask_2d, float("-inf"))

            m_i = torch.max(scores, dim=-1).values
            p_i = torch.exp(scores - m_i.unsqueeze(-1))
            l_i = torch.sum(p_i, dim=-1)
            p_i = p_i / l_i.unsqueeze(-1)
            p_i = p_i.to(v.dtype)

            out_i = torch.matmul(p_i, v_i)
            output[z, hq] = out_i
            lse[z, hq] = (m_i + torch.log(l_i)).float()

    return output, lse


def _flex_attention_torch_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    group = Hq // Hkv if Hq >= Hkv else 1

    with torch.enable_grad():
        q_g = q.detach().requires_grad_(True)
        k_g = k.detach().requires_grad_(True)
        v_g = v.detach().requires_grad_(True)

        q_g_exp = q_g
        k_g_exp = k_g
        v_g_exp = v_g
        if group > 1:
            k_g_exp = k_g.repeat_interleave(group, dim=1)
            v_g_exp = v_g.repeat_interleave(group, dim=1)

        scores = torch.matmul(q_g_exp, k_g_exp.transpose(-1, -2)).float() * sm_scale

        if block_mask is not None:
            dense_mask = getattr(block_mask, "dense_mask", None)
            if dense_mask is not None:
                if dense_mask.dim() >= 4:
                    mask_2d = dense_mask[0, 0, :M, :N]
                else:
                    mask_2d = dense_mask[:M, :N]
                scores = scores.masked_fill(~mask_2d.unsqueeze(0).unsqueeze(0), float("-inf"))

        m_i = torch.max(scores, dim=-1, keepdim=True).values
        p_i = torch.exp(scores - m_i)
        l_i = torch.sum(p_i, dim=-1, keepdim=True)
        p_i = (p_i / l_i).to(v_g.dtype)

        out = torch.matmul(p_i, v_g_exp)

        grad_q, grad_k, grad_v = torch.autograd.grad(
            out, (q_g_exp, k_g_exp, v_g_exp), grad_output, retain_graph=False, allow_unused=True
        )

        if group > 1:
            grad_k = grad_k.view(Z, group, Hkv, N, D).sum(dim=1)
            grad_v = grad_v.view(Z, group, Hkv, N, Dv).sum(dim=1)

    return grad_q, grad_k, grad_v


class MojoFlexAttentionFunction(MojoFunction):
    """FlexAttention with BlockMask sparse attention support.

    This is the reference (torch) implementation. Backend-specific
    implementations (e.g. TTX/Triton) override forward/backward.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask=None,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        Z, Hq, M, D = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        output, lse = _flex_attention_torch_forward(q, k, v, block_mask, sm_scale)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.block_mask = block_mask
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, output, lse = ctx.saved_tensors
        block_mask = ctx.block_mask
        sm_scale = ctx.sm_scale

        dq, dk, dv = _flex_attention_torch_backward(
            grad_output, q, k, v, output, lse, block_mask, sm_scale
        )
        return dq, dk, dv, None, None


def mojo_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask=None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Functional wrapper for MojoFlexAttentionFunction.

    Args:
        q: Query tensor [Z, Hq, M, D]
        k: Key tensor [Z, Hkv, N, D]
        v: Value tensor [Z, Hkv, N, Dv]
        block_mask: BlockMask from torch.nn.attention.flex_attention
        sm_scale: Optional softmax scale; defaults to 1/sqrt(D)

    Returns:
        Output tensor [Z, Hq, M, Dv]
    """
    return MojoFlexAttentionFunction.apply(q, k, v, block_mask, sm_scale)
