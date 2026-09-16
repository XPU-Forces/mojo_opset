from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must use packed [tokens, heads, head_dim] layout")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head_dim")
    if q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("the number of query heads must be divisible by the number of KV heads")
    if cu_q_lens.ndim != 1 or cu_k_lens.ndim != 1 or cu_q_lens.shape != cu_k_lens.shape:
        raise ValueError("cu_q_lens and cu_k_lens must be equally sized 1D tensors")


class SwaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        gqa_interleave,
        output_f32,
        implementation,
    ):
        forward, backward = load_impl("swa", implementation, require_backward=True)
        output, softmax_lse, output_cache = forward(
            q,
            k,
            v,
            cu_q_lens,
            cu_k_lens,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
            output_f32,
        )
        ctx.backward_kernel = backward
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.softmax_scale = softmax_scale
        ctx.gqa_interleave = gqa_interleave
        ctx.save_for_backward(q, k, v, output_cache if output_f32 else output, softmax_lse, cu_q_lens, cu_k_lens)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output_cache, softmax_lse, cu_q_lens, cu_k_lens = ctx.saved_tensors
        grad_q, grad_k, grad_v = ctx.backward_kernel(
            grad_output.contiguous(),
            q,
            k,
            v,
            output_cache,
            softmax_lse,
            cu_q_lens,
            cu_k_lens,
            ctx.is_causal,
            ctx.local_window_size,
            ctx.global_window_size,
            ctx.softmax_scale,
            ctx.gqa_interleave,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None, None


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attention_mask,
        cu_q_lens,
        cu_k_lens,
        max_q_len,
        max_k_len,
        dropout_p,
        softmax_scale,
        causal,
        deterministic,
        implementation,
    ):
        forward, backward = load_impl("flash_attention", implementation, require_backward=True)
        output, softmax_lse, output_f32 = forward(
            q,
            k,
            v,
            attention_mask,
            cu_q_lens,
            cu_k_lens,
            max_q_len,
            max_k_len,
            dropout_p,
            softmax_scale,
            causal,
            deterministic,
        )
        ctx.backward_kernel = backward
        ctx.has_attention_mask = attention_mask is not None
        ctx.max_q_len = max_q_len
        ctx.max_k_len = max_k_len
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        saved = [q, k, v, output_f32, softmax_lse, cu_q_lens, cu_k_lens]
        if attention_mask is not None:
            saved.append(attention_mask)
        ctx.save_for_backward(*saved)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output_f32, softmax_lse, cu_q_lens, cu_k_lens = ctx.saved_tensors[:7]
        attention_mask = ctx.saved_tensors[7] if ctx.has_attention_mask else None
        grad_q, grad_k, grad_v = ctx.backward_kernel(
            grad_output.contiguous(),
            q,
            k,
            v,
            attention_mask,
            output_f32,
            softmax_lse,
            cu_q_lens,
            cu_k_lens,
            ctx.max_q_len,
            ctx.max_k_len,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.deterministic,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None, None, None


def swa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    *,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    output_f32: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Packed variable-length sliding-window attention in TND layout.

    output_f32 retains an extra FP32 output for backward accuracy; otherwise
    backward reuses the returned output. The return dtype always matches q.
    """

    _validate_inputs(q, k, v, cu_q_lens, cu_k_lens)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return SwaFunction.apply(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        bool(is_causal),
        local_window_size,
        global_window_size,
        scale,
        bool(gqa_interleave),
        bool(output_f32),
        implementation,
    )


def swa_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    *,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only packed TND sliding-window attention, including cached KV.

    Local/global windows are used only for causal attention. GQA uses contiguous
    head groups (AABB) by default, or interleaved groups (ABAB) when requested.
    Use swa() for training; this path invokes the inference kernel directly.
    """
    _validate_inputs(q, k, v, cu_q_lens, cu_k_lens)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("swa_infer does not support autograd; use swa() or torch.no_grad()")
    forward, _ = load_impl("swa_infer", implementation)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return forward(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        bool(is_causal),
        local_window_size,
        global_window_size,
        scale,
        bool(gqa_interleave),
    )


def native_swa_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    *,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Temporary forward-only API for the original native packed TND SWA.

    Kept separate from swa() to preserve the native implementation's contract
    and restrictions. Currently provided by native on 950PR and torch_reference.
    This migration interface is not yet stable.
    Use swa_infer() for the original Mojo inference contract, or swa() to train.
    """
    _validate_inputs(q, k, v, cu_q_lens, cu_k_lens)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("native_swa_infer does not support autograd; use swa() or torch.no_grad()")
    forward, _ = load_impl("native_swa_infer", implementation)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return forward(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        bool(is_causal),
        local_window_size,
        global_window_size,
        scale,
        bool(gqa_interleave),
    )


def flash_attention_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    max_q_len: int,
    max_k_len: int,
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only packed flash attention; no dropout or explicit mask yet.

    GQA uses contiguous head groups (AABB), as in flash_attention().
    deterministic is retained for signature compatibility; no backward is run.
    """
    _validate_inputs(q, k, v, cu_q_lens, cu_k_lens)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("flash_attention_infer does not support autograd; use flash_attention() or torch.no_grad()")
    if dropout_p != 0.0 or attention_mask is not None:
        raise NotImplementedError("flash_attention_infer requires dropout_p=0 and attention_mask=None")
    if max_q_len <= 0 or max_k_len <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    forward, _ = load_impl("flash_attention_infer", implementation)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return forward(
        q,
        k,
        v,
        attention_mask,
        cu_q_lens,
        cu_k_lens,
        int(max_q_len),
        int(max_k_len),
        float(dropout_p),
        scale,
        bool(causal),
        bool(deterministic),
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    max_q_len: int,
    max_k_len: int,
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """First-order packed variable-length flash attention in TND layout."""

    _validate_inputs(q, k, v, cu_q_lens, cu_k_lens)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return FlashAttentionFunction.apply(
        q,
        k,
        v,
        attention_mask,
        cu_q_lens,
        cu_k_lens,
        int(max_q_len),
        int(max_k_len),
        float(dropout_p),
        scale,
        bool(causal),
        bool(deterministic),
        implementation,
    )


def varlen_fa_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: Optional[torch.Tensor] = None,
    *,
    is_causal: bool = True,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Temporary forward-only API for the original AscendC VarlenPrefillGQA.

    Kept separate for migration; this is not yet a stable public interface.

    Supports unequal Q/KV lengths and uneven GQA head groups (Hq >= Hkv).
    AABB gives the first Hq % Hkv groups one extra query head; ABAB maps
    query head h to KV head h % Hkv. Causal masking is bottom-right aligned.
    Native A2 requires contiguous BF16/FP16, D=128, int32 offsets on the input
    device and nonempty sequences (Q <= KV when causal). No dropout or mask.
    """
    cu_k_lens = cu_q_lens if cu_k_lens is None else cu_k_lens
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("expected q[Tq,Hq,D], k/v[Tkv,Hkv,D]")
    if k.shape[1] <= 0 or q.shape[1] < k.shape[1]:
        raise ValueError("GQA requires Hq >= Hkv > 0")
    if cu_q_lens.ndim != 1 or cu_k_lens.ndim != 1 or cu_q_lens.numel() < 2 or cu_q_lens.shape != cu_k_lens.shape:
        raise ValueError("cu_lens must have the same 1D shape and at least two offsets")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("varlen_fa_infer does not support autograd; use torch.no_grad()")
    forward, _ = load_impl("varlen_fa_infer", implementation)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return forward(q, k, v, cu_q_lens, cu_k_lens, bool(is_causal), scale, bool(gqa_interleave))
