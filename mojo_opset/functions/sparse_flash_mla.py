from typing import Optional

import torch

from ._dispatch import load_impl


def _validate_inputs(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_ori_kv: torch.Tensor,
    sinks: Optional[torch.Tensor],
    layout_kv: str,
    cu_seqlens_ori_kv: Optional[torch.Tensor],
    cmp_kv: Optional[torch.Tensor],
    cmp_block_table: Optional[torch.Tensor],
    seqused_cmp_kv: Optional[torch.Tensor],
    cu_seqlens_cmp_kv: Optional[torch.Tensor],
    cmp_residual_kv: Optional[torch.Tensor],
    cmp_mask_mode: int,
    cmp_ratio: int,
) -> None:
    if q.ndim != 3:
        raise ValueError("q must use packed [tokens, heads, head_dim] layout")
    if int(cu_seqlens_q[0]) != 0 or int(cu_seqlens_q[-1]) != q.shape[0]:
        raise ValueError("cu_seqlens_q must start at 0 and cover all query tokens")
    if sinks is not None and not torch.all(sinks == 0):
        raise NotImplementedError("sparse_flash_mla_infer supports zero sinks only")
    if layout_kv == "PA_BBND":
        if ori_kv.ndim != 4:
            raise ValueError("PA_BBND ori_kv must use [pages, page_size, kv_heads, head_dim] layout")
        if ori_block_table is None:
            raise ValueError("PA_BBND requires ori_block_table")
    elif layout_kv == "TND":
        if ori_kv.ndim != 3:
            raise ValueError("TND ori_kv must use [tokens, kv_heads, head_dim] layout")
        if cu_seqlens_ori_kv is None:
            raise ValueError("TND requires cu_seqlens_ori_kv")
    else:
        raise ValueError(f"unsupported layout_kv: {layout_kv!r}")
    if cmp_kv is not None:
        if layout_kv == "PA_BBND":
            if cmp_kv.ndim != 4:
                raise ValueError("PA_BBND cmp_kv must use [pages, page_size, kv_heads, head_dim] layout")
            if cmp_block_table is None:
                raise ValueError("PA_BBND cmp KV requires cmp_block_table")
        else:
            if cmp_kv.ndim != 3:
                raise ValueError("TND cmp_kv must use [tokens, kv_heads, head_dim] layout")
            if cu_seqlens_cmp_kv is None:
                raise ValueError("TND cmp KV requires cu_seqlens_cmp_kv")
        if seqused_cmp_kv is None:
            raise ValueError("cmp KV requires seqused_cmp_kv lengths")
        if cmp_mask_mode == 3 and cmp_ratio != 1 and cmp_residual_kv is None:
            raise ValueError("causal cmp_ratio != 1 requires cmp_residual_kv")


def sparse_flash_mla_infer(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_ori_kv: torch.Tensor,
    *,
    sinks: Optional[torch.Tensor] = None,
    win_left: int = 127,
    win_right: int = 0,
    softmax_scale: Optional[float] = None,
    layout_kv: str = "PA_BBND",
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_ratio: int = 1,
    cmp_mask_mode: int = 3,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only paged sparse MLA-style decode (DSA-flavoured).

    MQA shared paged KV + sliding-window mask + attention-sink seeded online
    softmax, with an optional compressed-KV branch. Supported subsets (kernel
    ``mode`` dispatch): SWA (ori-only window), HCA (dense compressed prefix,
    e.g. C128), CSA (per-query sparse topk selection, e.g. C4).

    Args:
        q: ``(T1, Hq, D)`` bf16/fp16 (TND layout, batched via cu_seqlens_q).
        ori_kv: PA_BBND ``(pages, page_size, Hkv, D)`` paged KV or TND
            ``(T2, Hkv, D)`` dense KV, MQA shared.
        ori_block_table: ``(B, pages_per_batch)`` int32 page ids (PA_BBND only).
        cu_seqlens_q: ``(B+1,)`` int32 cumulative query lens.
        seqused_ori_kv: ``(B,)`` int32 actual ori KV length per batch.
        sinks: optional per-head sink values; currently zero sinks only.
        win_left / win_right: sliding-window bounds (inclusive left, exclusive
            right offset).
        softmax_scale: scaling factor; defaults to ``1/sqrt(D)``.
        layout_kv: ``"PA_BBND"`` (paged, needs block tables) or ``"TND"``
            (dense, needs cu_seqlens_ori_kv / cu_seqlens_cmp_kv prefixes).
        cu_seqlens_ori_kv: ``(B+1,)`` int32 dense-storage prefixes (TND only).
        cmp_kv: optional compressed KV, same layout convention as ori_kv.
        cmp_block_table: ``(B, cmp_pages_per_batch)`` int32 (PA_BBND only).
        seqused_cmp_kv: ``(B,)`` int32 compressed KV length per batch.
        cu_seqlens_cmp_kv: ``(B+1,)`` int32 compressed-storage prefixes (TND only).
        cmp_residual_kv: ``(B,)`` int32 revert residual, required for causal
            cmp (cmp_mask_mode=3) with cmp_ratio != 1.
        cmp_sparse_indices: optional ``(T1, K)`` int32 per-query compressed
            token selection (CSA); None selects the dense prefix (HCA).
        cmp_ratio: compressed-KV compression ratio (1 for SWA).
        cmp_mask_mode: cmp branch mask mode; 3 = revert-causal (default).
        implementation: optional implementation override (e.g. "cannbotdsl").

    Returns:
        ``(T1, Hq, D)`` same dtype as q.
    """
    _validate_inputs(
        q,
        ori_kv,
        ori_block_table,
        cu_seqlens_q,
        seqused_ori_kv,
        sinks,
        layout_kv,
        cu_seqlens_ori_kv,
        cmp_kv,
        cmp_block_table,
        seqused_cmp_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        cmp_mask_mode,
        cmp_ratio,
    )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, ori_kv, cmp_kv) if tensor is not None
    ):
        raise RuntimeError("sparse_flash_mla_infer does not support autograd; use torch.no_grad()")
    forward, _ = load_impl("sparse_flash_mla_infer", implementation)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return forward(
        q,
        ori_kv,
        ori_block_table,
        cu_seqlens_q,
        seqused_ori_kv,
        sinks,
        win_left,
        win_right,
        scale,
        layout_kv,
        cu_seqlens_ori_kv,
        cmp_kv,
        cmp_block_table,
        seqused_cmp_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        cmp_sparse_indices,
        cmp_ratio,
        cmp_mask_mode,
    )
