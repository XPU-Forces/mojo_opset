# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen


from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_o import chunk_bwd_dqkwg
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_o import chunk_bwd_dv_local
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_o import chunk_fwd_o
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.utils import chunk_local_cumsum
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.utils import prepare_chunk_indices
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.utils import prepare_chunk_offsets
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.utils import solve_tril

fwd_hit_indexer_warning = False
bwd_hit_indexer_warning = False


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
):
    if cu_seqlens is None:
        chunk_indices, chunk_offsets = None, None
    else:
        global fwd_hit_indexer_warning
        if chunk_indices is None or chunk_offsets is None:
            if not fwd_hit_indexer_warning:
                print(
                    "[WARNING] chunk indices or offsets will be generated at runtime, will cause severe performance deterioration"
                )
                fwd_hit_indexer_warning = True

            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if chunk_indices is None else chunk_indices
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size) if chunk_offsets is None else chunk_offsets

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return g, o, A, w, u, h, v_new, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    v_new: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
    enable_gqa: bool = True,
):
    if cu_seqlens is None:
        chunk_indices, chunk_offsets = None, None
    else:
        global bwd_hit_indexer_warning
        if chunk_indices is None or chunk_offsets is None:
            if not bwd_hit_indexer_warning:
                print("[WARNING] chunk indices be generated at runtime, will cause severe performance deterioration")
                bwd_hit_indexer_warning = True
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if chunk_indices is None else chunk_indices
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size) if chunk_offsets is None else chunk_offsets
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"

    if enable_gqa and q.shape[2] > k.shape[2]:
        dk = dk.reshape(dk.shape[0], dk.shape[1], k.shape[2], q.shape[2] // k.shape[2], dk.shape[3]).sum(
            dim=3, keepdim=False
        )
        dv = dv.reshape(dv.shape[0], dv.shape[1], v.shape[2], q.shape[2] // v.shape[2], dv.shape[3]).sum(
            dim=3, keepdim=False
        )
        db = db.reshape(db.shape[0], db.shape[1], k.shape[2], q.shape[2] // k.shape[2]).sum(dim=3, keepdim=False)
        dg = dg.reshape(dg.shape[0], dg.shape[1], g.shape[2], q.shape[2] // k.shape[2]).sum(dim=3, keepdim=False)

    dg = chunk_local_cumsum(dg, chunk_size=16, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    return dq, dk, dv, db, dg, dh0
