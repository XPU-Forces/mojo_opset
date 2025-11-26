# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

import warnings

from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.ascend.linear_attn.modules.l2norm import l2norm_bwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.modules.l2norm import l2norm_fwd
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
from mojo_opset.backends.ttx.kernels.ascend.utils import autocast_custom_bwd
from mojo_opset.backends.ttx.kernels.ascend.utils import autocast_custom_fwd
from mojo_opset.backends.ttx.kernels.ascend.utils import input_guard

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


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
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
        use_qk_l2norm_in_kernel: bool = True,
        enable_gqa: bool = True,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, w, u, h, v_new, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A, w, u, h, v_new, initial_state, cu_seqlens, chunk_indices, chunk_offsets
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.enable_gqa = enable_gqa
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta,
            A,
            w,
            u,
            h,
            v_new,
            initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            w=w,
            u=u,
            h=h,
            v_new=v_new,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            enable_gqa=ctx.enable_gqa,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    enable_gqa: bool = True,
):
    r"""
        Args:
            q (torch.Tensor):
                queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            k (torch.Tensor):
                keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            v (torch.Tensor):
                values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
            g (torch.Tensor):
                (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
            beta (torch.Tensor):
                betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
            scale (Optional[int]):
                Scale factor for the RetNet attention scores.
                If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
            initial_state (Optional[torch.Tensor]):
                Initial state of shape `[N, H, K, V]` for `N` input sequences.
                For equal-length input sequences, `N` equals the batch size `B`.
                Default: `None`.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
            cu_seqlens (torch.LongTensor):
                Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
                consistent with the FlashAttention API.
            head_first (Optional[bool]):
                Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
                Default: `False`.

        Returns:
            o (torch.Tensor):
                Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

        Examples::
            >>> import torch
    import torch_npu
            >>> import torch
    import torch.nn.functional as F
            >>> from einops import rearrange
            >>> from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule import chunk_gated_delta_rule
            # inputs with equal lengths
            >>> B, T, H, K, V = 4, 2048, 4, 512, 512
            >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
            >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
            >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
            >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
            >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
            >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
            >>> o, ht = chunk_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=h0,
                output_final_state=True
            )
            # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
            >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
            # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
            >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
            >>> o_var, ht_var = chunk_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens
            )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    #  Handle GQA case where k,v have fewer heads than q
    head_dim = 1 if head_first else 2
    num_q_heads = q.shape[head_dim]
    num_kv_heads = k.shape[head_dim]
    num_gbeta_heads = g.shape[head_dim]

    if (not enable_gqa) and (num_q_heads > num_kv_heads):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Number of query heads ({num_q_heads}) must be divisible by "
                f"number of key/value heads ({num_kv_heads}) for GQA"
            )

        # Calculate expansion factor
        num_repeats = num_q_heads // num_kv_heads

        # Expand k and v to match number of query heads
        if head_first:
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)
        else:
            k = k.repeat_interleave(num_repeats, dim=2)
            v = v.repeat_interleave(num_repeats, dim=2)

    if (not enable_gqa) and (num_q_heads > num_gbeta_heads):
        if num_q_heads % num_gbeta_heads != 0:
            raise ValueError(
                f"Number of query heads ({num_q_heads}) must be divisible by "
                f"number of beta/gate heads ({num_gbeta_heads}) for GQA"
            )
        # Calculate expansion factor
        num_repeats = num_q_heads // num_gbeta_heads
        g = g.repeat_interleave(num_repeats, dim=2)
        beta = beta.repeat_interleave(num_repeats, dim=2)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        None,
        None,
        16,
        use_qk_l2norm_in_kernel,
        enable_gqa,
    )
    return o, final_state
