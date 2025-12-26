# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from einops import rearrange

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[triton.Config({"BD": BD}) for BD in [16, 32, 64, 128]],
#     key=["D", "W", "NB"],
# )
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            # p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # b_yi = tl.load(p_yi, boundary_check=(0, 1),padding_option="zero").to(tl.float32)
            yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]
            yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]
            mask = (yi_offset_0 < T) & (yi_offset_1 < D)
            # [BT, BD]
            b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(-W + 1, 1):
            # p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # b_yi = tl.load(p_yi, boundary_check=(0, 1),padding_option="zero").to(tl.float32)
            yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]
            yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]
            mask = (yi_offset_0 < T) & (yi_offset_1 < D)
            # [BT, BD]
            b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                tl.float32
            )

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BD": BD}) for BD in [16, 32, 64, 128]],
    key=["D", "W", "NB"],
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        # which may use initial state
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                # gradient comes from x：sum_t dy[t+i_w] * x[t]
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                # index of cache：c = W - i_w + t
                if USE_INITIAL_STATE:
                    mask_head_rows = o_t < i_w
                    # dy_head = dy[t]
                    b_dy_head = tl.load(
                        dy + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0
                    ).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":
                        # use y[t] （not y[t+i_w]）
                        b_y_head = tl.load(
                            y + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    # index 0 is padding 0
                    mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                    b_xc = tl.load(
                        initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                        mask=(mask_c[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    # add the gradient comes from initial_state
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

        if USE_INITIAL_STATE:
            p_dy0 = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_dy0 = tl.load(p_dy0, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y0 = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_y0 = tl.load(p_y0, boundary_check=(0, 1)).to(tl.float32)
                b_ys0 = tl.sigmoid(b_y0)
                b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))
            # index 0 is padding 0, skip calculation
            for i_w in tl.static_range(1, W):
                m_rows = o_t < i_w
                if HAS_WEIGHT:
                    # [BT]
                    w_idx_rows = i_w - 1 - o_t
                    # [BT, BW]
                    w_mask = o_w[None, :] == w_idx_rows[:, None]
                    w_pick = tl.sum(b_w[None, :, :] * w_mask[:, None, :], 2)
                else:
                    w_pick = 1.0
                contrib = (b_dy0 * w_pick).to(tl.float32)
                contrib = tl.where(m_rows[:, None] & m_d[None, :], contrib, 0.0)
                # [BD]
                b_dh0_s = tl.sum(contrib, 0)
                # dh0: [NT, B, D, W]
                tl.store(
                    dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                    b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
                    mask=m_d,
                )

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T - W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            b_dx += b_dht

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["cache"] is not None,
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
    }
)
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_c = o_w < W - 1

    # [BD]
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=0).to(tl.float32)

    if USE_INITIAL_STATE:
        # shift the cache by 1 with the last one being discarded
        p_cache = tl.make_block_ptr(cache + i_n * D * W, (D, W), (W, 1), (i_d * BD, W - BW + 1), (BD, BW), (1, 0))
        # [BD, BW]
        b_cache = tl.load(p_cache, boundary_check=(0, 1)).to(tl.float32)
        b_cache = tl.where(m_c[None, :], b_cache, b_x[:, None])
    else:
        b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)
    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(y + i_n * D + o_d, tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_d)

    if USE_INITIAL_STATE:
        b_cache = tl.cast(b_cache, dtype=cache.dtype.element_ty, fp_downcast_rounding="rtne")
        # update the cache in-place
        p_cache = tl.make_block_ptr(cache + i_n * D * W, (D, W), (W, 1), (i_d * BD, W - BW), (BD, BW), (1, 0))
        tl.store(p_cache, b_cache, boundary_check=(0, 1))


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D, W = *x.shape, weight.shape[1]
    BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B * T), get_num_cores())))
    BW = triton.next_power_of_2(W)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B * T, 1024)

    y = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
        BD=128,
    )
    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None
    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), get_num_cores())))

    BW = triton.next_power_of_2(W)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B * T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B * NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None
    dh0 = (
        initial_state.new_zeros(min(NT, triton.cdiv(W, BT)), *initial_state.shape)
        if initial_state is not None
        else None
    )

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dh0=dh0,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    if initial_state is not None:
        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)

    return dx.view(shape), dw, db, dr, dh0


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = eos - BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - BW + tl.arange(0, BW)
    m_t = o_t >= tl.maximum(bos, eos - W)
    m_d = o_d < D
    m_w = (o_w >= 0) & (o_w < W)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
    if USE_INITIAL_STATE:
        if T < BW:
            o_c = W - (BW - T) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(initial_state + i_n * D * W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
            b_x += b_cache

    tl.store(final_state + i_n * D * W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, D, W = *x.shape, state_len
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)
    grid = (triton.cdiv(D, BD), N)
    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        BW=BW,
        BD=BD,
    )
    return final_state


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(make_contiguous=True, auto_to_device=True)
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )
        return y, final_state

    @staticmethod
    @input_guard(make_contiguous=True, auto_to_device=True)
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor]):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dx, dw, db, dr, dh0, None, None, None


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: str = None,
    backend: str = "triton",
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` and `'triton'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
    """

    y, final_state = CausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
    )
    return y, final_state
