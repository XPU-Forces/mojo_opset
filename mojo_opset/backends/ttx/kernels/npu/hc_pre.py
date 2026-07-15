# Copyright 2026, The FlagOS Contributors.

import torch
import torch_npu
import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

try:
    import triton.experimental.tle as tle
    pipe = tle.dsa.ascend.PIPE
    HAS_TLE = True
except:
    HAS_TLE = False




@triton.jit
def _vector_process_single_row(
    mixes, 
    row_idx,
    x_ptr, 
    pre_ptr, 
    post_ptr, 
    comb_ptr, 
    y_ptr,
    stride_x_row, 
    stride_x_col,
    stride_pre_row, 
    stride_pre_col,
    stride_post_row, 
    stride_post_col,
    stride_comb_row, 
    stride_comb_col,
    stride_y_row, 
    stride_y_col,
    scale_pre_post, 
    base_pre_post, 
    scale_comb, 
    base_comb, 
    offs_hc,
    HC_MIX: tl.constexpr, 
    HC_MULT: tl.constexpr, 
    K: tl.constexpr,
    HC_EPS: tl.constexpr, 
    HC_SINKHORN_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    BLOCK_D: tl.constexpr,
):
    mixes_1d = tl.reshape(mixes, [BLOCK_N])
    pre_post_raw = tle.dsa.extract_slice(mixes_1d, (0,), (2 * HC_MULT,), (1,))
    comb_raw = tle.dsa.extract_slice(mixes_1d, (2 * HC_MULT,), (HC_MIX - 2 * HC_MULT,), (1,))

    # sigmoid
    # slice pre/post
    pre_post_sigmoid = tl.sigmoid(pre_post_raw * scale_pre_post + base_pre_post)
    pre_out = tle.dsa.extract_slice(pre_post_sigmoid, (0,), (HC_MULT,), (1,)) + HC_EPS
    post_out = 2.0 * tle.dsa.extract_slice(pre_post_sigmoid, (HC_MULT,), (HC_MULT,), (1,))

    # store pre/post
    tl.store(pre_ptr + row_idx * stride_pre_row + offs_hc * stride_pre_col, pre_out)
    tl.store(post_ptr + row_idx * stride_post_row + offs_hc * stride_post_col, post_out)

    comb = comb_raw * scale_comb + base_comb
    comb_2d = tl.reshape(comb, [HC_MULT, HC_MULT])

    # pad to 32-byte aligned: 32 / sizeof(float32) = 8
    PADDED: tl.constexpr = ((HC_MULT + 7) // 8) * 8
    NEG_INF: tl.constexpr = float('-inf')
    comb_padded = tl.full([PADDED, PADDED], NEG_INF, dtype=tl.float32)
    comb_padded = tle.dsa.insert_slice(comb_padded, comb_2d, (0, 0), (HC_MULT, HC_MULT), (1, 1))

    # softmax on axis=1 (row-wise)
    max_val = tl.max(comb_padded, axis=1)
    max_val = tl.where(max_val == NEG_INF, 0.0, max_val)
    comb_padded = comb_padded - max_val[:, None]
    exp_val = tl.exp(comb_padded)
    sum_val = tl.sum(exp_val, axis=1) + HC_EPS

    # Mask: perform softmax + eps on valid tokens, zero out padding positions
    offs_row = tl.arange(0, PADDED)
    offs_col = tl.arange(0, PADDED)
    valid_mask = tl.where((offs_row[:, None] < HC_MULT) & (offs_col[None, :] < HC_MULT), 1.0, 0.0)
    comb_sk = exp_val / sum_val[:, None] + HC_EPS
    comb_sk = comb_sk * valid_mask

    col_sum = tl.sum(comb_sk, axis=0) + HC_EPS
    comb_sk = comb_sk / col_sum[None, :]

    for _si in range(HC_SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb_sk, axis=1) + HC_EPS
        comb_sk = comb_sk / row_sum[:, None]
        col_sum = tl.sum(comb_sk, axis=0) + HC_EPS
        comb_sk = comb_sk / col_sum[None, :]

    # extract valid [HC_MULT, HC_MULT] → flatten to [HC_MULT*HC_MULT]
    comb_sk_valid = tle.dsa.extract_slice(comb_sk, (0, 0), (HC_MULT, HC_MULT), (1, 1))
    comb_sk_flat = tl.reshape(comb_sk_valid, [HC_MULT * HC_MULT])

    # store comb
    offs_comb_sk = tl.arange(0, HC_MULT * HC_MULT)
    tl.store(comb_ptr + row_idx * stride_comb_row + offs_comb_sk * stride_comb_col, comb_sk_flat)

    D: tl.constexpr = K // HC_MULT
    pre_2d = tl.reshape(pre_out, [HC_MULT, 1])
    for d_start in range(0, D, BLOCK_D):
        d_offset = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offset < D

        # load x [HC_MULT, BLOCK_D]
        h_offs = tl.arange(0, HC_MULT)
        x_offs = (row_idx * stride_x_row + (h_offs[:, None] * D + d_offset[None, :]) * stride_x_col)
        x_block = tl.load(x_ptr + x_offs, mask=d_mask[None, :], other=0.0).to(tl.float32)

        y_acc = tl.sum(pre_2d * x_block, axis=0)

        y_offs = row_idx * stride_y_row + d_offset * stride_y_col
        tl.store(y_ptr + y_offs, y_acc.to(y_ptr.dtype.element_ty), mask=d_mask)




@triton.jit
def _vector_process_multi_row(
    mixes,
    offs_row,
    mask_row,
    x_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_pre_row,
    stride_pre_col,
    stride_post_row,
    stride_post_col,
    stride_comb_row,
    stride_comb_col,
    stride_y_row,
    stride_y_col,
    scale_pre_post,
    base_pre_post,
    scale_comb,
    base_comb, offs_hc,
    HC_MIX: tl.constexpr,
    HC_MULT: tl.constexpr,
    K: tl.constexpr,
    HC_EPS: tl.constexpr,
    HC_SINKHORN_ITERS: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # slice mixes
    pre_post_raw = tle.dsa.extract_slice(mixes, (0, 0), (BLOCK_ROW, 2 * HC_MULT), (1, 1))
    comb_raw = tle.dsa.extract_slice(mixes, (0, 2 * HC_MULT), (BLOCK_ROW, HC_MIX - 2 * HC_MULT), (1, 1))

    # sigmoid
    pre_post_sigmoid = tl.sigmoid(pre_post_raw * scale_pre_post[None, :] + base_pre_post[None, :])
    pre_out = tle.dsa.extract_slice(pre_post_sigmoid, (0, 0), (BLOCK_ROW, HC_MULT), (1, 1)) + HC_EPS
    post_out = 2.0 * tle.dsa.extract_slice(pre_post_sigmoid, (0, HC_MULT), (BLOCK_ROW, HC_MULT), (1, 1))

    # store pre/post
    tl.store(pre_ptr + offs_row[:, None] * stride_pre_row + offs_hc[None, :] * stride_pre_col, pre_out, mask=mask_row[:, None])
    tl.store(post_ptr + offs_row[:, None] * stride_post_row + offs_hc[None, :] * stride_post_col, post_out, mask=mask_row[:, None])

    # ProcessComb: affine + reshape + pad to [BLOCK_ROW, PADDED, PADDED]
    comb = comb_raw * scale_comb + base_comb[None, :]
    comb_3d = tl.reshape(comb, [BLOCK_ROW, HC_MULT, HC_MULT])

    # pad to 32-byte aligned: 32 / sizeof(float32) = 8
    PADDED: tl.constexpr = ((HC_MULT + 7) // 8) * 8
    NEG_INF: tl.constexpr = float('-inf')
    comb_padded = tl.full([BLOCK_ROW, PADDED, PADDED], NEG_INF, dtype=tl.float32)
    comb_padded = tle.dsa.insert_slice(comb_padded, comb_3d, (0, 0, 0), (BLOCK_ROW, HC_MULT, HC_MULT), (1, 1, 1))

    # softmax on axis=2 (row-wise)
    max_val = tl.max(comb_padded, axis=2)
    max_val = tl.where(max_val == NEG_INF, 0.0, max_val)
    comb_padded = comb_padded - max_val[:, :, None]
    exp_val = tl.exp(comb_padded)
    sum_val = tl.sum(exp_val, axis=2) + HC_EPS

    # mask: perform softmax + eps on valid tokens, zero out padding positions
    _offs_r = tl.arange(0, PADDED)
    _offs_c = tl.arange(0, PADDED)
    _valid_mask = tl.where((_offs_r[:, None] < HC_MULT) & (_offs_c[None, :] < HC_MULT), 1.0, 0.0)
    comb_sk = exp_val / sum_val[:, :, None] + HC_EPS
    comb_sk = comb_sk * _valid_mask[None, :, :]
    col_sum = tl.sum(comb_sk, axis=1) + HC_EPS
    comb_sk = comb_sk / col_sum[:, None, :]

    for _si in range(HC_SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb_sk, axis=2) + HC_EPS
        comb_sk = comb_sk / row_sum[:, :, None]
        col_sum = tl.sum(comb_sk, axis=1) + HC_EPS
        comb_sk = comb_sk / col_sum[:, None, :]

    # store comb
    comb_sk_valid = tle.dsa.extract_slice(comb_sk, (0, 0, 0), (BLOCK_ROW, HC_MULT, HC_MULT), (1, 1, 1))
    comb_sk_flat = tl.reshape(comb_sk_valid, [BLOCK_ROW, HC_MULT * HC_MULT])  # [BLOCK_ROW, 16]
    offs_comb_sk = tl.arange(0, HC_MULT * HC_MULT)
    tl.store(comb_ptr
             + offs_row[:, None] * stride_comb_row
             + offs_comb_sk[None, :] * stride_comb_col,
             comb_sk_flat,
             mask=mask_row[:, None])

    # ProcessY: y[d] = sum_h(pre[h] * x[h, d])
    D: tl.constexpr = K // HC_MULT
    pre_3d = tl.reshape(pre_out, [BLOCK_ROW, HC_MULT, 1])
    for d_start in range(0, D, BLOCK_D):
        d_offset = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offset < D

        # load x [BLOCK_ROW, HC_MULT, BLOCK_D]
        h_offs = tl.arange(0, HC_MULT)
        x_offs = (offs_row[:, None, None] * stride_x_row
                  + (h_offs[None, :, None] * D
                  + d_offset[None, None, :]) * stride_x_col)
        x_mask = mask_row[:, None, None] & d_mask[None, None, :]
        x_block = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

        y_acc = tl.sum(pre_3d * x_block, axis=1)

        y_offs = offs_row[:, None] * stride_y_row + d_offset[None, :] * stride_y_col
        tl.store(y_ptr + y_offs, y_acc.to(y_ptr.dtype.element_ty), mask=mask_row[:, None] & d_mask[None, :])



@triton.autotune(
    configs=[
        triton.Config({"disable_auto_inject_block_sync": True, "unit_flag": True}),
    ],
    key=["K"],
)
@triton.jit
def _hc_pre_tle_kernel(
    x_ptr,
    hc_fn_ptr,
    mm_out_ptr,
    sq_out_ptr,
    x_ws_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_fn_row,
    stride_fn_col,
    stride_mm_k,
    stride_mm_row,
    stride_mm_col,
    stride_sq_k,
    stride_sq_row,
    stride_pre_row,
    stride_pre_col,
    stride_post_row,
    stride_post_col,
    stride_comb_row,
    stride_comb_col,
    stride_y_row,
    stride_y_col,
    bs,
    cube_block_dim_m,
    cube_block_dim_k,
    split_k_size,
    HC_MIX: tl.constexpr,
    HC_MULT: tl.constexpr,
    K: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    HC_SINKHORN_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    m_dim_idx = pid // cube_block_dim_k
    k_dim_idx = pid % cube_block_dim_k

    # K-range covered by the current program
    k_range_start = k_dim_idx * split_k_size
    k_range_end = tl.minimum(k_range_start + split_k_size, K)

    # M-range covered by the current program
    num_m_blocks = (bs + BLOCK_M - 1) // BLOCK_M
    single_core_m_blocks = (num_m_blocks + cube_block_dim_m - 1) // cube_block_dim_m
    m_block_start = m_dim_idx * single_core_m_blocks
    m_block_end = tl.minimum(m_block_start + single_core_m_blocks, num_m_blocks)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    ws_base = x_ws_ptr + pid * 2 * BLOCK_M * BLOCK_K
    ws_offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    tle.dsa.ascend.sync_block_set('cube', 'vector', 0, pipe.PIPE_MTE2, pipe.PIPE_MTE3)
    tle.dsa.ascend.sync_block_set('cube', 'vector', 1, pipe.PIPE_MTE2, pipe.PIPE_MTE3)

    buf_id = 0

    for block_iter in range(m_block_start, m_block_end):
        m_start = block_iter * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < bs

        mm_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        sq_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        for k_start in range(k_range_start, k_range_end, BLOCK_K):
            cur_k = k_start + offs_k
            k_mask = cur_k < K

            x_offs = offs_m[:, None] * stride_x_row + cur_k[None, :] * stride_x_col
            x_mask = mask_m[:, None] & k_mask[None, :]
            x_block = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(hc_fn_ptr.dtype.element_ty)

            sq_acc += tl.sum(x_block * x_block, axis=1)

            tle.dsa.ascend.sync_block_wait('cube', 'vector', buf_id, pipe.PIPE_MTE2, pipe.PIPE_MTE3)
            ws_cur = ws_base + buf_id * BLOCK_M * BLOCK_K
            tl.store(ws_cur + ws_offs, x_block, mask=x_mask)
            tle.dsa.ascend.sync_block_set('vector', 'cube', buf_id, pipe.PIPE_MTE3, pipe.PIPE_MTE2)

            tle.dsa.ascend.sync_block_wait('vector', 'cube', buf_id, pipe.PIPE_MTE3, pipe.PIPE_MTE2)
            x_cube = tl.load(ws_cur + ws_offs, mask=x_mask, other=0.0)

            fn_offs = offs_n[:, None] * stride_fn_row + cur_k[None, :] * stride_fn_col
            fn_mask = (offs_n[:, None] < HC_MIX) & k_mask[None, :]
            fn_block = tl.load(hc_fn_ptr + fn_offs, mask=fn_mask, other=0.0)

            mm_acc = tl.dot(x_cube, tl.trans(fn_block), acc=mm_acc, out_dtype=tl.float32)

            tle.dsa.ascend.sync_block_set('cube', 'vector', buf_id, pipe.PIPE_MTE2, pipe.PIPE_MTE3)
            buf_id ^= 1

        # store partial mm_out [cube_block_dim_k, bs, BLOCK_N]
        mm_store_offs = k_dim_idx * stride_mm_k + offs_m[:, None] * stride_mm_row + offs_n[None, :] * stride_mm_col
        tl.store(mm_out_ptr + mm_store_offs, mm_acc, mask=mask_m[:, None])

        # store partial sq_out [cube_block_dim_k, bs]
        tl.store(sq_out_ptr + k_dim_idx * stride_sq_k + offs_m * stride_sq_row, sq_acc, mask=mask_m)

    tle.dsa.ascend.sync_block_wait('cube', 'vector', 0, pipe.PIPE_MTE2, pipe.PIPE_MTE3)
    tle.dsa.ascend.sync_block_wait('cube', 'vector', 1, pipe.PIPE_MTE2, pipe.PIPE_MTE3)

    # Global sync: wait for all cores to complete matmul for all M-blocks
    tle.dsa.ascend.sync_block_all("all", 2)

    vec_num = tle.dsa.ascend.sub_vec_num()     # Number of vector cores per cube
    vec_id = tle.dsa.ascend.sub_vec_id()       # Local ID of the current vector sub-core

    # Global vector ID, partitioned by actual launch count
    global_vec_id = pid * vec_num + vec_id
    actual_vec_total = tl.num_programs(0) * vec_num

    # Evenly divide bs rows among the total number of vectors
    rows_per_vec = (bs + actual_vec_total - 1) // actual_vec_total
    row_start = global_vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, bs)

    offs_n = tl.arange(0, BLOCK_N)
    offs_hc = tl.arange(0, HC_MULT)

    hc_scale_vec = tl.load(hc_scale_ptr + tl.arange(0, 3))
    hc_base_vec  = tl.load(hc_base_ptr + tl.arange(0, HC_MIX))

    scale_pre  = tle.dsa.extract_element(hc_scale_vec, (0,))
    scale_post = tle.dsa.extract_element(hc_scale_vec, (1,))
    scale_comb = tle.dsa.extract_element(hc_scale_vec, (2,))
    base_pre_post = tle.dsa.extract_slice(hc_base_vec, (0,), (2 * HC_MULT,), (1,))
    base_comb = tle.dsa.extract_slice(hc_base_vec, (2 * HC_MULT,), (HC_MIX - 2 * HC_MULT,), (1,))

    offs_pp = tl.arange(0, 2 * HC_MULT)
    scale_pre_post = tl.where(offs_pp < HC_MULT, scale_pre, scale_post)

    # Reduce: accumulate K-split partial results by BLOCK_ROW + RMS norm
    for row_base in range(row_start, row_end, BLOCK_ROW):
        offs_row = row_base + tl.arange(0, BLOCK_ROW)
        mask_row = offs_row < row_end

        mm_reduced = tl.zeros([BLOCK_ROW, BLOCK_N], dtype=tl.float32)
        sq_reduced = tl.zeros([BLOCK_ROW], dtype=tl.float32)

        for ki in range(cube_block_dim_k):
            mm_reduced += tl.load(
                mm_out_ptr 
                + ki * stride_mm_k 
                + offs_row[:, None] * stride_mm_row 
                + offs_n[None, :] * stride_mm_col,
                mask=mask_row[:, None],
                other=0.0
            )
            sq_reduced += tl.load(
                sq_out_ptr 
                + ki * stride_sq_k 
                + offs_row * stride_sq_row,
                mask=mask_row,
                other=0.0
            )
            
        # RMS norm: rsqrt(sq_sum / K + eps) * mm_out
        rms_scale = 1.0 / tl.sqrt(sq_reduced / K + NORM_EPS)
        mixes = mm_reduced * rms_scale[:, None]

        if BLOCK_ROW == 1:
            _vector_process_single_row(
                mixes, row_base,
                x_ptr, pre_ptr, post_ptr, comb_ptr, y_ptr,
                stride_x_row, stride_x_col,
                stride_pre_row, stride_pre_col,
                stride_post_row, stride_post_col,
                stride_comb_row, stride_comb_col,
                stride_y_row, stride_y_col,
                scale_pre_post, base_pre_post, scale_comb, base_comb, offs_hc,
                HC_MIX=HC_MIX, HC_MULT=HC_MULT, K=K,
                HC_EPS=HC_EPS, HC_SINKHORN_ITERS=HC_SINKHORN_ITERS,
                BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            )
        else:
            _vector_process_multi_row(
                mixes, offs_row, mask_row,
                x_ptr, pre_ptr, post_ptr, comb_ptr, y_ptr,
                stride_x_row, stride_x_col,
                stride_pre_row, stride_pre_col,
                stride_post_row, stride_post_col,
                stride_comb_row, stride_comb_col,
                stride_y_row, stride_y_col,
                scale_pre_post, base_pre_post, scale_comb, base_comb, offs_hc,
                HC_MIX=HC_MIX, HC_MULT=HC_MULT, K=K,
                HC_EPS=HC_EPS, HC_SINKHORN_ITERS=HC_SINKHORN_ITERS,
                BLOCK_ROW=BLOCK_ROW, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            )


def _hc_pre_tle(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    hc_sinkhorn_iters: int = 20,
):
    input_dim = x.dim()
    if input_dim == 4:
        b, s, hc, d = x.shape
        bs = b * s
    else:
        bs, hc, d = x.shape
        b, s = 1, bs
    assert hc == hc_mult
    K = hc_mult * d
    hc_mix = hc_fn.shape[0]

    x_flat = x.reshape(bs, K)

    if input_dim == 4:
        pre_out = torch.empty((b, s, hc_mult), dtype=torch.float32, device=x.device)
        post_out = torch.empty((b, s, hc_mult), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((b, s, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
        y = torch.empty((b, s, d), dtype=torch.bfloat16, device=x.device)
    else:
        pre_out = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        post_out = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((bs, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
        y = torch.empty((bs, d), dtype=torch.bfloat16, device=x.device)

    # The setting is currently for the 910B, but the 910C can go larger.
    BLOCK_M = 48
    BLOCK_N = 32
    BLOCK_K = 256
    BLOCK_ROW, BLOCK_D = (1, 2048) if bs < 48 else (2, 1024)
    M_L1_MAX_SIZE = 256
    K_SPLIT_BASE_SIZE = 256

    num_cores_cube = get_num_cores("cube")

    # M+K splitting strategy
    cube_block_dim_m = min(num_cores_cube, (bs + M_L1_MAX_SIZE - 1) // M_L1_MAX_SIZE)
    cube_block_dim_k = num_cores_cube // cube_block_dim_m
    split_k_size = (
        (K + cube_block_dim_k - 1) // cube_block_dim_k
        + K_SPLIT_BASE_SIZE - 1
    ) // K_SPLIT_BASE_SIZE * K_SPLIT_BASE_SIZE
    cube_block_dim_k = (K + split_k_size - 1) // split_k_size
    total_programs = cube_block_dim_m * cube_block_dim_k

    # Workspace: layered by K-split
    mm_out = torch.empty((cube_block_dim_k, bs, BLOCK_N), dtype=torch.float32, device=x.device)
    sq_out = torch.empty((cube_block_dim_k, bs), dtype=torch.float32, device=x.device)

    # workspace for CV pipeline: [total_programs, 2, BLOCK_M, BLOCK_K]
    x_ws = torch.empty((total_programs, 2, BLOCK_M, BLOCK_K), dtype=hc_fn.dtype, device=x.device)

    grid_cube = (total_programs,)

    comb_flat = (
        comb_out.reshape(bs, hc_mult * hc_mult)
        if input_dim != 4
        else comb_out.reshape(b * s, hc_mult * hc_mult)
    )
    pre_flat = pre_out.reshape(bs, hc_mult)
    post_flat = post_out.reshape(bs, hc_mult)
    y_flat = y.reshape(bs, d)

    _hc_pre_tle_kernel[grid_cube](
        x_flat, hc_fn,
        mm_out, sq_out,
        x_ws,
        hc_scale, hc_base,
        pre_flat, post_flat, comb_flat, y_flat,
        x_flat.stride(0), x_flat.stride(1),
        hc_fn.stride(0), hc_fn.stride(1),
        mm_out.stride(0), mm_out.stride(1), mm_out.stride(2),
        sq_out.stride(0), sq_out.stride(1),
        pre_flat.stride(0), pre_flat.stride(1),
        post_flat.stride(0), post_flat.stride(1),
        comb_flat.stride(0), comb_flat.stride(1),
        y_flat.stride(0), y_flat.stride(1),
        bs,
        cube_block_dim_m, cube_block_dim_k, split_k_size,
        HC_MIX=hc_mix, HC_MULT=hc_mult, K=K, NORM_EPS=norm_eps, HC_EPS=hc_eps,
        HC_SINKHORN_ITERS=hc_sinkhorn_iters,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        BLOCK_ROW=BLOCK_ROW, BLOCK_D=BLOCK_D,
    )

    return y, post_out, comb_out



# ============================================================================
# DeepSeek V4 hc_pre: Two-stage implementation using only triton primitives.
# Stage 1 (Cube): matmul (x @ hc_fn^T) + row-wise square-sum, M*K split.
# Stage 2 (Vector): K-reduce, RMSNorm, pre/post/comb processing, weighted y.
# No tle primitives used. Relies on tl.dot for cube acceleration.
# ============================================================================


@triton.jit
def _hc_pre_cube_kernel(
    x_ptr,
    hc_fn_ptr,
    mm_out_ptr,
    sq_out_ptr,
    stride_x_row,
    stride_x_col,
    stride_fn_row,
    stride_fn_col,
    stride_mm_k,
    stride_mm_row,
    stride_mm_col,
    stride_sq_k,
    stride_sq_row,
    bs,
    K: tl.constexpr,
    HC_MIX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    cube_block_dim_k: tl.constexpr,
    split_k_size: tl.constexpr,
):
    pid = tl.program_id(0)
    m_dim_idx = pid // cube_block_dim_k
    k_dim_idx = pid % cube_block_dim_k

    k_range_start = k_dim_idx * split_k_size
    k_range_end = tl.minimum(k_range_start + split_k_size, K)

    cube_block_dim_m: tl.constexpr = tl.num_programs(0) // cube_block_dim_k
    num_m_blocks = (bs + BLOCK_M - 1) // BLOCK_M
    single_core_m_blocks = (num_m_blocks + cube_block_dim_m - 1) // cube_block_dim_m
    m_block_start = m_dim_idx * single_core_m_blocks
    m_block_end = tl.minimum(m_block_start + single_core_m_blocks, num_m_blocks)

    offs_n = tl.arange(0, BLOCK_N)

    for block_iter in range(m_block_start, m_block_end):
        m_start = block_iter * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < bs

        mm_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        sq_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        for k_start in tl.range(k_range_start, k_range_end, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x_offs = offs_m[:, None] * stride_x_row + offs_k[None, :] * stride_x_col
            x_mask = mask_m[:, None] & k_mask[None, :]
            x_block = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(hc_fn_ptr.dtype.element_ty)

            sq_acc += tl.sum(x_block * x_block, axis=1)

            fn_offs = offs_n[:, None] * stride_fn_row + offs_k[None, :] * stride_fn_col
            fn_mask = (offs_n[:, None] < HC_MIX) & k_mask[None, :]
            fn_block = tl.load(hc_fn_ptr + fn_offs, mask=fn_mask, other=0.0)

            mm_acc = tl.dot(x_block, tl.trans(fn_block), acc=mm_acc, out_dtype=tl.float32)

        # Store partial mm_out and sq_out
        mm_store_offs = (k_dim_idx * stride_mm_k
                         + offs_m[:, None] * stride_mm_row
                         + offs_n[None, :] * stride_mm_col)
        tl.store(mm_out_ptr + mm_store_offs, mm_acc,
                 mask=mask_m[:, None] & (offs_n[None, :] < HC_MIX))
        tl.store(sq_out_ptr + k_dim_idx * stride_sq_k + offs_m * stride_sq_row,
                 sq_acc, mask=mask_m)


@triton.jit
def _hc_pre_vector_kernel(
    x_ptr,
    mm_out_ptr,
    sq_out_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_mm_k,
    stride_mm_row,
    stride_mm_col,
    stride_sq_k,
    stride_sq_row,
    stride_pre_row,
    stride_pre_col,
    stride_post_row,
    stride_post_col,
    stride_comb_row,
    stride_comb_col,
    stride_y_row,
    stride_y_col,
    bs,
    cube_block_dim_k: tl.constexpr,
    K: tl.constexpr,
    HC_MIX: tl.constexpr,
    HC_MULT: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    HC_SINKHORN_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)

    hc_scale_vec = tl.load(hc_scale_ptr + tl.arange(0, 3))
    scale_pre = tl.get_element(hc_scale_vec, (0,))
    scale_post = tl.get_element(hc_scale_vec, (1,))
    scale_comb = tl.get_element(hc_scale_vec, (2,))

    PP: tl.constexpr = 2 * HC_MULT
    HC_COMB: tl.constexpr = HC_MULT * HC_MULT

    offs_hc = tl.arange(0, HC_MULT)
    base_pre = tl.load(hc_base_ptr + offs_hc)
    base_post = tl.load(hc_base_ptr + HC_MULT + offs_hc)
    offs_comb = tl.arange(0, HC_COMB)
    base_comb = tl.load(hc_base_ptr + PP + offs_comb)

    pre_reduced = tl.zeros([HC_MULT], dtype=tl.float32)
    post_reduced = tl.zeros([HC_MULT], dtype=tl.float32)
    comb_reduced = tl.zeros([HC_COMB], dtype=tl.float32)
    sq_reduced = 0.0

    post_col_offs = HC_MULT + offs_hc
    comb_col_offs = PP + offs_comb

    for k_idx in range(cube_block_dim_k):
        base_off = k_idx * stride_mm_k + row_idx * stride_mm_row

        pre_part = tl.load(mm_out_ptr + base_off + offs_hc * stride_mm_col)
        pre_reduced += pre_part

        post_part = tl.load(mm_out_ptr + base_off + post_col_offs * stride_mm_col)
        post_reduced += post_part

        comb_part = tl.load(mm_out_ptr + base_off + comb_col_offs * stride_mm_col)
        comb_reduced += comb_part

        sq_part = tl.load(sq_out_ptr + k_idx * stride_sq_k + row_idx * stride_sq_row)
        sq_reduced += sq_part

    # RMSNorm
    rms_scale = 1.0 / tl.sqrt(sq_reduced / K + NORM_EPS)
    pre_raw = pre_reduced * rms_scale
    post_raw = post_reduced * rms_scale
    comb_raw = comb_reduced * rms_scale

    # Pre: sigmoid(pre_raw * scale_pre + base_pre) + eps
    pre_out = tl.sigmoid(pre_raw * scale_pre + base_pre) + HC_EPS

    # Post: 2 * sigmoid(post_raw * scale_post + base_post)
    post_out = 2.0 * tl.sigmoid(post_raw * scale_post + base_post)

    # Store pre/post
    tl.store(pre_ptr + row_idx * stride_pre_row + offs_hc * stride_pre_col, pre_out)
    tl.store(post_ptr + row_idx * stride_post_row + offs_hc * stride_post_col, post_out)

    # Comb: affine -> reshape -> pad -> softmax -> mask -> Sinkhorn
    comb = comb_raw * scale_comb + base_comb
    comb_2d = tl.reshape(comb, [HC_MULT, HC_MULT])

    # pad to 32-byte aligned: 32 / sizeof(float32) = 8
    PADDED: tl.constexpr = ((HC_MULT + 7) // 8) * 8
    NEG_INF: tl.constexpr = float('-inf')
    comb_padded = tl.full([PADDED, PADDED], NEG_INF, dtype=tl.float32)
    comb_padded = tl.insert_slice(comb_padded, comb_2d, [0, 0], [HC_MULT, HC_MULT], [1, 1])

    # Softmax on axis=1 (row-wise)
    max_val = tl.max(comb_padded, axis=1)
    max_val = tl.where(max_val == NEG_INF, 0.0, max_val)
    comb_padded = comb_padded - max_val[:, None]
    exp_val = tl.exp(comb_padded)
    sum_val = tl.sum(exp_val, axis=1) + HC_EPS

    # Valid mask: zero out padding positions
    _offs_r = tl.arange(0, PADDED)
    _offs_c = tl.arange(0, PADDED)
    _valid_mask = tl.where((_offs_r[:, None] < HC_MULT) & (_offs_c[None, :] < HC_MULT), 1.0, 0.0)
    comb_sk = exp_val / sum_val[:, None] + HC_EPS
    comb_sk = comb_sk * _valid_mask

    # Sinkhorn: alternate col-norm and row-norm
    col_sum = tl.sum(comb_sk, axis=0) + HC_EPS
    comb_sk = comb_sk / col_sum[None, :]

    for _si in range(HC_SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb_sk, axis=1) + HC_EPS
        comb_sk = comb_sk / row_sum[:, None]
        col_sum = tl.sum(comb_sk, axis=0) + HC_EPS
        comb_sk = comb_sk / col_sum[None, :]

    # Extract valid [HC_MULT, HC_MULT] and store
    comb_sk_valid = tl.extract_slice(comb_sk, [0, 0], [HC_MULT, HC_MULT], [1, 1])
    comb_sk_flat = tl.reshape(comb_sk_valid, [HC_MULT * HC_MULT])
    tl.store(comb_ptr + row_idx * stride_comb_row + offs_comb * stride_comb_col, comb_sk_flat)

    # ProcessY: y[d] = sum_h(pre[h] * x[h, d])
    D: tl.constexpr = K // HC_MULT
    pre_2d = tl.reshape(pre_out, [HC_MULT, 1])

    for d_start in range(0, D, BLOCK_D):
        d_offset = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offset < D

        h_offs = tl.arange(0, HC_MULT)
        x_offs = (row_idx * stride_x_row
                  + (h_offs[:, None] * D + d_offset[None, :]) * stride_x_col)
        x_block = tl.load(x_ptr + x_offs, mask=d_mask[None, :], other=0.0).to(tl.float32)

        y_acc = tl.sum(pre_2d * x_block, axis=0)

        y_offs = row_idx * stride_y_row + d_offset * stride_y_col
        tl.store(y_ptr + y_offs, y_acc.to(y_ptr.dtype.element_ty), mask=d_mask)


def _hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    hc_sinkhorn_iters: int = 20,
):
    input_dim = x.dim()
    if input_dim == 4:
        b, s, hc, d = x.shape
        bs = b * s
    else:
        bs, hc, d = x.shape
        b, s = 1, bs
    assert hc == hc_mult
    K = hc_mult * d
    hc_mix = hc_fn.shape[0]

    x_flat = x.reshape(bs, K)

    if input_dim == 4:
        pre_out = torch.empty((b, s, hc_mult), dtype=torch.float32, device=x.device)
        post_out = torch.empty((b, s, hc_mult), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((b, s, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
        y = torch.empty((b, s, d), dtype=torch.bfloat16, device=x.device)
    else:
        pre_out = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        post_out = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((bs, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
        y = torch.empty((bs, d), dtype=torch.bfloat16, device=x.device)

    # Tiling parameters (910B baseline, 910C can go larger)
    BLOCK_M = 48
    BLOCK_N = 32
    BLOCK_K = 256
    BLOCK_D = 2048 if bs < 48 else 1024
    M_L1_MAX_SIZE = 256
    K_SPLIT_BASE_SIZE = 256

    num_cores_cube = get_num_cores("cube")

    # M+K splitting strategy
    cube_block_dim_m = min(num_cores_cube, (bs + M_L1_MAX_SIZE - 1) // M_L1_MAX_SIZE)
    cube_block_dim_k = num_cores_cube // cube_block_dim_m
    split_k_size = (
        (K + cube_block_dim_k - 1) // cube_block_dim_k
        + K_SPLIT_BASE_SIZE - 1
    ) // K_SPLIT_BASE_SIZE * K_SPLIT_BASE_SIZE
    cube_block_dim_k = (K + split_k_size - 1) // split_k_size
    total_programs = cube_block_dim_m * cube_block_dim_k

    # Workspace for K-split partial results
    mm_out = torch.empty((cube_block_dim_k, bs, BLOCK_N), dtype=torch.float32, device=x.device)
    sq_out = torch.empty((cube_block_dim_k, bs), dtype=torch.float32, device=x.device)

    # Flatten output views
    comb_flat = comb_out.reshape(bs, hc_mult * hc_mult)
    pre_flat = pre_out.reshape(bs, hc_mult)
    post_flat = post_out.reshape(bs, hc_mult)
    y_flat = y.reshape(bs, d)

    # Stage 1: Cube matmul kernel
    grid_cube = (total_programs,)
    _hc_pre_cube_kernel[grid_cube](
        x_flat, hc_fn,
        mm_out, sq_out,
        x_flat.stride(0), x_flat.stride(1),
        hc_fn.stride(0), hc_fn.stride(1),
        mm_out.stride(0), mm_out.stride(1), mm_out.stride(2),
        sq_out.stride(0), sq_out.stride(1),
        bs,
        K=K, HC_MIX=hc_mix,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        cube_block_dim_k=cube_block_dim_k,
        split_k_size=split_k_size,
    )

    # Stage 2: Vector post-processing kernel
    grid_vec = (bs,)

    _hc_pre_vector_kernel[grid_vec](
        x_flat,
        mm_out, sq_out,
        hc_scale, hc_base,
        pre_flat, post_flat, comb_flat, y_flat,
        x_flat.stride(0), x_flat.stride(1),
        mm_out.stride(0), mm_out.stride(1), mm_out.stride(2),
        sq_out.stride(0), sq_out.stride(1),
        pre_flat.stride(0), pre_flat.stride(1),
        post_flat.stride(0), post_flat.stride(1),
        comb_flat.stride(0), comb_flat.stride(1),
        y_flat.stride(0), y_flat.stride(1),
        bs,
        cube_block_dim_k=cube_block_dim_k,
        K=K, HC_MIX=hc_mix, HC_MULT=hc_mult,
        NORM_EPS=norm_eps, HC_EPS=hc_eps,
        HC_SINKHORN_ITERS=hc_sinkhorn_iters,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return y, post_out, comb_out


def hc_pre_impl(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    hc_sinkhorn_iters: int = 20,
):
    if HAS_TLE:
        return _hc_pre_tle(
            x, hc_fn, hc_scale, hc_base,
            hc_mult=hc_mult, norm_eps=norm_eps,
            hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
        )
    else:
        return _hc_pre(
            x, hc_fn, hc_scale, hc_base,
            hc_mult=hc_mult, norm_eps=norm_eps,
            hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
        )

