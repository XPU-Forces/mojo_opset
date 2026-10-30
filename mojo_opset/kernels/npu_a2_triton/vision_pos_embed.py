"""Ascend Triton fused bilinear vision position-embedding interpolation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_pos_embed_interp_kernel(
    embed_ptr,
    output_ptr,
    H,
    W,
    T,
    h_scale,
    w_scale,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_spatial = H * W
    j = tl.arange(0, ROWS)
    s = pid * ROWS + j
    block = s // (M_SIZE * M_SIZE)
    local = s % (M_SIZE * M_SIZE)
    lr = local // M_SIZE
    lc = local % M_SIZE
    num_blocks_w = W // M_SIZE
    br = block // num_blocks_w
    bc = block % num_blocks_w
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale
    hf = h_frac.to(tl.int32)
    wf = w_frac.to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM

    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    cols = tl.arange(0, BLOCK_D)
    mask = cols[None, :] < HIDDEN_DIM
    e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :], mask=mask)
    e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :], mask=mask)
    e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :], mask=mask)
    e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :], mask=mask)
    val = w00_c[:, None] * e00 + w01_c[:, None] * e01 + w10_c[:, None] * e10 + w11_c[:, None] * e11

    for f in range(T):
        out_off = (f * total_spatial + s[:, None]) * HIDDEN_DIM + cols[None, :]
        tl.store(output_ptr + out_off, val, mask=mask)


@triton.jit
def _vision_pos_embed_interp_multi_kernel(
    embed_ptr,
    output_ptr,
    cum_out_ptr,
    cum_sp_ptr,
    t_ptr,
    h_ptr,
    w_ptr,
    hs_ptr,
    ws_ptr,
    NUM_GRIDS: tl.constexpr,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    g = 0
    for i in range(1, NUM_GRIDS):
        ci = tl.load(cum_sp_ptr + i)
        g = tl.where(pid >= ci, i, g)

    sp_start = tl.load(cum_sp_ptr + g)
    out_start = tl.load(cum_out_ptr + g)
    T = tl.load(t_ptr + g)
    H = tl.load(h_ptr + g)
    W = tl.load(w_ptr + g)
    h_scale = tl.load(hs_ptr + g)
    w_scale = tl.load(ws_ptr + g)

    local = pid - sp_start
    j = tl.arange(0, 1)
    local_v = local + j
    total_spatial = H * W
    num_blocks_w = W // M_SIZE
    block = local_v // (M_SIZE * M_SIZE)
    inner = local_v % (M_SIZE * M_SIZE)
    lr = inner // M_SIZE
    lc = inner % M_SIZE
    br = block // num_blocks_w
    bc = block % num_blocks_w
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale
    hf = h_frac.to(tl.int32)
    wf = w_frac.to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM

    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    cols = tl.arange(0, BLOCK_D)
    mask = cols[None, :] < HIDDEN_DIM
    e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :], mask=mask)
    e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :], mask=mask)
    e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :], mask=mask)
    e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :], mask=mask)
    val = w00_c[:, None] * e00 + w01_c[:, None] * e01 + w10_c[:, None] * e10 + w11_c[:, None] * e11

    for f in range(T):
        out_off = (out_start + f * total_spatial + local_v[:, None]) * HIDDEN_DIM + cols[None, :]
        tl.store(output_ptr + out_off, val, mask=mask)


def _rows_per_prog(spatial):
    if spatial >= 8192 and spatial % 4 == 0:
        return 4
    return 1


def _launch_single(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    spatial = h * w
    rows = _rows_per_prog(spatial)
    output = torch.empty(t * spatial, hidden_dim, device=embed_weight.device, dtype=dtype)
    h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
    w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0
    block_d = triton.next_power_of_2(hidden_dim)
    _vision_pos_embed_interp_kernel[(spatial // rows,)](
        embed_weight,
        output,
        h,
        w,
        t,
        h_scale,
        w_scale,
        num_grid_per_side,
        m_size,
        hidden_dim,
        block_d,
        rows,
    )
    return output


def _launch_multi(embed_weight, grid_thw, num_grid_per_side, m_size, dtype):
    hidden_dim = embed_weight.shape[1]
    dev = embed_weight.device
    cum_out, cum_sp = [0], [0]
    ts, hs, ws = [], [], []
    for t, h, w in grid_thw:
        assert h % m_size == 0 and w % m_size == 0
        cum_out.append(cum_out[-1] + t * h * w)
        cum_sp.append(cum_sp[-1] + h * w)
        ts.append(t)
        hs.append(float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0)
        ws.append(float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0)

    output = torch.empty(cum_out[-1], hidden_dim, device=dev, dtype=dtype)

    def _t(vals, dt):
        return torch.tensor(vals, dtype=dt, device=dev)

    block_d = triton.next_power_of_2(hidden_dim)
    _vision_pos_embed_interp_multi_kernel[(cum_sp[-1],)](
        embed_weight,
        output,
        _t(cum_out, torch.int32),
        _t(cum_sp, torch.int32),
        _t(ts, torch.int32),
        _t([g[1] for g in grid_thw], torch.int32),
        _t([g[2] for g in grid_thw], torch.int32),
        _t(hs, torch.float32),
        _t(ws, torch.float32),
        len(grid_thw),
        num_grid_per_side,
        m_size,
        hidden_dim,
        block_d,
    )
    return output


def vision_pos_embed_interp_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return _launch_single(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


def vision_pos_embed_interp_multi_fwd(embed_weight, grid_thw, num_grid_per_side, m_size, dtype):
    return _launch_multi(embed_weight, grid_thw, num_grid_per_side, m_size, dtype)
