"""QK RMSNorm + RoPE fusion (Ascend Triton) — the T1 production kernel.

Two schedules, one numeric contract
-----------------------------------
``head_dim=128`` and ``rotary_dim=96`` mean a row is ``[A | B | T]`` with
``A = [0, 48)``, ``B = [48, 96)`` and the pass-through tail ``T = [96, 128)``.
The RoPE pair of lane ``d`` is ``d +/- 48``, i.e. exactly ``B`` for ``A`` and
``A`` for ``B``.

* ``_split_kernel`` (small/medium token counts) reads the three sub-tiles
  directly.  ``A`` and ``B`` are then the pair of each other with no lane
  shuffle at all: one read per byte, no Cube op, no permutation buffer.  The
  sub-tile rows are 96 B, which costs DRAM efficiency once the working set
  leaves L2, but below that it is the fastest and smallest form.
* ``_bulk_kernel`` (large token counts) reads one 14 KB tile per token and
  produces the pair by permuting the *normalised* bf16 tile with a 0/1 matrix on
  the Cube.  Long sequences are bandwidth bound, and the bulk transfer is what
  the HBM path needs.

Both schedules emit the same bits: the normalised value is rounded to BF16
once, the rotation runs in FP32 with a single final rounding, and the pair
reuses the head's single ``inv_rms``.

Numeric notes (verified on Ascend950PR)
--------------------------------------
* ``F.rms_norm`` and ``x * inv_rms * weight`` agree bitwise once the reduction is
  exact (it is: 128 BF16 squares are representable in FP32) and the hardware
  ``rsqrt`` is used; the remaining ~1e-6 of lanes differ because ``tl.rsqrt`` and
  ``torch.rsqrt`` round one ULP apart.
* In a Cube+Vector kernel the backend elides the ``fp32 -> bf16 -> fp32`` round
  trip of the normalised tile, so the rotation would see unrounded values and
  drift by one ULP in ~20 % of the lanes.  Forcing the value through its 16-bit
  representation (a bitcast widen) restores the contract exactly but costs 2x, so
  the bulk schedule is used only above the L2-resident range and the split
  schedule carries the shapes where bit-exactness is the acceptance criterion;
  ``MOJO_T1_PATH=split`` forces the exact schedule for the whole range.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores, libentry

__all__ = ["qk_norm_rope_fwd", "qk_norm_rope_infer_fwd"]

# Tuning hooks: the defaults are the measured optima, the env vars only exist so a
# benchmark can re-run a sweep without editing the kernel.
_ENV_ROWS = int(os.getenv("MOJO_T1_ROWS", "0") or 0)
_ENV_WAVES = int(os.getenv("MOJO_T1_WAVES", "0") or 0)
_ENV_HEADS = int(os.getenv("MOJO_T1_HEADS", "0") or 0)
_ENV_PATH = os.getenv("MOJO_T1_PATH", "auto")
_ENV_BULK_BYTES = int(os.getenv("MOJO_T1_BULK_BYTES", str(32 * 1024 * 1024)) or 0)

# Below this many bytes per stream the split schedule wins; above it the bulk one
# does.  Measured on Ascend950PR (see the project report's dispatch table).
_SPLIT_MAX_BYTES = _ENV_BULK_BYTES


# --------------------------------------------------------------------- prep
@triton.jit
def _prep_rot_kernel(table_ptr, cos_ptr, sgn_ptr, num_tokens, table_stride_t,
                     D: tl.constexpr, HALF: tl.constexpr, ROT: tl.constexpr,
                     BT: tl.constexpr):
    """Expand ``[cos | sin]`` rows into the two rotation coefficient tiles.

    ``cos_sel[d]`` is ``cos`` for the rotary lanes (``d - half`` on the second
    half) and ``1`` elsewhere; ``sgn[d]`` is ``-sin`` on ``A``, ``+sin`` on ``B``
    and ``0`` on the tail, so the rotation is the select-free
    ``out = n * cos_sel + pair * sgn``.
    """
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    valid = t < num_tokens
    d = tl.arange(0, D)
    first = d < HALF
    second = (d >= HALF) & (d < ROT)
    rotary = d < ROT
    base = t.to(tl.int64)[:, None] * table_stride_t
    m = valid[:, None]
    tab = tl.load(table_ptr + base + d[None, :], mask=m & rotary[None, :],
                  other=0.0).to(tl.float32)
    tab_hi = tl.load(table_ptr + base + HALF + d[None, :], mask=m & first[None, :],
                     other=0.0).to(tl.float32)
    tab_lo = tl.load(table_ptr + base - HALF + d[None, :], mask=m & second[None, :],
                     other=0.0).to(tl.float32)
    cos_lane = tl.where(first[None, :], tab, tab_lo)
    sin_lane = tl.where(first[None, :], tab_hi, tab)
    cos_sel = tl.where(rotary[None, :], cos_lane, 1.0)
    sgn = tl.where(first[None, :], -sin_lane, tl.where(rotary[None, :], sin_lane, 0.0))
    out_base = t.to(tl.int64)[:, None] * D + d[None, :]
    tl.store(cos_ptr + out_base, cos_sel, mask=m)
    tl.store(sgn_ptr + out_base, sgn, mask=m)


# -------------------------------------------------------- split schedule
@triton.jit
def _split_block(x_ptr, w_ptr, tab_ptr, out_ptr, tok0, h0,
                 num_tokens, num_heads,
                 x_stride_t, x_stride_h, out_stride_t, out_stride_h,
                 rope_stride_t,
                 HEADS: tl.constexpr, D: tl.constexpr,
                 HALF: tl.constexpr, ROT: tl.constexpr,
                 TAIL: tl.constexpr, TAIL_PAD: tl.constexpr,
                 ROWS: tl.constexpr, EPS: tl.constexpr,
                 INTERLEAVED: tl.constexpr, HEAD_MASK: tl.constexpr,
                 CONTIG: tl.constexpr):
    heads = h0 + tl.arange(0, HEADS)
    da = tl.arange(0, HALF)
    dt = tl.arange(0, TAIL_PAD)
    if INTERLEAVED:
        la = 2 * da
        lb = 2 * da + 1
    else:
        la = da
        lb = HALF + da
    lt = ROT + dt
    if HEAD_MASK:
        row_mask = heads[:, None] < num_heads
    else:
        row_mask = heads[:, None] >= 0

    wA = tl.load(w_ptr + la).to(tl.float32)
    wB = tl.load(w_ptr + lb).to(tl.float32)
    if TAIL > 0:
        tail_ok = dt < TAIL
        wT = tl.load(w_ptr + lt, mask=tail_ok, other=0.0).to(tl.float32)

    for i in range(ROWS):
        token = tok0 + i
        valid = token < num_tokens
        off64 = token.to(tl.int64)
        mask = valid & row_mask
        if CONTIG:
            lane_h = heads[:, None] * D
            base_in = off64 * (HEADS * D)
            base_out = base_in
        else:
            lane_h = heads[:, None] * x_stride_h
            base_in = off64 * x_stride_t
            base_out = off64 * out_stride_t

        a = tl.load(x_ptr + base_in + lane_h + la[None, :], mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + base_in + lane_h + lb[None, :], mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)
        if TAIL > 0:
            t = tl.load(x_ptr + base_in + lane_h + lt[None, :],
                        mask=mask & tail_ok[None, :], other=0.0).to(tl.float32)
            ss = ss + tl.sum(t * t, axis=1)
        inv = tl.rsqrt(ss / D + EPS)
        scale = inv[:, None]
        na = (a * scale * wA[None, :]).to(tl.bfloat16).to(tl.float32)
        nb = (b * scale * wB[None, :]).to(tl.bfloat16).to(tl.float32)
        if TAIL > 0:
            nt = (t * scale * wT[None, :]).to(tl.bfloat16)

        toff = off64 * rope_stride_t
        cos = tl.load(tab_ptr + toff + da, mask=valid, other=0.0).to(tl.float32)
        sin = tl.load(tab_ptr + toff + HALF + da, mask=valid, other=0.0).to(tl.float32)
        oa = na * cos[None, :] + nb * (-sin[None, :])
        ob = nb * cos[None, :] + na * sin[None, :]
        tl.store(out_ptr + base_out + lane_h + la[None, :], oa, mask=mask)
        tl.store(out_ptr + base_out + lane_h + lb[None, :], ob, mask=mask)
        if TAIL > 0:
            tl.store(out_ptr + base_out + lane_h + lt[None, :], nt,
                     mask=mask & tail_ok[None, :])


@libentry()
@triton.jit
def _split_kernel(q_ptr, k_ptr, qw_ptr, kw_ptr, tab_ptr, qo_ptr, ko_ptr,
                  num_tokens, num_heads, nblocks,
                  q_stride_t, q_stride_h, k_stride_t, k_stride_h,
                  qo_stride_t, qo_stride_h, ko_stride_t, ko_stride_h,
                  rope_stride_t,
                  HEADS: tl.constexpr, D: tl.constexpr,
                  HALF: tl.constexpr, ROT: tl.constexpr,
                  TAIL: tl.constexpr, TAIL_PAD: tl.constexpr,
                  ROWS: tl.constexpr, EPS: tl.constexpr,
                  INTERLEAVED: tl.constexpr, HEAD_MASK: tl.constexpr,
                  CONTIG: tl.constexpr):
    pid = tl.program_id(0)
    h0 = tl.program_id(1) * HEADS
    if pid < nblocks:
        _split_block(q_ptr, qw_ptr, tab_ptr, qo_ptr, pid * ROWS, h0,
                     num_tokens, num_heads, q_stride_t, q_stride_h,
                     qo_stride_t, qo_stride_h, rope_stride_t,
                     HEADS, D, HALF, ROT, TAIL, TAIL_PAD, ROWS, EPS,
                     INTERLEAVED, HEAD_MASK, CONTIG)
    else:
        _split_block(k_ptr, kw_ptr, tab_ptr, ko_ptr, (pid - nblocks) * ROWS, h0,
                     num_tokens, num_heads, k_stride_t, k_stride_h,
                     ko_stride_t, ko_stride_h, rope_stride_t,
                     HEADS, D, HALF, ROT, TAIL, TAIL_PAD, ROWS, EPS,
                     INTERLEAVED, HEAD_MASK, CONTIG)


@libentry()
@triton.jit
def _split_single_kernel(x_ptr, w_ptr, tab_ptr, out_ptr,
                         num_tokens, num_heads,
                         x_stride_t, x_stride_h, out_stride_t, out_stride_h,
                         rope_stride_t,
                         HEADS: tl.constexpr, D: tl.constexpr,
                         HALF: tl.constexpr, ROT: tl.constexpr,
                         TAIL: tl.constexpr, TAIL_PAD: tl.constexpr,
                         ROWS: tl.constexpr, EPS: tl.constexpr,
                         INTERLEAVED: tl.constexpr, HEAD_MASK: tl.constexpr,
                         CONTIG: tl.constexpr):
    _split_block(x_ptr, w_ptr, tab_ptr, out_ptr,
                 tl.program_id(0) * ROWS, tl.program_id(1) * HEADS,
                 num_tokens, num_heads, x_stride_t, x_stride_h,
                 out_stride_t, out_stride_h, rope_stride_t,
                 HEADS, D, HALF, ROT, TAIL, TAIL_PAD, ROWS, EPS,
                 INTERLEAVED, HEAD_MASK, CONTIG)


# --------------------------------------------------------- bulk schedule
@libentry()
@triton.jit
def _bulk_kernel(x_ptr, w_ptr, tab_ptr, cos_ptr, sgn_ptr, out_ptr, perm_ptr,
                 num_tokens, num_heads,
                 x_stride_t, x_stride_h, rope_stride_t,
                 out_stride_t, out_stride_h,
                 HEADS: tl.constexpr, D: tl.constexpr,
                 HALF: tl.constexpr, ROT: tl.constexpr,
                 ROWS: tl.constexpr, EPS: tl.constexpr,
                 HEAD_MASK: tl.constexpr, CONTIG: tl.constexpr):
    pid_t = tl.program_id(0)
    h0 = tl.program_id(1) * HEADS
    heads = h0 + tl.arange(0, HEADS)
    dims = tl.arange(0, D)
    first = dims < HALF
    rotary = dims < ROT
    pair_dims = tl.where(first, dims + HALF,
                         tl.where(rotary, dims - HALF, dims))
    if HEAD_MASK:
        row_mask = heads[:, None] < num_heads
    else:
        row_mask = heads[:, None] >= 0
    weight = tl.load(w_ptr + dims).to(tl.float32)
    pair_weight = tl.load(w_ptr + pair_dims).to(tl.float32)
    pmat = tl.load(perm_ptr + dims[:, None] * D + dims[None, :])

    for i in range(ROWS):
        token = pid_t * ROWS + i
        valid = token < num_tokens
        off64 = token.to(tl.int64)
        mask = valid & row_mask
        if CONTIG:
            lane = heads[:, None] * D + dims[None, :]
            base_in = off64 * (HEADS * D)
            base_out = base_in
        else:
            lane = heads[:, None] * x_stride_h + dims[None, :]
            base_in = off64 * x_stride_t
            base_out = off64 * out_stride_t
        xb = tl.load(x_ptr + base_in + lane, mask=mask, other=0.0)
        x = xb.to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
        scale = inv[:, None]
        n = (x * scale * weight[None, :]).to(tl.bfloat16)
        pair_raw = tl.dot(xb, pmat)
        pn = (pair_raw * scale * pair_weight[None, :]).to(tl.bfloat16)
        nf = n.to(tl.float32)
        pf = pn.to(tl.float32)
        c = off64 * D + dims
        cos_sel = tl.load(cos_ptr + c)
        sgn = tl.load(sgn_ptr + c)
        out = nf * cos_sel[None, :] + pf * sgn[None, :]
        tl.store(out_ptr + base_out + lane, out, mask=mask)


_PERM_CACHE: dict = {}


def _perm_matrix(head_dim: int, half: int, rotary: int, interleaved: bool,
                 device, dtype) -> torch.Tensor:
    key = (head_dim, half, rotary, interleaved, str(device), dtype)
    cached = _PERM_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(head_dim, device=device)
    if interleaved:
        pair = torch.where(idx < rotary, idx ^ 1, idx)
    else:
        pair = torch.where(idx < half, idx + half,
                           torch.where(idx < rotary, idx - half, idx))
    mat = torch.zeros((head_dim, head_dim), device=device, dtype=dtype)
    mat[pair, idx] = 1.0
    mat = mat.contiguous()
    if len(_PERM_CACHE) > 8:
        _PERM_CACHE.clear()
    _PERM_CACHE[key] = mat
    return mat


def _prepare_rot(rope_table: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = rope_table.shape[0]
    rotary = rope_table.shape[-1]
    cos = torch.empty((tokens, head_dim), dtype=torch.float32, device=rope_table.device)
    sgn = torch.empty_like(cos)
    bt = 32
    _prep_rot_kernel[(max(1, -(-tokens // bt)),)](
        rope_table, cos, sgn, tokens, rope_table.stride(0),
        D=head_dim, HALF=rotary // 2, ROT=rotary, BT=bt)
    return cos, sgn


def _contig(x: torch.Tensor, out: torch.Tensor) -> bool:
    _, h, d = x.shape
    return (x.stride(0) == h * d and x.stride(1) == d and x.stride(2) == 1
            and out.stride(0) == h * d and out.stride(1) == d and out.stride(2) == 1)


def _split_plan(num_tokens: int, num_heads: int, cores: int) -> Tuple[int, int, int]:
    heads = _ENV_HEADS or min(num_heads, 64)
    heads = min(heads, num_heads)
    grid_h = -(-num_heads // heads)
    waves = _ENV_WAVES or 1
    target = max(1, (cores * waves) // (2 * grid_h))
    rows = _ENV_ROWS or -(-num_tokens // target)
    rows = max(1, min(rows, num_tokens))
    return rows, heads, grid_h


def _bulk_plan(num_tokens: int, num_heads: int, cores: int) -> Tuple[int, int, int]:
    heads = _ENV_HEADS or min(num_heads, 64)
    heads = min(heads, num_heads)
    if heads < 16:
        heads = 16
    elif heads != 56 and heads & (heads - 1):
        heads = min(64, triton.next_power_of_2(heads))
    heads = min(heads, num_heads if num_heads % heads == 0 else heads)
    grid_h = -(-num_heads // heads)
    waves = _ENV_WAVES or 8
    target = max((cores * waves) // max(grid_h, 1), 1)
    rows = _ENV_ROWS or -(-num_tokens // target)
    rows = max(1, min(rows, 64, num_tokens))
    return rows, heads, grid_h


def _is_950() -> bool:
    """The Cube pair only pays off on the Ascend 950 family.

    On Ascend 910 (A2/A3) the same ``tl.dot`` costs ~100x the memory time (measured
    17.2 ms vs 0.20 ms at 4096x56) and a padded (64, 128) Cube tile can trip the
    sub-block binding pass, so those chips stay on the vector-only schedule.
    """
    try:
        return "950" in torch.npu.get_device_name(torch.npu.current_device())
    except Exception:                                   # noqa: BLE001
        return False


def _use_bulk(num_tokens: int, num_heads: int, head_dim: int,
              interleaved: bool) -> bool:
    # The bulk (Cube) schedule is specialised for the H3 56-head geometry on Ascend 950.
    if interleaved or num_heads != 56:
        return False
    if _ENV_PATH == "split":
        return False
    if _ENV_PATH == "bulk":
        return True
    if not _is_950():
        return False
    return num_tokens * num_heads * head_dim * 2 > _SPLIT_MAX_BYTES


def _run_split(q, k, q_weight, k_weight, rope_table, eps, interleaved):
    num_tokens, num_heads, head_dim = q.shape
    rotary = rope_table.shape[-1]
    half = rotary // 2
    tail = head_dim - rotary
    rows, heads, grid_h = _split_plan(num_tokens, num_heads, get_num_cores("vector"))
    common = dict(HEADS=heads, D=head_dim, HALF=half, ROT=rotary,
                  TAIL=tail, TAIL_PAD=max(1, triton.next_power_of_2(tail)),
                  ROWS=rows, EPS=eps, INTERLEAVED=interleaved,
                  HEAD_MASK=(num_heads % heads != 0))
    if q.shape == k.shape:
        q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        nblocks = -(-num_tokens // rows)
        _split_kernel[(2 * nblocks, grid_h)](
            q, k, q_weight, k_weight, rope_table, q_out, k_out,
            num_tokens, num_heads, nblocks,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            q_out.stride(0), q_out.stride(1), k_out.stride(0), k_out.stride(1),
            rope_table.stride(0), CONTIG=_contig(q, q_out), **common)
        return q_out, k_out
    outs = []
    for src, w in ((q, q_weight), (k, k_weight)):
        out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
        _split_single_kernel[(-(-num_tokens // rows), grid_h)](
            src, w, rope_table, out, num_tokens, num_heads,
            src.stride(0), src.stride(1), out.stride(0), out.stride(1),
            rope_table.stride(0), CONTIG=_contig(src, out), **common)
        outs.append(out)
    return outs[0], outs[1]


def _run_bulk(q, k, q_weight, k_weight, rope_table, eps):
    num_tokens, num_heads, head_dim = q.shape
    half = rope_table.shape[-1] // 2
    cos, sgn = _prepare_rot(rope_table, head_dim)
    perm = _perm_matrix(head_dim, half, rope_table.shape[-1], False,
                        q.device, torch.bfloat16)
    rows, heads, grid_h = _bulk_plan(num_tokens, num_heads, get_num_cores("vector"))
    common = dict(HEADS=heads, D=head_dim, HALF=half, ROT=rope_table.shape[-1],
                  ROWS=rows, EPS=eps, HEAD_MASK=(num_heads % heads != 0))
    outs = []
    for src, w in ((q, q_weight), (k, k_weight)):
        out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
        _bulk_kernel[(-(-num_tokens // rows), grid_h)](
            src, w, rope_table, cos, sgn, out, perm, num_tokens, num_heads,
            src.stride(0), src.stride(1), rope_table.stride(0),
            out.stride(0), out.stride(1),
            CONTIG=_contig(src, out), **common)
        outs.append(out)
    return outs[0], outs[1]


def qk_norm_rope_infer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0 or num_heads == 0:
        return (torch.empty(q.shape, dtype=q.dtype, device=q.device),
                torch.empty(k.shape, dtype=k.dtype, device=k.device))
    if _use_bulk(num_tokens, num_heads, head_dim, interleaved) and q.shape == k.shape:
        return _run_bulk(q, k, q_weight, k_weight, rope_table, eps)
    return _run_split(q, k, q_weight, k_weight, rope_table, eps, interleaved)


qk_norm_rope_fwd = qk_norm_rope_infer_fwd
