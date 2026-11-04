import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._npu_triton_utils import get_num_cores

"""
GmmSwigluQuant for NPU A5 (forward only): one fused launch (ops-transformer
gmm/grouped_matmul_swiglu_quant_v2 semantics, 950PR MX path):

    C     = (x . w) with MXFP8 dequant fused into the dot
    act, lin = C[:, :I], C[:, I:]          (weight halves, I = N // 2)
    S     = silu(clamp(act, max=L)) * clamp(lin, +-L),  L=0 disables
    y, y_scale = MX quant of S             (OCP: e = floor(log2 amax) - 8,
                                            32-element groups, e8m0 scales)

I/O (all contiguous; one e8m0 byte per 32 K elements; x/y scales are
row-major, weight_scale is N-major so rhs_scale tiles load in the
[BN, BK // 32] orientation tl.dot_scaled consumes, zero transposes;
K must be a multiple of 32 so the scale k-axis is exact):
    x           [M, K]         float8_e4m3fn
    weight      [E, K, N]      float8_e4m3fn   (N = 2I; first half = act)
    weight_scale[E, N, K/32]   uint8 e8m0
    x_scale     [M, K/32]      uint8 e8m0
    group_list  [E]            int64           (row cumsum, last == M)
    y           [M, I]         float8_e4m3fn
    y_scale     [M, ceil(I/32)]          uint8 e8m0

Rows [cumsum[g-1], cumsum[g]) of x multiply against weight[g]; E=1 is the
dense case.

The SwiGLU + MX quant epilogue runs on the vector cores of the same launch,
so the SwiGLU halves never hit GM.
"""

# Tile selection, gmm-style; BM keeps real (non-pow2) values, but BN/BK are
# powers of two (non-pow2 scale-tile extents fault the AIC cube instruction).
_TILE_BUDGET_BYTES = 512 * 1024 // 2  # L1 / 2
_BK_MAX = 512
_BK_MIN = 32  # one MX scale block: tl.dot_scaled K granularity
_TileBKMaxWide = 256
_BK512_MIN_K = 6144
_WIDE_N_BK_MAX = 512
_DECODE_ROWS = 16
_BM_CAP = 128
_NUM_B = 2  # dual accumulators (act / lin halves)
_BAND_M = 8


def _align32(value: int) -> int:
    return (value + 31) // 32 * 32


def _pow2_floor(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _stage_tiles(rows: int, n: int, k: int, cores: int = 28):
    """Deterministic (BM, BN, BK) for the dual-dot MX kernel, gmm-style."""
    if rows <= _DECODE_ROWS:
        block_m, bn_cap = 16, 128
    else:
        block_m = min(rows, _BM_CAP)
        bn_cap = 128
    block_n = max(min(triton.next_power_of_2(n), bn_cap), 32)

    per_k = _align32(block_m) + _NUM_B * _align32(block_n)  # fp8: 1 byte/elem
    block_k = min(_TILE_BUDGET_BYTES // per_k, _BK_MAX, k & ~(_BK_MIN - 1) or _BK_MIN)
    if block_n >= 128:
        block_k = min(block_k, _WIDE_N_BK_MAX if k >= _BK512_MIN_K else _TileBKMaxWide)
    # Keep BLOCK_K a power-of-two divisor of K: the K loop then tiles K
    # exactly, so the kb sweep ends on the k_blocks'th scale column.
    return block_m, block_n, min(max(_pow2_floor(block_k), _BK_MIN), k & -k)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "HALF_N", "NUM_GROUPS"])
def _gmm_swiglu_quant_kernel(
    x, weight, x_scale, weight_scale, y, y_scale, group_list,
    M, N, K, HALF_N, NUM_GROUPS,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_M: tl.constexpr,
):
    """y, y_scale = mxfp8_quant(silu(clamp(x @ W_g[:, :I])) * clamp(x @ W_g[:, I:])).

    Expert g owns rows [cumsum[g-1], cumsum[g]); m-blocks are group-relative
    and tail rows are masked at the group end.
    """
    K_GROUPS: tl.constexpr = BLOCK_K // 32
    N_GROUPS: tl.constexpr = BLOCK_N // 32

    pid_n = tl.program_id(axis=0)  # n-block over I (output cols)
    pid_c = tl.program_id(axis=1)  # m-chunk over flat (group, m-block) blocks
    half_n = HALF_N
    i_blocks = tl.cdiv(half_n, 32)
    k_blocks = tl.cdiv(K, 32)
    k_iters = tl.cdiv(K, BLOCK_K)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kb = tl.arange(0, K_GROUPS)
    n_mask = offs_n < half_n

    # Flat m-block space, group-major: this chunk covers [b_lo, b_hi); the
    # cumsum walk needs no host-side group sizes and stops once covered.
    b_lo = pid_c * CHUNK_M
    b_hi = b_lo + CHUNK_M
    g = 0
    start = 0
    flat = 0
    while (flat < b_hi) & (g < NUM_GROUPS):
        end = tl.load(group_list + g).to(tl.int32)
        blocks_g = tl.cdiv(end - start, BLOCK_M)
        flat_next = flat + blocks_g
        if flat_next > b_lo:
            # Expert bases in int64: E * K * N can exceed int32.
            w_g = weight + g.to(tl.int64) * K * N
            ws_g = weight_scale + g.to(tl.int64) * N * k_blocks
            for pid_m in range(tl.maximum(b_lo - flat, 0), tl.minimum(b_hi - flat, blocks_g)):
                offs_m = start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                m_mask = offs_m < end

                acc_act = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                acc_lin = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # No K-axis masks: K % 32 == 0 is input-validated and BLOCK_K
                # divides K, so the kb sweep stays inside the scale k-axis.
                for k in range(0, k_iters):
                    k_ofs = k * BLOCK_K
                    kb = k * K_GROUPS + offs_kb
                    x_tile = tl.load(
                        x + offs_m[:, None] * K + (k_ofs + offs_k)[None, :],
                        mask=m_mask[:, None], other=0.0)
                    w1_tile = tl.load(
                        w_g + (k_ofs + offs_k)[:, None] * N + offs_n[None, :],
                        mask=n_mask[None, :], other=0.0)
                    # Second half (lin) at column n + I.
                    w2_tile = tl.load(
                        w_g + (k_ofs + offs_k)[:, None] * N + (half_n + offs_n)[None, :],
                        mask=n_mask[None, :], other=0.0)
                    xs_tile = tl.load(
                        x_scale + offs_m[:, None] * k_blocks + kb[None, :],
                        mask=m_mask[:, None], other=0)
                    ws1_tile = tl.load(
                        ws_g + offs_n[:, None] * k_blocks + kb[None, :],
                        mask=n_mask[:, None], other=0)
                    ws2_tile = tl.load(
                        ws_g + (half_n + offs_n)[:, None] * k_blocks + kb[None, :],
                        mask=n_mask[:, None], other=0)
                    tl.extra.cann.extension.compile_hint(x_tile, "dot_pad_only_k")
                    acc_act = tl.dot_scaled(x_tile, xs_tile, "e4m3", w1_tile,
                                            ws1_tile, "e4m3", acc=acc_act)
                    acc_lin = tl.dot_scaled(x_tile, xs_tile, "e4m3", w2_tile,
                                            ws2_tile, "e4m3", acc=acc_lin)

                act = acc_act
                lin = acc_lin
                if HAS_LIMIT:
                    act = tl.minimum(act, limit)
                    lin = tl.minimum(tl.maximum(lin, -limit), limit)
                s = act * tl.sigmoid(act) * lin
                s = tl.where(n_mask[None, :], s, 0.0)  # tail cols must not inflate amax
                # V3 numerics: quantize the BF16-rounded view of S.
                s = s.to(tl.bfloat16).to(tl.float32)

                # ---- MX dynamic quant, 32-element groups, OCP scale algorithm ----
                groups = s.reshape(BLOCK_M, N_GROUPS, 32)
                amax = tl.max(tl.abs(groups), axis=2)
                e = tl.where(amax > 0.0, tl.floor(tl.log2(amax)) - 8.0, -127.0)
                e = tl.maximum(e, -127.0)  # e8m0 encoding floor
                q = groups * tl.exp2(-e)[:, :, None]
                q = tl.minimum(tl.maximum(q, -448.0), 448.0)  # E4M3FN max, satfinite
                y_ptrs = y + offs_m[:, None] * half_n + offs_n[None, :]
                tl.store(y_ptrs, q.reshape(BLOCK_M, BLOCK_N).to(y.dtype.element_ty),
                         mask=m_mask[:, None] & n_mask[None, :])
                offs_gp = tl.arange(0, N_GROUPS)
                g_col = pid_n * N_GROUPS + offs_gp  # first column of group j is g_col * 32
                ys_ptrs = y_scale + offs_m[:, None] * i_blocks + g_col[None, :]
                tl.store(ys_ptrs, (e + 127.0).to(tl.int32).to(tl.uint8),
                         mask=m_mask[:, None] & (g_col[None, :] * 32 < half_n))
        flat = flat_next
        start = end
        g += 1


def _check_gmm_swiglu_quant_inputs(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
    if x.dtype is not torch.float8_e4m3fn:
        raise ValueError("gmm_swiglu_quant x must be float8_e4m3fn.")
    if x.dim() < 1:
        raise ValueError("gmm_swiglu_quant input must have at least one dimension.")
    k = x.shape[-1]
    rows = x.numel() // k if k else 0
    e = weight.shape[0] if weight.dim() == 3 else 0
    if weight.dim() != 3 or weight.shape[1] != k or e < 1:
        raise ValueError(
            f"gmm_swiglu_quant weight must be [E, K, N]=[E>=1, {k}, N], got {list(weight.shape)}."
        )
    n = weight.shape[2]
    if n % 2:
        raise ValueError("gmm_swiglu_quant weight N must be even (two SwiGLU halves).")
    if k % 32:
        raise ValueError(
            f"gmm_swiglu_quant K must be a multiple of 32 (one MX group), got K={k}."
        )
    k_blocks = (k + 31) // 32
    if weight_scale.dim() != 3 or weight_scale.shape != (e, n, k_blocks) \
            or weight_scale.dtype not in (torch.uint8, torch.int8):
        raise ValueError(
            f"gmm_swiglu_quant weight_scale must be uint8/int8 e8m0 of shape [{e}, {n}, {k_blocks}] (N-major), "
            f"got {tuple(weight_scale.shape)} {weight_scale.dtype}."
        )
    if x_scale.dim() != 2 or x_scale.shape != (rows, k_blocks) \
            or x_scale.dtype not in (torch.uint8, torch.int8):
        raise ValueError(
            f"gmm_swiglu_quant x_scale must be uint8/int8 e8m0 of shape [{rows}, {k_blocks}], "
            f"got {tuple(x_scale.shape)} {x_scale.dtype}."
        )
    if group_list.dtype != torch.int64 or group_list.numel() != e:
        raise ValueError(
            f"gmm_swiglu_quant group_list must be int64 [{e}] (per-group row cumsum, "
            f"last == {rows})."
        )
    if not (x.device == weight.device == weight_scale.device == x_scale.device == group_list.device):
        raise ValueError("gmm_swiglu_quant tensors must share device.")
    if swiglu_limit < 0:
        raise ValueError("gmm_swiglu_quant swiglu_limit must be nonnegative.")


def gmm_swiglu_quant_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, y_scale) = mxfp8_quant(silu(clamp(x@W_left_g)) * clamp(x@W_right_g))."""
    _check_gmm_swiglu_quant_inputs(x, weight, weight_scale, x_scale, group_list, swiglu_limit)
    k = x.shape[-1]
    e = weight.shape[0]
    n = weight.shape[2]
    inter = n // 2
    rows = x.numel() // k
    if rows == 0:
        return (
            x.new_empty((0, inter), dtype=torch.float8_e4m3fn),
            x.new_zeros((0, triton.cdiv(inter, 64), 2), dtype=torch.uint8),
        )

    x_2d = x.reshape(rows, k).contiguous()
    weight = weight.contiguous().view(e * k, n)
    x_scale = x_scale.contiguous().view(rows, -1)
    group_list = group_list.contiguous()
    if weight_scale.dtype is not torch.uint8:
        weight_scale = weight_scale.view(torch.uint8)
    if x_scale.dtype is not torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    weight_scale = weight_scale.contiguous().view(torch.uint8).view(-1)

    # Tile by the average per-group row count (rows // e, no host-side group
    # sizes), snapped to a power of two for grouped dispatch: routed rows
    # spread thinly across experts, and non-pow2 BM at BN=128 overflows the
    # quant epilogue UB.
    rows_eff = max(rows // e, 1)
    if e > 1:
        rows_eff = min(triton.next_power_of_2(rows_eff), _BM_CAP)
    block_m, block_n, block_k = _stage_tiles(rows_eff, inter, k)

    y = x_2d.new_empty((rows, inter), dtype=torch.float8_e4m3fn)
    y_scale = x_2d.new_zeros((rows, triton.cdiv(inter, 32)), dtype=torch.uint8)

    num_pid_n = triton.cdiv(inter, block_n)
    cores = get_num_cores("cube")
    chunk_m = _BAND_M if num_pid_n >= cores else 1
    # Flat m-block upper bound: groups round up (< 1 extra block each) and
    # blocks never exceed rows, so sparse decode collapses to the real count.
    m_blocks = min(rows, triton.cdiv(rows, block_m) + e - 1)
    _gmm_swiglu_quant_kernel[(num_pid_n, triton.cdiv(m_blocks, chunk_m))](
        x_2d, weight, x_scale, weight_scale, y, y_scale, group_list,
        rows, n, k, inter, e,
        float(swiglu_limit), swiglu_limit > 0.0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, CHUNK_M=chunk_m,
    )
    return y, y_scale


@torch.library.custom_op("mojo_npu_triton_a5::gmm_swiglu_quant_fwd", mutates_args=())
def gmm_swiglu_quant_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gmm_swiglu_quant_fwd_impl(x, weight, weight_scale, x_scale, group_list, swiglu_limit)


@gmm_swiglu_quant_fwd.register_fake
def _gmm_swiglu_quant_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.shape[-1]
    inter = weight.shape[2] // 2
    blocks = -(-inter // 32)
    return (
        torch.empty((rows, inter), dtype=torch.float8_e4m3fn, device=x.device,
                    memory_format=torch.contiguous_format),
        torch.empty((rows, blocks), dtype=torch.uint8, device=x.device,
                    memory_format=torch.contiguous_format),
    )
