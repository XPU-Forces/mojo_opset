"""Triton INT8 GEMM + fused dequantization kernel for Ascend 910B NPU.

Computes:
    output = (A_int8 @ B_int8) * input_scale * weight_scale [+ bias]

with INT32 accumulation and fused epilogue (scale, bias, dtype cast).
Optimized with persistent scheduling, B-transposed layout, double-buffering,
host-side padding, and @triton.autotune tile selection.
"""

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

# Padding alignment — must be divisible by every BLOCK size in the autotune configs.
# BLOCK_M ∈ {64, 128}  → pad M to 128
# BLOCK_N ∈ {64, 128, 256}  → pad N to 256
# BLOCK_K ∈ {256, 512}  → pad K to 512
_PAD_M = 128
_PAD_N = 256
_PAD_K = 512


def _get_autotune_configs():
    _vmix = {"tile_mix_vector_loop": 2}
    return [
        # Large tiles — best for M≥2048, K≥4096
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 512}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 512}, _vmix),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256}, _vmix),
        # Medium tiles — vmix2 gives +17% here at M=4096 by interleaving
        # vector epilogue (scale, bias, cast) with cube drain
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, _vmix),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512}),
        # Small M tiles
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256}, _vmix),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256}, _vmix),
    ]


@triton.autotune(configs=_get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _int8_gemm_dequant_kernel(
    a_ptr, bt_ptr, c_ptr,
    input_scale_ptr, weight_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btn, stride_btk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_sms = tl.num_programs(axis=0)
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n

    tiles_per_sm = num_tiles // num_sms
    if start_pid < num_tiles % num_sms:
        tiles_per_sm += 1

    tile_id = start_pid - num_sms

    for _ in range(0, tiles_per_sm):
        tile_id += num_sms
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        bt_ptrs = bt_ptr + offs_n[:, None] * stride_btn + offs_k[None, :] * stride_btk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            bt = tl.load(bt_ptrs)
            tl.multibuffer(a, 2)
            tl.multibuffer(bt, 2)
            acc = tl.dot(a, tl.trans(bt), acc=acc)
            a_ptrs += BLOCK_K * stride_ak
            bt_ptrs += BLOCK_K * stride_btk

        # Fused epilogue: int32 → fp32 → scale → [+bias] → output_dtype
        c_fp32 = acc.to(tl.float32)
        in_scale = tl.load(input_scale_ptr + offs_m)
        wt_scale = tl.load(weight_scale_ptr + offs_n)
        c_fp32 = c_fp32 * in_scale[:, None] * wt_scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n).to(tl.float32)
            c_fp32 = c_fp32 + bias[None, :]

        c = c_fp32.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)


def _pad_to(x, mult):
    return ((x + mult - 1) // mult) * mult


def prepare_b(b: torch.Tensor) -> torch.Tensor:
    """Transpose B to (N, K) row-major and pad to block boundaries.

    For inference: weight B is fixed, call once and reuse.

    Args:
        b: (K, N) int8

    Returns:
        bt: (N_padded, K_padded) int8, contiguous
    """
    K_orig, N_orig = b.shape
    bt = b.T.contiguous()

    pN = _pad_to(N_orig, _PAD_N)
    pK = _pad_to(K_orig, _PAD_K)

    if pN != N_orig or pK != K_orig:
        bt_pad = torch.zeros(pN, pK, device=bt.device, dtype=torch.int8)
        bt_pad[:N_orig, :K_orig] = bt
        bt = bt_pad
    return bt


def int8_gemm_dequant_impl(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    M_orig: int,
    N_orig: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused Triton INT8 GEMM + dequantization.

    Computes: output = (a @ b) * input_scale * weight_scale [+ bias]

    Args:
        a: (M, K) int8, contiguous
        b_transposed: (N_padded, K_padded) int8, from prepare_b
        input_scale: (M,) float32, per-token scale
        weight_scale: (N,) float32, per-channel scale
        bias: (N,) output_dtype or None
        M_orig: original M before padding
        N_orig: original N before padding
        output_dtype: torch.float16 / torch.bfloat16 / torch.float32

    Returns:
        c: (M_orig, N_orig) in output_dtype
    """
    M, K = a.shape
    num_sms = get_num_cores("cube")

    pM = _pad_to(max(M, 1), _PAD_M)
    pN = b_transposed.shape[0]
    pK_bt = b_transposed.shape[1]
    pK = _pad_to(K, _PAD_K)
    pK = max(pK, pK_bt)

    if pM != M or pK != K:
        a_pad = torch.zeros(pM, pK, device=a.device, dtype=torch.int8)
        a_pad[:M, :K] = a
        a = a_pad

    if pK != pK_bt:
        bt_pad = torch.zeros(pN, pK, device=b_transposed.device, dtype=torch.int8)
        bt_pad[:, :pK_bt] = b_transposed
        b_transposed = bt_pad

    if pM != M_orig:
        is_pad = torch.zeros(pM, device=input_scale.device, dtype=input_scale.dtype)
        is_pad[:M_orig] = input_scale
        input_scale = is_pad

    if pN != N_orig:
        ws_pad = torch.zeros(pN, device=weight_scale.device, dtype=weight_scale.dtype)
        ws_pad[:N_orig] = weight_scale
        weight_scale = ws_pad

    has_bias = bias is not None
    if has_bias and pN != N_orig:
        b_pad = torch.zeros(pN, device=bias.device, dtype=bias.dtype)
        b_pad[:N_orig] = bias
        bias = b_pad

    bias_ptr = bias if has_bias else weight_scale

    c = torch.zeros(pM, pN, device=a.device, dtype=output_dtype)

    _int8_gemm_dequant_kernel[(num_sms,)](
        a, b_transposed, c,
        input_scale, weight_scale, bias_ptr,
        pM, pN, pK,
        a.stride(0), a.stride(1),
        b_transposed.stride(0), b_transposed.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
        multibuffer=True,
    )

    return c[:M_orig, :N_orig]
