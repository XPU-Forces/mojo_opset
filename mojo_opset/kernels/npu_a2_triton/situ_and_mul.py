from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.cann.libdevice as libdevice
except ModuleNotFoundError:
    libdevice = tl

from mojo_opset.kernels._npu_triton_utils import libentry

"""
Kimi-K3 SiTU gated activation (SituAndMul) for NPU:
c = (beta * tanh(gate / beta) * sigmoid(gate)) * up over concatenated [gate | up]
rows; when linear_beta is set, up is bounded to linear_beta * tanh(up / linear_beta).
Reference: Kimi-K3 modeling_kimi_linear.SituAndMul (beta=4.0, linear_beta=25.0).

Modifications for NPU architecture by triton-x team, 2025.
"""

UB_BYTES_A2 = 196608
UB_RESERVE_BYTES = 1024
_BLOCK_BYTES = 32


def _ub_max_pair(x: torch.Tensor) -> int:
    """Gate+up element pairs fitting one AIV's UB (situ_glu CountMaxPair).

    fp32: 7 floats/pair (x/y/tmp queues); fp16/bf16: 4*T + T + 5 floats/pair
    (adds gateF/upF/yF cast scratch).
    """
    esize = x.element_size()
    if esize == 4:
        bytes_per_pair = 7 * 4
        align_elem = _BLOCK_BYTES // 4
    else:
        bytes_per_pair = 4 * esize + esize + 5 * 4
        align_elem = _BLOCK_BYTES // 2  # 32B-aligned for both T(2B) and float(4B)
    pairs = (UB_BYTES_A2 - UB_RESERVE_BYTES) // bytes_per_pair
    return pairs // align_elem * align_elem


def _ub_max_pair_grad(x: torch.Tensor) -> int:
    """Element pairs fitting one AIV's UB for the backward kernel
    (situ_glu_grad CountMaxPair): fp32 13 floats/pair; fp16/bf16 8*T + 8 floats/pair.
    """
    esize = x.element_size()
    if esize == 4:
        bytes_per_pair = 13 * 4
        align_elem = _BLOCK_BYTES // 4
    else:
        bytes_per_pair = 8 * esize + 8 * 4
        align_elem = _BLOCK_BYTES // 2
    pairs = (UB_BYTES_A2 - UB_RESERVE_BYTES) // bytes_per_pair
    return pairs // align_elem * align_elem


def _largest_aligned_divisor(n: int, cap: int, align: int) -> int:
    """Largest d <= cap with d | n and d 32B-aligned; 0 if none exists."""
    for cand in range(cap - cap % align, 0, -align):
        if n % cand == 0:
            return cand
    return 0


def _situ_tiling(x: torch.Tensor, n_cols: int, ub_max_pair: int) -> Tuple[int, int]:
    """situ_glu(_grad)'s CalTilingParam rule: short-H (half-row fits the UB budget
    and is 32B-aligned) uses full-width tiles with ubMaxPair // n_cols rows;
    long-H uses single rows. Constexpr Triton blocks cannot size to the exact
    remainder like AscendC's dynamic blockLen, so long-H takes the widest
    waste-free aligned divisor of n_cols within the budget (fallback: even split).
    """
    half_row_aligned = (n_cols * x.element_size()) % _BLOCK_BYTES == 0
    if n_cols <= ub_max_pair and half_row_aligned:
        return max(1, ub_max_pair // n_cols), n_cols
    align_elem = _BLOCK_BYTES // min(x.element_size(), 4)
    block = _largest_aligned_divisor(n_cols, ub_max_pair, align_elem)
    if block <= 0:
        iters = -(-n_cols // ub_max_pair)
        block = -(-n_cols // iters)
        block = (block + align_elem - 1) // align_elem * align_elem
        block = min(block, ub_max_pair)
    return 1, block


def _situ_fwd_tiling(x: torch.Tensor, n_cols: int) -> Tuple[int, int]:
    return _situ_tiling(x, n_cols, _ub_max_pair(x))


def _situ_bwd_tiling(x: torch.Tensor, n_cols: int) -> Tuple[int, int]:
    return _situ_tiling(x, n_cols, _ub_max_pair_grad(x))


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def _situ_and_mul_fwd_kernel(
    x,
    c,
    beta,
    linear_beta,
    stride_row,
    stride_out_row,
    n_rows,
    n_cols,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            gate_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            up_ptrs = gate_ptrs + n_cols
            c_ptrs = c + rows_off[:, None] * stride_out_row + cols_off[None, :]

            gate_chunk = tl.load(gate_ptrs, mask=block_mask, other=0.0).to(tl.float32)
            up_chunk = tl.load(up_ptrs, mask=block_mask, other=0.0).to(tl.float32)

            tanh_gate = libdevice.tanh(gate_chunk / beta)
            sigmoid_gate = tl.sigmoid(gate_chunk)
            situ_a = beta * tanh_gate * sigmoid_gate
            if HAS_LINEAR_BETA:
                up_chunk = linear_beta * libdevice.tanh(up_chunk / linear_beta)

            c_chunk = situ_a * up_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def _situ_and_mul_bwd_kernel(
    dc,
    x,
    dx,
    beta,
    linear_beta,
    stride_row,
    stride_dc_row,
    n_rows,
    n_cols,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dc_ptrs = dc + rows_off[:, None] * stride_dc_row + cols_off[None, :]
            gate_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            up_ptrs = gate_ptrs + n_cols
            dgate_ptrs = dx + rows_off[:, None] * stride_row + cols_off[None, :]
            dup_ptrs = dgate_ptrs + n_cols

            gate_chunk = tl.load(gate_ptrs, mask=block_mask, other=0.0).to(tl.float32)
            up_chunk = tl.load(up_ptrs, mask=block_mask, other=0.0).to(tl.float32)

            tanh_gate = libdevice.tanh(gate_chunk / beta)
            sigmoid_gate = tl.sigmoid(gate_chunk)
            situ_a = beta * tanh_gate * sigmoid_gate
            d_situ = (1 - tanh_gate * tanh_gate) * sigmoid_gate + beta * tanh_gate * sigmoid_gate * (1 - sigmoid_gate)

            if HAS_LINEAR_BETA:
                tanh_up = libdevice.tanh(up_chunk / linear_beta)
                up_transformed = linear_beta * tanh_up
                d_up = 1 - tanh_up * tanh_up
            else:
                up_transformed = up_chunk
                d_up = 1.0

            dc_chunk = tl.load(dc_ptrs, mask=block_mask, other=0.0).to(tl.float32)

            dgate_chunk = dc_chunk * d_situ * up_transformed
            dup_chunk = dc_chunk * situ_a * d_up

            tl.store(dgate_ptrs, dgate_chunk, mask=block_mask)
            tl.store(dup_ptrs, dup_chunk, mask=block_mask)


def situ_and_mul_train_fwd_impl(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Forward: c = (beta * tanh(gate / beta) * sigmoid(gate)) * up."""
    _check_situ_and_mul_inputs(x, beta, linear_beta)

    ori_shape = x.shape
    n_cols = ori_shape[-1] // 2

    x_2d = x.reshape(-1, 2 * n_cols).contiguous()
    n_rows = x_2d.shape[0]

    c = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M, BLOCK_SIZE_N = _situ_fwd_tiling(x_2d, n_cols)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _situ_and_mul_fwd_kernel[grid](
        x_2d,
        c,
        float(beta),
        0.0 if linear_beta is None else float(linear_beta),
        x_2d.stride(0),
        c.stride(0),
        n_rows,
        n_cols,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return c.reshape(*ori_shape[:-1], n_cols)


def situ_and_mul_train_bwd_impl(
    dc: torch.Tensor,
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Backward: gradient w.r.t. the concatenated [gate | up] input."""
    _check_situ_and_mul_inputs(x, beta, linear_beta)
    expected_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    if dc.shape != expected_shape:
        raise ValueError(f"situ_and_mul grad_output must have shape {expected_shape}, got {dc.shape}.")
    if dc.dtype != x.dtype or dc.device != x.device:
        raise ValueError("situ_and_mul grad_output must share dtype and device with the input.")

    ori_shape = x.shape
    n_cols = ori_shape[-1] // 2

    x_2d = x.reshape(-1, 2 * n_cols).contiguous()
    dc_2d = dc.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]

    dx = torch.empty_like(x_2d)

    BLOCK_SIZE_M, BLOCK_SIZE_N = _situ_bwd_tiling(dc_2d, n_cols)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _situ_and_mul_bwd_kernel[grid](
        dc_2d,
        x_2d,
        dx,
        float(beta),
        0.0 if linear_beta is None else float(linear_beta),
        x_2d.stride(0),
        dc_2d.stride(0),
        n_rows,
        n_cols,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return dx.reshape(*ori_shape)


def _check_situ_and_mul_inputs(x: torch.Tensor, beta: float, linear_beta: Optional[float]) -> None:
    if x.dim() < 1:
        raise ValueError("situ_and_mul input must have at least one dimension.")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"situ_and_mul expects an even last dimension, got {x.shape[-1]}.")
    if beta == 0:
        raise ValueError("situ_and_mul beta must be nonzero.")
    if linear_beta == 0:
        raise ValueError("situ_and_mul linear_beta must be nonzero.")


def situ_and_mul_fwd_impl(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    return situ_and_mul_train_fwd_impl(x, beta, linear_beta)


def situ_and_mul_bwd_impl(
    dc: torch.Tensor,
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    return situ_and_mul_train_bwd_impl(dc, x, beta, linear_beta)


@torch.library.custom_op("mojo_npu_triton_a2::situ_and_mul_fwd", mutates_args=())
def situ_and_mul_fwd(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    return situ_and_mul_train_fwd_impl(x, beta, linear_beta)


@situ_and_mul_fwd.register_fake
def _situ_and_mul_fwd_fake(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    del beta, linear_beta
    return torch.empty(
        x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format
    )


@torch.library.custom_op("mojo_npu_triton_a2::situ_and_mul_bwd", mutates_args=())
def situ_and_mul_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    return situ_and_mul_train_bwd_impl(grad_output, x, beta, linear_beta)


@situ_and_mul_bwd.register_fake
def _situ_and_mul_bwd_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    del grad_output, beta, linear_beta
    return torch.empty_like(x, memory_format=torch.contiguous_format)
