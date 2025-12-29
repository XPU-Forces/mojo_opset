import math

from typing import Any
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

from triton.runtime.libentry import libentry
from scipy.linalg import hadamard

from mojo_opset.backends.ttx.kernels.ascend.layernorm import (
    layer_norm_fwd,
)

class TritonConfig:
    def __init__(self, 
                 dim, 
                 n_heads, 
                 n_local_heads, 
                 head_dim, 
                 rope_head_dim, 
                 topk, 
                 q_lora_rank):
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.topk = topk
        self.q_lora_rank = q_lora_rank

# linear
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}),
]
@triton.autotune(configs=configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel_2d(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    acc_dtype: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_total = num_pid_m * num_pid_n
    pid_m = (pid % num_pid_total) // num_pid_n
    pid_n = pid % num_pid_n

    # offset cal
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A point: [M, K]
    a_ptrs = a_ptr +\
             offs_am[:, None] * stride_am + \
             offs_k[None, :] * stride_ak

    # B point: [K, N]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
             offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + \
             offs_am[:, None] * stride_cm + \
             offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def linear(a, b):
    # Check constraints.
    assert a.shape[-1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    input_shape = a.shape
    a = a.reshape(-1, b.shape[0])
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    acc_dtype = tl.float32
    matmul_kernel_2d[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        acc_dtype,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    new_shape = input_shape[:-1] + (b.shape[1],)
    return c.reshape(new_shape)

# layernrom
def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    device: torch.device = None,
):
    return layer_norm_fwd(x, weight, bias, eps)

# 3d
@triton.autotune(
    configs=[
        triton.Config({"TOKEN_BLOCK_SIZE": 1}),
        triton.Config({"TOKEN_BLOCK_SIZE": 2}),
        triton.Config({"TOKEN_BLOCK_SIZE": 4}),
        triton.Config({"TOKEN_BLOCK_SIZE": 8}),
        triton.Config({"TOKEN_BLOCK_SIZE": 16}),
        triton.Config({"TOKEN_BLOCK_SIZE": 24}),
        triton.Config({"TOKEN_BLOCK_SIZE": 32}),
        triton.Config({"TOKEN_BLOCK_SIZE": 64}),
    ],
    key=["hd", "rope_dim", "seq_len"],
    restore_value=["q_ptr"],
)
@triton.jit
def _rope_forward_kernel_3d(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    q_dim_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    q_rope_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE kernel for 3D input tensor [batch, seq_len, dim]
    
    Args:
        q_ptr: Pointer to input q tensor [bs, seq_len, hd]
        rope_dim: Number of dimensions to apply RoPE (must be even)
        half_rope_dim: rope_dim // 2
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    
    # Validate rope_dim parameters
    # rope_dim must be even and <= hd
    # tl.static_assert(rope_dim % 2 == 0, "rope_dim must be even")
    # tl.static_assert(rope_dim <= hd, "rope_dim must be <= head_dim")

    for batch_idx in range(bs):
        num_seq_blocks = (seq_len + TOKEN_BLOCK_SIZE - 1) // TOKEN_BLOCK_SIZE
        for seq_block_id in range(pid, num_seq_blocks, grid_size):
            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len

            # Calculate batch offset for cos/sin
            sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
            cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
            sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

            # RoPE only applied to first rope_dim dimensions
            rope_dim_offsets = tl.arange(0, half_rope_dim)
            rope_dim_mask = rope_dim_offsets < half_rope_dim

            # Load cos/sin data, only first rope_dim dimensions
            cos_block_2d = tl.load(
                cos_token_ptr + rope_dim_offsets[None, :], 
                mask=seq_mask[:, None] & rope_dim_mask[None, :], 
                other=0
            )
            sin_block_2d = tl.load(
                sin_token_ptr + rope_dim_offsets[None, :], 
                mask=seq_mask[:, None] & rope_dim_mask[None, :], 
                other=0
            )

            # Calculate q tensor offsets - only process first rope_dim dimensions
            # First half dimensions (0:half_rope_dim)
            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + seq_offsets[:, None] * q_seq_stride
                + rope_dim_offsets[None, :] * q_dim_stride
            )
            
            # Second half dimensions (half_rope_dim:rope_dim)
            q_offsets_half2 = (
                batch_idx * q_batch_stride
                + seq_offsets[:, None] * q_seq_stride
                + (rope_dim_offsets[None, :] + half_rope_dim) * q_dim_stride
            )
            
            # Create mask: sequence mask & dimension mask
            q_mask = seq_mask[:, None] & rope_dim_mask[None, :]

            # Load original q values
            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_mask, other=0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_mask, other=0).to(sin_block_2d.dtype)

            # Reshape cos/sin for broadcasting
            cos_row = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, half_rope_dim), can_reorder=True)
            sin_row = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, half_rope_dim), can_reorder=True)

            # Apply RoPE rotation
            new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
            new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row

            # Store results
            tl.store(q_rope_ptr + q_offsets_half1, new_q_tile_1, mask=q_mask)
            tl.store(q_rope_ptr + q_offsets_half2, new_q_tile_2, mask=q_mask)
            
            # If rope_dim < hd, copy remaining dimensions (no RoPE applied)
            if rope_dim < hd:
                # Calculate offsets for remaining dimensions
                remaining_dim_start = rope_dim
                remaining_dim_count = hd - rope_dim
                
                # Process remaining dimensions
                for rem_block_start in range(0, remaining_dim_count, TOKEN_BLOCK_SIZE):
                    rem_dim_offsets = tl.arange(0, TOKEN_BLOCK_SIZE) + rem_block_start
                    rem_dim_mask = rem_dim_offsets < remaining_dim_count
                    
                    # Calculate offsets for remaining dimensions
                    q_offsets_remaining = (
                        batch_idx * q_batch_stride
                        + seq_offsets[:, None] * q_seq_stride
                        + (rem_dim_offsets[None, :] + remaining_dim_start) * q_dim_stride
                    )
                    
                    # Create mask
                    rem_mask = seq_mask[:, None] & rem_dim_mask[None, :]
                    
                    # Load original values
                    q_remaining = tl.load(q_ptr + q_offsets_remaining, mask=rem_mask, other=0)
                    
                    # Direct copy (no RoPE applied)
                    tl.store(q_rope_ptr + q_offsets_remaining, q_remaining, mask=rem_mask)


def rope_3d(
    q: torch.Tensor,           # [batch_size, seq_len, dim]
    cos: torch.Tensor,         # [bs, seq_len, rope_dim//2] or [1, seq_len, rope_dim//2]
    sin: torch.Tensor,         # [bs, seq_len, rope_dim//2] or [1, seq_len, rope_dim//2]
    rope_dim: int = None,      # Number of dimensions to apply RoPE, defaults to dim
) -> torch.Tensor:            # [batch_size, seq_len, dim]
    """
    Apply RoPE to 3D q tensor, supports partial dimension RoPE
    
    Args:
        q: Input query tensor, shape [batch_size, seq_len, dim]
        cos, sin: RoPE cos/sin matrices
        rope_dim: Number of dimensions to apply RoPE, must be even
    """
    # Ensure input is contiguous
    q_t = q.contiguous()
    
    # Determine rope_dim, default is dim
    batch_size, seq_len, dim = q_t.shape
    if rope_dim is None:
        rope_dim = dim
    
    # Validate parameters
    assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
    assert rope_dim <= dim, f"rope_dim ({rope_dim}) must be <= dim ({dim})"
    
    # Validate cos/sin shapes
    cos_batch_size = cos.shape[0]
    sin_batch_size = sin.shape[0]
    assert cos_batch_size == sin_batch_size, "cos and sin must have same batch size"
    assert cos.shape[1] == seq_len, f"cos seq_len mismatch: {cos.shape[1]} vs {seq_len}"
    assert sin.shape[1] == seq_len, f"sin seq_len mismatch: {sin.shape[1]} vs {seq_len}"
    assert cos.shape[2] == rope_dim // 2, f"cos dim mismatch: {cos.shape[2]} vs {rope_dim // 2}"
    assert sin.shape[2] == rope_dim // 2, f"sin dim mismatch: {sin.shape[2]} vs {rope_dim // 2}"
    
    # Create output tensor
    q_rope = torch.empty_like(q_t)
    
    # Calculate grid size
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)
    
    # Ensure cos/sin are contiguous
    cos_contig = cos.contiguous()
    sin_contig = sin.contiguous()
    
    # Call kernel
    _rope_forward_kernel_3d[grid](
        q_t,
        q_t.stride(0),      # batch stride
        q_t.stride(1),      # seq stride
        q_t.stride(2),      # dim stride
        cos_contig,
        cos_contig.stride(1),  # cos seq stride (cos.stride(-2) for [bs, seq_len, dim//2])
        sin_contig,
        sin_contig.stride(1),  # sin seq stride
        seq_len,
        q_rope,
        batch_size,
        cos_batch_size,
        dim,
        rope_dim,
        rope_dim // 2,
    )
    
    return q_rope

# rope
@triton.autotune(
    configs=[
        triton.Config({"TOKEN_BLOCK_SIZE": 1}),
        triton.Config({"TOKEN_BLOCK_SIZE": 2}),
        triton.Config({"TOKEN_BLOCK_SIZE": 4}),
        triton.Config({"TOKEN_BLOCK_SIZE": 8}),
        triton.Config({"TOKEN_BLOCK_SIZE": 16}),
        triton.Config({"TOKEN_BLOCK_SIZE": 24}),
        triton.Config({"TOKEN_BLOCK_SIZE": 32}),
        triton.Config({"TOKEN_BLOCK_SIZE": 64}),
    ],
    key=["n_qh", "hd", "rope_dim", "seq_len"],
    restore_value=["q_ptr"],
)
@triton.jit
def _rope_forward_kernel_4d(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    q_head_stride,
    q_head_dim_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    q_rope_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE kernel only apply rope in [bs, seq_len, n_qh, :rope_dim]
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    
    for batch_idx in range(bs):
        num_seq_blocks = (seq_len + TOKEN_BLOCK_SIZE - 1) // TOKEN_BLOCK_SIZE
        for seq_block_id in range(pid, num_seq_blocks, grid_size):
            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len

            sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
            cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
            sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

            rope_dim_offsets = tl.arange(0, half_rope_dim)
            rope_dim_mask = rope_dim_offsets < half_rope_dim

            cos_block_2d = tl.load(
                cos_token_ptr + rope_dim_offsets[None, :], 
                mask=seq_mask[:, None] & rope_dim_mask[None, :], 
                other=0
            )
            sin_block_2d = tl.load(
                sin_token_ptr + rope_dim_offsets[None, :], 
                mask=seq_mask[:, None] & rope_dim_mask[None, :], 
                other=0
            )

            head_offsets = tl.arange(0, n_qh)
            head_mask = head_offsets < n_qh

            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + seq_offsets[:, None, None] * q_seq_stride
                + head_offsets[None, :, None] * q_head_stride
                + rope_dim_offsets[None, None, :] * q_head_dim_stride
            )
            
            q_offsets_half2 = (
                batch_idx * q_batch_stride
                + seq_offsets[:, None, None] * q_seq_stride
                + head_offsets[None, :, None] * q_head_stride
                + (rope_dim_offsets[None, None, :] + half_rope_dim) * q_head_dim_stride
            )
            
            q_mask = seq_mask[:, None, None] & head_mask[None, :, None] & rope_dim_mask[None, None, :]

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_mask, other=0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_mask, other=0).to(sin_block_2d.dtype)

            # reshape cos sin
            cos_row = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
            sin_row = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

            # apply rope
            new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
            new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row

            tl.store(q_rope_ptr + q_offsets_half1, new_q_tile_1, mask=q_mask)
            tl.store(q_rope_ptr + q_offsets_half2, new_q_tile_2, mask=q_mask)
            
            if rope_dim < hd:
                remaining_dim_start = rope_dim
                remaining_dim_count = hd - rope_dim
                
                for rem_block_start in range(0, remaining_dim_count, TOKEN_BLOCK_SIZE):
                    rem_dim_offsets = tl.arange(0, TOKEN_BLOCK_SIZE) + rem_block_start
                    rem_dim_mask = rem_dim_offsets < remaining_dim_count
                    
                    q_offsets_remaining = (
                        batch_idx * q_batch_stride
                        + seq_offsets[:, None, None] * q_seq_stride
                        + head_offsets[None, :, None] * q_head_stride
                        + (rem_dim_offsets[None, None, :] + remaining_dim_start) * q_head_dim_stride
                    )
                    
                    rem_mask = seq_mask[:, None, None] & head_mask[None, :, None] & rem_dim_mask[None, None, :]
                    
                    q_remaining = tl.load(q_ptr + q_offsets_remaining, mask=rem_mask, other=0)
                    
                    tl.store(q_rope_ptr + q_offsets_remaining, q_remaining, mask=rem_mask)

def rope_4d(
    q: torch.Tensor,           # [batch_size, n_q_head, seq_len, head_dim]
    cos: torch.Tensor,         # [bs, seq_len, rope_dim//2]
    sin: torch.Tensor,         # [bs, seq_len, rope_dim//2]
    rope_dim: int = None,      # dim need be roped
) -> torch.Tensor:            # [batch_size, n_q_head, seq_len, head_dim]
    q = q.contiguous()
    
    batch_size, seq_len, n_q_head, head_dim = q.shape
    if rope_dim is None:
        rope_dim = head_dim
    
    assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
    assert rope_dim <= head_dim, f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
    
    cos_batch_size = cos.shape[0]
    sin_batch_size = sin.shape[0]
    assert cos_batch_size == sin_batch_size, "cos and sin must have same batch size"
    assert cos.shape[1] == seq_len, f"cos seq_len mismatch: {cos.shape[1]} vs {seq_len}"
    assert sin.shape[1] == seq_len, f"sin seq_len mismatch: {sin.shape[1]} vs {seq_len}"
    assert cos.shape[2] == rope_dim // 2, f"cos dim mismatch: {cos.shape[2]} vs {rope_dim // 2}"
    assert sin.shape[2] == rope_dim // 2, f"sin dim mismatch: {sin.shape[2]} vs {rope_dim // 2}"
    
    q_rope = torch.empty_like(q)
    
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)
    
    cos_contig = cos.contiguous()
    sin_contig = sin.contiguous()
    
    _rope_forward_kernel_4d[grid](
        q,
        q.stride(0),      # batch stride
        q.stride(1),      # seq stride
        q.stride(2),      # head stride
        q.stride(3),      # head_dim stride
        cos_contig,
        cos_contig.stride(-2),  # cos row stride
        sin_contig,
        sin_contig.stride(-2),  # sin row stride
        seq_len,
        q_rope,
        batch_size,
        cos_batch_size,
        n_q_head,
        head_dim,
        rope_dim,
        rope_dim // 2,
    )
    
    return q_rope

def rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int = None,
) -> torch.Tensor:
    """
    Wrapper function that handles both 3D and 4D inputs
    
    Args:
        q: Input tensor, can be [batch, seq_len, dim] or [batch, n_heads, seq_len, dim]
        cos, sin: RoPE cos/sin matrices
        rope_dim: Number of dimensions to apply RoPE
    
    Returns:
        RoPE-transformed tensor with same shape as input
    """
    # Check input dimensions
    if q.dim() == 3:
        # 3D input: [batch, seq_len, dim]
        return rope_3d(q, cos, sin, rope_dim)
    elif q.dim() == 4:
        # 4D input: [batch, n_heads, seq_len, dim]
        return rope_4d(q, cos, sin, rope_dim)
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, got {q.dim()}D")

#quant
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 8, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 16, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 32, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 64, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 8, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 16, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 32, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 64, 'multibuffer': False}),
    ],
    key=['cols'],
)
@triton.jit
def scale_dynamic_quant_kernel_4d(
    input, 
    scale, 
    output, 
    quant_scale, 
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    rows: tl.constexpr, 
    cols: tl.constexpr, 
    align_cols: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    input shpae: [batch, seq_len, rows, cols]
    scale shape: [cols]
    output shape: [batch, seq_len, rows, cols] (int8)
    quant_scale shape: [batch, seq_len, rows]
    """
    
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_elements = batch * seq_len * rows
    num_tasks = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for task_id in range(pid, num_tasks, grid_size):
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)

        batch_idx = element_off // (seq_len * rows)
        seq_idx = (element_off // rows) % seq_len
        row_idx = element_off % rows

        
        element_mask = element_off < total_elements
        
        max_abs_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_block_offset in range(0, align_cols, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < cols
            
            # get input position
            # mem layout: [batch, seq_len, rows, cols]
            input_offset = (batch_idx[:, None] * seq_len * rows * cols + 
                           seq_idx[:, None] * rows * cols + 
                           row_idx[:, None] * cols + 
                           cols_off[None, :])
            
            input_ptr = input + input_offset
            scale_ptr = scale + cols_off
            
            block_mask = element_mask[:, None] & cols_mask[None, :]
            
            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)
            
            scaled_vals = input_vals * scale_vals
            current_max = tl.max(tl.abs(scaled_vals), axis=1)
            max_abs_accumulator = tl.maximum(max_abs_accumulator, current_max)
        
        final_max_abs = max_abs_accumulator
        current_quant_scale = final_max_abs / 127.0
        
        quant_scale_ptr = quant_scale + element_off
        tl.store(quant_scale_ptr, current_quant_scale, mask=element_mask)
        
        for col_block_offset in range(0, align_cols, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < cols
            
            input_offset = (batch_idx[:, None] * seq_len * rows * cols + 
                           seq_idx[:, None] * rows * cols + 
                           row_idx[:, None] * cols + 
                           cols_off[None, :])
            
            input_ptr = input + input_offset
            output_ptr = output + input_offset
            scale_ptr = scale + cols_off
            
            block_mask = element_mask[:, None] & cols_mask[None, :]
            
            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)
            
            scaled_vals = input_vals * scale_vals
            quant_vals = scaled_vals / current_quant_scale[:, None]
            quant_vals_int8 = quant_vals.to(tl.int8)
            
            tl.store(output_ptr, quant_vals_int8, mask=block_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 8, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 16, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 32, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 64, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE_M': 8, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 16, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 32, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE_M': 64, 'multibuffer': False}),
    ],
    key=['cols'],
)
@triton.jit
def scale_dynamic_quant_kernel_3d(
    input, 
    scale, 
    output, 
    quant_scale, 
    batch: tl.constexpr,
    rows: tl.constexpr, 
    cols: tl.constexpr, 
    align_cols: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    3D dynamic quantization kernel
    
    Args:
        input: Pointer to input tensor with shape [batch, rows, cols]
        scale: Pointer to scale tensor with shape [cols]
        output: Pointer to output tensor with shape [batch, rows, cols] (int8)
        quant_scale: Pointer to quantization scale tensor with shape [batch, rows]
        batch: Number of batches
        rows: Number of rows per batch
        cols: Number of columns
        align_cols: Aligned column size (power of 2)
        BLOCK_SIZE_M: Block size for M dimension (rows)
        BLOCK_SIZE_N: Block size for N dimension (cols)
    
    Memory layout: [batch, rows, cols]
    Scale shape: [cols] (broadcast to all batches and rows)
    Output shape: [batch, rows, cols] (int8)
    Quantization scale shape: [batch, rows]
    """
    
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    # Calculate total number of elements to process: batch * rows
    total_elements = batch * rows
    num_tasks = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # Process each task
    for task_id in range(pid, num_tasks, grid_size):
        # Calculate block start position
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)
        
        # Convert linear index to 2D indices (batch, row)
        batch_idx = element_off // rows
        row_idx = element_off % rows
        
        # Create mask for valid elements
        element_mask = element_off < total_elements
        
        # Initialize accumulator for max absolute values
        max_abs_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        # First pass: find max absolute value for each row
        for col_block_offset in range(0, align_cols, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < cols
            
            # Calculate input pointer offset for current block
            # Memory layout: [batch, rows, cols]
            input_offset = (batch_idx[:, None] * rows * cols + 
                           row_idx[:, None] * cols + 
                           cols_off[None, :])
            
            input_ptr = input + input_offset
            scale_ptr = scale + cols_off
            
            # Create combined mask
            block_mask = element_mask[:, None] & cols_mask[None, :]
            
            # Load input values
            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            
            # Load scale values
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)
            
            # Apply scaling
            scaled_vals = input_vals * scale_vals
            
            # Calculate max absolute value for this column block
            current_max = tl.max(tl.abs(scaled_vals), axis=1)
            
            # Update accumulator with maximum values
            max_abs_accumulator = tl.maximum(max_abs_accumulator, current_max)
        
        # Calculate quantization scale for each row
        final_max_abs = max_abs_accumulator
        current_quant_scale = final_max_abs / 127.0
        
        # Store quantization scale
        quant_scale_ptr = quant_scale + element_off
        tl.store(quant_scale_ptr, current_quant_scale, mask=element_mask)
        
        # Second pass: perform quantization
        for col_block_offset in range(0, align_cols, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < cols
            
            # Calculate input and output pointer offsets
            input_offset = (batch_idx[:, None] * rows * cols + 
                           row_idx[:, None] * cols + 
                           cols_off[None, :])
            
            input_ptr = input + input_offset
            output_ptr = output + input_offset
            scale_ptr = scale + cols_off
            
            # Create combined mask
            block_mask = element_mask[:, None] & cols_mask[None, :]
            
            # Load input values
            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            
            # Load scale values
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)
            
            # Apply scaling and quantization
            scaled_vals = input_vals * scale_vals
            quant_vals = scaled_vals / current_quant_scale[:, None]
            quant_vals_int8 = quant_vals.to(tl.int8)
            
            # Store quantized values
            tl.store(output_ptr, quant_vals_int8, mask=block_mask)

def act_quant_3d(
    input_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    """
    3D dynamic quantization function
    
    Args:
        input_tensor: Input tensor with shape [batch, rows, cols]
        scale_tensor: Scale tensor with shape [cols]
    
    Returns:
        output_tensor: Quantized tensor with shape [batch, rows, cols] (int8)
        quant_scale_tensor: Quantization scale tensor with shape [batch, rows]
    """
    assert input_tensor.dim() == 3, f"Input tensor must be 3D, got {input_tensor.dim()}D"
    assert scale_tensor.dim() == 1, f"Scale tensor must be 1D, got {scale_tensor.dim()}D"
    assert input_tensor.shape[2] == scale_tensor.shape[0], \
        f"Input cols {input_tensor.shape[2]} must match scale length {scale_tensor.shape[0]}"
    
    batch, rows, cols = input_tensor.shape
    device = input_tensor.device
    
    # Create output tensors
    output_tensor = torch.empty_like(input_tensor, dtype=torch.int8)
    quant_scale_tensor = torch.empty(batch, rows, device=device, dtype=torch.float32)
    
    # Align columns for better memory access
    align_cols = triton.next_power_of_2(cols)
    
    # Get number of programs
    try:
        import triton.runtime.driver as driver
        num_programs = driver.active.utils.get_device_properties(torch.npu.current_device())[
            "num_vectorcore"
        ]
    except AttributeError:
        num_programs = 48
    
    grid = (num_programs,)
    
    # Call kernel
    scale_dynamic_quant_kernel_3d[grid](
        input_tensor,
        scale_tensor,
        output_tensor,
        quant_scale_tensor.view(-1),  # Flatten for easier indexing
        batch=batch,
        rows=rows,
        cols=cols,
        align_cols=align_cols,
        BLOCK_SIZE_N=256,
    )
    
    return output_tensor, quant_scale_tensor

def act_quant_4d(input_tensor: torch.Tensor, scale_tensor: torch.Tensor):
    """
    4d quant
    input shape: [batch, seq_len, rows, cols]
    scale shape: [cols]
    output shape: [batch, seq_len, rows, cols] (int8)
    scale_output shape: [batch, seq_len, rows]
    """
    assert input_tensor.dim() == 4 and scale_tensor.dim() == 1
    assert input_tensor.shape[-1] == scale_tensor.shape[0]
    
    batch, seq_len, rows, cols = input_tensor.shape
    device = input_tensor.device
    
    # create output tensor
    output_tensor = torch.empty_like(input_tensor, dtype=torch.int8)
    quant_scale_tensor = torch.empty(batch, seq_len, rows, device=device, dtype=torch.float32)
    
    align_cols = triton.next_power_of_2(cols)
    
    total_elements = batch * seq_len * rows
    
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)
    
    scale_dynamic_quant_kernel_4d[grid](
        input_tensor,
        scale_tensor,
        output_tensor,
        quant_scale_tensor.view(-1),
        batch=batch,
        seq_len=seq_len,
        rows=rows,
        cols=cols,
        align_cols=align_cols,
        BLOCK_SIZE_N=256,
    )

    return output_tensor, quant_scale_tensor

def act_quant(input_tensor: torch.Tensor, scale_tensor: torch.Tensor):
    if input_tensor.dim() == 3:
        output_tensor, quant_scale_tensor = act_quant_3d(input_tensor, scale_tensor)
    elif input_tensor.dim() == 4:
        output_tensor, quant_scale_tensor = act_quant_4d(input_tensor, scale_tensor)
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, got {q.dim()}D")
    return output_tensor, quant_scale_tensor

# hadamard
@triton.jit
def hadamard_kernel(
    output_ptr,
    n: tl.constexpr,
    log2_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    output shape: [n, n]
    """
    pid = tl.program_id(axis=0)
    
    row_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offs < n
    
    for col_block in range(0, n, BLOCK_SIZE):
        col_offs = col_block + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < n
        
        row_indices = row_offs[:, None]
        col_indices = col_offs[None, :]
        
        result = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1, dtype=tl.int8)
        
        for k in range(log2_n):
            row_bit = (row_indices >> k) & 1
            col_bit = (col_indices >> k) & 1
            
            condition = (row_bit == 1) & (col_bit == 1)
            result = tl.where(condition, -result, result)
        
        output_offset = row_offs[:, None] * n + col_offs[None, :]
        output_ptr_block = output_ptr + output_offset
        
        block_mask = row_mask[:, None] & col_mask[None, :]
        
        tl.store(output_ptr_block, result, mask=block_mask)

def hadamard_triton(n: int, dtype, device):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, but got {n}")
    
    log2_n = int(math.log2(n))
    
    output = torch.empty((n, n), dtype=dtype, device=device)
    
    if n <= 32:
        BLOCK_SIZE = 16
    elif n <= 128:
        BLOCK_SIZE = 32
    elif n <= 512:
        BLOCK_SIZE = 64
    else:
        BLOCK_SIZE = 128
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    hadamard_kernel[grid](
        output,
        n=n,
        log2_n=log2_n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def rotate_activation(x: torch.Tensor):
    hidden_size = x.size(-1)
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = linear(x, hadamard_triton(dim_padded, dtype=x.dtype, device=x.device))
    out = out * hidden_size**-0.5
    return out[..., :dim].reshape(*x_shape)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_N": 512}),
    ],
    key=["H", "N", "K"],
)
@libentry()
@triton.jit
def lightning_index_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    q_s_ptr,
    k_s_ptr,
    B,
    H: tl.constexpr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        q_ptr (tl.tensor): Pointer to the Q matrix.
        k_ptr (tl.tensor): Pointer to the K matrix.
        o_ptr (tl.tensor): Pointer to the index score output.
        q_s_ptr (tl.tensor): Pointer to scaling factors for Q (float).
        k_s_ptr (tl.tensor): Pointer to scaling factors for K (float).
        B : Batch size.
        H (tl.constexpr): Number of Q heads.
        M : Q sequence length.
        N : K sequence length.
        K (tl.constexpr): Dimension length.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.

    Returns:
        None
    """

    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = B * M * NUM_BLOCKS_N

    # Current M-dimension block index
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        batch_idx = block_idx // (M * NUM_BLOCKS_N)
        bm_idx = tl.cast(block_idx // NUM_BLOCKS_N, tl.int64)
        n_idx = block_idx % NUM_BLOCKS_N

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        # Load Q and its scale
        q_ptrs = q_ptr + bm_idx * H * K + offs_h[:, None] * K + offs_k[None, :]
        q_s_ptrs = q_s_ptr + bm_idx * H + offs_h
        q = tl.load(q_ptrs)
        q_s = tl.load(q_s_ptrs)

        # Load K and its scale
        # Calculate K scale pointer offset based on its dimensions
        # K scale can be [B, N, K], [B, N], [N, K], or [N]
        # Here we assume K scale shape is [B, N, K] for per-element scaling
        
        # Calculate K pointer position
        k_batch_offset = batch_idx * N * K
        k_n_offset = n_idx * BLOCK_SIZE_N * K
        k_ptrs = (
            k_ptr
            + k_batch_offset
            + k_n_offset
            + offs_n[:, None] * K
            + offs_k[None, :]
        )
        
        # Create mask for K
        n_mask = n_idx * BLOCK_SIZE_N + offs_n < N
        mask_2d = n_mask[:, None]  # Expand to 2D mask
        
        # Load K values
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0)
        
        # Load K scale (handle different scale shapes)
        # Case 1: K scale is [B, N, K] - per-element scaling
        # Case 2: K scale is [B, N] - per-token scaling
        # Case 3: K scale is [N, K] - per-token per-dimension scaling
        # Case 4: K scale is [N] - per-token scaling
        
        # Assume K scale shape is [B, N, K] (most flexible case)
        k_s_ptrs = (
            k_s_ptr
            + k_batch_offset
            + k_n_offset
            + offs_n[:, None] * K
            + offs_k[None, :]
        )
        k_s = tl.load(k_s_ptrs, mask=mask_2d, other=1.0)  # Default scale is 1.0
        
        # Apply K scale (per-element scaling)
        k_scaled = k * k_s

        # Calculate dot product and apply ReLU
        # q: [H, K], k_scaled: [BLOCK_SIZE_N, K]
        # Transpose k_scaled first: [K, BLOCK_SIZE_N]
        k_t = tl.trans(k_scaled)
        
        # Calculate dot product: q @ k_t = [H, BLOCK_SIZE_N]
        dot_product = tl.dot(q, k_t)
        
        # Apply ReLU
        relu_dot = tl.maximum(dot_product, 0.0)
        
        # Apply Q scale: [H, BLOCK_SIZE_N] * q_s[:, None] = [H, BLOCK_SIZE_N]
        scaled_relu = relu_dot * q_s[:, None]
        
        # Sum along head dimension: [BLOCK_SIZE_N]
        o = tl.sum(scaled_relu, axis=0)

        # Store result
        o_ptrs = o_ptr + bm_idx * N + n_idx * BLOCK_SIZE_N + offs_n
        tl.store(o_ptrs, o, mask=n_mask)

def lightning_index(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
):
    """
    Lightning index calculation with both query and key scaling.
    
    Args:
        query: [B, M, H, K] query tensor
        query_scale: [B, M, H] query scaling factors
        key: [B, N, K] key tensor
        key_scale: Optional scaling factors for key. 
                   Can be [B, N, K], [B, N], [N, K], or [N]
    
    Returns:
        index scores: [B, M, N]
    """
    assert query.is_contiguous() and key.is_contiguous(), "Input tensors must be contiguous"
    
    B, M, H, K = query.size()
    N = key.size(1)
    
    # Validate query_scale shape
    assert query_scale.size() == (B, M, H), f"query_scale must be [B, M, H], got {query_scale.size()}"
    
    # Handle key_scale
    if key_scale is None:
        # If key_scale is not provided, create all-ones tensor
        key_scale = torch.ones((B, N, K), dtype=torch.float32, device=query.device)
    else:
        # Validate key_scale shape and potentially broadcast
        key_scale_shape = key_scale.size()
        if len(key_scale_shape) == 1:
            # [N] -> expand to [B, N, K]
            assert key_scale_shape[0] == N, f"key_scale [N] must have N={N}, got {key_scale_shape[0]}"
            key_scale = key_scale.unsqueeze(0).unsqueeze(-1).expand(B, -1, K)
        else:
            raise ValueError(f"Invalid key_scale shape {key_scale_shape}")
        
        # Ensure key_scale is contiguous
        key_scale = key_scale.contiguous()
    
    # Ensure all inputs are contiguous
    query = query.contiguous()
    query_scale = query_scale.contiguous()
    key = key.contiguous()
    
    # Create output tensor
    o = torch.zeros((B, M, N), dtype=torch.float32, device=query.device)
    
    # Get number of compute cores
    try:
        import triton.runtime.driver as driver
        NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())[
            "num_vectorcore"
        ]
    except AttributeError:
        NUM_CORE = 48
        
    grid = (NUM_CORE // 2,)
    
    # Call kernel
    lightning_index_kernel[grid](
        query,  # q_ptr
        key,    # k_ptr
        o,      # o_ptr
        query_scale,  # q_s_ptr
        key_scale,    # k_s_ptr
        B,      # B
        H,      # H
        M,      # M
        N,      # N
        K,      # K
    )
    return o

def indexer_forward(
    x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, config: TritonConfig, k_cache, k_scale_cache, mask: Optional[torch.Tensor] = None
):
    # unpack config
    dim = config.dim
    head_dim = config.head_dim
    n_heads = config.n_heads
    topk = config.topk
    rope_head_dim = config.rope_head_dim
    q_lora_rank = config.q_lora_rank

    x_device = x.device
    x_dtype = x.dtype
    q_device = qr.device
    q_dtype = qr.dtype
    bsz, seqlen, _ = x.shape
    end_pos = start_pos + seqlen

    cos = freqs_cis.real.unsqueeze(0).expand(bsz, -1, -1) # BSD
    sin = freqs_cis.imag.unsqueeze(0).expand(bsz, -1, -1) # BSD

    # handle k
    w = torch.ones(dim, head_dim, dtype=x_dtype, device=x_device)
    k = linear(x, w)
    wk = torch.ones(k.size(-1), dtype=x_dtype, device=x_device)
    bias = torch.zeros(k.size(-1), dtype=x_dtype, device=x_device)
    k, _, _, _ = layer_norm(k.float(), wk, bias, device=x_device)
    k = rope(k, cos, sin, rope_head_dim)
    k = rotate_activation(k)

    # handle q
    wq = torch.ones(q_lora_rank, (n_heads*head_dim), dtype=q_dtype, device=q_device)
    q = linear(qr, wq)
    q = q.view(bsz, seqlen, n_heads, head_dim)
    q = rope(q, cos, sin, rope_head_dim)
    q = rotate_activation(q)

    # get scale
    scale = torch.ones(q.shape[-1])
    q_int8, q_scale = act_quant(q, scale)
    k_int8, k_scale = act_quant(k, scale)

    k_cache[:bsz, start_pos:end_pos] = k_int8
    k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    weights = torch.ones(dim, n_heads, dtype=x_dtype)
    q_int8_scale = linear(x, weights) * n_heads**-0.5 # q_int8_scale shape [bsz, seq, n_heads]

    # get topk
    index_scores = lightning_index(
        q_int8.contiguous(),
        q_int8_scale,
        k_cache[:bsz, :end_pos].contiguous(),
        k_scale_cache[:bsz, :end_pos].contiguous(),
    )
    if mask is not None:
            index_score += mask
    topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1]
    return topk_indices