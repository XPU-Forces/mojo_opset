from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry


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
    q_t = q.contiguous()
    
    # Determine rope_dim, default is dim
    batch_size, seq_len, dim = q_t.shape
    if rope_dim is None:
        rope_dim = dim
    
    # Validate cos/sin shapes
    cos_batch_size = cos.shape[0]
    
    q_rope = torch.empty_like(q_t)
    
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)
    
    cos_contig = cos.contiguous()
    sin_contig = sin.contiguous()
    
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
    
    cos_batch_size = cos.shape[0]
    
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

def indexer_rope_impl(
        q: torch.Tensor,  # [BNSD]
        k: torch.Tensor,  # [BSD]
        cos: torch.Tensor,  # [BSD]
        sin: torch.Tensor,  # [BSD]
        rope_head_dim: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rp = rope(q, cos, sin, rope_head_dim)
    k_rp = rope(k, cos, sin, rope_head_dim)
    return q_rp, k_rp