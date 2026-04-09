from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.utils import prepare_lens
from mojo_opset.backends.ttx.kernels.utils import tensor_cache

from .utils import ilu_grid_dim_from_row_tasks
from .utils import libentry

ROPE_TOKEN_BLOCK_SIZE_TABLE = {
    (2, 1): 36,
    (4, 1): 16,
    (8, 1): 10,
    (16, 16): 5,
    (32, 32): 2,
    (64, 64): 1,
}


def _normalize_to_bsnd(
    q: torch.Tensor,
    k: torch.Tensor,
    head_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int, int, int, int]:
    """Normalize q/k to [B, S, N, D] layout, returning strides and metadata (same contract as NPU rope)."""

    if q.dim() == 3:
        assert k.dim() == 3
        if head_first:
            q = q.transpose(0, 1).clone(memory_format=torch.contiguous_format)
            k = k.transpose(0, 1).clone(memory_format=torch.contiguous_format)
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)
        batch_size = 1
        seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[1]
        q_batch_stride, q_seq_stride = 0, q.stride(0)
        k_batch_stride, k_seq_stride = 0, k.stride(0)
    else:
        assert q.dim() == 4 and k.dim() == 4
        if head_first:
            q = q.transpose(1, 2).clone(memory_format=torch.contiguous_format)
            k = k.transpose(1, 2).clone(memory_format=torch.contiguous_format)
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)

        batch_size, seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[2]
        q_batch_stride, q_seq_stride = q.stride(0), q.stride(1)
        k_batch_stride, k_seq_stride = k.stride(0), k.stride(1)

    return (
        q,
        k,
        batch_size,
        seq_len,
        n_q_head,
        n_kv_head,
        head_dim,
        q_batch_stride,
        q_seq_stride,
        k_batch_stride,
        k_seq_stride,
    )


def _ilu_cos_batch_size(cos: torch.Tensor, q_like: torch.Tensor, *, is_varlen: bool) -> int:
    """How many cos/sin rows exist along the batch-like leading dim for offset math in the kernel.

    Padded prefill and varlen pass cos as ``[S, rope]`` or ``[T, rope]`` (2D); the reference
    unsqueezes to ``[1, …, rope]`` so all logical batches share the same cos rows. Using
    ``cos.shape[0]`` on 2D cos would wrongly treat sequence length as batch size (breaks bs>1).
    Decode uses ``[B, rope]`` where ``cos.shape[0] == B``. Mirrors NPU ``cos_batch_stride = 0``
    when ``cos.dim() != 3``.
    """
    if cos.dim() == 3:
        return cos.shape[0]
    if cos.dim() == 2:
        if is_varlen or q_like.dim() == 4:
            return 1
        return cos.shape[0]
    raise AssertionError(f"cos/sin must be 2D or 3D, got dim={cos.dim()} shape={tuple(cos.shape)}")


def _get_token_block_size(n_qh: int, n_kh: int) -> int:
    assert n_qh <= 84 and n_kh <= 84, "don't support head_num > 84, please raise an issue."

    if (n_qh, n_kh) in ROPE_TOKEN_BLOCK_SIZE_TABLE:
        return ROPE_TOKEN_BLOCK_SIZE_TABLE[(n_qh, n_kh)]

    for (q_thresh, k_thresh), block_size in sorted(
        ROPE_TOKEN_BLOCK_SIZE_TABLE.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        if n_qh <= q_thresh and n_kh <= k_thresh:
            return block_size

    return 1


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    kv_lens: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    lens = prepare_lens(cu_seqlens)
    num_chunks = triton.cdiv(lens, chunk_size)
    total = num_chunks.sum()
    flat = torch.arange(total, device=cu_seqlens.device)
    seq_ids = torch.repeat_interleave(torch.arange(num_chunks.numel(), device=cu_seqlens.device), num_chunks)
    offsets = torch.cumsum(num_chunks, 0) - num_chunks
    chunk_indices = flat - offsets[seq_ids]

    seq_starts = cu_seqlens[:-1]
    seq_start_per_block = seq_starts[seq_ids]

    if kv_lens is not None:
        sin_cos_offset_per_block = kv_lens[seq_ids]
    else:
        sin_cos_offset_per_block = torch.zeros_like(seq_ids)

    return torch.stack([seq_ids, chunk_indices, seq_start_per_block, sin_cos_offset_per_block, lens[seq_ids]], dim=1)


@triton.jit
def _compute_rope_separated(
    x1,
    x2,
    sin_tile,
    cos_tile,
    inverse: tl.constexpr,
):
    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile
    return roped_x1, roped_x2


@libentry()
@triton.jit(do_not_specialize=["seq_len"])
def _rope_forward_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    TOKEN_PAD: tl.constexpr,
    HALF_ROPE_PAD: tl.constexpr,
    N_QH_PAD: tl.constexpr,
    N_KH_PAD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    offs_t = tl.arange(0, TOKEN_PAD)
    half_rope_dim_offsets = tl.arange(0, HALF_ROPE_PAD)
    half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
    head_q_offsets = tl.arange(0, N_QH_PAD)
    head_k_offsets = tl.arange(0, N_KH_PAD)

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + offs_t
            seq_mask = (offs_t < TOKEN_BLOCK_SIZE) & (seq_offsets < actual_seq_len)

            global_seq_offsets = seq_start + seq_offsets

            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
            sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + offs_t
            seq_mask = (offs_t < TOKEN_BLOCK_SIZE) & (seq_offsets < seq_len)

            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
                cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
                sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
                sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )

        # Avoid tl.reshape(..., can_reorder=True): on ILU it can scramble cos/sin layout vs rows.
        cos_tile = cos_block_2d[:, None, :]
        sin_tile = sin_block_2d[:, None, :]

        q_offsets_half1 = (
            batch_idx * q_batch_stride
            + global_seq_offsets[:, None, None] * q_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        q_offsets_half2 = (
            batch_idx * q_batch_stride
            + global_seq_offsets[:, None, None] * q_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        q_half_mask = (
            seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
        )

        q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
        q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
        new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, False)
        tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
        tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

        k_offsets_half1 = (
            batch_idx * k_batch_stride
            + global_seq_offsets[:, None, None] * k_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        k_offsets_half2 = (
            batch_idx * k_batch_stride
            + global_seq_offsets[:, None, None] * k_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        k_half_mask = (
            seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
        )

        k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
        k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
        new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, False)
        tl.store(k_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
        tl.store(k_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)


@libentry()
@triton.jit(do_not_specialize=["seq_len"])
def _rope_backward_kernel(
    dq_ptr,
    dq_batch_stride,
    dq_seq_stride,
    dk_ptr,
    dk_batch_stride,
    dk_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    TOKEN_PAD: tl.constexpr,
    HALF_ROPE_PAD: tl.constexpr,
    N_QH_PAD: tl.constexpr,
    N_KH_PAD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    offs_t = tl.arange(0, TOKEN_PAD)
    half_rope_dim_offsets = tl.arange(0, HALF_ROPE_PAD)
    half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
    head_q_offsets = tl.arange(0, N_QH_PAD)
    head_k_offsets = tl.arange(0, N_KH_PAD)

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + offs_t
            seq_mask = (offs_t < TOKEN_BLOCK_SIZE) & (seq_offsets < actual_seq_len)

            global_seq_offsets = seq_start + seq_offsets

            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
            sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + offs_t
            seq_mask = (offs_t < TOKEN_BLOCK_SIZE) & (seq_offsets < seq_len)

            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
                cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
                sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
                sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )

        cos_tile = cos_block_2d[:, None, :]
        sin_tile = sin_block_2d[:, None, :]

        dq_offsets_half1 = (
            batch_idx * dq_batch_stride
            + global_seq_offsets[:, None, None] * dq_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        dq_offsets_half2 = (
            batch_idx * dq_batch_stride
            + global_seq_offsets[:, None, None] * dq_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        q_half_mask = (
            seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
        )

        dq_tile_1 = tl.load(dq_ptr + dq_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
        dq_tile_2 = tl.load(dq_ptr + dq_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
        new_dq_1, new_dq_2 = _compute_rope_separated(dq_tile_1, dq_tile_2, sin_tile, cos_tile, True)
        tl.store(dq_ptr + dq_offsets_half1, new_dq_1, mask=q_half_mask)
        tl.store(dq_ptr + dq_offsets_half2, new_dq_2, mask=q_half_mask)

        dk_offsets_half1 = (
            batch_idx * dk_batch_stride
            + global_seq_offsets[:, None, None] * dk_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        dk_offsets_half2 = (
            batch_idx * dk_batch_stride
            + global_seq_offsets[:, None, None] * dk_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + half_rope_dim
            + half_rope_dim_offsets[None, None, :]
        )
        k_half_mask = (
            seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
        )

        dk_tile_1 = tl.load(dk_ptr + dk_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
        dk_tile_2 = tl.load(dk_ptr + dk_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
        new_dk_1, new_dk_2 = _compute_rope_separated(dk_tile_1, dk_tile_2, sin_tile, cos_tile, True)
        tl.store(dk_ptr + dk_offsets_half1, new_dk_1, mask=k_half_mask)
        tl.store(dk_ptr + dk_offsets_half2, new_dk_2, mask=k_half_mask)


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    is_varlen = cu_seqlens is not None
    has_kv_lens = kv_lens is not None

    orig_q_shape = q.shape
    orig_k_shape = k.shape
    (
        q,
        k,
        batch_size,
        seq_len,
        n_q_head,
        n_kv_head,
        head_dim,
        q_batch_stride,
        q_seq_stride,
        k_batch_stride,
        k_seq_stride,
    ) = _normalize_to_bsnd(q, k, head_first)
    q_for_cos_bs = q

    if is_varlen:
        assert q.dim() == 3 and k.dim() == 3, "q and k must be [total_seq_len, n_head, head_dim] after normalize."

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim >= 0, "cos/sin last dim must not exceed head_dim"
    half_rope_dim = rope_dim // 2

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    # ILU Triton: tl.arange(0, N) requires N to be a power of 2; pad with masks in kernel.
    token_pad = triton.next_power_of_2(token_block_size)
    half_rope_pad = triton.next_power_of_2(max(1, half_rope_dim))
    n_qh_pad = triton.next_power_of_2(max(1, n_q_head))
    n_kh_pad = triton.next_power_of_2(max(1, n_kv_head))

    chunk_indices = prepare_chunk_indices(cu_seqlens, token_block_size, kv_lens) if is_varlen else None

    # 沿序列维按 TOKEN_BLOCK_SIZE 切 tile：非 varlen 时 num_seq_blocks = ceil(seq_len / TOKEN) = triton.cdiv(seq_len, TOKEN)。
    # 内核里 total_blocks = bs * num_seq_blocks，block_id 遍历每个 (batch, seq_tile)。
    num_seq_blocks = chunk_indices.shape[0] if is_varlen else triton.cdiv(seq_len, token_block_size)
    total_tile_blocks = batch_size * num_seq_blocks
    grid = (ilu_grid_dim_from_row_tasks(total_tile_blocks),)

    cos_batch_size = _ilu_cos_batch_size(cos, q_for_cos_bs, is_varlen=is_varlen)
    cos = cos.contiguous()
    sin = sin.contiguous()

    _rope_forward_kernel[grid](
        q,
        q_batch_stride,
        q_seq_stride,
        k,
        k_batch_stride,
        k_seq_stride,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        chunk_indices,
        kv_lens,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        token_pad,
        half_rope_pad,
        n_qh_pad,
        n_kh_pad,
        is_varlen,
        has_kv_lens,
    )

    if head_first:
        q = q.transpose(-2, -3).contiguous()
        k = k.transpose(-2, -3).contiguous()
    q = q.reshape(*orig_q_shape)
    k = k.reshape(*orig_k_shape)
    return q, k


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same argument order as NPU ``rope_bwd_impl`` (dq, dk, cos, sin, head_first, ...)."""
    is_varlen = cu_seqlens is not None
    has_kv_lens = kv_lens is not None

    orig_dq_shape = dq.shape
    orig_dk_shape = dk.shape
    (
        dq,
        dk,
        batch_size,
        seq_len,
        n_q_head,
        n_kv_head,
        head_dim,
        dq_batch_stride,
        dq_seq_stride,
        dk_batch_stride,
        dk_seq_stride,
    ) = _normalize_to_bsnd(dq, dk, head_first)
    dq_for_cos_bs = dq

    if is_varlen:
        assert dq.dim() == 3 and dk.dim() == 3, "dq and dk must be [total_seq_len, n_head, head_dim] after normalize."

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim >= 0, "cos/sin last dim must not exceed head_dim"
    half_rope_dim = rope_dim // 2

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    token_pad = triton.next_power_of_2(token_block_size)
    half_rope_pad = triton.next_power_of_2(max(1, half_rope_dim))
    n_qh_pad = triton.next_power_of_2(max(1, n_q_head))
    n_kh_pad = triton.next_power_of_2(max(1, n_kv_head))

    chunk_indices = prepare_chunk_indices(cu_seqlens, token_block_size, kv_lens) if is_varlen else None

    num_seq_blocks = chunk_indices.shape[0] if is_varlen else triton.cdiv(seq_len, token_block_size)
    total_tile_blocks = batch_size * num_seq_blocks
    grid = (ilu_grid_dim_from_row_tasks(total_tile_blocks),)

    cos_batch_size = _ilu_cos_batch_size(cos, dq_for_cos_bs, is_varlen=is_varlen)
    cos = cos.contiguous()
    sin = sin.contiguous()

    _rope_backward_kernel[grid](
        dq,
        dq_batch_stride,
        dq_seq_stride,
        dk,
        dk_batch_stride,
        dk_seq_stride,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        chunk_indices,
        kv_lens,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        token_pad,
        half_rope_pad,
        n_qh_pad,
        n_kh_pad,
        is_varlen,
        has_kv_lens,
    )

    if head_first:
        dq = dq.transpose(-2, -3).contiguous()
        dk = dk.transpose(-2, -3).contiguous()
    dq = dq.reshape(*orig_dq_shape)
    dk = dk.reshape(*orig_dk_shape)
    return dq, dk


def rot_pos_embed_impl(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqlens_kv: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather cos/sin from the RoPE table (ILU).

    Mirrors NPU behavior: ``position_ids`` may be 1D ``[S]`` or 2D ``[B, S]`` (pos3d), so
    ``x`` can stay ``[S, H]`` while indices are batched — no need to match core
    ``MojoRotaryEmbedding`` shape checks when using ``TTXRotaryEmbedding`` on ILU.
    """
    if position_ids is not None:
        return cos[position_ids], sin[position_ids]
    if cu_seqlens_q is None:
        seq_len = x.shape[-2]
        return cos[:seq_len], sin[:seq_len]

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if seqlens_kv is not None:
        context_lens = seqlens_kv - seqlens_q
    else:
        context_lens = torch.zeros_like(seqlens_q, dtype=seqlens_q.dtype, device=seqlens_q.device)

    total = x.shape[0]
    rope_dim = cos.shape[-1]
    device = x.device
    dtype = cos.dtype
    cos_out = torch.empty((total, rope_dim), device=device, dtype=dtype)
    sin_out = torch.empty((total, rope_dim), device=device, dtype=dtype)
    for i in range(seqlens_q.numel()):
        start = int(cu_seqlens_q[i].item())
        end = int(cu_seqlens_q[i + 1].item())
        q_len = end - start
        ctx = int(context_lens[i].item())
        positions = torch.arange(ctx, ctx + q_len, device=device, dtype=torch.int64)
        cos_out[start:end] = cos[positions.to(cos.device)]
        sin_out[start:end] = sin[positions.to(sin.device)]
    return cos_out, sin_out
