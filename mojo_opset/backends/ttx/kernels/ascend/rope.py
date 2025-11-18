from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry


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
    key=["n_qh", "n_kh", "hd", "seq_len"],
    restore_value=["q_ptr", "k_ptr"],
)
@libentry()
@triton.jit
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
    q_rope_ptr,
    k_rope_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    half_hd: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
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

            dim_offsets = tl.arange(0, half_hd)
            dim_mask = dim_offsets < half_hd

            cos_block_2d = tl.load(
                cos_token_ptr + dim_offsets[None, :], mask=seq_mask[:, None] & dim_mask[None, :], other=0
            )
            sin_block_2d = tl.load(
                sin_token_ptr + dim_offsets[None, :], mask=seq_mask[:, None] & dim_mask[None, :], other=0
            )

            head_q_offsets = tl.arange(0, n_qh)
            head_k_offsets = tl.arange(0, n_kh)

            # q_base_ptr = q_ptr + batch_idx * q_batch_stride
            q_offsets_half1 = (
                # q_base_ptr
                batch_idx * q_batch_stride
                + seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + dim_offsets[None, None, :]
            )
            q_offsets_half2 = q_offsets_half1 + (half_hd)
            q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & dim_mask[None, None, :]

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_mask, other=0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_mask, other=0).to(sin_block_2d.dtype)

            cos_row = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_hd), can_reorder=True)
            sin_row = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_hd), can_reorder=True)

            new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
            new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row

            # q_base_ptr = q_rope_ptr + batch_idx * q_batch_stride
            # q_offsets_half1 = (
            #     q_base_ptr
            #     + seq_offsets[:, None, None] * q_seq_stride
            #     + head_q_offsets[None, :, None] * hd
            #     + dim_offsets[None, None, :]
            # )
            # q_offsets_half2 = q_offsets_half1 + (half_hd)
            tl.store(q_rope_ptr + q_offsets_half1, new_q_tile_1, mask=q_mask)
            tl.store(q_rope_ptr + q_offsets_half2, new_q_tile_2, mask=q_mask)

            # k_base_ptr = k_ptr + batch_idx * k_batch_stride
            k_offsets_half1 = (
                batch_idx * k_batch_stride
                + seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + dim_offsets[None, None, :]
            )
            k_offsets_half2 = k_offsets_half1 + (half_hd)
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & dim_mask[None, None, :]

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_mask, other=0).to(sin_block_2d.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_mask, other=0).to(sin_block_2d.dtype)

            new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
            new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row

            tl.store(k_rope_ptr + k_offsets_half1, new_k_tile_1, mask=k_mask)
            tl.store(k_rope_ptr + k_offsets_half2, new_k_tile_2, mask=k_mask)


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
    key=["n_qh", "n_kh", "hd", "seq_len"],
    restore_value=["dq_ptr", "dk_ptr"],
)
@libentry()
@triton.jit
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
    grad_q_ptr,
    grad_k_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    half_hd: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
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

            dim_offsets = tl.arange(0, half_hd)
            dim_mask = dim_offsets < half_hd

            cos_block_2d = tl.load(
                cos_token_ptr + dim_offsets[None, :], mask=seq_mask[:, None] & dim_mask[None, :], other=0
            )
            sin_block_2d = tl.load(
                sin_token_ptr + dim_offsets[None, :], mask=seq_mask[:, None] & dim_mask[None, :], other=0
            )

            head_q_offsets = tl.arange(0, n_qh)
            head_k_offsets = tl.arange(0, n_kh)

            # dq_base_ptr = dq_ptr + batch_idx * dq_batch_stride
            dq_offsets_half1 = (
                # dq_base_ptr
                batch_idx * dq_batch_stride
                + seq_offsets[:, None, None] * dq_seq_stride
                + head_q_offsets[None, :, None] * hd
                + dim_offsets[None, None, :]
            )
            dq_offsets_half2 = dq_offsets_half1 + (half_hd)
            q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & dim_mask[None, None, :]

            dq_tile_1 = tl.load(dq_ptr + dq_offsets_half1, mask=q_mask, other=0).to(sin_block_2d.dtype)
            dq_tile_2 = tl.load(dq_ptr + dq_offsets_half2, mask=q_mask, other=0).to(sin_block_2d.dtype)

            cos_row = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_hd), can_reorder=True)
            sin_row = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_hd), can_reorder=True)

            new_dq_tile_1 = dq_tile_1 * cos_row + dq_tile_2 * sin_row
            new_dq_tile_2 = dq_tile_2 * cos_row - dq_tile_1 * sin_row

            tl.store(grad_q_ptr + dq_offsets_half1, new_dq_tile_1, mask=q_mask)
            tl.store(grad_q_ptr + dq_offsets_half2, new_dq_tile_2, mask=q_mask)

            # dk_base_ptr = dk_ptr + batch_idx * dk_batch_stride
            dk_offsets_half1 = (
                batch_idx * dk_batch_stride
                + seq_offsets[:, None, None] * dk_seq_stride
                + head_k_offsets[None, :, None] * hd
                + dim_offsets[None, None, :]
            )
            dk_offsets_half2 = dk_offsets_half1 + (half_hd)
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & dim_mask[None, None, :]

            dk_tile_1 = tl.load(dk_ptr + dk_offsets_half1, mask=k_mask, other=0).to(sin_block_2d.dtype)
            dk_tile_2 = tl.load(dk_ptr + dk_offsets_half2, mask=k_mask, other=0).to(sin_block_2d.dtype)

            new_dk_tile_1 = dk_tile_1 * cos_row + dk_tile_2 * sin_row
            new_dk_tile_2 = dk_tile_2 * cos_row - dk_tile_1 * sin_row

            tl.store(grad_k_ptr + dk_offsets_half1, new_dk_tile_1, mask=k_mask)
            tl.store(grad_k_ptr + dk_offsets_half2, new_dk_tile_2, mask=k_mask)


@torch.library.custom_op("ttx::rope", mutates_args={})
def rome_fwd(
    q: torch.Tensor,  # [BNSD]
    k: torch.Tensor,  # [BNSD]
    cos: torch.Tensor,  # [BSD]
    sin: torch.Tensor,  # [BSD]
) -> Tuple[torch.Tensor, torch.Tensor]:  # [BNSD]
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    q_rope = torch.empty_like(q_t)
    k_rope = torch.empty_like(k_t)

    batch_size, seq_len, n_q_head, head_dim = q_t.shape
    n_kv_head = k_t.shape[2]

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    cos_batch_size = cos.shape[0]
    cos = cos.contiguous()
    sin = sin.contiguous()

    _rope_forward_kernel[grid](
        q_t,
        q_t.stride(0),
        q_t.stride(1),
        k_t,
        k_t.stride(0),
        k_t.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        q_rope,
        k_rope,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        head_dim // 2,
    )

    # return q_rope.transpose(1, 2).contiguous(), k_rope.transpose(1, 2).contiguous()
    return q_rope.transpose(1, 2), k_rope.transpose(1, 2)


@rome_fwd.register_fake
def rope_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("ttx::rope_bwd", mutates_args={})
def rope_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dq_t = dq.transpose(1, 2).contiguous()
    dk_t = dk.transpose(1, 2).contiguous()

    grad_q = torch.empty_like(dq_t)
    grad_k = torch.empty_like(dk_t)

    batch_size, seq_len, n_q_head, head_dim = dq_t.shape
    n_kv_head = dk_t.shape[2]
    cos_batch_size = cos.shape[0]
    cos = cos.contiguous()
    sin = sin.contiguous()

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _rope_backward_kernel[grid](
        dq_t,
        dq_t.stride(0),
        dq_t.stride(1),
        dk_t,
        dk_t.stride(0),
        dk_t.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        grad_q,
        grad_k,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        head_dim // 2,
    )

    return grad_q.transpose(1, 2).contiguous(), grad_k.transpose(1, 2).contiguous()


@rope_bwd.register_fake
def rope_bwd_fake(
    dq: torch.Tensor,
    dk: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(dq), torch.empty_like(dk)
