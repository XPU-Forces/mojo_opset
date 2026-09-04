# Copyright 2026, The FlagOS Contributors.

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


@triton.jit
def _hc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    BS,
    HC: tl.constexpr,
    D,
    x_stride_0,
    residual_stride_0,
    residual_stride_1,
    post_stride_0,
    comb_stride_0,
    comb_stride_1,
    y_stride_0,
    y_stride_1,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    d_blocks = tl.cdiv(D, BLOCK_D)

    for bs_idx in range(pid, BS, grid_size):
        post_vec = tl.load(post_ptr + bs_idx * post_stride_0 + tl.arange(0, HC)).to(tl.float32)

        comb_block = tl.load(
            comb_ptr + bs_idx * comb_stride_0
            + tl.arange(0, HC)[:, None] * comb_stride_1
            + tl.arange(0, HC)[None, :]
        ).to(tl.float32)

        for d_block_idx in range(d_blocks):
            d_off = d_block_idx * BLOCK_D
            d_indices = d_off + tl.arange(0, BLOCK_D)
            mask = d_indices < D

            x_block = tl.load(x_ptr + bs_idx * x_stride_0 + d_indices, mask=mask, other=0.0).to(tl.float32)
            out_buf = post_vec[:, None] * x_block[None, :]

            for j in range(HC):
                res_row = tl.load(
                    residual_ptr + bs_idx * residual_stride_0 + j * residual_stride_1 + d_indices,
                    mask=mask, other=0.0
                ).to(tl.float32)

                comb_row = tl.extract_slice(comb_block, (j, 0), (1, HC), (1, 1))
                comb_row = tl.reshape(comb_row, (HC,))
                out_buf += comb_row[:, None] * res_row[None, :]

            y_offsets = (
                bs_idx * y_stride_0
                + tl.arange(0, HC)[:, None] * y_stride_1
                + d_indices[None, :]
            )
            y_mask = mask[None, :]
            tl.store(y_ptr + y_offsets, out_buf.to(y_ptr.dtype.element_ty), mask=y_mask)


def hc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:

    orig_x_ndim = x.dim()
    orig_residual_shape = residual.shape
    out_dtype = x.dtype

    # Reshape 3D -> 2D
    if orig_x_ndim == 3:
        B, S = x.shape[0], x.shape[1]
        BS = B * S
        x = x.reshape(BS, x.shape[-1])
        residual = residual.reshape(BS, residual.shape[-2], residual.shape[-1])
        post = post.reshape(BS, post.shape[-1])
        comb = comb.reshape(BS, comb.shape[-2], comb.shape[-1])
    else:
        BS = x.shape[0]

    D = x.shape[-1]
    HC = residual.shape[1]
    # The setting is currently for the 910B, but the 910C can go larger.
    BLOCK_D = min(triton.next_power_of_2(D), 4096)

    y = torch.empty(BS, HC, D, dtype=out_dtype, device=x.device)

    num_cores = get_num_cores("vector")
    grid = (min(BS, num_cores),)

    _hc_post_kernel[grid](
        x,
        residual,
        post,
        comb,
        y,
        BS,
        HC,
        D,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        post.stride(0),
        comb.stride(0),
        comb.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_D=BLOCK_D,
    )

    y = y.reshape(orig_residual_shape)
    return y
