import torch
import triton
import triton.language as tl
from math import ceil

from mojo_opset.core.operators.over_encoding import get_nf4_codebook

__910B_UB_MAX_SIZE__ = 192 * 2**10
__MAX_UB_TILING_SIZE__ = triton.next_power_of_2(__910B_UB_MAX_SIZE__ // 4)


@triton.heuristics(
    values={
        "BLOCK_EMBEDDING_DIM": lambda args: (
            8192
            if args["LUT"].size(1) >= __MAX_UB_TILING_SIZE__ // 4
            else args["LUT"].size(1)
        )
    }
)
@triton.jit
def embedding_kernel(
    output: torch.Tensor,
    input: torch.Tensor,
    batch_size: int,
    LUT: torch.Tensor,
    embedding_dim: int,
    LUT_stride_0: int,
    LUT_stride_1: int,
    output_stride_0: int,
    output_stride_1: int,
    vocab_start_id: int,  # TODO(liuyuan): tl.constexpr would be better?
    vocab_size: int,  # TODO(liuyuan): tl.constexpr would be better?
    BLOCK_TOKEN_SIZE: tl.constexpr,
    BLOCK_EMBEDDING_DIM: tl.constexpr,
):
    tid = tl.program_id(0)  # token batch id
    eid = tl.program_id(1)  # embedding slicing id

    token_start = tid * BLOCK_TOKEN_SIZE
    embedding_vec_start = eid * BLOCK_EMBEDDING_DIM
    embedding_vec_enums = tl.arange(0, BLOCK_EMBEDDING_DIM)
    embedding_vec_idx = embedding_vec_start + embedding_vec_enums
    embedding_vec_mask = embedding_vec_idx < embedding_dim

    for idx in tl.static_range(BLOCK_TOKEN_SIZE):
        __token_idx = token_start + idx
        __token_mask = __token_idx < batch_size
        # WARNING(liuyuan): 0 <= __token_id < vocab_size should be guaranteed by user.
        __token_id = tl.load(input + __token_idx, mask=__token_mask, other=-1)
        __token_id -= vocab_start_id
        __token_mask &= __token_id >= 0
        __token_mask &= __token_id < vocab_size

        __embedding_vec_mask = __token_mask & embedding_vec_mask
        __embedding_vec = tl.load(
            LUT + __token_id * LUT_stride_0 + embedding_vec_idx * LUT_stride_1,
            mask=__embedding_vec_mask,
            other=0,
        )

        __output_vec_ptr = (
            output
            + __token_idx * output_stride_0
            + embedding_vec_idx * output_stride_1
        )
        tl.store(__output_vec_ptr, __embedding_vec)


@triton.jit
def __embedding_nf4_dequant__(
    input_idx,
    token_id,
    output:torch.Tensor,
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    codebook: torch.Tensor,
    embedding_dim: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    LUT_qweight_stride_0: tl.constexpr,
    LUT_qweight_stride_1: tl.constexpr,
    LUT_scale_stride_0: tl.constexpr,
    LUT_scale_stride_1: tl.constexpr,
    LUT_mean_stride_0: tl.constexpr,
    LUT_mean_stride_1: tl.constexpr,
    vocab_start_id: tl.constexpr,
    vocab_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_EMBEDDING_DIM: tl.constexpr,
):
    token_id -= vocab_start_id
    token_mask = token_id >= 0
    token_mask &= token_id < vocab_size

    for __embedding_vec_start in tl.range(0, embedding_dim, BLOCK_EMBEDDING_DIM):
        __packed_vec_offsets = tl.arange(0, BLOCK_EMBEDDING_DIM // 2)
        __packed_vec_idx = __embedding_vec_start // 2 + __packed_vec_offsets
        __packed_vec_mask = __packed_vec_idx < (embedding_dim // 2)
        __low_embedding_vec_idx = __embedding_vec_start + __packed_vec_offsets * 2
        __high_embedding_vec_idx = __low_embedding_vec_idx + 1
        __low_group_vec_idx = __low_embedding_vec_idx // GROUP_SIZE
        __high_group_vec_idx = __high_embedding_vec_idx // GROUP_SIZE
        __embedding_vec_idx = __embedding_vec_start + tl.arange(0, BLOCK_EMBEDDING_DIM)


        __packed_mask = token_mask & __packed_vec_mask

        __packed_qweight = tl.load(
            LUT_qweight + token_id * LUT_qweight_stride_0 + __packed_vec_idx * LUT_qweight_stride_1,
            mask=__packed_mask,
            other=0,
        ).to(tl.int32)
        __low_nf4_idx = __packed_qweight & 0x0F
        __high_nf4_idx = (__packed_qweight >> 4) & 0x0F

        __low_output_mask = token_mask & (__low_embedding_vec_idx < embedding_dim)
        __low_nf4_val = tl.load(
            codebook + __low_nf4_idx, mask=__low_output_mask, other=0
        ).to(tl.float32)
        __low_scale = tl.load(
            LUT_scale + token_id * LUT_scale_stride_0 + __low_group_vec_idx * LUT_scale_stride_1,
            mask=__low_output_mask,
            other=0,
        ).to(tl.float32)
        __low_mean = tl.load(
            LUT_mean + token_id * LUT_mean_stride_0 + __low_group_vec_idx * LUT_mean_stride_1,
            mask=__low_output_mask,
            other=0,
        ).to(tl.float32)
        __low_output_vec = __low_nf4_val * __low_scale + __low_mean

        __high_output_mask = token_mask & (__high_embedding_vec_idx < embedding_dim)
        __high_nf4_val = tl.load(
            codebook + __high_nf4_idx, mask=__high_output_mask, other=0
        ).to(tl.float32)
        __high_scale = tl.load(
            LUT_scale + token_id * LUT_scale_stride_0 + __high_group_vec_idx * LUT_scale_stride_1,
            mask=__high_output_mask,
            other=0,
        ).to(tl.float32)
        __high_mean = tl.load(
            LUT_mean + token_id * LUT_mean_stride_0 + __high_group_vec_idx * LUT_mean_stride_1,
            mask=__high_output_mask,
            other=0,
        ).to(tl.float32)
        __high_output_vec = __high_nf4_val * __high_scale + __high_mean
        __output_vec = tl.reshape(
            tl.join(__low_output_vec, __high_output_vec),
            (BLOCK_EMBEDDING_DIM,),
        )
        __output_mask = token_mask & (__embedding_vec_idx < embedding_dim)
        __output_vec_ptr = (
            output
            + input_idx * output_stride_0
            + __embedding_vec_idx * output_stride_1
        )
        tl.store(__output_vec_ptr, __output_vec, mask=__output_mask)


@triton.heuristics(
    values={
        "BLOCK_EMBEDDING_DIM": lambda args: (
            __MAX_UB_TILING_SIZE__ // 8
            if args["embedding_dim"]  >= __MAX_UB_TILING_SIZE__ // 8
            else args["embedding_dim"]
        )
    }
)
@triton.jit
def embedding_nf4_dequant_kernel(
    output: torch.Tensor,
    input: torch.Tensor,
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    codebook: torch.Tensor,
    token_num: tl.constexpr,
    embedding_dim: tl.constexpr,
    LUT_qweight_stride_0: tl.constexpr,
    LUT_qweight_stride_1: tl.constexpr,
    LUT_scale_stride_0: tl.constexpr,
    LUT_scale_stride_1: tl.constexpr,
    LUT_mean_stride_0: tl.constexpr,
    LUT_mean_stride_1: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    vocab_start_id: tl.constexpr,
    vocab_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_TOKEN_SIZE: tl.constexpr,
    BLOCK_EMBEDDING_DIM: tl.constexpr,
):
    tid = tl.program_id(0)  # token batch id
    token_start = tid * BLOCK_TOKEN_SIZE

    for idx in tl.range(BLOCK_TOKEN_SIZE):
        __token_idx = token_start + idx
        __token_mask = __token_idx < token_num

        __token_id = tl.load(input + __token_idx, mask=__token_mask, other=-1).to(tl.int64)
        __embedding_nf4_dequant__(
            __token_idx,
            __token_id,
            output,
            LUT_qweight,
            LUT_scale,
            LUT_mean,
            codebook,
            embedding_dim,
            output_stride_0,
            output_stride_1,
            LUT_qweight_stride_0,
            LUT_qweight_stride_1,
            LUT_scale_stride_0,
            LUT_scale_stride_1,
            LUT_mean_stride_0,
            LUT_mean_stride_1,
            vocab_start_id,
            vocab_size,
            GROUP_SIZE,
            BLOCK_EMBEDDING_DIM,
        )


def embedding_nf4_dequant_impl(
    input: torch.Tensor,
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    *,
    group_size: int = 1,
    codebook: torch.Tensor = None,
    vocab_start_id: int = 0,
    vocab_size: int = None,
    output_dtype: torch.dtype = torch.bfloat16,
):
    if input.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        raise ValueError(f"`input` must be an integer tensor, got {input.dtype}.")

    if LUT_qweight.ndim != 2 or LUT_scale.ndim != 2 or LUT_mean.ndim != 2:
        raise ValueError(
            "NF4 embedding tensors must all be 2D, "
            f"got qweight={tuple(LUT_qweight.shape)}, "
            f"scale={tuple(LUT_scale.shape)}, mean={tuple(LUT_mean.shape)}."
        )

    if LUT_scale.shape != LUT_mean.shape:
        raise ValueError(
            f"`LUT_scale` and `LUT_mean` must have the same shape, got {LUT_scale.shape} and {LUT_mean.shape}."
        )

    if group_size <= 0:
        raise ValueError(f"`group_size` must be > 0, got {group_size}.")

    if len({input.device, LUT_qweight.device, LUT_scale.device, LUT_mean.device}) != 1:
        raise ValueError(
            "`input`, `LUT_qweight`, `LUT_scale`, and `LUT_mean` must be on the same device."
        )

    embedding_dim = LUT_scale.size(1) * group_size
    if LUT_qweight.size(1) * 2 != embedding_dim:
        raise ValueError(
            f"`LUT_qweight` shape {tuple(LUT_qweight.shape)} is incompatible with "
            f"`LUT_scale` shape {tuple(LUT_scale.shape)} and group_size={group_size}."
        )

    vocab_size = LUT_qweight.size(0) if vocab_size is None else vocab_size
    input_flat = input.contiguous().view(-1)

    if input_flat.numel() == 0:
        return torch.empty(
            (*input.shape, embedding_dim),
            dtype=output_dtype,
            device=input.device,
        )

    if codebook is None:
        codebook = get_nf4_codebook(device=LUT_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=LUT_qweight.device, dtype=torch.float16)

    output = torch.empty(
        (input_flat.numel(), embedding_dim),
        dtype=output_dtype,
        device=input.device,
    )

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")[
        "num_vectorcore"
    ]

    grid = (num_programs,)
    embedding_nf4_dequant_kernel[grid](
        output=output,
        input=input_flat,
        LUT_qweight=LUT_qweight,
        LUT_scale=LUT_scale,
        LUT_mean=LUT_mean,
        codebook=codebook,
        token_num=input_flat.numel(),
        embedding_dim=embedding_dim,
        LUT_qweight_stride_0=LUT_qweight.stride(0),
        LUT_qweight_stride_1=LUT_qweight.stride(1),
        LUT_scale_stride_0=LUT_scale.stride(0),
        LUT_scale_stride_1=LUT_scale.stride(1),
        LUT_mean_stride_0=LUT_mean.stride(0),
        LUT_mean_stride_1=LUT_mean.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        vocab_start_id=vocab_start_id,
        vocab_size=vocab_size,
        GROUP_SIZE=group_size,
        BLOCK_TOKEN_SIZE=ceil(input_flat.numel() / num_programs),
        multibuffer=False,
    )
    return output.view(*input.shape, embedding_dim)
