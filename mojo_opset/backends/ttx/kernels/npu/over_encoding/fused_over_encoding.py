import math
import torch
import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.npu.over_encoding.embedding import __embedding_nf4_dequant__
from mojo_opset.core.operators.over_encoding import get_nf4_codebook

__910B_UB_MAX_SIZE__ = 192 * 2**10
__MAX_UB_TILING_SIZE__ = triton.next_power_of_2(__910B_UB_MAX_SIZE__ // 2)


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
def over_encoding_decode_kernel(
    output: torch.Tensor,  # [bs * MTP_STEP * oe_vocab_sizes.size(0), embedding_dim]
    input_ids: torch.Tensor,  # [bs, 1]
    oe_history: torch.Tensor,  # [bs, MAX_N_GRAM - 1]
    n_grams: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    oe_vocab_sizes: torch.Tensor,  #  [ (N - 1) * K]
    oe_vocab_offsets: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    codebook: torch.Tensor,
    ori_vocab_size: tl.constexpr,
    mega_vocab_size: tl.constexpr,
    batch_size: tl.constexpr,
    n_grams_size: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    oe_history_dim_1: tl.constexpr,
    oe_history_stride_0: tl.constexpr,
    oe_history_stride_1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_BATCH_SIZE: tl.constexpr,
    MAX_N_GRAM: tl.constexpr,
    MTP_STEP: tl.constexpr,
    embedding_dim: tl.constexpr,
    LUT_qweight_stride_0: tl.constexpr,
    LUT_qweight_stride_1: tl.constexpr,
    LUT_scale_stride_0: tl.constexpr,
    LUT_scale_stride_1: tl.constexpr,
    LUT_mean_stride_0: tl.constexpr,
    LUT_mean_stride_1: tl.constexpr,
    vocab_start_id: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_EMBEDDING_DIM: tl.constexpr,
):
    start_bid = tl.program_id(0) * BLOCK_BATCH_SIZE
    if start_bid >= batch_size:
        return

    block_offsets = tl.arange(0, BLOCK_SIZE_N)
    block_mask = block_offsets < n_grams_size

    oe_vocab_sizes = tl.load(
        oe_vocab_sizes + block_offsets, mask=block_mask, other=1
    ).to(tl.int64)
    oe_vocab_offsets = tl.load(
        oe_vocab_offsets + block_offsets, mask=block_mask, other=0
    ).to(tl.int64)
    if MTP_STEP > 1:
        oe_vocab_offsets = tl.broadcast_to(
            tl.view(oe_vocab_offsets, (1, BLOCK_SIZE_N)), (MTP_STEP, BLOCK_SIZE_N)
        )
    n_grams = tl.load(n_grams + block_offsets, mask=block_mask, other=-1).to(tl.int64)

    for bid_cnt in (tl.static_range if BLOCK_BATCH_SIZE < 4 else tl.range)(
        0, BLOCK_BATCH_SIZE
    ):
        bid = start_bid + bid_cnt
        if bid < batch_size:
            if MTP_STEP > 1:
                input_indices = tl.arange(0, MTP_STEP)
                __input_ids = tl.load(input_ids + bid * MTP_STEP + input_indices).to(tl.int64)

                n_gram_ids = tl.view(__input_ids, (MTP_STEP, 1))
                n_gram_ids = tl.broadcast_to(n_gram_ids, (MTP_STEP, BLOCK_SIZE_N))

                oe_carry = tl.full((MTP_STEP, BLOCK_SIZE_N,), ori_vocab_size, dtype=tl.int64)

                history_ptr = oe_history + oe_history_stride_0 * bid
                n_gram_offsets = tl.flip(tl.arange(0, MAX_N_GRAM))

                history_id = tl.load(
                    history_ptr + (oe_history_dim_1 - n_gram_offsets - 1) * oe_history_stride_1
                ).to(tl.int64)

                # WARNING(liuyuan): tl.cat required the same shapes of lhs and rhs in triton-npu. WTF?
                # history_id = tl.cat(history_id, __input_ids, can_reorder=True)
                __tmp = tl.zeros((MTP_STEP + MAX_N_GRAM,), dtype=tl.int64)
                __tmp = tl.insert_slice(__tmp, history_id, (0,), (MAX_N_GRAM,), (1,))
                __tmp = tl.insert_slice(__tmp, __input_ids, (MAX_N_GRAM,), (MTP_STEP,), (1,))
                history_id = __tmp

                for i in tl.static_range(1, MAX_N_GRAM):
                    __cal_mask = n_grams >= (i + 1)
                    __history_ids = tl.extract_slice(
                        history_id, (MAX_N_GRAM - i,), (MTP_STEP,), (1,)
                    )
                    __history_ids = (
                        tl.broadcast_to(
                            tl.view(__history_ids, (MTP_STEP, 1)), (MTP_STEP, BLOCK_SIZE_N)
                        )
                        * __cal_mask
                    )

                    n_gram_ids = (n_gram_ids + oe_carry * __history_ids) % oe_vocab_sizes
                    oe_carry = oe_carry * ori_vocab_size % oe_vocab_sizes
                n_gram_ids += oe_vocab_offsets

                for ele_idx in (tl.static_range if BLOCK_BATCH_SIZE < 4 else tl.range)(
                    0, MTP_STEP * n_grams_size
                ):
                    __id = tl.get_element(n_gram_ids, (ele_idx,))
                    __embedding_nf4_dequant__(
                        ele_idx + bid * MTP_STEP * MAX_N_GRAM,
                        __id,
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
                        mega_vocab_size,
                        GROUP_SIZE,
                        BLOCK_EMBEDDING_DIM,
                    )

            else:
                input_id = tl.load(input_ids + bid)
                n_gram_ids = tl.full((BLOCK_SIZE_N,), input_id, dtype=tl.int64)

                oe_carry = tl.full((BLOCK_SIZE_N,), ori_vocab_size, dtype=tl.int64)

                history_ptr = oe_history + oe_history_stride_0 * bid

                for i in tl.static_range(1, MAX_N_GRAM):
                    __history_id = tl.load(
                        history_ptr + (oe_history_dim_1 - i) * oe_history_stride_1
                    ).to(tl.int64)
                    __cal_mask = n_grams >= (i + 1)
                    __history_ids = (
                        tl.full((BLOCK_SIZE_N,), __history_id, dtype=__history_id.dtype)
                        * __cal_mask
                    )
                    n_gram_ids = (n_gram_ids + oe_carry * __history_ids) % oe_vocab_sizes
                    oe_carry = oe_carry * ori_vocab_size % oe_vocab_sizes
                n_gram_ids += oe_vocab_offsets


                for ele_idx in (tl.static_range if BLOCK_BATCH_SIZE < 4 else tl.range)(
                    0, MTP_STEP * n_grams_size
                ):
                    __id = tl.get_element(n_gram_ids, (ele_idx,))
                    __embedding_nf4_dequant__(
                        ele_idx + bid * MTP_STEP * n_grams_size,
                        __id,
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
                        mega_vocab_size,
                        GROUP_SIZE,
                        BLOCK_EMBEDDING_DIM,
                    )


def over_encoding_decode_impl(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    *,
    group_size: int = 1,
    codebook: torch.Tensor = None,
    ori_vocab_size: int = None,
    mega_vocab_start_id: int = 0,
    mega_vocab_size: int = None,
    output_dtype: torch.dtype = torch.bfloat16,
):

    if input_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        raise ValueError(f"`input_ids` must be an integer tensor, got {input_ids.dtype}.")

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

    if len({input_ids.device, LUT_qweight.device, LUT_scale.device, LUT_mean.device}) != 1:
        raise ValueError(
            "`input_ids`, `LUT_qweight`, `LUT_scale`, and `LUT_mean` must be on the same device."
        )

    embedding_dim = LUT_scale.size(1) * group_size
    if LUT_qweight.size(1) * 2 != embedding_dim:
        raise ValueError(
            f"`LUT_qweight` shape {tuple(LUT_qweight.shape)} is incompatible with "
            f"`LUT_scale` shape {tuple(LUT_scale.shape)} and group_size={group_size}."
        )

    mega_vocab_size = LUT_qweight.size(0) if mega_vocab_size is None else mega_vocab_size
    input_ids_flat = input_ids.contiguous().view(-1)

    if input_ids_flat.numel() == 0:
        return torch.empty(
            (*input_ids.shape, embedding_dim),
            dtype=output_dtype,
            device=input_ids.device,
        )

    if codebook is None:
        codebook = get_nf4_codebook(device=LUT_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=LUT_qweight.device, dtype=torch.float16)

    output = torch.zeros(
        input_ids.numel() * oe_vocab_sizes.size(0),
        embedding_dim,
        dtype=output_dtype,
        device="npu",
    )

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")[
        "num_vectorcore"
    ]

    grid = (num_programs,)
    over_encoding_decode_kernel[grid](
        output=output,
        input_ids=input_ids,
        oe_history=oe_history_inputs,
        n_grams=n_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        oe_vocab_offsets=oe_vocab_offsets,
        LUT_qweight=LUT_qweight,
        LUT_scale=LUT_scale,
        LUT_mean=LUT_mean,
        codebook=codebook,
        ori_vocab_size=ori_vocab_size,
        mega_vocab_size=mega_vocab_size,
        batch_size=input_ids.size(0),
        n_grams_size=n_grams.size(0),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        oe_history_dim_1=oe_history_inputs.size(1),
        oe_history_stride_0=oe_history_inputs.stride(0),
        oe_history_stride_1=oe_history_inputs.stride(1),
        BLOCK_SIZE_N=triton.next_power_of_2(n_grams.size(0)),
        BLOCK_BATCH_SIZE=triton.next_power_of_2(
            math.ceil(input_ids.size(0) / num_programs)
        ),
        MAX_N_GRAM=max(n_grams).item(),
        MTP_STEP=input_ids.size(1),
        embedding_dim=embedding_dim,
        LUT_qweight_stride_0=LUT_qweight.stride(0),
        LUT_qweight_stride_1=LUT_qweight.stride(1),
        LUT_scale_stride_0=LUT_scale.stride(0),
        LUT_scale_stride_1=LUT_scale.stride(1),
        LUT_mean_stride_0=LUT_mean.stride(0),
        LUT_mean_stride_1=LUT_mean.stride(1),
        vocab_start_id=mega_vocab_start_id,
        GROUP_SIZE=group_size,
    )
    return output.view(
        *input_ids.shape, oe_vocab_sizes.size(0), embedding_dim
    )
