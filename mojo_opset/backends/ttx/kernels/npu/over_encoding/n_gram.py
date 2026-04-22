import math
import torch
import triton
import triton.language as tl
from .embedding import __embedding_nf4_dequant__


@triton.heuristics(
    values={
        "BLOCK_SIZE_S": lambda args: min(triton.next_power_of_2(
            args["input_ids"].size(0) // args["oe_history"].size(0)
        ), 2048 * (2 if args["input_ids"].dtype == torch.int32 else 1))
    }
)
@triton.jit
# NOTE(liuyuan): split each sequence and calculated the n_gram ids.
def n_gram_prefill_kerenl(
    output_ids: torch.Tensor,  # [oe_vocab_sizes.size(0), total_seq_len]
    input_ids: torch.Tensor,  # [total_seq_len]
    input_offsets: torch.Tensor,  # [seq_num*2]
    oe_history: torch.Tensor,  # [seq_num, MAX_N_GRAM - 1]
    n_grams: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    oe_vocab_sizes: torch.Tensor,  # [ (N-1) * K]
    oe_vocab_offsets: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    output_stride_0: int,
    output_stride_1: int,
    oe_history_dim_1: int,
    oe_history_stride_0: int,
    oe_history_stride_1: int,
    vocab_size: int,
    BLOCK_IDX_SIZE: tl.constexpr,
    N_GRAM_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_IDX_SIZE

    for k in tl.static_range(BLOCK_IDX_SIZE):
        __idx = idx + k
        sid = __idx // N_GRAM_SIZE
        nid = __idx % N_GRAM_SIZE
        if sid >= SEQ_LEN:
            return

        block_seq_indices = tl.arange(0, BLOCK_SIZE_S)
        cur_seq_start = tl.load(input_offsets + sid)
        cur_seq_end = tl.load(input_offsets + sid + 1)
        block_history_ptr = oe_history + oe_history_stride_0 * sid

        for __global_start in range(cur_seq_start, cur_seq_end, BLOCK_SIZE_S):
            __block_offset = __global_start - cur_seq_start + block_seq_indices
            __global_offset = __global_start + block_seq_indices
            __block_mask = __global_offset < cur_seq_end

            __cur_n_gram = tl.load(n_grams + nid)
            __cur_oe_vocab_size = tl.load(oe_vocab_sizes + nid).to(tl.int64)
            __cur_oe_vocab_offset = tl.load(oe_vocab_offsets + nid).to(tl.int64)

            # WARNING(liuyuan): MUST recompute for the different oe_vocab_sizes.
            __n_gram_ids = tl.load(
                input_ids + __global_offset, mask=__block_mask, other=0
            ).to(tl.int64)
            __oe_carry = vocab_size.to(tl.int64)
            for i in range(1, __cur_n_gram):
                __n_gram_window_idx = __block_offset - i
                __mask = __n_gram_window_idx >= 0
                __ids = tl.load(
                    input_ids + __global_offset - i, mask=__block_mask & __mask, other=0
                )

                __history_ptr = (
                    block_history_ptr
                    + (oe_history_dim_1 + __n_gram_window_idx) * oe_history_stride_1
                )
                __history_ids = tl.load(__history_ptr, mask=~__mask, other=0)

                __merged = tl.where(__mask, __ids, __history_ids)
                __n_gram_ids = (__n_gram_ids + __merged * __oe_carry) % __cur_oe_vocab_size
                __oe_carry = __oe_carry * vocab_size % __cur_oe_vocab_size

            __n_gram_ids += __cur_oe_vocab_offset
            __output_ptr = (
                output_ids + nid * output_stride_0 + __global_offset * output_stride_1
            )
            tl.store(
                __output_ptr,
                __n_gram_ids.to(__output_ptr.dtype.element_ty),
                mask=__block_mask,
            )


def n_gram_prefill_impl(
    input_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    vocab_size: int,
):

    input_offsets = torch.cumsum(
        torch.cat(
            (torch.tensor([0], dtype=seq_lens.dtype, device="npu"), seq_lens), dim=0
        ),
        dim=0,
    )

    output = torch.empty(
        oe_vocab_sizes.size(0), input_ids.size(0), dtype=input_ids.dtype, device="npu"
    )
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")[
        "num_vectorcore"
    ]

    grid = (num_programs,)
    n_gram_prefill_kerenl[grid](
        output_ids=output,
        input_ids=input_ids,
        input_offsets=input_offsets,
        oe_history=oe_history_inputs,
        n_grams=n_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        oe_vocab_offsets=oe_vocab_offsets,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        oe_history_dim_1=oe_history_inputs.size(1),
        oe_history_stride_0=oe_history_inputs.stride(0),
        oe_history_stride_1=oe_history_inputs.stride(1),
        vocab_size=vocab_size,
        BLOCK_IDX_SIZE=math.ceil(n_grams.size(0) * seq_lens.size(0) / num_programs),
        N_GRAM_SIZE=n_grams.size(0),
        SEQ_LEN=seq_lens.size(0),
    )
    # NOTE(liuyuan): stored by layout [oe_vocab_sizes.size(0), total_seq_len] but we need [total_seq_len, oe_vocab_sizes.size(0)]
    return output.transpose(0, 1).contiguous()


@triton.jit
def n_gram_decode_kernel(
    output_ids: torch.Tensor,  # [bs, oe_vocab_sizes.size(0)]
    input_ids: torch.Tensor,  # [bs, 1]
    oe_history: torch.Tensor,  # [bs, MAX_N_GRAM - 1]
    n_grams: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    oe_vocab_sizes: torch.Tensor,  #  [ (N - 1) * K]
    oe_vocab_offsets: torch.Tensor,  # [oe_vocab_sizes.size(0)]
    vocab_size: tl.constexpr,
    batch_size: tl.constexpr,
    n_grams_size: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    output_stride_2: tl.constexpr,
    oe_history_dim_1: tl.constexpr,
    oe_history_stride_0: tl.constexpr,
    oe_history_stride_1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_BATCH_SIZE: tl.constexpr,
    MAX_N_GRAM: tl.constexpr,
    MTP_STEP: tl.constexpr,
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
                tl.device_print("__input_ids:", __input_ids)

                n_gram_ids = tl.view(__input_ids, (MTP_STEP, 1))
                n_gram_ids = tl.broadcast_to(n_gram_ids, (MTP_STEP, BLOCK_SIZE_N))

                oe_carry = tl.full((MTP_STEP, BLOCK_SIZE_N,), vocab_size, dtype=tl.int64)

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
                    oe_carry = oe_carry * vocab_size % oe_vocab_sizes
                n_gram_ids += oe_vocab_offsets

                tl.store(
                    output_ids + bid * output_stride_0 + tl.arange(0, MTP_STEP)[:, None] * output_stride_1 + block_offsets[None, :] * output_stride_2,
                    # output_ids + bid * output_stride_0 + output_offsets,
                    n_gram_ids.to(output_ids.dtype.element_ty),
                    mask=tl.view(block_mask, (1, BLOCK_SIZE_N)),
                )

            else:
                input_id = tl.load(input_ids + bid)
                n_gram_ids = tl.full((BLOCK_SIZE_N,), input_id, dtype=tl.int64)

                oe_carry = tl.full((BLOCK_SIZE_N,), vocab_size, dtype=tl.int64)

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
                    oe_carry = oe_carry * vocab_size % oe_vocab_sizes
                n_gram_ids += oe_vocab_offsets

                tl.store(
                    output_ids + bid * output_stride_0 + block_offsets * output_stride_2,
                    n_gram_ids.to(output_ids.dtype.element_ty),
                    mask=block_mask,
                )


def n_gram_decode_impl(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offsets: torch.Tensor,
    n_grams: torch.Tensor,
    vocab_size: int,
):


    output = torch.zeros(
        input_ids.size(0),
        input_ids.size(1),
        oe_vocab_sizes.size(0),
        dtype=input_ids.dtype,
        device="npu",
    )

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")[
        "num_vectorcore"
    ]

    grid = (num_programs,)
    n_gram_decode_kernel[grid](
        output_ids=output,
        input_ids=input_ids,
        oe_history=oe_history_inputs,
        n_grams=n_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        oe_vocab_offsets=oe_vocab_offsets,
        vocab_size=vocab_size,
        batch_size=input_ids.size(0),
        n_grams_size=n_grams.size(0),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        oe_history_dim_1=oe_history_inputs.size(1),
        oe_history_stride_0=oe_history_inputs.stride(0),
        oe_history_stride_1=oe_history_inputs.stride(1),
        BLOCK_SIZE_N=triton.next_power_of_2(n_grams.size(0)),
        BLOCK_BATCH_SIZE=triton.next_power_of_2(
            math.ceil(input_ids.size(0) / num_programs)
        ),
        MAX_N_GRAM=max(n_grams).item(),
        MTP_STEP=input_ids.size(1),
    )
    return output