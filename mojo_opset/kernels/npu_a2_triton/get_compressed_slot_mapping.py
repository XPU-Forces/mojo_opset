"""Ascend A2 (910B) Triton compressed-KV slot mapping.

Structure relative to the vllm original (one program per request, serial query
loop):

1. Pure decode (``num_tokens == num_reqs``, one token per request): a dedicated
   one-dimensional kernel; the token index *is* the request index, so the
   addressing is contiguous and no ``query_start_loc`` / column mask is needed.
2. Long queries (``query_len > batched_q_threshold``) split the query axis over a
   second grid dimension; empty chunks return early.
3. Short queries (``2 <= query_len <= batched_q_threshold``) merge ``RPP`` requests
   into one ``[RPP, BLOCK_Q]`` tile, but only when enough programs remain to fill
   the cores (few requests collapse to ``grid=1`` and leave most cores idle).
4. A persistent ``out`` buffer only needs its tail cleared: every dispatch path
   overwrites ``[0, num_tokens)``.
5. An empty batch skips the launch (a fully masked store on an empty A2 tensor
   hits an MTE illegal address).
6. ``max_query_len=None`` falls back to the conservative per-request structure.
7. An optional ``slot_contig`` flag buffer (with the token ``positions`` tensor)
   marks which group-tail tokens continue the previous token's slot at a
   consecutive position; ``rope_quant_insert`` consumes the flags to take its
   2D contiguous-run fast path.

A2 calibration (910B): ``MIN_PROGRAMS=24``, ``SPLITQ_BLOCK=16`` and
``DECODE_BLOCK=8``; the one-dimensional decode kernel is enabled on this arch
only.
"""

import torch
import triton
import triton.language as tl

_LANE_BUDGET = 512
_BATCHED_Q_THRESHOLD = 128
_MIN_PROGRAMS = 24
_SPLITQ_BLOCK = 16
_DECODE_BLOCK = 8


@triton.jit(do_not_specialize=["block_table_stride"])
def _compressed_slot_mapping_native(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    dcp_world_size,
    cp_interleave,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # Row offsets are int64: a 32-bit batch index * stride wraps past 2^31 on
    # large page tables.
    batch_idx = tl.program_id(0).to(tl.int64)
    block_table_stride = block_table_stride.to(tl.int64)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        # A replayed token has a PAD slot (SWA bounded replay: its KV and its
        # compressed KV are cached already), so it maps to PAD here.
        slot = tl.load(slot_mapping_ptr + query_start + offset, mask=mask, other=PAD_ID)
        is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
        # Under DCP a PAD slot also marks tokens owned by other ranks; the
        # interleave is a multiple of COMPRESS_RATIO, so a state lives entirely
        # on the rank owning its last token.
        local_pos = pos // (cp_interleave * dcp_world_size) * cp_interleave + pos % cp_interleave
        pos_after_compress = local_pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers.to(tl.int64) * block_size + (pos_after_compress % block_size)
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(compressed_slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["block_table_stride"])
def _compressed_slot_mapping_batched(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    dcp_world_size,
    cp_interleave,
    num_reqs,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    RPP: tl.constexpr,
):
    # Row offsets are int64: a 32-bit pid * stride wraps past 2^31 on large
    # page tables.
    pid = tl.program_id(0).to(tl.int64)
    block_table_stride = block_table_stride.to(tl.int64)
    reqs = pid * RPP + tl.arange(0, RPP)
    req_mask = reqs < num_reqs

    q_start = tl.load(query_start_loc_ptr + reqs, mask=req_mask, other=0)
    q_end = tl.load(query_start_loc_ptr + reqs + 1, mask=req_mask, other=0)
    q_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + reqs, mask=req_mask, other=0)
    start_pos = seq_len - q_len

    cols = tl.arange(0, BLOCK_Q)
    m = req_mask[:, None] & (cols[None, :] < q_len[:, None])
    tok = q_start[:, None] + cols[None, :]
    pos = start_pos[:, None] + cols[None, :]

    slot = tl.load(slot_mapping_ptr + tok, mask=m, other=PAD_ID)
    is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
    local_pos = pos // (cp_interleave * dcp_world_size) * cp_interleave + pos % cp_interleave
    pos_after_compress = local_pos // COMPRESS_RATIO

    block_ids = pos_after_compress // block_size
    block_numbers = tl.load(
        block_table_ptr + reqs[:, None] * block_table_stride + block_ids,
        mask=m & is_valid,
        other=0,
    )
    slot_ids = block_numbers.to(tl.int64) * block_size + (pos_after_compress % block_size)
    slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
    tl.store(compressed_slot_mapping_ptr + tok, slot_ids, mask=m)


@triton.jit(do_not_specialize=["block_table_stride"])
def _compressed_slot_mapping_decode(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    dcp_world_size,
    cp_interleave,
    num_tokens,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row offsets are int64: a 32-bit pid * stride wraps past 2^31 on large page
    # tables.
    pid = tl.program_id(0).to(tl.int64)
    block_table_stride = block_table_stride.to(tl.int64)
    tok = pid * BLOCK + tl.arange(0, BLOCK)
    m = tok < num_tokens

    pos = tl.load(seq_lens_ptr + tok, mask=m, other=0) - 1
    slot = tl.load(slot_mapping_ptr + tok, mask=m, other=PAD_ID)
    is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
    # dcp=cp=1 makes the DCP terms identity; Triton's equal_to_1 specialisation
    # folds them at compile time.
    local_pos = pos // (cp_interleave * dcp_world_size) * cp_interleave + pos % cp_interleave
    pos_after_compress = local_pos // COMPRESS_RATIO

    block_numbers = tl.load(
        block_table_ptr + tok * block_table_stride + pos_after_compress // block_size,
        mask=m & is_valid,
        other=0,
    )
    slot_ids = block_numbers.to(tl.int64) * block_size + (pos_after_compress % block_size)
    slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
    tl.store(compressed_slot_mapping_ptr + tok, slot_ids, mask=m)


@triton.jit(do_not_specialize=["block_table_stride"])
def _compressed_slot_mapping_splitq(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    dcp_world_size,
    cp_interleave,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row offsets are int64: a 32-bit batch index * stride wraps past 2^31 on
    # large page tables.
    batch_idx = tl.program_id(0).to(tl.int64)
    block_table_stride = block_table_stride.to(tl.int64)
    chunk = tl.program_id(1)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start
    if chunk * BLOCK >= query_len:
        return

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    offset = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offset < query_len

    pos = start_pos + offset
    slot = tl.load(slot_mapping_ptr + query_start + offset, mask=mask, other=PAD_ID)
    is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
    local_pos = pos // (cp_interleave * dcp_world_size) * cp_interleave + pos % cp_interleave
    pos_after_compress = local_pos // COMPRESS_RATIO

    block_ids = pos_after_compress // block_size
    block_numbers = tl.load(
        block_table_ptr + batch_idx * block_table_stride + block_ids,
        mask=mask & is_valid,
        other=0,
    )
    slot_ids = block_numbers.to(tl.int64) * block_size + (pos_after_compress % block_size)
    slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
    tl.store(compressed_slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def _pow2_floor(value):
    return 1 << (value.bit_length() - 1)


def _pick_rpp(block_q, num_reqs):
    """Lane-budget cap and program-count floor, quantised to a power of two."""
    rpp_cap = max(1, _LANE_BUDGET // block_q)
    rpp_min = max(1, _pow2_floor((num_reqs + _MIN_PROGRAMS - 1) // _MIN_PROGRAMS))
    return max(1, min(rpp_cap, rpp_min))



@triton.jit
def _slot_contig_kernel(out_ptr, positions_ptr, slot_contig_ptr, num_tokens,
                        COMPRESS_RATIO: tl.constexpr):
    """Paired-line producer: 1 iff slot AND position continue the previous token."""
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    m = offs < num_tokens
    cur = tl.load(out_ptr + offs, mask=m, other=-1)
    prev = tl.load(out_ptr + offs - 1, mask=m & (offs > 0), other=-2)
    pos = tl.load(positions_ptr + offs, mask=m, other=-1)
    prev_pos = tl.load(positions_ptr + offs - 1, mask=m & (offs > 0), other=-2)
    cont = (
        (cur != -1)
        & (cur == prev + 1)
        & (pos == prev_pos + 1)
        & ((pos + 1) % COMPRESS_RATIO == 0)
    )
    tl.store(slot_contig_ptr + offs, cont.to(tl.uint8), mask=m)


def _get_compressed_slot_mapping_impl(
    num_tokens,
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_size,
    compress_ratio,
    out,
    dcp_world_size,
    cp_interleave,
    max_query_len,
    slot_contig=None,
    positions=None,
):
    if out.numel() > num_tokens:
        # Every path overwrites [0, num_tokens); only the tail needs clearing to
        # keep graph replay / padding residue out of the persistent buffer.
        out[num_tokens:].fill_(-1)
    compressed_slot_mapping = out[:num_tokens]

    num_reqs = block_table.shape[0]
    stride = block_table.stride(0)

    if num_tokens == 0 or num_reqs == 0:
        return compressed_slot_mapping

    if max_query_len is None:
        _compressed_slot_mapping_native[(num_reqs,)](
            compressed_slot_mapping,
            slot_mapping,
            query_start_loc,
            seq_lens,
            block_table,
            stride,
            block_size,
            dcp_world_size,
            cp_interleave,
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=1024,
        )
    elif num_tokens == num_reqs:
        # Pure decode: contiguous one-dimensional addressing.
        _compressed_slot_mapping_decode[(triton.cdiv(num_tokens, _DECODE_BLOCK),)](
            compressed_slot_mapping,
            slot_mapping,
            seq_lens,
            block_table,
            stride,
            block_size,
            dcp_world_size,
            cp_interleave,
            num_tokens,
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            BLOCK=_DECODE_BLOCK,
        )
    else:
        # BLOCK_Q only needs next_pow2(max_query_len); a lower bound would burn
        # lanes for q_len=1 shapes.
        block_q = max(triton.next_power_of_2(max_query_len), 1)
        rpp = _pick_rpp(block_q, num_reqs)
        if max_query_len <= _BATCHED_Q_THRESHOLD and triton.cdiv(num_reqs, rpp) >= _MIN_PROGRAMS:
            _compressed_slot_mapping_batched[(triton.cdiv(num_reqs, rpp),)](
                compressed_slot_mapping,
                slot_mapping,
                query_start_loc,
                seq_lens,
                block_table,
                stride,
                block_size,
                dcp_world_size,
                cp_interleave,
                num_reqs,
                COMPRESS_RATIO=compress_ratio,
                PAD_ID=-1,
                BLOCK_Q=block_q,
                RPP=rpp,
            )
        else:
            block = min(_SPLITQ_BLOCK, block_q)
            _compressed_slot_mapping_splitq[(num_reqs, triton.cdiv(max_query_len, block))](
                compressed_slot_mapping,
                slot_mapping,
                query_start_loc,
                seq_lens,
                block_table,
                stride,
                block_size,
                dcp_world_size,
                cp_interleave,
                COMPRESS_RATIO=compress_ratio,
                PAD_ID=-1,
                BLOCK=block,
            )
    if slot_contig is not None:
        _slot_contig_kernel[(triton.cdiv(num_tokens, 256),)](
            compressed_slot_mapping, positions, slot_contig, num_tokens,
            COMPRESS_RATIO=compress_ratio,
        )
    return compressed_slot_mapping


@torch.library.custom_op(
    "mojo_npu_triton_a2::get_compressed_slot_mapping_fwd",
    mutates_args=("out", "slot_contig"),
)
def get_compressed_slot_mapping_fwd(
    num_tokens: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor,
    dcp_world_size: int,
    cp_interleave: int,
    max_query_len: int | None,
    slot_contig: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> None:
    _get_compressed_slot_mapping_impl(
        num_tokens,
        slot_mapping.contiguous(),
        query_start_loc.contiguous(),
        seq_lens.contiguous(),
        block_table,
        block_size,
        compress_ratio,
        out,
        dcp_world_size,
        cp_interleave,
        max_query_len,
        slot_contig,
        positions,
    )


@get_compressed_slot_mapping_fwd.register_fake
def _get_compressed_slot_mapping_fake(
    num_tokens, slot_mapping, query_start_loc, seq_lens, block_table,
    block_size, compress_ratio, out, dcp_world_size, cp_interleave, max_query_len,
    slot_contig, positions,
) -> None:
    return None
