"""Torch-only packed BlockMask construction with bounded temporary memory."""

import torch
import torch.nn.functional as _F

from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import _convert_mask_to_block_mask
from torch.nn.attention.flex_attention import _dense_to_ordered
from torch.nn.attention.flex_attention import create_mask as _torch_create_mask
from torch.nn.attention.flex_attention import noop_mask

from mojo_opset.utils.target import get_torch_device as _get_torch_device

_STRIPE_TARGET_BYTES = 256 * 1024**2

_BYTES_PER_MASK_ELEMENT = 8


def _create_sparse_block_from_block_mask(block_mask, mask_mod, seq_lengths, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """Build both index directions directly, without a strided reverse-index view."""

    def ordered(mask):
        # Explicit copy avoids singleton-view argsort failures on some torch_npu/CANN versions.
        return _dense_to_ordered(mask.clone(memory_format=torch.contiguous_format))

    partial, full = block_mask
    kv_num_blocks, kv_indices = ordered(partial)
    q_num_blocks, q_indices = ordered(partial.transpose(-2, -1))
    full_kv_num_blocks, full_kv_indices = ordered(full) if full is not None else (None, None)
    full_q_num_blocks, full_q_indices = ordered(full.transpose(-2, -1)) if full is not None else (None, None)
    if seq_lengths is None:
        seq_lengths = (partial.shape[-2] * Q_BLOCK_SIZE, partial.shape[-1] * KV_BLOCK_SIZE)
    return BlockMask(
        seq_lengths=seq_lengths,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
        mask_mod=mask_mod if mask_mod is not None else noop_mask,
    )


def _round_up_to_multiple(x, multiple):
    """Round *x* up to the nearest multiple of *multiple*."""
    return (x + multiple - 1) // multiple * multiple


def _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device):
    """Evaluate ``mask_mod`` for a horizontal stripe of Q rows.

    Returns a bool tensor of shape ``[B, H, actual_q, KV_LEN]``.

    For the common B=1/H=1 case we call ``mask_mod`` directly with 2-D index
    tensors, which avoids the Python overhead of ``create_mask``'s vmap stack.
    For multi-batch/multi-head we fall back to ``create_mask`` with a shifted
    mask_mod closure.
    """
    if B == 1 and H == 1:
        q_idx = torch.arange(q_start, q_start + actual_q, device=device, dtype=torch.int64)[:, None]
        kv_idx = torch.arange(0, KV_LEN, device=device, dtype=torch.int64)[None, :]
        mask_2d = mask_mod(0, 0, q_idx, kv_idx)
        return mask_2d.view(1, 1, actual_q, KV_LEN)

    def _shifted_mm(b, h, q_idx, kv_idx, _mm=mask_mod, _offset=q_start):
        return _mm(b, h, q_idx + _offset, kv_idx)

    return _torch_create_mask(_shifted_mm, B, H, actual_q, KV_LEN, device=device)


def _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Classify each (Q-block, KV-block) tile as full / partial / empty.

    Args:
        stripe_mask: bool tensor ``[B, H, stripe_q, KV_LEN_PADDED]`` whose Q and
            KV dimensions are already padded to multiples of block sizes.

    Returns:
        flags: int8 tensor ``[stripe_q_nb, KV_num_blocks]`` where
            0 = empty, 1 = partial, 2 = full.
    """
    stripe_q_nb = stripe_mask.shape[2] // Q_BLOCK_SIZE
    kv_num_blocks = stripe_mask.shape[3] // KV_BLOCK_SIZE
    partial_dense, full_dense = _convert_mask_to_block_mask(
        stripe_mask, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE, separate_full_blocks=True
    )
    flags = torch.zeros((stripe_q_nb, kv_num_blocks), dtype=torch.int8, device=stripe_mask.device)
    flags[partial_dense[0, 0] == 1] = 1
    flags[full_dense[0, 0] == 1] = 2
    return flags


def _pack_stripe_partial_blocks(
    stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE, running_total, partial_block_table
):
    """Extract partial blocks from a stripe into the packed cache.

    For every (Q-block, KV-block) classified as partial, copy the
    ``[Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` tile from ``stripe_mask`` into a flat list
    and record its packed index in ``partial_block_table``.

    Returns:
        packed_tiles: ``[num_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool tensor.
        num_partial:  count of partial blocks in this stripe.
    """
    partial_bool = flags == 1
    num_partial = int(partial_bool.sum().item())
    if num_partial == 0:
        empty = torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=stripe_mask.device)
        return (empty, 0)
    stripe_q_nb = flags.shape[0]
    kv_num_blocks = flags.shape[1]
    sq_idx, kv_blk_idx = partial_bool.nonzero(as_tuple=True)
    blocks = stripe_mask.view(stripe_q_nb, Q_BLOCK_SIZE, kv_num_blocks, KV_BLOCK_SIZE)
    packed_tiles = blocks[sq_idx, :, kv_blk_idx, :]
    cumsum_per_row = partial_bool.to(torch.int32).cumsum(dim=-1)
    per_row_count = cumsum_per_row.max(dim=-1).values
    row_offset_local = per_row_count.cumsum(dim=-1) - per_row_count
    local_idx = cumsum_per_row[sq_idx, kv_blk_idx] - 1
    packed_idx = (row_offset_local[sq_idx] + local_idx + running_total).to(torch.int32)
    partial_block_table[qs_block + sq_idx, kv_blk_idx] = packed_idx
    return (packed_tiles, num_partial)


def _assemble_packed_block_mask(
    block_flags,
    packed_partial_mask,
    partial_block_table,
    global_per_row_count,
    Q_LEN,
    KV_LEN,
    Q_BLOCK_SIZE,
    KV_BLOCK_SIZE,
):
    """Assemble the final BlockMask from streaming stripe outputs.

    Args:
        block_flags:          ``[B, H, Q_nb, KV_nb]`` int8 (0/1/2).
        packed_partial_mask:  ``[total_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool.
        partial_block_table:  ``[Q_nb, KV_nb]`` int32 (packed index or -1).
        global_per_row_count: ``[Q_nb]`` int32, partial count per Q-block row.
    """
    Q_num_blocks = block_flags.shape[2]
    partial_mask_offsets = (
        (global_per_row_count.cumsum(dim=-1) - global_per_row_count).view(1, 1, Q_num_blocks).contiguous()
    )
    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (Q_LEN, KV_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets
    packed_block_mask.partial_block_table = partial_block_table
    return packed_block_mask


def create_block_mask_patched(
    mask_mod, B=1, H=1, Q_LEN=None, KV_LEN=None, device=None, BLOCK_SIZE=128, stripe_q_blocks=None
):
    """Build a packed BlockMask with streaming stripe processing.

    Parameters are aligned with ``torch.nn.attention.flex_attention.create_block_mask``.

    The mask is built incrementally: Q rows are processed in horizontal stripes,
    each stripe small enough to keep peak HBM bounded. For every stripe we
    evaluate ``mask_mod``, classify blocks (full/partial/empty), and immediately
    pack partial blocks into a flat cache. This avoids materialising the full
    ``[Q_LEN, KV_LEN]`` dense mask at any point.

    Args:
        mask_mod: A mask_mod callable ``(b, h, q_idx, kv_idx) -> bool``.
            Supports any flexible mask pattern, e.g. ``_full_mask_mod``,
            ``_cross_sample_causal_video_bidir_mask_mod``, ``_sparse_mask_mod``, etc.
        B: Batch size (default 1).
        H: Number of heads (default 1).
        Q_LEN: Query sequence length. If None, inferred from KV_LEN.
        KV_LEN: Key/value sequence length. If None, inferred from Q_LEN.
        device: Device for tensor allocation. If None, uses NPU/CUDA.
        BLOCK_SIZE: Block size as int (square) or ``(Q_BLOCK_SIZE, KV_BLOCK_SIZE)`` tuple.
        stripe_q_blocks: Number of Q blocks per streaming stripe. If None, auto-computed
            to target ~256MB per stripe. Controls HBM peak consumption.

    Returns:
        BlockMask with ``packed_partial_mask``, ``partial_mask_offsets``,
        and ``partial_block_table`` attributes set.
    """
    if B != 1 or H != 1:
        raise NotImplementedError(
            "NPU packed BlockMask currently supports only batch/head-independent masks with B=1 and H=1"
        )
    if device is None:
        device = _get_torch_device()
    if Q_LEN is None and KV_LEN is not None:
        Q_LEN = KV_LEN
    if KV_LEN is None and Q_LEN is not None:
        KV_LEN = Q_LEN
    assert Q_LEN is not None and KV_LEN is not None, "Q_LEN and KV_LEN must be provided"
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = (BLOCK_SIZE, BLOCK_SIZE)
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE
    Q_num_blocks = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_num_blocks = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    KV_LEN_padded = KV_num_blocks * KV_BLOCK_SIZE
    if stripe_q_blocks is None:
        max_rows = max(1, _STRIPE_TARGET_BYTES // (KV_LEN_padded * _BYTES_PER_MASK_ELEMENT))
        stripe_q_blocks = max(1, max_rows // Q_BLOCK_SIZE)
    stripe_q_blocks = min(stripe_q_blocks, Q_num_blocks)
    block_flags = torch.zeros((B, H, Q_num_blocks, KV_num_blocks), device=device, dtype=torch.int8)
    partial_block_table = torch.full((Q_num_blocks, KV_num_blocks), -1, dtype=torch.int32, device=device)
    global_per_row_count = torch.zeros(Q_num_blocks, dtype=torch.int32, device=device)
    packed_tiles_list = []
    running_total = 0
    for qs_block in range(0, Q_num_blocks, stripe_q_blocks):
        qe_block = min(qs_block + stripe_q_blocks, Q_num_blocks)
        q_start = qs_block * Q_BLOCK_SIZE
        stripe_q = (qe_block - qs_block) * Q_BLOCK_SIZE
        actual_q = min(stripe_q, Q_LEN - q_start)
        stripe_mask = _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device)
        pad_q = stripe_q - actual_q
        pad_kv = KV_LEN_padded - KV_LEN
        if pad_q > 0 or pad_kv > 0:
            stripe_mask = _F.pad(stripe_mask, (0, pad_kv, 0, pad_q))
        flags = _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        block_flags[:, :, qs_block:qe_block, :] = flags
        partial_bool = flags == 1
        per_row_count = partial_bool.to(torch.int32).cumsum(dim=-1).max(dim=-1).values
        global_per_row_count[qs_block:qe_block] = per_row_count.to(torch.int32)
        packed_tiles, num_partial = _pack_stripe_partial_blocks(
            stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE, running_total, partial_block_table
        )
        packed_tiles_list.append(packed_tiles)
        running_total += num_partial
        del stripe_mask, flags, partial_bool, per_row_count
    if running_total > 0:
        packed_partial_mask = torch.cat(packed_tiles_list, dim=0)
    else:
        packed_partial_mask = torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device)
    del packed_tiles_list
    packed_block_mask = _assemble_packed_block_mask(
        block_flags,
        packed_partial_mask,
        partial_block_table,
        global_per_row_count,
        Q_LEN,
        KV_LEN,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    del block_flags, global_per_row_count
    return packed_block_mask


create_flex_block_mask = create_block_mask_patched
