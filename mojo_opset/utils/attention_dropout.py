"""Sparse dropout state prepared outside FlexAttention execution/capture.

Only nonempty 128x128 blocks have payload. Raw payload holds uint8 random
values; packed payload holds keep bits in Q planes: bit i of [r,k] is row
r + 16*i. Keep a state unchanged until its attention backward completes.
"""

import math
import struct

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch


def _dropout_parameters(dropout_p):
    """Match CUDA FA2: FP32 keep probability, floor(keep * 255.0), FP32 reciprocal.

    The effective keep probability is (threshold + 1) / 256, but the scale
    intentionally uses the nominal keep probability, matching flash_api.cpp.
    """

    def fp32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    if not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must satisfy 0 <= p < 1")
    keep_probability = fp32(1.0 - fp32(float(dropout_p)))
    if keep_probability == 0:
        raise ValueError("dropout_p rounds to 1 in FP32")
    return math.floor(keep_probability * 255.0), fp32(1.0 / keep_probability)


def _check_grid_shape(block_mask, nq, nk):
    for prefix, rows in (("kv", nq), ("q", nk)):
        counts = getattr(block_mask, f"{prefix}_num_blocks")
        indices = getattr(block_mask, f"{prefix}_indices")
        full_counts = getattr(block_mask, f"full_{prefix}_num_blocks", None)
        full_indices = getattr(block_mask, f"full_{prefix}_indices", None)
        if (full_counts is None) != (full_indices is None):
            raise ValueError("BlockMask full counts and full indices must be provided together")
        pairs = ((counts, indices),) if full_counts is None else ((counts, indices), (full_counts, full_indices))
        for cnt, idx in pairs:
            if cnt.ndim != 3 or idx.ndim != 4 or cnt.shape[:2] != (1, 1) or idx.shape[:2] != (1, 1):
                raise ValueError("Only batch/head-independent BlockMask is supported")
            if cnt.shape[-1] != rows or idx.shape[2] != rows:
                raise ValueError("Sequence lengths do not match the BlockMask grid")


def _unroll_half(counts, indices, num_rows, num_cols, *, check_bounds=True):
    """Unpack one partial or full BlockMask schedule into int32 (row_counts, rows, cols)."""
    if counts is None:
        empty = np.empty(0, dtype=np.int32)
        return np.zeros(num_rows, dtype=np.int32), empty, empty

    row_counts = np.asarray(counts.detach().cpu(), dtype=np.int32).reshape(num_rows)
    width = int(row_counts.max()) if num_rows else 0
    if check_bounds and (row_counts.min(initial=0) < 0 or width > indices.shape[-1]):
        raise ValueError("Invalid or duplicate BlockMask entry")
    if width == 0:
        empty = np.empty(0, dtype=np.int32)
        return row_counts, empty, empty

    if indices.device.type == "cpu":
        cropped = np.asarray(indices.detach(), dtype=np.int32).reshape(num_rows, indices.shape[-1])[:, :width]
    else:
        cropped = np.asarray(indices[:, :, :, :width].detach().cpu(), dtype=np.int32).reshape(num_rows, width)
    cols = cropped[np.arange(width, dtype=np.int32)[None, :] < row_counts[:, None]]
    if check_bounds and cols.size and (cols.min() < 0 or cols.max() >= num_cols):
        raise ValueError("Invalid or duplicate BlockMask entry")
    return row_counts, np.repeat(np.arange(num_rows, dtype=np.int32), row_counts), cols


def _unroll_direction(block_mask, prefix, num_rows, num_cols, *, check_bounds=True):
    counts = getattr(block_mask, f"{prefix}_num_blocks")
    indices = getattr(block_mask, f"{prefix}_indices")
    full_counts = getattr(block_mask, f"full_{prefix}_num_blocks", None)
    full_indices = getattr(block_mask, f"full_{prefix}_indices", None)
    return (
        _unroll_half(counts, indices, num_rows, num_cols, check_bounds=check_bounds),
        _unroll_half(full_counts, full_indices, num_rows, num_cols, check_bounds=check_bounds),
    )


def _schedule_signature(block_mask):
    names = (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    )
    tensors = tuple(
        None
        if (t := getattr(block_mask, name, None)) is None
        else (t.device, t.shape, t.stride(), t.data_ptr(), t._version)
        for name in names
    )
    return (tuple(block_mask.BLOCK_SIZE), tensors)


def _mark_verified_block_mask(block_mask):
    """Mark a BlockMask whose disjoint forward/reverse schedules match by construction."""
    try:
        setattr(block_mask, "_mojo_verified_signature", _schedule_signature(block_mask))
    except (AttributeError, TypeError):
        pass
    return block_mask


def _cache_key(block_mask, q_len, kv_len):
    return (int(q_len), int(kv_len), _schedule_signature(block_mask))


@dataclass
class _CachedLayout:
    """Cycle-free per-BlockMask cache entry storing reusable device table and dKV plans."""

    block_table: torch.Tensor
    num_active: int
    reverse_counts: tuple[list[int], list[int]]
    dkv_plans: dict = field(default_factory=dict)


@dataclass
class _DropoutLayout:
    """One dropout slot per full OR partial block; empty blocks have slot -1."""

    block_mask: object
    q_len: int
    kv_len: int
    block_table: torch.Tensor
    num_active: int
    dkv_schedule: torch.Tensor | None = None
    dkv_ranges: torch.Tensor | None = None
    dkv_num_tasks: int = 0

    @classmethod
    def _build(cls, block_mask, q_len, kv_len):
        if q_len <= 0 or kv_len <= 0:
            raise ValueError("Sequence lengths must be positive")
        bm, bn = block_mask.BLOCK_SIZE
        nq, nk = (q_len + bm - 1) // bm, (kv_len + bn - 1) // bn
        _check_grid_shape(block_mask, nq, nk)

        verified = getattr(block_mask, "_mojo_verified_signature", None) == _schedule_signature(block_mask)
        device = block_mask.kv_indices.device
        if verified and device.type == "npu":
            from mojo_opset.utils.target import detect_target

            if detect_target().arch_key == "npu.a5":
                from mojo_opset.kernels.npu_a5_triton._attention_dropout import build_dropout_layout_entry

                return build_dropout_layout_entry(block_mask, nq, nk)

        (p_cnt, p_rows, p_cols), (f_cnt, f_rows, f_cols) = _unroll_direction(
            block_mask, "kv", nq, nk, check_bounds=not verified
        )
        total_cnt = p_cnt + f_cnt
        row_ends = np.cumsum(total_cnt, dtype=np.int32)
        num_active = int(row_ends[-1]) if nq else 0
        row_starts = row_ends - total_cnt

        table = np.full((nq, nk), -1, dtype=np.int32)
        if p_cols.size:
            p_offsets = (np.cumsum(p_cnt, dtype=np.int32) - p_cnt)[p_rows]
            p_slots = row_starts[p_rows] + np.arange(p_cols.size, dtype=np.int32) - p_offsets
            table[p_rows, p_cols] = p_slots
            if not verified and not np.array_equal(table[p_rows, p_cols], p_slots):
                raise ValueError("Invalid or duplicate BlockMask entry")
        if f_cols.size:
            f_offsets = (np.cumsum(f_cnt, dtype=np.int32) - f_cnt)[f_rows]
            f_slots = row_starts[f_rows] + p_cnt[f_rows] + np.arange(f_cols.size, dtype=np.int32) - f_offsets
            table[f_rows, f_cols] = f_slots
            if not verified and (
                not np.array_equal(table[f_rows, f_cols], f_slots)
                or (p_cols.size and not np.array_equal(table[p_rows, p_cols], p_slots))
            ):
                raise ValueError("Invalid or duplicate BlockMask entry")

        if verified:
            rp_cnt = np.asarray(block_mask.q_num_blocks.detach().cpu(), dtype=np.int32).reshape(nk)
            if block_mask.full_q_num_blocks is None:
                rt_cnt = rp_cnt
            else:
                rf_cnt = np.asarray(block_mask.full_q_num_blocks.detach().cpu(), dtype=np.int32).reshape(nk)
                rt_cnt = rp_cnt + rf_cnt
            reverse_counts = (rp_cnt.tolist(), rt_cnt.tolist())
            return _CachedLayout(torch.from_numpy(table).to(device), num_active, reverse_counts)

        (rp_cnt, rp_kb, rp_qb), (rf_cnt, rf_kb, rf_qb) = _unroll_direction(block_mask, "q", nk, nq)
        if num_active == 0:
            if rp_qb.size or rf_qb.size:
                raise ValueError("Forward and reverse BlockMask schedules disagree")
        else:
            seen = np.zeros(num_active, dtype=np.bool_)
            if rp_qb.size:
                rp_slots = table[rp_qb, rp_kb]
                if not np.all((rp_slots >= row_starts[rp_qb]) & (rp_slots < row_starts[rp_qb] + p_cnt[rp_qb])):
                    raise ValueError("Forward and reverse BlockMask schedules disagree")
                seen[rp_slots] = True
            if rf_qb.size:
                rf_slots = table[rf_qb, rf_kb]
                if not np.all((rf_slots >= row_starts[rf_qb] + p_cnt[rf_qb]) & (rf_slots < row_ends[rf_qb])):
                    raise ValueError("Forward and reverse BlockMask schedules disagree")
                seen[rf_slots] = True
            if not seen.all():
                raise ValueError("Forward and reverse BlockMask schedules disagree")

        reverse_counts = (rp_cnt.tolist(), (rp_cnt + rf_cnt).tolist())
        return _CachedLayout(torch.from_numpy(table).to(device), num_active, reverse_counts)

    @classmethod
    def resolve(cls, block_mask, q_len, kv_len):
        """Return a fresh ``_DropoutLayout`` and cached ``_CachedLayout`` entry for ``block_mask``."""
        if q_len <= 0 or kv_len <= 0:
            raise ValueError("Sequence lengths must be positive")
        key = _cache_key(block_mask, q_len, kv_len)
        cache = getattr(block_mask, "_mojo_dropout_cache", None)
        entry = cache.get(key) if isinstance(cache, dict) else None
        if entry is None:
            entry = cls._build(block_mask, q_len, kv_len)
            if not isinstance(cache, dict):
                cache = {}
                try:
                    setattr(block_mask, "_mojo_dropout_cache", cache)
                except (AttributeError, TypeError):
                    cache = None
            if cache is not None:
                if len(cache) >= 8:
                    cache.clear()
                cache[key] = entry
        return cls(block_mask, q_len, kv_len, entry.block_table, entry.num_active), entry

    @classmethod
    def from_block_mask(cls, block_mask, q_len, kv_len):
        """Return the persistent layout and reverse-row counts for backend planning."""
        layout, entry = cls.resolve(block_mask, q_len, kv_len)
        return layout, entry.reverse_counts


@dataclass
class SparseDropoutMask:
    layout: _DropoutLayout
    batch: int
    heads: int
    dropout_p: float
    bitpacked: bool
    tiles: torch.Tensor | None
    threshold: int
    scale: float

    def to_dense(self):
        """Expand keep bits for debugging/reference use, not the accelerator kernel."""
        return _dropout_keep(self, 0, self.layout.q_len)


def _dropout_keep(dropout, start, stop):
    """Decode one Q stripe, bounded by stripe length rather than the full Q axis."""
    layout = dropout.layout
    shape = (dropout.batch, dropout.heads, stop - start, layout.kv_len)
    if dropout.tiles is None:
        return torch.ones(shape, dtype=torch.bool, device=layout.block_table.device)
    if layout.num_active == 0:
        return torch.zeros(shape, dtype=torch.bool, device=layout.block_table.device)
    qi = torch.arange(start, stop, device=dropout.tiles.device)[:, None]
    ki = torch.arange(layout.kv_len, device=dropout.tiles.device)[None, :]
    slots = layout.block_table[qi // 128, ki // 128].long()
    rows = qi % (16 if dropout.bitpacked else 128)
    tile = dropout.tiles[:, :, slots.clamp_min(0), rows, ki % 128]
    if dropout.bitpacked:
        keep = ((tile.to(torch.int32) >> ((qi % 128) // 16)) & 1).bool()
    else:
        keep = tile <= dropout.threshold
    return keep & (slots >= 0)


def _validate_shape(q, k, block_mask):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K [B,H,S,D] with matching batch and head dimension")
    if q.shape[0] <= 0 or k.shape[1] <= 0 or q.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid batch/head count or GQA ratio")
    if q.device != k.device or q.device != block_mask.kv_indices.device:
        raise ValueError("Q/K and BlockMask must be on the same device")
    if block_mask.BLOCK_SIZE != (128, 128):
        raise ValueError("Sparse dropout requires 128x128 BlockMask tiles")


def create_flex_dropout_mask(
    q, k, block_mask, dropout_p, *, bitpacked=False, seed=None, offset=0, generator=None, dkv_chunk_size=None
):
    """Prepare a mask to pass as ``flex_attention_v2(..., dropout=state)``.

    An explicit seed uses A5 Triton Philox4x32-10, with offset in 16-byte
    counters. Without seed, use Torch RNG (optionally a torch.Generator).
    Threshold/scale match CUDA FA2's uint8 semantics, not its Philox7 stream.
    Metadata construction runs on the host and must precede graph capture.
    """
    _validate_shape(q, k, block_mask)
    if q.device.type == "npu":
        from mojo_opset.utils.target import detect_target

        if detect_target().arch_key == "npu.a5":
            from mojo_opset.kernels.npu_a5_triton._attention_dropout import allocate_dropout_mask

            state = allocate_dropout_mask(
                q, k, block_mask, dropout_p, bitpacked=bitpacked, dkv_chunk_size=dkv_chunk_size
            )
            return refill_flex_dropout_mask(state, seed=seed, offset=offset, generator=generator)
    if dkv_chunk_size is not None:
        raise ValueError("dKV split tuning is only supported by the A5 provider")
    threshold, scale = _dropout_parameters(dropout_p)
    layout, _ = _DropoutLayout.resolve(block_mask, q.shape[2], k.shape[2])
    tiles = (
        None
        if dropout_p == 0
        else torch.empty(
            (q.shape[0], q.shape[1], layout.num_active, 16 if bitpacked else 128, 128),
            dtype=torch.uint8,
            device=q.device,
        )
    )
    state = SparseDropoutMask(layout, q.shape[0], q.shape[1], dropout_p, bitpacked, tiles, threshold, scale)
    return refill_flex_dropout_mask(state, seed=seed, offset=offset, generator=generator)


def _pack_random_bytes_torch(raw, packed, threshold):
    """Pack eight Q planes into the existing uint8 payload using Torch only."""
    if raw.numel() != packed.numel() * 8 or raw.dtype != torch.uint8 or packed.dtype != torch.uint8:
        raise ValueError("Expected raw uint8 bytes and a packed buffer eight times smaller")
    if not raw.is_contiguous() or not packed.is_contiguous() or raw.device != packed.device:
        raise ValueError("Dropout buffers must be contiguous and on the same device")
    planes = raw.view(-1, 8, 16, 128)
    output = packed.view(-1, 16, 128)
    output.zero_()
    for bit in range(8):
        output.bitwise_or_(((planes[:, bit] <= threshold).to(torch.int32) << bit).to(torch.uint8))


def refill_flex_dropout_mask(dropout, *, generator=None, seed=None, offset=0):
    """Refill between completed training steps; never modify an outstanding mask."""
    if dropout.layout.block_table.device.type == "npu":
        from mojo_opset.utils.target import detect_target

        if detect_target().arch_key == "npu.a5":
            from mojo_opset.kernels.npu_a5_triton._attention_dropout import refill_dropout_mask

            return refill_dropout_mask(dropout, generator=generator, seed=seed, offset=offset)
    if seed is not None or offset != 0:
        raise ValueError("Explicit Philox seed/offset requires the A5 Triton provider; use a Torch generator otherwise")
    if dropout.tiles is not None and dropout.tiles.numel():
        if dropout.bitpacked:
            raw = torch.empty(dropout.tiles.numel() * 8, dtype=torch.uint8, device=dropout.tiles.device)
            raw.random_(0, 256, generator=generator)
            _pack_random_bytes_torch(raw, dropout.tiles, dropout.threshold)
        else:
            dropout.tiles.random_(0, 256, generator=generator)
    return dropout
