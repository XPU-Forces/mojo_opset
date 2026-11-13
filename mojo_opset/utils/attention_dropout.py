"""Sparse dropout state prepared outside FlexAttention execution/capture.

Only nonempty 128x128 blocks have payload. Raw payload holds uint8 random
values; packed payload holds keep bits in Q planes: bit i of [r,k] is row
r + 16*i. Keep a state unchanged until its attention backward completes.
"""

import math
import struct

from dataclasses import dataclass

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


def _rows(block_mask, reverse=False):
    prefix = "q" if reverse else "kv"
    counts = getattr(block_mask, f"{prefix}_num_blocks").cpu()
    indices = getattr(block_mask, f"{prefix}_indices").cpu()
    full_counts = getattr(block_mask, f"full_{prefix}_num_blocks", None)
    full_indices = getattr(block_mask, f"full_{prefix}_indices", None)
    if counts.shape[:2] != (1, 1):
        raise ValueError("Only batch/head-independent BlockMask is supported")
    if full_counts is not None:
        full_counts, full_indices = full_counts.cpu(), full_indices.cpu()
    rows = []
    for row in range(counts.shape[-1]):
        entries = [(int(col), False) for col in indices[0, 0, row, : int(counts[0, 0, row])]]
        if full_counts is not None:
            entries += [(int(col), True) for col in full_indices[0, 0, row, : int(full_counts[0, 0, row])]]
        rows.append(entries)
    return rows


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
    def from_block_mask(cls, block_mask, q_len, kv_len):
        """Return the persistent layout and temporary rows for backend planning."""
        if q_len <= 0 or kv_len <= 0:
            raise ValueError("Sequence lengths must be positive")
        bm, bn = block_mask.BLOCK_SIZE
        nq, nk = (q_len + bm - 1) // bm, (kv_len + bn - 1) // bn
        rows, reverse = _rows(block_mask), _rows(block_mask, reverse=True)
        if len(rows) != nq or len(reverse) != nk:
            raise ValueError("Sequence lengths do not match the BlockMask grid")
        table = torch.full((nq, nk), -1, dtype=torch.int32)
        num_active = 0
        for qb, entries in enumerate(rows):
            for kb, _ in entries:
                if not 0 <= kb < nk or table[qb, kb] >= 0:
                    raise ValueError("Invalid or duplicate BlockMask entry")
                table[qb, kb] = num_active
                num_active += 1
        forward_edges = {(qb, kb, flag) for qb, row in enumerate(rows) for kb, flag in row}
        reverse_edges = {(qb, kb, flag) for kb, row in enumerate(reverse) for qb, flag in row}
        if forward_edges != reverse_edges:
            raise ValueError("Forward and reverse BlockMask schedules disagree")
        device = block_mask.kv_indices.device
        return cls(block_mask, q_len, kv_len, table.to(device), num_active), reverse


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
    layout, _ = _DropoutLayout.from_block_mask(block_mask, q.shape[2], k.shape[2])
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
