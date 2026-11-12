"""Triton-Ascend tail seeding for GLM-5.3-Flash (D=128, KPOOL=4)."""

from functools import lru_cache

import torch
import triton
import triton.backends.ascend.runtime  # noqa: F401
import triton.language as tl
import triton.runtime.driver as driver


@triton.autotune(
    configs=[triton.Config({"BLOCK_T": b}, num_warps=32) for b in (1, 2, 4, 8, 16, 32)],
    key=["N", "RING", "STRIDE_B", "STRIDE_KG", "SLOT_BITS", "INDEX32", "VECTOR_CORES", "PACK_BITS", "WORD32"],
    cache_results=True,
)
@triton.jit
def _seed_tail_fused_kernel(
    KEY,
    GATE,
    SLOT,
    CACHE,
    N: tl.constexpr,
    RING: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_KG: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INDEX32: tl.constexpr,
    VECTOR_CORES: tl.constexpr,
    PACK_BITS: tl.constexpr,
    WORD32: tl.constexpr,
    RING_SHIFT: tl.constexpr,
):
    COPY_WIDTH: tl.constexpr = 64 if WORD32 else 128
    COPY_STRIDE_B: tl.constexpr = STRIDE_B // 2 if WORD32 else STRIDE_B
    COPY_STRIDE_KG: tl.constexpr = STRIDE_KG // 2 if WORD32 else STRIDE_KG
    if WORD32:
        KEY = KEY.to(tl.pointer_type(tl.int32))
        GATE = GATE.to(tl.pointer_type(tl.int32))
        CACHE = CACHE.to(tl.pointer_type(tl.int32))
    else:
        KEY = KEY.to(tl.pointer_type(tl.int16))
        GATE = GATE.to(tl.pointer_type(tl.int16))
        CACHE = CACHE.to(tl.pointer_type(tl.int16))
    TILES: tl.constexpr = (N + BLOCK_T - 1) // BLOCK_T
    STEP: tl.constexpr = 1 if TILES <= 8 else (VECTOR_CORES if VECTOR_CORES < TILES else TILES)
    lane = tl.arange(0, BLOCK_T).to(tl.int32)
    cols = tl.arange(0, COPY_WIDTH).to(tl.int32)
    for tile in range(tl.program_id(0), TILES, STEP):
        rows = tile.to(tl.int32) * BLOCK_T + lane
        raw = tl.load(SLOT + rows, mask=rows < N, other=-1)
        ahead = tl.load(SLOT + rows + 4, mask=rows + 4 < N, other=-1)
        valid = ((raw >> (SLOT_BITS - 1)).to(tl.int32) == 0) & (rows < N)
        ahead_valid = (ahead >> (SLOT_BITS - 1)).to(tl.int32) == 0
        slot = tl.maximum(raw.to(tl.int32), 0)
        ahead_slot = tl.maximum(ahead.to(tl.int32), 0)
        block = slot >> RING_SHIFT if RING_SHIFT >= 0 else slot // RING
        ahead_block = ahead_slot >> RING_SHIFT if RING_SHIFT >= 0 else ahead_slot // RING
        selected = valid & ~(ahead_valid & (ahead_block == block))
        ring_slot = slot - block * RING
        if INDEX32:
            src = rows[:, None] * COPY_WIDTH + cols[None, :]
            dst = (block * COPY_STRIDE_B + ring_slot * COPY_WIDTH)[:, None] + cols[None, :]
        else:
            src = rows.to(tl.int64)[:, None] * COPY_WIDTH + cols[None, :]
            dst = (block.to(tl.int64) * COPY_STRIDE_B + ring_slot * COPY_WIDTH)[:, None] + cols[None, :]
        # Full contiguous row-tile reads trade bandwidth for eliminating the
        # scalar-selected-run copy sequence. Stores touch selected rows only.
        key_words = tl.load(KEY + src, mask=(rows < N)[:, None], other=0)
        gate_words = tl.load(GATE + src, mask=(rows < N)[:, None], other=0)
        tl.store(CACHE + dst, key_words, mask=selected[:, None])
        tl.store(CACHE + dst + COPY_STRIDE_KG, gate_words, mask=selected[:, None])


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    return int(driver.active.utils.get_device_properties(device_index)["num_vectorcore"])


def _route(tail_kv_cache, key, gate_score, tslot, vector_cores):
    if key.dtype != torch.bfloat16:
        raise ValueError("K, gate and cache must use BF16")
    if gate_score.dtype != key.dtype or tail_kv_cache.dtype != key.dtype:
        raise ValueError("K, gate and cache must have identical dtype")
    if not (key.device == gate_score.device == tail_kv_cache.device == tslot.device):
        raise ValueError("All tensors must be on the same NPU")
    if key.device.type != "npu":
        raise ValueError("This implementation requires an Ascend NPU")
    if key.ndim != 2 or key.shape[1] != 128 or gate_score.shape != key.shape:
        raise ValueError("K and gate must have shape [N, 128]")
    if tail_kv_cache.ndim != 4 or tail_kv_cache.shape[1] != 2 or tail_kv_cache.shape[3] != 128:
        raise ValueError("Cache must have shape [num_blocks, 2, RING, 128]")
    if tail_kv_cache.shape[2] < 4 or tail_kv_cache.stride(3) != 1 or tail_kv_cache.stride(2) != 128:
        raise ValueError("RING >= 4 and contiguous cache rows are required")
    if tail_kv_cache.shape[0] * tail_kv_cache.shape[2] >= 2**31:
        raise ValueError("The addressable slot range must fit signed int32")
    if tslot.ndim != 1 or tslot.shape[0] != key.shape[0] or tslot.dtype not in (torch.int32, torch.int64):
        raise ValueError("tslot must be signed int32/int64 with shape [N]")
    if not key.is_contiguous() or not gate_score.is_contiguous() or not tslot.is_contiguous():
        raise ValueError("K, gate and tslot must be contiguous")
    n = key.shape[0]
    if n == 0:
        return None
    cache_extent = (
        (tail_kv_cache.shape[0] - 1) * tail_kv_cache.stride(0) + tail_kv_cache.stride(1) + tail_kv_cache.shape[2] * 128
    )
    index32 = (
        n * 128 < 2**31 and cache_extent < 2**31 and tail_kv_cache.stride(0) < 2**31 and tail_kv_cache.stride(1) < 2**31
    )
    # E114 generic-only alignment proof from effective pointers/metadata.
    generic_word32 = (
        tail_kv_cache.stride(0) % 2 == 0
        and tail_kv_cache.stride(1) % 2 == 0
        and key.data_ptr() % 4 == 0
        and gate_score.data_ptr() % 4 == 0
        and tail_kv_cache.data_ptr() % 4 == 0
    )
    # Also prove safe offsets for the masked extra lanes of a512value run tile.
    run_index32 = index32 and n * 128 + 512 < 2**31 and cache_extent + 512 < 2**31

    # All padded metadata rows (<N+511) and valid slots fit positive int32.
    slot_count = tail_kv_cache.shape[0] * tail_kv_cache.shape[2]
    pack_bits = max(1, (slot_count - 1).bit_length())
    if (n + 511).bit_length() + pack_bits > 31:
        pack_bits = 0

    def grid(meta):
        total_tiles = triton.cdiv(n, meta["BLOCK_T"])
        return (1 if total_tiles <= 8 else min(vector_cores, total_tiles),)

    _seed_tail_fused_kernel[grid](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        tail_kv_cache.shape[2],
        tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
        tslot.element_size() * 8,
        INDEX32=run_index32,
        VECTOR_CORES=vector_cores,
        PACK_BITS=pack_bits,
        WORD32=generic_word32,
        RING_SHIFT=(
            tail_kv_cache.shape[2].bit_length() - 1
            if (tail_kv_cache.shape[2] & (tail_kv_cache.shape[2] - 1)) == 0
            else -1
        ),
        compile_mode="simt_only",
    )
    return None


def _kpool_seed_tail_cache_impl(tail_kv_cache, key, gate_score, tslot, kpool=4, head_dim=128):
    """Mutate tail cache and return None; selected destinations must be unique.

    Non-negative slots must be in [0, num_blocks*RING). Input/cache aliasing
    is unsupported. Negative slots of either signed index dtype are skipped.
    """
    if kpool != 4 or head_dim != 128:
        raise ValueError("This GLM-5.3-Flash kernel fixes kpool=4 and head_dim=128")
    return _route(tail_kv_cache, key, gate_score, tslot, _vector_core_count(key.device.index))


@torch.library.custom_op("mojo_npu_triton_a5::kpool_seed_tail_cache", mutates_args=("tail_kv_cache",))
def _kpool_seed_tail_cache_leaf(
    tail_kv_cache: torch.Tensor, key: torch.Tensor, gate_score: torch.Tensor, tslot: torch.Tensor
) -> None:
    _kpool_seed_tail_cache_impl(tail_kv_cache, key, gate_score, tslot)


@_kpool_seed_tail_cache_leaf.register_fake
def _kpool_seed_tail_cache_fake(tail_kv_cache, key, gate_score, tslot):
    return None


def kpool_seed_tail_cache_fwd(tail_kv_cache, key, gate_score, tslot, kpool=4, head_dim=128):
    if kpool != 4 or head_dim != 128:
        raise ValueError("This GLM-5.3-Flash kernel fixes kpool=4 and head_dim=128")
    # Keep pointer checks, hardware queries and Triton launch outside graph capture.
    return _kpool_seed_tail_cache_leaf(tail_kv_cache, key, gate_score, tslot)
