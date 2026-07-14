import gc
import math
import os

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask

from mojo_opset.experimental import mojo_flex_attention
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device


# NPU device validation monkey-patch (same as original test)
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass


# ============================================================================
# Global configuration
# ============================================================================
DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
SLIDING_WINDOW = 1024
GLOBAL_WINDOW = 4

DATA_LENGTH = [[2000, 22000, 2000], [2000, 22000, 2000]]
DATA_INPUT_TYPE = [["text", "image_gen", "text"], ["text", "image_gen", "text"]]
FULL_MASK_MODALITIES = ("image_gen", "image_vae")

DATA_LENGTH_VIDEO = [
    [6500, 6500, 6500, 6500],
    [6500, 6500, 6500, 6500],
]
VIDEO_FRAME_LENGTH = [
    [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
    [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],
]

SEED = 0
APPLY_Q_CHUNK = 2048
Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128

_WARMUP = 1
_ITERS = 3
_MB = 1024 ** 2


def _device():
    return get_torch_device()


def _sync():
    if _device() == "npu":
        torch.npu.synchronize()
    elif _device() == "cuda":
        torch.cuda.synchronize()


def _get_num_vector_core():
    try:
        dev = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(dev)
        return max(int(props.get("num_vectorcore", 1)), 1)
    except Exception:
        return 1


# ============================================================================
# Kernel 1: bool_count_nonzero_kernel
# ============================================================================
@triton.jit
def bool_count_nonzero_kernel(
    MASK, OUT,
    stride_mb, stride_mh, stride_mqb, stride_mkb, stride_mqi, stride_mki,
    stride_ob, stride_oh, stride_oqb, stride_okb,
    NUM_TASKS,
    H: tl.constexpr,
    Q_NUM_BLOCKS, KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        mask_base = MASK + off_b * stride_mb + off_h * stride_mh + off_qb * stride_mqb + off_kb * stride_mkb
        q_inner = tl.arange(0, Q_BLOCK_SIZE)[:, None]
        kv_inner = tl.arange(0, KV_BLOCK_SIZE)[None, :]
        ptrs = mask_base + q_inner * stride_mqi + kv_inner * stride_mki
        vals = tl.load(ptrs)
        count = tl.sum(tl.sum(vals.to(tl.int32), axis=1), axis=0)

        out_ptr = OUT + off_b * stride_ob + off_h * stride_oh + off_qb * stride_oqb + off_kb * stride_okb
        tl.store(out_ptr, count)


def triton_count_nonzero_last(new_mask):
    new_mask = new_mask.contiguous()
    B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS, Q_BLOCK_SIZE, KV_BLOCK_SIZE = new_mask.shape
    out_i32 = torch.empty((B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS), device=new_mask.device, dtype=torch.int32)
    num_tasks = B * H * Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    bool_count_nonzero_kernel[grid](
        new_mask, out_i32,
        new_mask.stride(0), new_mask.stride(1), new_mask.stride(2), new_mask.stride(3),
        new_mask.stride(4), new_mask.stride(5),
        out_i32.stride(0), out_i32.stride(1), out_i32.stride(2), out_i32.stride(3),
        NUM_TASKS=num_tasks, H=H, Q_NUM_BLOCKS=Q_NUM_BLOCKS, KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    return out_i32


def _round_up_to_multiple(x, multiple):
    return (x + multiple - 1) // multiple * multiple


# ============================================================================
# Kernel 2: create_mask_kernel
# ============================================================================
@triton.jit
def create_mask_kernel(
    OUT, stride_ob, stride_oh, stride_oq, stride_ok,
    TABLE1, stride_t1, TABLE2, stride_t2, TABLE3, stride_t3,
    Q_LEN, KV_LEN, W, G,
    MASK_TYPE: tl.constexpr, TILE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int32)
    pid_k = tl.program_id(1).to(tl.int32)
    q_off = pid_q * TILE + tl.arange(0, TILE)
    k_off = pid_k * TILE + tl.arange(0, TILE)
    q_idx = q_off[:, None]
    k_idx = k_off[None, :]

    if MASK_TYPE == 0:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=0)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-1)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        window = causal & ((q_idx - k_idx) <= W)
        ds_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        glob = causal & (k_idx >= ds_q) & (k_idx < ds_q + G)
        sparse = same_doc & (window | glob)
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = sparse | same_img
    elif MASK_TYPE == 1:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        frame_causal = fid_q >= fid_k
        result = same_doc & frame_causal
    elif MASK_TYPE == 2:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_video = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        same_frame = fid_q == fid_k
        prev_frame = fid_q > fid_k
        result = same_video & (same_frame | prev_frame)
    elif MASK_TYPE == 3:
        causal = q_idx >= k_idx
        mod_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        is_video = mod_q > 0
        same_video = is_video & (mod_q == mod_k)
        result = causal | same_video
    elif MASK_TYPE == 4:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        samedoc_causal = same_doc & causal
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = samedoc_causal | same_img
    else:
        result = tl.full([TILE, TILE], False, tl.int1)

    valid = (q_idx < Q_LEN) & (k_idx < KV_LEN)
    ptrs = OUT + q_idx * stride_oq + k_idx * stride_ok
    tl.store(ptrs, result, mask=valid)


_MASK_TYPE_MAP = {
    "sparse": 0, "stair": 1, "video_stair": 2,
    "cross_sample_causal_video_bidir": 3, "full": 4,
}


def triton_create_mask(problem, mask_type, tile_size=128):
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    out = torch.empty(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device)
    mt = _MASK_TYPE_MAP[mask_type]
    t1 = t2 = t3 = torch.empty(0, device=device)
    s1 = s2 = s3 = 0
    W_val = G_val = 0

    if mt == 0:
        t1, t2, t3 = problem["segment_ids"], problem["doc_start"], problem["modality"]
        s1, s2, s3 = t1.stride(0), t2.stride(0), t3.stride(0)
        W_val = problem["sliding_window"]
        G_val = problem["global_window"]
    elif mt in (1, 2):
        t1, t2 = problem["video_ids"], problem["frame_ids"]
        s1, s2 = t1.stride(0), t2.stride(0)
    elif mt == 3:
        t1 = problem["modality"]
        s1 = t1.stride(0)
    elif mt == 4:
        t1, t3 = problem["segment_ids"], problem["modality"]
        s1, s3 = t1.stride(0), t3.stride(0)

    n_tiles = (SEQ_LEN + tile_size - 1) // tile_size
    create_mask_kernel[(n_tiles, n_tiles)](
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        t1, s1, t2, s2, t3, s3,
        SEQ_LEN, SEQ_LEN, W_val, G_val,
        MASK_TYPE=mt, TILE=tile_size,
    )
    return out


# ============================================================================
# Kernel 3: block_classify_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=["stride_mq", "stride_mk", "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "NUM_TASKS"]
)
def block_classify_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk,
    BLOCK_FLAGS, stride_fb, stride_fh, stride_fqb, stride_fkb,
    Q_LEN, KV_LEN, NUM_TASKS,
    H: tl.constexpr, Q_NUM_BLOCKS, KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    TILE_M: tl.constexpr = 64
    TILE_N: tl.constexpr = 64

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        has_one = tl.full((), 0, dtype=tl.int32)
        all_one = tl.full((), 1, dtype=tl.int32)
        mask_base = DENSE_MASK + off_b * stride_mb + off_h * stride_mh

        for m0 in range(0, Q_BLOCK_SIZE, TILE_M):
            offs_m = off_qb * Q_BLOCK_SIZE + m0 + tl.arange(0, TILE_M)
            valid_m = offs_m < Q_LEN
            for n0 in range(0, KV_BLOCK_SIZE, TILE_N):
                offs_n = off_kb * KV_BLOCK_SIZE + n0 + tl.arange(0, TILE_N)
                valid_n = offs_n < KV_LEN
                valid = valid_m[:, None] & valid_n[None, :]
                ptrs = mask_base + offs_m[:, None] * stride_mq + offs_n[None, :] * stride_mk
                vals = tl.load(ptrs, mask=valid, other=0).to(tl.int32)
                tile_any = tl.max(tl.max(tl.where(valid, vals, 0), axis=1), axis=0)
                tile_all = tl.min(tl.min(tl.where(valid, vals, 0), axis=1), axis=0)
                has_one = tl.where(tile_any != 0, 1, has_one)
                all_one = tl.where(tile_all == 0, 0, all_one)

        partial = (has_one == 1) & (all_one == 0)
        full = all_one == 1
        flag = tl.where(full, 2, tl.where(partial, 1, 0))
        out_ptr = BLOCK_FLAGS + off_b * stride_fb + off_h * stride_fh + off_qb * stride_fqb + off_kb * stride_fkb
        tl.store(out_ptr, flag.to(tl.int8))


def classify_mask_blocks(dense_mask, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    dense_mask = dense_mask.contiguous()
    B, H, Q_PAD, KV_PAD = dense_mask.shape
    Q_NUM_BLOCKS = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    block_flags = torch.zeros((B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS), device=dense_mask.device, dtype=torch.int8)
    num_tasks = B * H * Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    block_classify_kernel[grid](
        dense_mask, dense_mask.stride(0), dense_mask.stride(1), dense_mask.stride(2), dense_mask.stride(3),
        block_flags, block_flags.stride(0), block_flags.stride(1), block_flags.stride(2), block_flags.stride(3),
        Q_LEN, KV_LEN, NUM_TASKS=num_tasks, H=H,
        Q_NUM_BLOCKS=Q_NUM_BLOCKS, KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    return block_flags


def _convert_mask_to_block_mask(mask, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128, separate_full_blocks=False):
    def padding_needed(x, m):
        return _round_up_to_multiple(x, m) - x

    mask_pad = F.pad(mask, (0, padding_needed(mask.shape[-1], KV_BLOCK_SIZE), 0, padding_needed(mask.shape[-2], Q_BLOCK_SIZE)))
    B, H, Q, KV = mask_pad.shape
    mask_pad = mask_pad.view(B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE)
    new_mask = mask_pad.permute(0, 1, 2, 4, 3, 5)
    mask_block_sum = triton_count_nonzero_last(new_mask)
    del new_mask, mask_pad

    if separate_full_blocks:
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        return partial_blocks.to(dtype=torch.int8), full_blocks.to(dtype=torch.int8)
    else:
        return (mask_block_sum > 0).to(dtype=torch.int8), None


def _compute_partial_offsets(block_flags):
    flags = (block_flags == 1).to(torch.int32)
    A = flags.cumsum(dim=-1)
    B = A.max(dim=-1).values
    C = B.cumsum(dim=-1) - B
    total_partial = int(B.sum().item())
    return C.contiguous().to(torch.int32), A, total_partial


# ============================================================================
# Kernel 4: pack_partial_blocks_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mq", "stride_mk", "stride_offset_q",
        "stride_local_q", "stride_local_k",
        "stride_flag_q", "stride_flag_k",
        "stride_table_q", "stride_table_k",
        "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "TOTAL_PARTIAL",
    ]
)
def pack_partial_blocks_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk,
    BLOCK_FLAGS, stride_flag_q, stride_flag_k,
    PARTIAL_OFFSETS, stride_offset_q,
    LOCAL_IDX, stride_local_q, stride_local_k,
    PACKED_MASK, stride_packed_p, stride_packed_m, stride_packed_n,
    BLOCK_TABLE, stride_table_q, stride_table_k,
    Q_LEN, KV_LEN, Q_NUM_BLOCKS, KV_NUM_BLOCKS, TOTAL_PARTIAL,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    if pid_q >= Q_NUM_BLOCKS:
        return

    row_offset = tl.load(PARTIAL_OFFSETS + pid_q * stride_offset_q).to(tl.int32)
    offs_m_local = tl.arange(0, Q_BLOCK_SIZE)[:, None].to(tl.int64)
    offs_n_local = tl.arange(0, KV_BLOCK_SIZE)[None, :].to(tl.int64)

    for kv_idx in range(KV_NUM_BLOCKS):
        flag = tl.load(BLOCK_FLAGS + pid_q * stride_flag_q + kv_idx * stride_flag_k).to(tl.int32)
        is_partial = flag == 1
        if is_partial:
            local_idx = tl.load(LOCAL_IDX + pid_q * stride_local_q + kv_idx * stride_local_k).to(tl.int32)
            packed_idx = (row_offset + local_idx - 1).to(tl.int64)
            offs_m = (pid_q * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE))[:, None].to(tl.int64)
            offs_n = (kv_idx * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE))[None, :].to(tl.int64)
            valid_src = (offs_m < Q_LEN) & (offs_n < KV_LEN)
            src_ptrs = DENSE_MASK + offs_m * stride_mq + offs_n * stride_mk
            block = tl.load(src_ptrs, mask=valid_src, other=0)
            dst_ptrs = PACKED_MASK + packed_idx * stride_packed_p + offs_m_local * stride_packed_m + offs_n_local * stride_packed_n
            tl.store(dst_ptrs, block)
            tl.store(BLOCK_TABLE + pid_q * stride_table_q + kv_idx * stride_table_k, packed_idx.to(tl.int32))


def pack_partial_blocks(dense_mask, block_flags, partial_mask_offsets, local_idx, total_partial, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    dense_mask = dense_mask.contiguous()
    B, H, Q_NB, KV_NB = block_flags.shape
    device = dense_mask.device

    if total_partial == 0:
        return (
            torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device),
            torch.full((Q_NB, KV_NB), -1, dtype=torch.int32, device=device),
            torch.zeros((B, H, Q_NB), dtype=torch.int32, device=device),
        )

    packed_partial_mask = torch.zeros((total_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device)
    partial_block_table = torch.full((Q_NB, KV_NB), -1, dtype=torch.int32, device=device)
    offsets_1d = partial_mask_offsets[0, 0].contiguous()
    local_idx_2d = local_idx[0, 0].contiguous()
    flags_2d = block_flags[0, 0].contiguous()

    pack_partial_blocks_kernel[(Q_NB,)](
        dense_mask, dense_mask.stride(0), dense_mask.stride(1), dense_mask.stride(2), dense_mask.stride(3),
        flags_2d, flags_2d.stride(0), flags_2d.stride(1),
        offsets_1d, offsets_1d.stride(0),
        local_idx_2d, local_idx_2d.stride(0), local_idx_2d.stride(1),
        packed_partial_mask, packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
        partial_block_table, partial_block_table.stride(0), partial_block_table.stride(1),
        Q_LEN, KV_LEN, Q_NUM_BLOCKS=Q_NB, KV_NUM_BLOCKS=KV_NB, TOTAL_PARTIAL=total_partial,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    return packed_partial_mask, partial_block_table, partial_mask_offsets.contiguous()


def _build_packed_block_mask(dense_mask, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    block_flags = classify_mask_blocks(dense_mask, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    partial_mask_offsets, local_idx, total_partial = _compute_partial_offsets(block_flags)
    packed_partial_mask, partial_block_table, partial_mask_offsets_3d = pack_partial_blocks(
        dense_mask, block_flags, partial_mask_offsets, local_idx, total_partial, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (SEQ_LEN, SEQ_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets_3d
    packed_block_mask.partial_block_table = partial_block_table
    del block_flags, partial_mask_offsets, local_idx, partial_bm, full_bm
    return packed_block_mask


# ============================================================================
# Mask function definitions
# ============================================================================
def _sparse_mask_mod(problem):
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= W)
        glob = causal & (kv_idx >= doc_start[q_idx]) & (kv_idx < doc_start[q_idx] + G)
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img
    return mask_mod


def _stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = video_ids[q_idx] == video_ids[kv_idx]
        frame_causal = frame_ids[q_idx] >= frame_ids[kv_idx]
        return same_doc & frame_causal
    return mask_mod


def _video_stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_video = video_ids[q_idx] == video_ids[kv_idx]
        same_frame = frame_ids[q_idx] == frame_ids[kv_idx]
        prev_frame = frame_ids[q_idx] > frame_ids[kv_idx]
        return same_video & (same_frame | prev_frame)
    return mask_mod


def _cross_sample_causal_video_bidir_mask_mod(problem):
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_video = modality[q_idx] > 0
        same_video = is_video & (modality[q_idx] == modality[kv_idx])
        return causal | same_video
    return mask_mod


def _full_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img
    return mask_mod


_MASK_FUNC_TO_TYPE = {
    id(_sparse_mask_mod): "sparse",
    id(_stair_mask_mod): "stair",
    id(_video_stair_mask_mod): "video_stair",
    id(_cross_sample_causal_video_bidir_mask_mod): "cross_sample_causal_video_bidir",
    id(_full_mask_mod): "full",
}


# ============================================================================
# Attention wrappers
# ============================================================================
def _flex_attention_mojo(q, k, v, mask, block_mask, dropout_rate=0.0, input_format=None):
    if input_format == "head-first":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if mask is not None:
        block_mask.dense_mask = mask
    output = mojo_flex_attention(q, k, v, block_mask=block_mask)
    return output.transpose(1, 2)


def _sdpa_with_dense_mask(query_states, key_states, value_states, attention_mask, dropout_rate, input_format):
    if input_format == "head-first":
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    mask_2d = attention_mask
    while mask_2d.dim() > 2:
        mask_2d = mask_2d[0]

    q_len_total = query_states.size(2)
    block_q = APPLY_Q_CHUNK if APPLY_Q_CHUNK is not None else q_len_total
    chunks = []
    for qs in range(0, q_len_total, block_q):
        qe = min(qs + block_q, q_len_total)
        row = mask_2d[qs:qe]
        col_any = row.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            chunks.append(query_states.new_zeros((query_states.size(0), query_states.size(1), qe - qs, query_states.size(3))))
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        chunks.append(
            F.scaled_dot_product_attention(
                query_states[:, :, qs:qe], key_states[:, :, kmin:kmax], value_states[:, :, kmin:kmax],
                attn_mask=row[None, None, :, kmin:kmax], dropout_p=dropout_rate, enable_gqa=False,
            )
        )
    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous()


# ============================================================================
# Data building
# ============================================================================
def _build_video_indicators(device):
    segment_ids, doc_start, video_ids, frame_ids, modality = [], [], [], [], []
    sample_start = 0
    next_video_id = 0

    for sample_id, sample_videos in enumerate(VIDEO_FRAME_LENGTH):
        for frame_lens in sample_videos:
            cur_video_id = next_video_id
            next_video_id += 1
            for frame_id, frame_len in enumerate(frame_lens):
                segment_ids.append(torch.full((frame_len,), sample_id, dtype=torch.long))
                doc_start.append(torch.full((frame_len,), sample_start, dtype=torch.long))
                video_ids.append(torch.full((frame_len,), cur_video_id, dtype=torch.long))
                frame_ids.append(torch.full((frame_len,), frame_id, dtype=torch.long))
                modality.append(torch.full((frame_len,), cur_video_id + 1, dtype=torch.long))
        sample_start += sum(sum(fl) for fl in sample_videos)

    return {
        "segment_ids": torch.cat(segment_ids).to(device),
        "doc_start": torch.cat(doc_start).to(device),
        "video_ids": torch.cat(video_ids).to(device),
        "frame_ids": torch.cat(frame_ids).to(device),
        "modality": torch.cat(modality).to(device),
    }


def _build_modality_indicators(device):
    indicator = []
    iidx = 1
    for sample_types, sample_lens in zip(DATA_INPUT_TYPE, DATA_LENGTH):
        for sample_type, sample_len in zip(sample_types, sample_lens):
            if sample_type in FULL_MASK_MODALITIES:
                indicator.append(torch.full((sample_len,), iidx, dtype=torch.long))
                iidx += 1
            else:
                indicator.append(torch.full((sample_len,), -1, dtype=torch.long))
    return torch.cat(indicator).to(device)


def build_problem(mask_mod):
    device = _device()
    torch.manual_seed(SEED)
    local_data_len = DATA_LENGTH
    if mask_mod in [_video_stair_mask_mod, _stair_mask_mod]:
        local_data_len = DATA_LENGTH_VIDEO

    sample_lens = [sum(s) for s in local_data_len]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int32),
        torch.tensor(sample_lens, device=device),
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)

    q = torch.rand(1, NUM_Q_HEADS, total_s, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.rand(1, NUM_KV_HEADS, total_s, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.rand(1, NUM_KV_HEADS, total_s, HEAD_DIM, device=device, dtype=DTYPE)

    if mask_mod in [_video_stair_mask_mod, _stair_mask_mod]:
        meta = _build_video_indicators(device=device)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": meta["segment_ids"], "doc_start": meta["doc_start"],
            "video_ids": meta["video_ids"], "frame_ids": meta["frame_ids"], "modality": meta["modality"],
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": SLIDING_WINDOW, "global_window": GLOBAL_WINDOW,
            "num_q_heads": NUM_Q_HEADS, "num_kv_heads": NUM_KV_HEADS, "head_dim": HEAD_DIM,
        }
    else:
        modality = _build_modality_indicators(device=device)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": segment_ids.long(), "modality": modality, "doc_start": doc_start,
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": SLIDING_WINDOW, "global_window": GLOBAL_WINDOW,
            "num_q_heads": NUM_Q_HEADS, "num_kv_heads": NUM_KV_HEADS, "head_dim": HEAD_DIM,
        }


# ============================================================================
# Test cases
# ============================================================================
@pytest.mark.parametrize(
    "mask_func",
    [
        _sparse_mask_mod,
        _full_mask_mod,
        _stair_mask_mod,
        _video_stair_mask_mod,
        _cross_sample_causal_video_bidir_mask_mod,
    ],
    ids=["sparse", "stair", "full", "video_stair", "cross_sample_causal_video_bidir"],
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
@bypass_not_implemented
def test_flex_attention(mask_func):
    problem = build_problem(mask_func)

    q_base = problem["q"]
    k_base = problem["k"]
    v_base = problem["v"]

    q_mojo = q_base.detach().clone().requires_grad_(True)
    k_mojo = k_base.detach().clone().requires_grad_(True)
    v_mojo = v_base.detach().clone().requires_grad_(True)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)

    SEQ_LEN = problem["total_s"]

    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
    dense_mask = triton_create_mask(problem, mask_type_str)
    _sync()

    packed_block_mask = _build_packed_block_mask(dense_mask, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    return_grid = torch.tensor(520000, dtype=DTYPE, device=torch.device(_device()))

    mojo_output = _flex_attention_mojo(q_mojo, k_mojo, v_mojo, None, packed_block_mask, 0.0, None)
    _sync()

    ref_output = _sdpa_with_dense_mask(q_ref, k_ref, v_ref, dense_mask, 0.0, None)
    _sync()

    assert mojo_output.shape == ref_output.shape
    torch.testing.assert_close(mojo_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    _sync()

    mojo_output.float().mean().backward(return_grid)
    _sync()

    ref_output.float().mean().backward(return_grid)
    _sync()

    torch.testing.assert_close(q_mojo.grad.cpu(), q_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k_mojo.grad.cpu(), k_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v_mojo.grad.cpu(), v_ref.grad.cpu(), atol=5e-3, rtol=5e-3)


# ============================================================================
# Performance benchmark (torch_npu.profiler based)
# ============================================================================
def _perf_benchmark(label, build_mask_fn, fwd_fn, q, k, v, prof_dir_root, mask_func):
    import torch_npu

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    return_grid = torch.tensor(520000, dtype=DTYPE, device=torch.device(_device()))

    # mask build measurement: peak + stable
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    _sync()
    mask = build_mask_fn()
    mask_peak = torch.npu.max_memory_allocated() / _MB
    torch.npu.empty_cache()
    gc.collect()
    _sync()
    mask_mem = torch.npu.memory_allocated() / _MB

    # fwd measurement (includes grad graph)
    torch.npu.reset_peak_memory_stats()
    _sync()
    out = fwd_fn(q, k, v, mask)
    _sync()
    fwd_mem = torch.npu.max_memory_allocated() / _MB

    # bwd measurement
    torch.npu.reset_peak_memory_stats()
    _sync()
    out.float().mean().backward(return_grid)
    _sync()
    bwd_mem = torch.npu.max_memory_allocated() / _MB

    q.grad = k.grad = v.grad = None
    peak_mem = max(fwd_mem, bwd_mem)
    print(f"[{label}] mask: {mask_mem:.1f}MB(peak:{mask_peak:.1f}MB), fwd_mem: {fwd_mem:.1f}MB, bwd_mem: {bwd_mem:.1f}MB, peak: {peak_mem:.1f}MB")

    # ===== torch_npu.profiler based timing =====
    prof_dir = os.path.join(prof_dir_root, label)
    os.makedirs(prof_dir, exist_ok=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    print(f"\n======================== prof begin ({label}) ====================")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=10, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    ) as prof:
        for i in range(12):
            # 重新构造数据，避免L2 Cache影响
            problem = build_problem(mask_func)
            q = problem["q"].detach().clone().requires_grad_(True)
            k = problem["k"].detach().clone().requires_grad_(True)
            v = problem["v"].detach().clone().requires_grad_(True)

            out = fwd_fn(q, k, v, mask)
            _sync()

            # 插入其他算子，重置L2 Cache, 112M，总访存224MB覆盖112MB L2, 模拟整网调用场景
            for j in range(5):
                a = torch.randn(19573419, dtype=torch.float32, device="cpu").to(q.device)
                b = torch.randn(19573419, dtype=torch.float32, device="cpu").to(q.device)
                c = a + b       # 冲刷全部L2
            _sync()

            out.float().mean().backward(return_grid)
            _sync()
            prof.step()
            del a,b,c
    print(f"======================== prof end ({label}) ====================")

    del out, mask
    torch.npu.empty_cache()
    return {"mask_mem_mb": mask_mem, "mask_peak_mb": mask_peak,
            "fwd_mem_mb": fwd_mem, "bwd_mem_mb": bwd_mem,
            "peak_mem_mb": peak_mem}


def _perf_flex_attention(mask_func, problem=None):
    if problem is None:
        problem = build_problem(mask_func)
    SEQ_LEN = problem["total_s"]
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]

    prof_dir_root = os.path.join("./prof_dir", mask_type_str)
    os.makedirs(prof_dir_root, exist_ok=True)

    results = {}

    # mojo_packed: dense_mask → classify + pack → del dense_mask
    gc.collect()
    torch.npu.empty_cache()

    def _build_packed_mask():
        dm = triton_create_mask(problem, mask_type_str)
        pbm = _build_packed_block_mask(dm, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        del dm
        torch.npu.empty_cache()
        return pbm

    results["mojo_packed"] = _perf_benchmark(
        "mojo_packed",
        _build_packed_mask,
        lambda q, k, v, bm: _flex_attention_mojo(q, k, v, None, bm, 0.0, None),
        problem["q"], problem["k"], problem["v"],
        prof_dir_root,
        mask_func,
    )

    # ascendc: torch SDPA + dense_mask
    gc.collect()
    torch.npu.empty_cache()

    results["ascendc"] = _perf_benchmark(
        "ascendc",
        lambda: triton_create_mask(problem, mask_type_str),
        lambda q, k, v, m: _sdpa_with_dense_mask(q, k, v, m, 0.0, None),
        problem["q"], problem["k"], problem["v"],
        prof_dir_root,
        mask_func,
    )
    return results


@pytest.mark.parametrize(
    "mask_func",
    [
        _sparse_mask_mod,
        _full_mask_mod,
        _stair_mask_mod,
        _video_stair_mask_mod,
        _cross_sample_causal_video_bidir_mask_mod,
    ],
    ids=["sparse", "full", "stair", "video_stair", "cross_sample_causal_video_bidir"],
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
def test_flex_attention_perf(mask_func):
    problem = build_problem(mask_func)
    results = _perf_flex_attention(mask_func, problem)
    print(f"\n{'=' * 60}")
    print(f"Performance results for {_MASK_FUNC_TO_TYPE[id(mask_func)]}:")
    for label, r in results.items():
        print(f"  [{label}] mask: {r['mask_mem_mb']:.1f}MB(peak:{r['mask_peak_mb']:.1f}MB), "
              f"fwd_mem: {r['fwd_mem_mb']:.1f}MB, bwd_mem: {r['bwd_mem_mb']:.1f}MB, "
              f"peak: {r['peak_mem_mb']:.1f}MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys

    mask_map = {
        "sparse": _sparse_mask_mod,
        "stair": _stair_mask_mod,
        "video_stair": _video_stair_mask_mod,
        "cross_sample_causal_video_bidir": _cross_sample_causal_video_bidir_mask_mod,
        "full": _full_mask_mod,
    }
    name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    if name == "all":
        for n, fn in mask_map.items():
            print(f"\n{'=' * 60}")
            print(f"Testing: {n}")
            print(f"{'=' * 60}")
            test_flex_attention(fn)
            test_flex_attention_perf(fn)
    else:
        test_flex_attention(mask_map[name])
        test_flex_attention_perf(mask_map[name])
