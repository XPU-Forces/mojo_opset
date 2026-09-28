# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""SparseFlashMlaMetadata AICPU operator for the A A5 SMLA DSL kernel.

The public ABI follows ops-transformer SparseFlashMlaMetadata.  The scheduling
algorithm is ported from its arch35 AICPU source.  This target deliberately keeps
FlashDecode and batch-consistency disabled, matching the supported SMLA kernel.
"""
import os
import tempfile
from functools import lru_cache

from cannbotdsl.aicpu import GmIn, GmOut, I32, I64, U32, aicpu_kernel

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
SMLA_METADATA_TOTAL_SIZE = 1024
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7
FA_S2_MAX_NUM = 8

SPARSE_DEFAULT_MASK = 0
SPARSE_ALL_MASK = 1
SPARSE_LEFT_UP_CAUSAL = 2
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4
LAYOUT_BSND = 0
LAYOUT_BNSD = 1
LAYOUT_BSH = 2
LAYOUT_NBSD = 3
LAYOUT_TND = 4
LAYOUT_NTD = 5
LAYOUT_PA_BBND = 6
_LAYOUT_CODES = {
    "BSND": LAYOUT_BSND, "BNSD": LAYOUT_BNSD, "BSH": LAYOUT_BSH,
    "NBSD": LAYOUT_NBSD, "TND": LAYOUT_TND, "NTD": LAYOUT_NTD,
    "PA": LAYOUT_PA_BBND, "PA_BBND": LAYOUT_PA_BBND,
}

INT64_MAX = 9223372036854775807
UINT32_MAX = 4294967295
FA_TOLERANCE_RATIO = 2
COST_WEIGHT_M = 6
COST_WEIGHT_S2 = 10
S2_BASE_SIZE = 128


class _SparseFlashMlaMetadataArgs:
    cu_seqlens_q: GmIn(I32)
    cu_seqlens_ori_kv: GmIn(I32)
    cu_seqlens_cmp_kv: GmIn(I32)
    seqused_q: GmIn(I32)
    seqused_ori_kv: GmIn(I32)
    seqused_cmp_kv: GmIn(I32)
    cmp_residual_kv: GmIn(I32)
    ori_topk_length: GmIn(I32)
    cmp_topk_length: GmIn(I32)
    metadata: GmOut(I32)

    cu_q_len: U32
    cu_ori_len: U32
    cu_cmp_len: U32
    sq_len: U32
    sori_len: U32
    scmp_len: U32
    residual_len: U32
    ori_topk_len: U32
    cmp_topk_len: U32

    num_heads_q: U32
    num_heads_kv: U32
    head_dim: U32
    batch_size: U32
    max_seqlen_q: U32
    max_seqlen_ori_kv: U32
    max_seqlen_cmp_kv: U32
    ori_topk: U32
    cmp_topk: U32
    cmp_ratio: U32
    ori_mask_mode: U32
    cmp_mask_mode: U32
    ori_win_left: I64
    ori_win_right: I64
    layout_q: U32
    layout_kv: U32
    has_ori_kv: U32
    has_cmp_kv: U32
    aic_core_num: U32
    aiv_core_num: U32
    is_batch_consistency: U32


def _ceil_div(x, y):
    return (x + y - 1) // y


def _seq_q(a, b):
    if a.sq_len != 0:
        return a.seqused_q[b]
    if a.layout_q == LAYOUT_TND and a.cu_q_len != 0:
        return a.cu_seqlens_q[b + 1] - a.cu_seqlens_q[b]
    return a.max_seqlen_q


def _seq_ori(a, b):
    if a.sori_len != 0:
        return a.seqused_ori_kv[b]
    if a.layout_kv == LAYOUT_TND and a.cu_ori_len != 0:
        return a.cu_seqlens_ori_kv[b + 1] - a.cu_seqlens_ori_kv[b]
    if a.ori_topk != 0 and (a.layout_kv == LAYOUT_PA_BBND or a.max_seqlen_ori_kv == 0):
        return UINT32_MAX
    return a.max_seqlen_ori_kv


def _seq_cmp(a, b):
    if a.scmp_len != 0:
        return a.seqused_cmp_kv[b]
    if a.layout_kv == LAYOUT_TND and a.cu_cmp_len != 0:
        return a.cu_seqlens_cmp_kv[b + 1] - a.cu_seqlens_cmp_kv[b]
    if a.cmp_topk != 0 and (a.layout_kv == LAYOUT_PA_BBND or a.max_seqlen_cmp_kv == 0):
        return UINT32_MAX
    return a.max_seqlen_cmp_kv


def _bs_stride(a, b, row):
    if a.layout_q == LAYOUT_TND and a.cu_q_len != 0:
        return a.cu_seqlens_q[b] + row
    return b * a.max_seqlen_q + row


def _ori_topk(a, b, row):
    if a.ori_topk != 0 and a.ori_mask_mode == SPARSE_DEFAULT_MASK and a.ori_topk_len != 0:
        return a.ori_topk_length[_bs_stride(a, b, row)]
    return a.ori_topk


def _cmp_topk(a, b, row):
    if a.cmp_topk != 0 and a.cmp_mask_mode == SPARSE_DEFAULT_MASK and a.cmp_topk_len != 0:
        return a.cmp_topk_length[_bs_stride(a, b, row)]
    return a.cmp_topk


def _token_range(a, row, s1, s2, mode, is_cmp):
    first = 0
    last = s2 - 1
    if mode != SPARSE_DEFAULT_MASK:
        pre = INT64_MAX
        nxt = INT64_MAX
        if mode == SPARSE_RIGHT_DOWN_CAUSAL:
            nxt = s2 - s1
        elif mode == SPARSE_BAND:
            if a.ori_win_left > -1:
                pre = s1 - s2 + a.ori_win_left
            if a.ori_win_right > -1:
                nxt = s2 - s1 + a.ori_win_right
        else:
            if a.ori_win_left > -1:
                pre = a.ori_win_left
            if a.ori_win_right > -1:
                nxt = a.ori_win_right
        first = row - pre
        if nxt == INT64_MAX:
            last = INT64_MAX
        else:
            last = row + nxt
    return first, last


def _active_tokens(a, b, row, is_cmp):
    s1 = _seq_q(a, b)
    if s1 == 0:
        return 0
    if is_cmp != 0:
        cmp_s2 = _seq_cmp(a, b)
        residual = 0
        if a.residual_len != 0:
            residual = a.cmp_residual_kv[b]
        revert_s2 = cmp_s2 * a.cmp_ratio + residual
        if revert_s2 == 0:
            return 0
        first = 0
        last = revert_s2 - 1
        mode = a.cmp_mask_mode
        if mode != SPARSE_DEFAULT_MASK:
            pre = INT64_MAX
            nxt = INT64_MAX
            if mode == SPARSE_RIGHT_DOWN_CAUSAL:
                nxt = revert_s2 - s1
            elif mode == SPARSE_BAND:
                if a.ori_win_left > -1:
                    pre = s1 - revert_s2 + a.ori_win_left
                if a.ori_win_right > -1:
                    nxt = revert_s2 - s1 + a.ori_win_right
            else:
                if a.ori_win_left > -1:
                    pre = a.ori_win_left
                if a.ori_win_right > -1:
                    nxt = a.ori_win_right
            first = row - pre
            last = INT64_MAX if nxt == INT64_MAX else row + nxt
        if first >= revert_s2 or last < 0 or last < first:
            return 0
        if first < 0:
            first = 0
        if first >= revert_s2:
            first = revert_s2 - 1
        if last < 0:
            last = 0
        if last >= revert_s2:
            last = revert_s2 - 1
        if (last + 1) // a.cmp_ratio == 0:
            return 0
        cmp_first = 0
        if (first + 1) // a.cmp_ratio != 0:
            cmp_first = (first + 1) // a.cmp_ratio - 1
        cmp_last = (last + 1) // a.cmp_ratio - 1
        active = cmp_last - cmp_first + 1
        if a.cmp_topk != 0:
            topk = _cmp_topk(a, b, row)
            if active > topk:
                active = topk
        return active
    ori_s2 = _seq_ori(a, b)
    if ori_s2 == 0:
        return 0
    first = 0
    last = ori_s2 - 1
    mode = a.ori_mask_mode
    if mode != SPARSE_DEFAULT_MASK:
        pre = INT64_MAX
        nxt = INT64_MAX
        if mode == SPARSE_RIGHT_DOWN_CAUSAL:
            nxt = ori_s2 - s1
        elif mode == SPARSE_BAND:
            if a.ori_win_left > -1:
                pre = s1 - ori_s2 + a.ori_win_left
            if a.ori_win_right > -1:
                nxt = ori_s2 - s1 + a.ori_win_right
        else:
            if a.ori_win_left > -1:
                pre = a.ori_win_left
            if a.ori_win_right > -1:
                nxt = a.ori_win_right
        first = row - pre
        last = INT64_MAX if nxt == INT64_MAX else row + nxt
    if first >= ori_s2 or last < 0 or last < first:
        return 0
    if first < 0:
        first = 0
    if first >= ori_s2:
        first = ori_s2 - 1
    if last < 0:
        last = 0
    if last >= ori_s2:
        last = ori_s2 - 1
    active = last - first + 1
    if a.ori_topk != 0:
        topk = _ori_topk(a, b, row)
        if active > topk:
            active = topk
    return active


def _block_cost(group_size, tokens):
    return COST_WEIGHT_M * _ceil_div(group_size, 16) + COST_WEIGHT_S2 * _ceil_div(tokens, 64)


def _calc_row(a, b, row, out):
    # out = cost, blocks, last_block_cost, s2_loop
    ori = 0
    cmp = 0
    if a.has_ori_kv != 0:
        ori = _active_tokens(a, b, row, 0)
    if a.has_cmp_kv != 0:
        cmp = _active_tokens(a, b, row, 1)
    group_size = a.num_heads_q // a.num_heads_kv
    ori_blocks = _ceil_div(ori, S2_BASE_SIZE) if ori != 0 else 0
    cmp_blocks = _ceil_div(cmp, S2_BASE_SIZE) if cmp != 0 else 0
    ori_cost = 0
    ori_last = 0
    if ori_blocks != 0:
        ori_tail = ori % S2_BASE_SIZE
        ori_last = _block_cost(group_size, ori_tail if ori_tail != 0 else S2_BASE_SIZE)
        ori_cost = (ori_blocks - 1) * _block_cost(group_size, S2_BASE_SIZE) + ori_last
    cmp_cost = 0
    cmp_last = 0
    if cmp_blocks != 0:
        cmp_tail = cmp % S2_BASE_SIZE
        cmp_last = _block_cost(group_size, cmp_tail if cmp_tail != 0 else S2_BASE_SIZE)
        cmp_cost = (cmp_blocks - 1) * _block_cost(group_size, S2_BASE_SIZE) + cmp_last
    out[0] = ori_cost + cmp_cost
    out[1] = ori_blocks + cmp_blocks
    out[2] = cmp_last if cmp_blocks != 0 else ori_last
    out[3] = ori_blocks + cmp_blocks


@aicpu_kernel
def _sparse_flash_mla_metadata_kernel(a: _SparseFlashMlaMetadataArgs):
    batch_cost = zeros(I64, a.batch_size + 1)
    batch_blocks = zeros(U32, a.batch_size + 1)
    batch_loops = zeros(U32, a.batch_size + 1)
    batch_last = zeros(I64, a.batch_size + 1)
    row_info = array(I64, 4)
    bn2_end = zeros(U32, a.aic_core_num + 1)
    gs1_end = zeros(U32, a.aic_core_num + 1)
    s2_end = zeros(U32, a.aic_core_num + 1)

    for i in range(0, SMLA_METADATA_TOTAL_SIZE):
        a.metadata[i] = 0

    total_cost = 0
    max_row_cost = 0
    for b in range(0, a.batch_size):
        s1 = _seq_q(a, b)
        for row in range(0, s1):
            _calc_row(a, b, row, row_info)
            batch_cost[b] = batch_cost[b] + row_info[0]
            batch_blocks[b] = batch_blocks[b] + row_info[1]
            batch_loops[b] = batch_loops[b] + row_info[3]
            if row_info[0] > max_row_cost:
                max_row_cost = row_info[0]
            if row_info[1] != 0:
                batch_last[b] = row_info[2]
        total_cost = total_cost + batch_cost[b] * a.num_heads_kv

    group_size = a.num_heads_q // a.num_heads_kv
    split_g = 1 if group_size > 64 else 0
    schedule_cores = a.aic_core_num
    if split_g != 0:
        schedule_cores = schedule_cores // 2

    used_core_num = 0
    max_s2_loop_num = 0
    bn2_total = a.batch_size * a.num_heads_kv
    cur_bn2 = 0
    cur_row = 0
    remaining_cost = total_cost
    remaining_bn_cost = batch_cost[0] if a.batch_size != 0 else 0
    remaining_bn_blocks = batch_blocks[0] if a.batch_size != 0 else 0
    remaining_bn_loops = batch_loops[0] if a.batch_size != 0 else 0

    if total_cost == 0:
        used_core_num = 1
        bn2_end[0] = bn2_total
        gs1_end[0] = 0
        s2_end[0] = 0
    else:
        for core in range(0, schedule_cores):
            if remaining_cost <= 0:
                break
            remaining_cores = schedule_cores - core
            limit = remaining_cost // remaining_cores
            if limit < max_row_cost:
                limit = max_row_cost
            core_cost = 0
            core_blocks = 0
            core_loops = 0

            assign_batch = 1
            while assign_batch != 0 and cur_bn2 < bn2_total:
                b = cur_bn2 // a.num_heads_kv
                tolerance = batch_last[b] // FA_TOLERANCE_RATIO
                if remaining_bn_cost == 0 or limit + tolerance >= core_cost + remaining_bn_cost:
                    core_cost = core_cost + remaining_bn_cost
                    core_blocks = core_blocks + remaining_bn_blocks
                    core_loops = core_loops + remaining_bn_loops
                    cur_bn2 = cur_bn2 + 1
                    cur_row = 0
                    if cur_bn2 < bn2_total:
                        nb = cur_bn2 // a.num_heads_kv
                        remaining_bn_cost = batch_cost[nb]
                        remaining_bn_blocks = batch_blocks[nb]
                        remaining_bn_loops = batch_loops[nb]
                else:
                    assign_batch = 0

            if cur_bn2 < bn2_total:
                b = cur_bn2 // a.num_heads_kv
                s1 = _seq_q(a, b)
                assign_row = 1
                while assign_row != 0 and cur_row < s1:
                    _calc_row(a, b, cur_row, row_info)
                    tolerance = row_info[2] // FA_TOLERANCE_RATIO
                    if limit + tolerance >= core_cost + row_info[0]:
                        core_cost = core_cost + row_info[0]
                        core_blocks = core_blocks + row_info[1]
                        core_loops = core_loops + row_info[3]
                        if remaining_bn_cost > row_info[0]:
                            remaining_bn_cost = remaining_bn_cost - row_info[0]
                        else:
                            remaining_bn_cost = 0
                        if remaining_bn_blocks > row_info[1]:
                            remaining_bn_blocks = remaining_bn_blocks - row_info[1]
                        else:
                            remaining_bn_blocks = 0
                        if remaining_bn_loops > row_info[3]:
                            remaining_bn_loops = remaining_bn_loops - row_info[3]
                        else:
                            remaining_bn_loops = 0
                        cur_row = cur_row + 1
                    else:
                        assign_row = 0

            bn2_end[core] = cur_bn2
            gs1_end[core] = cur_row
            s2_end[core] = 0
            remaining_cost = remaining_cost - core_cost
            if core_loops > max_s2_loop_num:
                max_s2_loop_num = core_loops
            used_core_num = core + 1

    if split_g != 0:
        for i in range(0, schedule_cores):
            c0 = i * 2
            c1 = c0 + 1
            a.metadata[c0 * FA_METADATA_SIZE + FA_S2_MAX_NUM] = max_s2_loop_num
            a.metadata[c1 * FA_METADATA_SIZE + FA_S2_MAX_NUM] = max_s2_loop_num
            if i < used_core_num:
                start_bn = 0 if i == 0 else bn2_end[i - 1]
                start_m = 0 if i == 0 else gs1_end[i - 1]
                for c in range(c0, c1 + 1):
                    base = c * FA_METADATA_SIZE
                    a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
                    a.metadata[base + FA_BN2_START_INDEX] = start_bn
                    a.metadata[base + FA_M_START_INDEX] = start_m
                    a.metadata[base + FA_S2_START_INDEX] = 0
                    a.metadata[base + FA_BN2_END_INDEX] = bn2_end[i]
                    a.metadata[base + FA_M_END_INDEX] = gs1_end[i]
                    a.metadata[base + FA_S2_END_INDEX] = 0
                    a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = 0
    else:
        for i in range(0, a.aic_core_num):
            if i < used_core_num:
                base = i * FA_METADATA_SIZE
                a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
                a.metadata[base + FA_BN2_START_INDEX] = 0 if i == 0 else bn2_end[i - 1]
                a.metadata[base + FA_M_START_INDEX] = 0 if i == 0 else gs1_end[i - 1]
                a.metadata[base + FA_S2_START_INDEX] = 0
                a.metadata[base + FA_BN2_END_INDEX] = bn2_end[i]
                a.metadata[base + FA_M_END_INDEX] = gs1_end[i]
                a.metadata[base + FA_S2_END_INDEX] = 0
                a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = 0
    return 0


def _layout_code(layout):
    return _LAYOUT_CODES.get(layout, 9)


def _effective_core_counts(device=None):
    import torch
    if device is None:
        device = torch.npu.current_device()
    props = torch.npu.get_device_properties(device)
    aic = int(props.cube_core_num)
    aiv = int(props.vector_core_num) or 2 * aic
    if not 0 < aic <= AIC_CORE_MAX_NUM:
        raise ValueError(f"aic_core_num must be in [1,{AIC_CORE_MAX_NUM}], got {aic}")
    if not 0 < aiv <= AIV_CORE_MAX_NUM:
        raise ValueError(f"aiv_core_num must be in [1,{AIV_CORE_MAX_NUM}], got {aiv}")
    return aic, aiv


@lru_cache(maxsize=1)
def _compiled_metadata():
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
    directory = tempfile.TemporaryDirectory(prefix="cannbot_sparse_flash_mla_metadata_")
    compiled = compile_aicpu_kernel(
        _sparse_flash_mla_metadata_kernel,
        workdir=directory.name,
        launch_mode="interface",
    )
    return directory, compiled


def sparse_flash_mla_metadata(
    cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
    seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
    cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None, *,
    num_heads_q, num_heads_kv=1, head_dim=512, batch_size=0,
    max_seqlen_q=0, max_seqlen_ori_kv=0, max_seqlen_cmp_kv=0,
    ori_topk=0, cmp_topk=0, cmp_ratio=1, ori_mask_mode=4,
    cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
    layout_q="BSND", layout_kv="PA_BBND", has_ori_kv=True,
    has_cmp_kv=True, soc_version="Ascend950", aic_core_num=None,
    aiv_core_num=None, is_batch_consistency=False, device=None,
):
    """Generate transformer-compatible SMLA metadata on AICPU."""
    import torch
    from cannbotdsl.aicpu import current_raw_stream

    tensors = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "ori_topk_length": ori_topk_length,
        "cmp_topk_length": cmp_topk_length,
    }
    if seqused_q is not None:
        actual_batch = int(seqused_q.numel())
    elif cu_seqlens_q is not None:
        actual_batch = int(cu_seqlens_q.numel()) - 1
    else:
        actual_batch = int(batch_size)
    if actual_batch < 0:
        raise ValueError("batch_size must be non-negative")
    if num_heads_kv <= 0 or num_heads_q < num_heads_kv or num_heads_q % num_heads_kv:
        raise ValueError("num_heads_q must be a positive multiple of num_heads_kv")
    if cmp_ratio <= 0:
        raise ValueError("cmp_ratio must be positive")
    if is_batch_consistency:
        raise ValueError("batch consistency is not supported in this A5 SMLA target")
    if "Ascend950" not in str(soc_version):
        raise ValueError("only Ascend950/A5 is supported")

    device_id = torch.npu.current_device() if device is None else int(device)
    npu_device = torch.device("npu", device_id)
    stream = torch.npu.current_stream(device_id)
    for name, tensor in tensors.items():
        if tensor is not None and (
            tensor.dtype != torch.int32 or tensor.device != npu_device
            or tensor.ndim != 1 or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous 1D int32 tensor on {npu_device}")

    dyn_aic, dyn_aiv = _effective_core_counts(device_id)
    aic = dyn_aic if aic_core_num is None else int(aic_core_num)
    aiv = dyn_aiv if aiv_core_num is None else int(aiv_core_num)
    if not 0 < aic <= AIC_CORE_MAX_NUM or not 0 < aiv <= AIV_CORE_MAX_NUM:
        raise ValueError(f"invalid core counts AIC={aic}, AIV={aiv}")

    metadata = torch.empty(SMLA_METADATA_TOTAL_SIZE, dtype=torch.int32, device=npu_device)
    _, compiled = _compiled_metadata()
    compiled.launch(
        current_raw_stream(device_id),
        **{name: 0 if tensor is None else tensor.data_ptr() for name, tensor in tensors.items()},
        metadata=metadata.data_ptr(),
        cu_q_len=0 if cu_seqlens_q is None else cu_seqlens_q.numel(),
        cu_ori_len=0 if cu_seqlens_ori_kv is None else cu_seqlens_ori_kv.numel(),
        cu_cmp_len=0 if cu_seqlens_cmp_kv is None else cu_seqlens_cmp_kv.numel(),
        sq_len=0 if seqused_q is None else seqused_q.numel(),
        sori_len=0 if seqused_ori_kv is None else seqused_ori_kv.numel(),
        scmp_len=0 if seqused_cmp_kv is None else seqused_cmp_kv.numel(),
        residual_len=0 if cmp_residual_kv is None else cmp_residual_kv.numel(),
        ori_topk_len=0 if ori_topk_length is None else ori_topk_length.numel(),
        cmp_topk_len=0 if cmp_topk_length is None else cmp_topk_length.numel(),
        num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim,
        batch_size=actual_batch, max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv, max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        ori_topk=ori_topk, cmp_topk=cmp_topk, cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode, cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left, ori_win_right=ori_win_right,
        layout_q=_layout_code(layout_q), layout_kv=_layout_code(layout_kv),
        has_ori_kv=int(bool(has_ori_kv)), has_cmp_kv=int(bool(has_cmp_kv)),
        aic_core_num=aic, aiv_core_num=aiv, is_batch_consistency=0,
    )
    for tensor in tensors.values():
        if tensor is not None:
            tensor.record_stream(stream)
    return metadata
