      
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Non-quantized SMLA with explicit Channel slot selection.
A5 SWA/HCA/CSA: BF16/FP16, N1=64/128; one fused kernel with runtime shapes.
The original SMLA golden and test entry are unchanged.
"""

import math
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx
from cannbotdsl import select as dyn_select
from cannbotdsl.types._integer import Int32, Int64
from cannbotdsl import dtypes
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import matmul
from cannbotdsl import ChannelKind, MemLoc, PIPE, Tensor, get_platform_info
from cannbotdsl.tensor import reinterpret, tile_slice, make_tiler, permute
from cannbotdsl import cv_valid_extent
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.sync import (
    cube_sync_intra_wait,
    vec_sync_intra_arrive,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.sync import (
    cube_sync_intra_arrive,
    global_sync_all,
    vec_sync_intra_wait,
)

D = 512
TILE = 128
D_TILES = D // TILE
_KV_RING = 3
# Cross-core channels use 0..4; keep the explicit GM workspace protocol
# disjoint: ready=9, free slots=10..12.
# The address-publication barrier uses separate flags 13..15.
_WORKSPACE_READY_FLAG = 9
_WORKSPACE_FREE_FLAG = 10
_Q_RING = 3


def _clamp_nonneg(value):
    if isinstance(value, int):
        return max(0, value)
    return Int64(dyn_select(value < 0, 0, value))


def _rt_min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    a64 = a if not isinstance(a, int) else Int64(a)
    b64 = b if not isinstance(b, int) else Int64(b)
    return Int64(dyn_select(a64 < b64, a64, b64))


def _rt_max(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    a64 = a if not isinstance(a, int) else Int64(a)
    b64 = b if not isinstance(b, int) else Int64(b)
    return Int64(dyn_select(a64 > b64, a64, b64))


class MqsmlaMatmul:
    """Three-slot Q/KV rings; selected KV survives both QK and PV reads."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_n, dtype=dtypes.bfloat16):
        self.dtype = dtype
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)
        self.q_ring = Channel(MemLoc.L1, (tile_cube_m, D // 2), dtype,
                              depth=_Q_RING, addr=32768)
        self.kv_ring = Channel(MemLoc.L1, (tile_n, D), dtype,
                               depth=_KV_RING, addr=131072)
        self.l0a = Channel(MemLoc.L0A, (tile_cube_m, TILE), dtype, depth=2)
        self.l0a_p = Channel(MemLoc.L0A, (tile_cube_m, tile_n), dtype, depth=2)
        self.l0b = Channel(MemLoc.L0B, (TILE, TILE), dtype, depth=2)
        self.l0c_qk = Channel(MemLoc.L0C, (tile_cube_m, TILE), dtypes.float32, depth=2)
        self.l0c = Channel(MemLoc.L0C, (tile_cube_m, TILE), dtypes.float32, depth=2)

    @jit
    def load_q_wide(self, q_tile_gm, m_seq: Int64, rows=None):
        # Two independent half-slots, each reused only after its final QK read.
        for half in range_constexpr(2):
            mem_copy(self.q_ring.produce(),
                     tile_slice(q_tile_gm, (self.tile_cube_m, D // 2), (0, half)),
                     engine=self.nd2nz)

    def _kv_chunk(self, kv_slot, chunk):
        return tile_slice(kv_slot, (self.tile_n, TILE), (0, chunk))

    def bmm1_fanout_q(self, kv_slot, q_half0, q_half1, actual_n=None):
        # The physical NZ frame stays full width. Softmax masks columns
        # beyond actual_n, so narrowing this load must not repack the frame.
        acc = self.l0c_qk.produce()
        for chunk in range_constexpr(D_TILES):
            q_half = q_half0 if chunk < 2 else q_half1
            a = self.l0a.produce()
            b = self.l0b.produce()
            mem_copy(a, tile_slice(q_half, (self.tile_cube_m, TILE), (0, chunk % 2)))
            mem_copy(b, self._kv_chunk(kv_slot, chunk))
            matmul(acc, a, b, init=(chunk == 0))

    def store_s(self, qk_ub_ch):
        mem_copy(qk_ub_ch.produce(), self.l0c_qk.consume(), engine=self.fixpipe)

    def compute_pv_fanout(self, p_l1_ch, pv_ub_ch, kv_slot, actual_n=None):
        rows = actual_n if actual_n is not None else self.tile_n
        p = p_l1_ch.consume()
        out = pv_ub_ch.produce()
        a = self.l0a_p.produce()
        mem_copy(a, p)
        a_dyn = reinterpret(a, shape=(self.tile_cube_m, rows))
        for chunk in range_constexpr(D_TILES):
            b = self.l0b.produce()
            mem_copy(b, self._kv_chunk(kv_slot, chunk), transpose=True)
            acc = self.l0c.produce()
            matmul(acc, a_dyn, reinterpret(b, shape=(rows, TILE)), init=True)
            mem_copy(tile_slice(out, (self.tile_vec_m, TILE), (0, chunk)),
                     acc, engine=self.fixpipe)
        # kv_slot remains the same alias used by QK in the previous tick.
        # Automatic synchronization protects reuse after this last L1 read.


class MqsmlaVector:

    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        preload_num=3,
        n_heads=0,
        sinks_span=0,
        use_sinks=True,
        ablate_no_softmax=False,
        ablate_no_rescale=False,
        ablate_no_pv_acc=False,
        ablate_no_out_cast=False,
        runtime_n1=False,
        rt_split=False,
        dtype=dtypes.bfloat16,
    ):
        self.dtype = dtype
        self.ablate_no_softmax = bool(ablate_no_softmax)
        self.ablate_no_rescale = bool(ablate_no_rescale)
        self.ablate_no_pv_acc = bool(ablate_no_pv_acc)
        self.ablate_no_out_cast = bool(ablate_no_out_cast)
        self.rt_split = bool(rt_split)
        self.n_heads = n_heads
        self.use_sinks = bool(use_sinks) and n_heads > 0
        self.sinks_span = sinks_span or n_heads
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx
        assert (
            tile_d % (2048 // 32) == 0
        ), f"tile_d({tile_d}) 必须被 b32 的 VL(64) 整除；不整除时 O 路径的列循环最后一块是残块，不能复用 full_mask，需退回逐块 `rr.update_mask(tile_d - col)`。"
        self.sm_max_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_sum_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_exp_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        # Retain the output completion protocol while migrating slot ownership.
        self.out_mte3_event = 0
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        self.res_o_b16 = self.res_o.reinterpret(self.dtype, shape=(tile_vec_m, tile_d))
        p_n1_pad = 32 // 2
        self.p_ub = Channel(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=self.dtype,
            depth=2,
            data_format="nz",
            n1_pad=p_n1_pad,
        )
        if n_heads and self.use_sinks:
            self.sinks_ub = Channel(
                MemLoc.UB, shape=(self.sinks_span,), dtype=dtypes.float32, depth=1
            )

    def _nz_params(self):
        s = self.p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = (s[0], s[1], s[2], s[3])
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return (m0, s_m1, s_m0, s_n1 // n0)

    def _softmax_fold_row(
        self,
        qk_ch,
        p_slot,
        base,
        nz_off,
        max_brc_buf,
        row,
        ve_mask,
        vo_mask,
        b16,
        b16_full,
        full,
        block_stride,
    ):
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, self.dtype, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, self.dtype, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        if const_expr(self.rt_split):
            zero16 = rr.vdups(0.0, self.dtype, mask=b16_full)
            invalid16 = rr.mask_xor(b16_full, b16, exec_mask=b16_full)
            merged = rr.vselect(zero16, merged, cond_mask=invalid16)
        rr.vstore_strided(
            p_slot,
            nz_off,
            merged,
            b16_full,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return (ve, vo)

    def _pass_a_row(
        self,
        qk_ch,
        scale,
        sm_max_dst,
        row,
        row_stride,
        VL_T,
        half0_mask,
        half1_mask,
        full_mask,
        half_only=False,
    ):
        base = row * row_stride
        if const_expr(half_only):
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rmax = rr.vreduce_max(v0, mask=full_mask)
        else:
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            v1 = rr.vmuls(rr.vload(qk_ch, base + VL_T), scale, mask=half1_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rr.vstore(qk_ch, base + VL_T, v1, half1_mask)
            if const_expr(self.rt_split):
                r0 = rr.vreduce_max(v0, mask=half0_mask)
                r1 = rr.vreduce_max(v1, mask=half1_mask)
                rmax = rr.vmax(r0, r1, mask=full_mask)
            else:
                rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        rr.vstore_first(sm_max_dst, row, rmax)

    @staticmethod
    def _half_only(actual_n, VL_T):
        return isinstance(actual_n, int) and actual_n <= VL_T

    @jit
    def softmax_first(self, qk_ch, scale, m_axis_triple: Int64, actual_n):
        p_slot = self.p_ub.produce()
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        VL_T = 2048 // 32
        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        half_only = self._half_only(actual_n, VL_T)
        with vf(mode="raw"):
            rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(
                        _clamp_nonneg(actual_n - VL_T), elem_bits=32
                    )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    sm_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                    half_only=half_only,
                )
            rr.vmem_bar("vst_vld")
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
                vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
                b16, _ = rr.update_mask(actual_n, elem_bits=16)
                base = row * src_row_stride
                nz_off = row // m0 * s_m1 + row % m0 * s_m0
                ve, vo = self._softmax_fold_row(
                    qk_ch,
                    p_slot,
                    base,
                    nz_off,
                    sm_max,
                    row,
                    ve_mask,
                    vo_mask,
                    b16,
                    b16_full,
                    full,
                    block_stride,
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
                rr.vstore_first(sm_sum, row, rsum)
            rr.vmem_bar("vst_vld")

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)

    @jit
    def softmax_rest(
        self, qk_ch, scale, m_axis_triple: Int64, tile_triple: Int64, actual_n
    ):
        p_slot = self.p_ub.produce()
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]
        VL_T = 2048 // 32
        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        half_only = self._half_only(actual_n, VL_T)
        with vf(mode="raw"):
            rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(
                        _clamp_nonneg(actual_n - VL_T), elem_bits=32
                    )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                    half_only=half_only,
                )
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask
            )
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
                vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
                b16, _ = rr.update_mask(actual_n, elem_bits=16)
                base = row * src_row_stride
                nz_off = row // m0 * s_m1 + row % m0 * s_m0
                ve, vo = self._softmax_fold_row(
                    qk_ch,
                    p_slot,
                    base,
                    nz_off,
                    self.tmp_new_max,
                    row,
                    ve_mask,
                    vo_mask,
                    b16,
                    b16_full,
                    full,
                    block_stride,
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
                rr.vstore_first(self.tmp_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def seed_from_sinks(self, sinks_view, m_axis_triple: Int64, sink_base=None):
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        span = self.sinks_span
        with vf(mode="raw"):
            full, _ = rr.update_mask(2048 // 32, elem_bits=32)
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            if const_expr(sink_base is None):
                for row in tuple(range(self.tile_vec_m)):
                    rr.vstore_first(
                        sm_max, row, rr.vload_broadcast(sinks_view, row % span)
                    )
                    rr.vstore_first(sm_sum, row, one)
            else:
                span_rt = self._g_rt
                for row in tuple(range(self.tile_vec_m)):
                    idx = _rt_min(sink_base + row, span_rt - 1)
                    rr.vstore_first(sm_max, row, rr.vload_broadcast(sinks_view, idx))
                    rr.vstore_first(sm_sum, row, one)
            rr.vmem_bar("vst_vld")

    @jit
    def softmax_stub(self, qk_ch, m_axis_triple: Int64, actual_n):
        p_slot = self.p_ub.produce()
        m0, s_m1, s_m0, block_stride = self._nz_params()
        VL_T = 2048 // 32
        with vf(mode="raw"):
            for row in range(self.tile_vec_m):
                b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
                m32, _ = rr.update_mask(VL_T, elem_bits=32)
                base = row * qk_ch.stride[0]
                ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
                he = rr.vcast(ve, self.dtype, mask=m32, reg_layout=rr.RegLayout.ZERO)
                ho = rr.vcast(vo, self.dtype, mask=m32, reg_layout=rr.RegLayout.ONE)
                merged = rr.vbitwise_or(he, ho, mask=b16_full)
                nz_off = row // m0 * s_m1 + row % m0 * s_m0
                rr.vstore_strided(
                    p_slot,
                    nz_off,
                    merged,
                    b16_full,
                    block_stride=block_stride,
                    repeat_stride=0,
                )

    @jit
    def softmax_p_stub(self):
        p_slot = self.p_ub.produce()
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="raw"):
            b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
            zero = rr.vdups(0.0, self.dtype, mask=b16_full)
            for row in range(self.tile_vec_m):
                nz_off = row // m0 * s_m1 + row % m0 * s_m0
                rr.vstore_strided(
                    p_slot,
                    nz_off,
                    zero,
                    b16_full,
                    block_stride=block_stride,
                    repeat_stride=0,
                )

    def store_p(self, p_l1_ch):
        mem_copy(
            p_l1_ch.produce(),
            self.p_ub.consume(),
            engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx,
        )

    @jit
    def init_o(self, pv_ch):
        mem_copy(self.res_o, pv_ch)

    @jit
    def update_o(self, pv_ch, exp_idx: Int64):
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    if const_expr(self.ablate_no_pv_acc):
                        rr.vstore(self.res_o, off, cur, full)
                    else:
                        o = rr.vmadd(pre, exp_b, cur, mask=full)
                        rr.vstore(self.res_o, off, o, full)

    @jit
    def update_o_last(self, pv_ch, exp_idx: Int64, sum_idx: Int64):
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    if const_expr(self.ablate_no_pv_acc):
                        o = cur
                    else:
                        o = rr.vmadd(pre, exp_b, cur, mask=full)
                    if const_expr(not self.ablate_no_rescale):
                        o = rr.vdiv(o, sum_b, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def init_o_last(self, pv_ch, sum_idx: Int64):
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    cur = rr.vload(pv_ch, off)
                    if const_expr(self.ablate_no_rescale):
                        rr.vstore(self.res_o, off, cur, full)
                    else:
                        rr.vstore(self.res_o, off, rr.vdiv(cur, sum_b, mask=full), full)

    @jit
    def _finalize_div_vf(self, sum_idx: Int64, actual_vec_m):
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(actual_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    rr.vstore(
                        self.res_o,
                        off,
                        rr.vdiv(rr.vload(self.res_o, off), sum_b, mask=full),
                        full,
                    )

    @jit
    def _cast_o_bf16_vf(self):
        # Match the old tensor-cast lowering: short runtime loop, one contiguous
        # f32 load/cast and one packed bf16 store per 64 source elements.
        with vf(mode="raw"):
            for off in dsl_range(0, self.tile_vec_m * self.tile_d, 64):
                m32, _ = rr.update_mask(64, elem_bits=32)
                src = rr.vload(self.res_o, off)
                dst = rr.vcast(src, self.dtype, mask=m32, rounding=rr.RoundingMode.RN)
                rr.vstore_pack(self.res_o_b16, off, dst, m32, pack_mode="b32_to_b16")
            rr.vmem_bar("vst_vld")

    def finalize_o(self, o_tile_gm, sum_idx: Int64, div_done=False):
        actual_vec_m = cv_valid_extent(
            o_tile_gm,
            axis=0,
            alignment=1,
            part_id=self.subblock_idx,
        )
        if not div_done:
            self._finalize_div_vf(sum_idx, actual_vec_m)
        if const_expr(not self.ablate_no_out_cast):
            self._cast_o_bf16_vf()
        vec_sync_notify(PIPE.V, PIPE.MTE3, self.out_mte3_event)
        vec_sync_wait(PIPE.V, PIPE.MTE3, self.out_mte3_event)
        mem_copy(
            o_tile_gm,
            self.res_o_b16,
            engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx,
        )
        vec_sync_notify(PIPE.MTE3, PIPE.V, self.out_mte3_event)
        vec_sync_wait(PIPE.MTE3, PIPE.V, self.out_mte3_event)

    @jit
    def finalize_o_rt(self, out_gm, row0, sum_idx: Int64, div_done=False):
        if not div_done:
            self._finalize_div_vf(sum_idx, self.tile_vec_m)
        sub_off = self.subblock_idx * self.tile_vec_m
        rest = self._m_real_rt - sub_off
        rows = _rt_min(_rt_max(rest, 0), self.tile_vec_m)
        if rows > 0:
            if const_expr(not self.ablate_no_out_cast):
                self._cast_o_bf16_vf()
            vec_sync_notify(PIPE.V, PIPE.MTE3, self.out_mte3_event)
            vec_sync_wait(PIPE.V, PIPE.MTE3, self.out_mte3_event)
            dst = out_gm[row0 + sub_off : row0 + sub_off + rows, None]
            mem_copy(
                dst,
                reinterpret(
                    self.res_o_b16,
                    shape=(self.tile_vec_m, self.tile_d),
                    stride=(self.tile_d, 1),
                ),
            )
            vec_sync_notify(PIPE.MTE3, PIPE.V, self.out_mte3_event)
            vec_sync_wait(PIPE.MTE3, PIPE.V, self.out_mte3_event)


# SMLA adaptations: non-quantized PA DMA, sparse gather, original FA metadata.
import functools
import torch
from cannbotdsl import Dim, TensorSpec


class SmlaMatmul(MqsmlaMatmul):
    @jit
    def load_kv(
        self,
        workspace: Tensor,
        ori: Tensor,
        block_table: Tensor,
        batch: Int64,
        start: Int64,
        rows: Int64,
        page_stride: Int64,
        slot: Int64,
        is_cmp: Int64,
        page_size: Int64,
    ):
        kv_slot = self.kv_ring.produce()
        if is_cmp != 0:
            mem_copy(
                kv_slot,
                workspace,
                engine=self.nd2nz,
            )
            # Both vector writers may reuse this slot only after MTE2 has
            # finished reading it into this Cube's private L1 ring.
            cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + slot)
            cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + 16 + slot)
        else:
            # Assemble the page-table-selected GM segments directly in L1.
            copied = 0
            while copied < rows:
                token = start + copied
                page_rows = 0
                physical_row = 0
                if page_size > 0:
                    page = Int64(block_table[batch, token // page_size])
                    in_page = token % page_size
                    page_rows = min(rows - copied, page_size - in_page)
                    physical_row = page * page_stride + in_page
                else:
                    prefix = 0
                    if page_size == 0:
                        prefix = Int64(block_table[0, batch])
                    else:
                        prefix = batch * (-page_size)
                    page_rows = rows - copied
                    physical_row = prefix + token
                # Copy the exact page tail into the shared 128-row NZ frame.
                # Each NZ row is one 32-byte DataBlock. The Tensor storage
                # layout pins column pitch independently of segment height.
                src = ori[physical_row : physical_row + page_rows, None]
                if copied == 0:
                    # A tile window preserves the parent's 128-row C0 pitch.
                    mem_copy(tile_slice(kv_slot, (128, 512), (0, 0)),
                             src, engine=self.nd2nz)
                else:
                    # Treat C0 columns as a regular batch of 16-wide matrices.
                    # One ND2NZ instruction writes an arbitrarily positioned
                    # page segment while retaining the 128-row matrix stride.
                    stripes = kv_slot.reinterpret(shape=(D // 16, 128, 16),
                                                   data_format="nz")
                    dst = reinterpret(stripes, shape=(D // 16, page_rows, 16),
                                      stride=(128 * 16, 16, 1), offset=copied * 32)
                    src_stripes = permute(src.view(page_rows, D // 16, 16), (1, 0, 2))
                    mem_copy(dst, src_stripes, engine=self.nd2nz)
                copied = copied + page_rows


class SmlaKVPhyAddr:
    """MQSMLA physical-row table, adapted to non-quantized PA page stride."""

    def __init__(self):
        self.block_table = Buffer(MemLoc.UB, (1, 2048), dtypes.int32)
        self.sparse_indices = Buffer(MemLoc.UB, (1, 1024), dtypes.int32)
        self.physical_rows = Buffer(MemLoc.UB, (1, 2048), dtypes.int32)

    @jit
    def compute_vf(
        self,
        page_stride: Int64,
        sparse_count: Int64,
        shift_bits,
        table_start: Int64,
        table_count: Int64,
    ):
        with vf(mode="raw"):
            zero = rr.vdups(0, dtypes.int32)
            for index in range(0, sparse_count, 64):
                mask, _ = rr.update_mask(min(64, sparse_count - index), elem_bits=32)
                sparse = rr.vmaxs(rr.vload(self.sparse_indices, index), 0, mask=mask)
                block = rr.vshr(sparse, shift_bits, mask=mask)
                offset = rr.vsub(
                    sparse, rr.vmuls(block, 1 << shift_bits, mask=mask), mask=mask
                )
                local_block = rr.vadds(block, -table_start, mask=mask)
                in_window = rr.vges(local_block, 0, mask=mask)
                in_window = rr.vlts(local_block, table_count, mask=in_window)
                safe_block = rr.vmaxs(
                    rr.vmins(local_block, table_count - 1, mask=mask),
                    0,
                    mask=mask,
                )
                physical = rr.vgather(
                    self.block_table,
                    rr.vreinterpret(safe_block, dtypes.uint32),
                    mask=mask,
                )
                row = rr.vadd(
                    rr.vmuls(physical, page_stride, mask=mask), offset, mask=mask
                )
                previous = zero
                if table_start != 0:
                    previous, unused = rr.vload_deinterleave(
                        self.physical_rows, index * 2
                    )
                row = rr.vselect(row, previous, cond_mask=in_window)
                rr.vstore_interleave(self.physical_rows, index * 2, row, zero)

    @jit
    def compute_linear_vf(self, prefix: Int64, sparse_count: Int64):
        with vf(mode="raw"):
            zero = rr.vdups(0, dtypes.int32)
            for index in range(0, sparse_count, 64):
                mask, _ = rr.update_mask(min(64, sparse_count - index), elem_bits=32)
                sparse = rr.vmaxs(rr.vload(self.sparse_indices, index), 0, mask=mask)
                row = rr.vadds(sparse, prefix, mask=mask)
                rr.vstore_interleave(self.physical_rows, index * 2, row, zero)

    @jit
    def compute_vectorized(
        self,
        out: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        count = indices.shape[1]
        for start in range(0, count, 1024):
            rows = min(1024, count - start)
            indices_tile = make_tiler((1, rows), alignment=(1, 1))
            mem_copy(
                reinterpret(self.sparse_indices, shape=(1, rows), stride=(1024, 1)),
                indices[query, start : start + rows],
            )
            if page_size > 0:
                for table_start in range(0, table.shape[1], 2048):
                    width = min(2048, table.shape[1] - table_start)
                    table_tile = make_tiler((1, width), alignment=(1, 1))
                    mem_copy(
                        reinterpret(
                            self.block_table, shape=(1, width), stride=(2048, 1)
                        ),
                        table[batch, table_start : table_start + width],
                    )
                    for shift_bits in range_constexpr(11):
                        if page_size == 1 << shift_bits:
                            self.compute_vf(
                                page_stride, rows, shift_bits, table_start, width
                            )
            else:
                prefix = 0
                if page_size == 0:
                    prefix = Int64(table[0, batch])
                else:
                    prefix = batch * (-page_size)
                self.compute_linear_vf(prefix, rows)
            result_tile = make_tiler((1, rows * 2), alignment=(1, 1))
            mem_copy(
                out[query, start * 2 : start * 2 + rows * 2],
                reinterpret(self.physical_rows, shape=(1, rows * 2), stride=(2048, 1)),
            )

    @jit
    def compute_enabled(
        self,
        out: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        page_stride: Int64,
        page_size: Int64,
        vectorize: Int64,
    ):
        if vectorize != 0:
            self.compute_vectorized(
                out, indices, table, batch, query, page_stride, page_size
            )
        else:
            self.compute_general(
                out, indices, table, batch, query, page_stride, page_size
            )

    @jit
    def store_physical_row(self, row: Int64, index: Int64):
        with vf(mode="raw"):
            mask, _ = rr.update_mask(1, elem_bits=32)
            low = rr.vdups(Int32(row), dtypes.int32)
            high = rr.vdups(0, dtypes.int32)
            rr.vscatter(
                self.physical_rows,
                low,
                rr.vreinterpret(
                    rr.vdups(Int32(index * 2), dtypes.int32), dtypes.uint32
                ),
                mask=mask,
            )
            rr.vscatter(
                self.physical_rows,
                high,
                rr.vreinterpret(
                    rr.vdups(Int32(index * 2 + 1), dtypes.int32),
                    dtypes.uint32,
                ),
                mask=mask,
            )

    @jit
    def compute_general(
        self,
        out: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        sparse_count = indices.shape[1]
        for start in range(0, sparse_count, 1024):
            rows = min(1024, sparse_count - start)
            for index in range(rows):
                sparse = max(0, Int64(indices[query, start + index]))
                physical = 0
                if page_size > 0:
                    physical = (
                        Int64(table[batch, sparse // page_size]) * page_stride
                        + sparse % page_size
                    )
                else:
                    prefix = 0
                    if page_size == 0:
                        prefix = Int64(table[0, batch])
                    else:
                        prefix = batch * (-page_size)
                    physical = prefix + sparse
                self.store_physical_row(physical, index)
            bounded = make_tiler((1, rows * 2), alignment=(1, 1))
            mem_copy(
                out[query, start * 2 : start * 2 + rows * 2],
                reinterpret(self.physical_rows, shape=(1, rows * 2), stride=(2048, 1)),
            )

    @jit
    def compute_page128(
        self,
        out: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        page_stride: Int64,
    ):
        width = table.shape[1]
        count = indices.shape[1]
        table_tile = make_tiler((1, width), alignment=(1, 1))
        mem_copy(
            reinterpret(self.block_table, shape=(1, width), stride=(64, 1)),
            tile_slice(table, table_tile, (batch, 0)),
        )
        indices_tile = make_tiler((1, count), alignment=(1, 1))
        mem_copy(
            reinterpret(self.sparse_indices, shape=(1, count), stride=(1024, 1)),
            tile_slice(indices, indices_tile, (query, 0)),
        )
        self.compute_vf(page_stride, count, 7, 0, width)
        result_tile = make_tiler((1, count * 2), alignment=(1, 1))
        mem_copy(
            tile_slice(out, result_tile, (query, 0)),
            reinterpret(self.physical_rows, shape=(1, count * 2), stride=(2048, 1)),
        )

    @jit
    def compute(
        self,
        out: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        if page_size == 128 and indices.shape[1] <= 1024 and table.shape[1] <= 64:
            self.compute_page128(out, indices, table, batch, query, page_stride)
        else:
            self.compute_general(
                out, indices, table, batch, query, page_stride, page_size
            )


def copy_sparse_pair_gm_to_ub(dst, src, row, physical0, physical1):
    # Preserve the original pair order and two-copy behavior for this Channel
    # migration; sparse DMA merging can be measured as a separate change.
    mem_copy(
        tile_slice(dst, (1, 512), (row, 0)), tile_slice(src, (1, 512), (physical0, 0))
    )
    mem_copy(
        tile_slice(dst, (1, 512), (row + 1, 0)),
        tile_slice(src, (1, 512), (physical1, 0)),
    )


class SmlaVec0:
    def __init__(self, n1, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.stage0InBuf0 = Buffer(MemLoc.UB, (16, 512), dtype)
        self.stage0InBuf1 = Buffer(MemLoc.UB, (16, 512), dtype)
        self.ready_flag = _WORKSPACE_READY_FLAG

    @jit
    def initialize(self):
        # 空闲令牌：初始UB未被MTE3占用，可以首次搬入。
        vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
        vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)
        for slot in range_constexpr(_KV_RING):
            cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + slot)
            cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + 16 + slot)

    @jit
    def finish(self):
        vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
        vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
        for slot in range_constexpr(_KV_RING):
            vec_sync_intra_wait(PIPE.MTE3, _WORKSPACE_FREE_FLAG + slot)

    @jit
    def _load_full16(
        self, dst, kv, physical, query: Int64, start: Int64, sub_row: Int64
    ):
        for row in range_constexpr(0, 16, 2):
            physical0 = Int64(physical[query, (start + sub_row + row) * 2])
            physical1 = Int64(physical[query, (start + sub_row + row + 1) * 2])
            distance = max(physical0, physical1) - min(physical0, physical1)
            if distance > 0 and distance < 2097152:
                copy_sparse_pair_gm_to_ub(dst, kv, row, physical0, physical1)
            else:
                mem_copy(
                    tile_slice(dst, (1, 512), (row, 0)),
                    tile_slice(kv, (1, 512), (physical0, 0)),
                )
                mem_copy(
                    tile_slice(dst, (1, 512), (row + 1, 0)),
                    tile_slice(kv, (1, 512), (physical1, 0)),
                )

    @jit
    def _load_tail(
        self,
        dst,
        kv,
        physical,
        query: Int64,
        start: Int64,
        sub_row: Int64,
        tail_rows: Int64,
    ):
        # Exact tail: paired valid rows keep the merged DMA; an odd final row
        # uses one row copy. Invalid sparse indices are never read.
        for row in range_constexpr(0, 16, 2):
            if row + 1 < tail_rows:
                physical0 = Int64(physical[query, (start + sub_row + row) * 2])
                physical1 = Int64(physical[query, (start + sub_row + row + 1) * 2])
                distance = max(physical0, physical1) - min(physical0, physical1)
                if distance > 0 and distance < 2097152:
                    copy_sparse_pair_gm_to_ub(dst, kv, row, physical0, physical1)
                else:
                    mem_copy(
                        tile_slice(dst, (1, 512), (row, 0)),
                        tile_slice(kv, (1, 512), (physical0, 0)),
                    )
                    mem_copy(
                        tile_slice(dst, (1, 512), (row + 1, 0)),
                        tile_slice(kv, (1, 512), (physical1, 0)),
                    )
            elif row < tail_rows:
                physical0 = Int64(physical[query, (start + sub_row + row) * 2])
                mem_copy(
                    tile_slice(dst, (1, 512), (row, 0)),
                    tile_slice(kv, (1, 512), (physical0, 0)),
                )

    @jit
    def gather(
        self,
        workspace: Tensor,
        kv: Tensor,
        physical: Tensor,
        query: Int64,
        start: Int64,
        rows: Int64,
        slot: Int64,
    ):
        vec_sync_intra_wait(PIPE.MTE3, _WORKSPACE_FREE_FLAG + slot)
        lane = get_subblock_id()
        rows_per_vec = 64
        lane_row = lane * rows_per_vec
        lane_rows = min(max(rows - lane_row, 0), rows_per_vec)
        full_chunks = lane_rows // 16
        tail_rows = lane_rows - full_chunks * 16
        # Full chunks and the one possible tail are separate control paths.
        for chunk in range_constexpr(rows_per_vec // 16):
            if chunk < full_chunks:
                dst = self.stage0InBuf0 if chunk % 2 == 0 else self.stage0InBuf1
                event = 2 + chunk % 2
                vec_sync_wait(PIPE.MTE3, PIPE.MTE2, event)
                sub_row = lane_row + chunk * 16
                self._load_full16(dst, kv, physical, query, start, sub_row)
                vec_sync_notify(PIPE.MTE2, PIPE.MTE3, event)
                vec_sync_wait(PIPE.MTE2, PIPE.MTE3, event)
                mem_copy(tile_slice(workspace, (16, 512), (sub_row // 16, 0)), dst)
                vec_sync_notify(PIPE.MTE3, PIPE.MTE2, event)
        if tail_rows > 0:
            tail_sub_row = lane_row + full_chunks * 16
            if full_chunks % 2 == 0:
                vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
                self._load_tail(
                    self.stage0InBuf0,
                    kv,
                    physical,
                    query,
                    start,
                    tail_sub_row,
                    tail_rows,
                )
                vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 2)
                vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 2)
                dst_tile = workspace[tail_sub_row : tail_sub_row + tail_rows, None]
                mem_copy(
                    dst_tile,
                    reinterpret(
                        self.stage0InBuf0,
                        shape=(tail_rows, 512),
                        stride=(512, 1),
                    ),
                )
                vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
            else:
                vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
                self._load_tail(
                    self.stage0InBuf1,
                    kv,
                    physical,
                    query,
                    start,
                    tail_sub_row,
                    tail_rows,
                )
                vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 3)
                vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 3)
                dst_tile = workspace[tail_sub_row : tail_sub_row + tail_rows, None]
                mem_copy(
                    dst_tile,
                    reinterpret(
                        self.stage0InBuf1,
                        shape=(tail_rows, 512),
                        stride=(512, 1),
                    ),
                )
                vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)
        vec_sync_intra_arrive(PIPE.MTE3, self.ready_flag)

    @jit
    def wait(self):
        cube_sync_intra_wait(PIPE.MTE2, self.ready_flag)
        cube_sync_intra_wait(PIPE.MTE2, self.ready_flag + 16)


@jit
def smla_actual_kv_length(
    lengths: Tensor, batch: Int64, length_mode: Int64, page_size: Int64
):
    length = 0
    if length_mode == 0:
        length = Int64(lengths[batch])
    elif length_mode == 1:
        length = Int64(lengths[batch + 1]) - Int64(lengths[batch])
    else:
        length = -page_size
    return length


class SmlaZeroOutput:
    """Initialize only unused or empty query rows, without an auxiliary kernel."""

    def __init__(self, n1, dtype, mode):
        self.n1 = n1
        self.dtype = dtype
        self.mode = mode
        self.zero_row = Buffer(MemLoc.UB, (1, 512), dtype)

    @jit
    def write(self, out: Tensor, query: Int64):
        with vf(mode="raw"):
            mask, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, self.dtype)
            for col in range(0, 512, 128):
                rr.vstore(self.zero_row, col, zero, mask)
        for head in range(self.n1):
            mem_copy(
                tile_slice(out, (1, 512), (query * self.n1 + head, 0)),
                self.zero_row,
            )

    @jit
    def initialize(
        self,
        out: Tensor,
        cu: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        q_lengths: Tensor,
        topk: Tensor,
        indices: Tensor,
        batch_count: Int64,
        q_size: Int64,
        has_q_length: Int64,
        has_topk: Int64,
        ori_mask: Int64,
        cmp_mask: Int64,
        left: Int64,
        right: Int64,
        ratio: Int64,
        block_dim: Int64,
        ori_length_mode: Int64,
        cmp_length_mode: Int64,
        ori_page: Int64,
        cmp_page: Int64,
    ):
        lane = get_block_idx() * 2 + get_subblock_id()
        for query in range(lane, out.shape[0] // self.n1, block_dim * 2):
            for batch in range(batch_count):
                prefix = 0
                storage_length = 0
                if q_size == 0:
                    prefix = Int64(cu[batch])
                    storage_length = Int64(cu[batch + 1]) - prefix
                else:
                    prefix = batch * q_size
                    storage_length = q_size
                if query >= prefix and query < prefix + storage_length:
                    s1 = storage_length
                    if has_q_length != 0:
                        s1 = Int64(q_lengths[batch])
                    position = query - prefix
                    ori_len = smla_actual_kv_length(
                        ori_lengths, batch, ori_length_mode, ori_page
                    )
                    begin = 0
                    end = ori_len
                    if ori_mask == 3:
                        end = max(0, min(ori_len, ori_len - s1 + position + 1))
                    if ori_mask == 4:
                        if left != -1:
                            begin = max(0, min(ori_len, ori_len - s1 + position - left))
                        if right != -1:
                            end = max(
                                0,
                                min(ori_len, ori_len - s1 + position + right + 1),
                            )
                    cmp_valid = 0
                    if const_expr(self.mode != "SWA"):
                        cmp_len = smla_actual_kv_length(
                            cmp_lengths, batch, cmp_length_mode, cmp_page
                        )
                        cmp_valid = cmp_len
                        if cmp_mask == 3:
                            cmp_valid = min(
                                cmp_len,
                                max(
                                    0,
                                    (
                                        cmp_len * ratio
                                        + dyn_select(
                                            ratio == 1,
                                            0,
                                            Int64(residual[batch]),
                                        )
                                        - s1
                                        + position
                                        + 1
                                    )
                                    // ratio,
                                ),
                            )
                        if const_expr(self.mode == "CSA"):
                            if has_topk != 0:
                                cmp_valid = min(cmp_valid, Int64(topk[query]))
                            if Int32(indices[query, 0]) < 0:
                                cmp_valid = 0
                    if position >= s1 or (begin >= end and cmp_valid == 0):
                        self.write(out, query)


@kernel
class SparseFlashMlaCsaKernel:
    def __init__(self, n1, block_dim, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.zero_output = SmlaZeroOutput(n1, dtype, "CSA")
        self.qk_ub = Channel(
            MemLoc.UB, (32, 128), dtypes.float32, depth=2, kind=ChannelKind.CrossCore
        )
        self.pv_ub = Channel(
            MemLoc.UB, (32, 512), dtypes.float32, depth=1, kind=ChannelKind.CrossCore
        )
        self.p_l1 = Channel(
            MemLoc.L1, (64, 128), dtype, depth=2, kind=ChannelKind.CrossCore, addr=0
        )
        self.matmul = SmlaMatmul(64, 32, 128, dtype)
        self.vector = MqsmlaVector(
            32,
            128,
            512,
            get_subblock_id(),
            n_heads=n1,
            sinks_span=32,
            use_sinks=True,
            rt_split=True,
            dtype=dtype,
        )
        self.phy = SmlaKVPhyAddr()
        self.vec0 = SmlaVec0(n1, dtype)
        self.run_info = DelayLineGroup(
            4, "batch", "query", "m", "n", "last", "start", "rows", "cmp"
        )

    @jit
    def stage_softmax(
        self,
        tick: Int64,
        m_seq: Int64,
        n: Int64,
        rows: Int64,
        sinks: Tensor,
        scale: dtypes.float32,
        g_half: Int64,
    ):
        # Reuse MQSMLA's query-axis max/sum and task-axis exp banking.
        m_bank = m_seq % 2
        exp_bank = tick % 2
        slot = self.qk_ub.consume()
        qk = reinterpret(slot, shape=(32, 128), stride=(128, 1))
        if n == 0:
            dst = self.vector.sinks_ub.produce()
            mem_copy(dst, tile_slice(sinks, (32,), (g_half * 2 + get_subblock_id(),)))
            src = self.vector.sinks_ub.consume()
            self.vector.seed_from_sinks(src, m_bank)
        self.vector.softmax_rest(qk, scale, m_bank, exp_bank, rows)
        self.vector.store_p(self.p_l1)

    @jit
    def stage_update(
        self,
        tick: Int64,
        m_seq: Int64,
        n: Int64,
        last: Int64,
        query: Int64,
        out: Tensor,
        g_half: Int64,
    ):
        pv = self.pv_ub.consume()
        sum_idx = m_seq % 2
        exp_idx = tick % 2
        if n == 0:
            if last != 0:
                self.vector.init_o_last(pv, sum_idx)
            else:
                self.vector.init_o(pv)
        else:
            if last != 0:
                self.vector.update_o_last(pv, exp_idx, sum_idx)
            else:
                self.vector.update_o(pv, exp_idx)
        if last != 0:
            tile = tile_slice(out, (64, 512), (query * self.n1 // 64 + g_half, 0))
            self.vector.finalize_o(tile, sum_idx, div_done=True)

    @jit
    def pipeline(
        self,
        tick: Int64,
        count: Int64,
        out: Tensor,
        q: Tensor,
        ws: Tensor,
        ori: Tensor,
        table: Tensor,
        sinks: Tensor,
        scale: dtypes.float32,
        page_stride: Int64,
        ori_page: Int64,
        group: Int64,
        g_half: Int64,
        kv_qk: Tensor, kv_pv: Tensor, q_half0: Tensor, q_half1: Tensor,
    ):
        next_pv = kv_pv
        if tick >= 1 and tick < count + 1:
            info = self.run_info.tap(1)
            load_tick = tick - 1
            if load_tick == 0:
                self.matmul.load_q_wide(
                    tile_slice(q, (64, 512), (info.query * self.n1 // 64 + g_half, 0)),
                    info.m,
                )
            if info.cmp != 0:
                self.vec0.wait()
            self.matmul.load_kv(
                tile_slice(ws, (128, 512), (group * 3 + load_tick % 3, 0)),
                ori,
                table,
                info.batch,
                info.start,
                info.rows,
                page_stride,
                load_tick % 3,
                info.cmp,
                ori_page,
            )
        if tick >= 2 and tick < count + 2:
            info = self.run_info.tap(2)
            qk_tick = tick - 2
            self.matmul.bmm1_fanout_q(kv_qk, q_half0, q_half1, actual_n=info.rows)
            next_pv = kv_qk
            kv_qk = self.matmul.kv_ring.consume()
            if info.last != 0:
                q_half0 = self.matmul.q_ring.consume()
                q_half1 = self.matmul.q_ring.consume()
            if info.last != 0 and tick < count + 1:
                next_info = self.run_info.tap(1)
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (64, 512),
                        (next_info.query * self.n1 // 64 + g_half, 0),
                    ),
                    next_info.m,
                )
            self.matmul.store_s(self.qk_ub)
            self.stage_softmax(qk_tick, info.m, info.n, info.rows, sinks, scale, g_half)
        if tick >= 3 and tick < count + 3:
            info = self.run_info.tap(3)
            pv_tick = tick - 3
            self.matmul.compute_pv_fanout(
                self.p_l1, self.pv_ub, kv_pv, actual_n=info.rows
            )
            self.stage_update(
                pv_tick, info.m, info.n, info.last, info.query, out, g_half
            )
        self.run_info.advance()
        return kv_qk, next_pv, q_half0, q_half1

    def __call__(
        self,
        out: Tensor,
        q: Tensor,
        ws: Tensor,
        ori: Tensor,
        cmp: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        indices: Tensor,
        physical: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        cu_q: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        scale: dtypes.float32,
        ratio: dtypes.int64,
        left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        kv_qk = self.matmul.kv_ring.consume()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.consume()
        q_half1 = self.matmul.q_ring.consume()
        block = get_block_idx()
        base = block * 9
        # Head halves read the same KV values, but own independent workspace
        # slots so an intra-block handshake covers every writer and reader.
        group = block
        g_half = block % 2 if const_expr(self.n1 == 128) else 0
        self.vec0.initialize()
        # arch35 CalcVectorizeKvPhyAddrWorkspaceSize: CSA, power-of-two PA pages,
        # max block-table bytes + align128(K)*(int32 index + int64 address) <=184 KiB.
        aligned_sparse_count = ((indices.shape[1] + 127) // 128) * Int64(128)
        vectorize_ub_size = aligned_sparse_count * 12
        blocksize_flag = 1
        if cmp_page > 0:
            vectorize_ub_size = (
                vectorize_ub_size + max(ori_table.shape[1], cmp_table.shape[1]) * 4
            )
            ori_power = 0
            cmp_power = 0
            for shift_bits in range_constexpr(11):
                if ori_page == 1 << shift_bits:
                    ori_power = 1
                if cmp_page == 1 << shift_bits:
                    cmp_power = 1
            blocksize_flag = ori_power * cmp_power
        vectorize = 0
        if vectorize_ub_size <= 184 * 1024 and blocksize_flag != 0:
            vectorize = 1
        # Preliminary address pass; one writer per row, no auxiliary kernel launch.
        vec_idx = block * 2 + get_subblock_id()
        for query in range(vec_idx, indices.shape[0], self.block_dim * 2):
            for batch in range(batch_count):
                prefix = 0
                end = 0
                if q_size == 0:
                    prefix = Int64(cu_q[batch])
                    end = Int64(cu_q[batch + 1])
                else:
                    prefix = batch * q_size
                    end = prefix + q_size
                if query >= prefix and query < end:
                    self.phy.compute_enabled(
                        physical,
                        indices,
                        cmp_table,
                        batch,
                        query,
                        cmp_stride,
                        cmp_page,
                        vectorize,
                    )
        # Metadata may assign a query to a different block than the address
        # pass. Publish every physical-row entry before any gather reads it.
        global_sync_all(flag_ids=(13, 14, 15))
        self.zero_output.initialize(
            out,
            cu_q,
            ori_lengths,
            cmp_lengths,
            residual,
            q_lengths,
            topk_length,
            indices,
            batch_count,
            q_size,
            has_q_length,
            has_topk,
            ori_mask,
            cmp_mask,
            left,
            right,
            ratio,
            self.block_dim,
            ori_length_mode,
            cmp_length_mode,
            ori_page,
            cmp_page,
        )
        tick = 0
        m_seq = 0
        if Int32(metadata[base]) != 0:
            first_batch = Int64(metadata[base + 1])
            first_query = Int64(metadata[base + 2])
            last_batch = Int64(metadata[base + 4])
            last_query = Int64(metadata[base + 5])
            for batch in range(first_batch, min(last_batch + 1, batch_count)):
                prefix = 0
                s1 = 0
                if q_size == 0:
                    prefix = Int64(cu_q[batch])
                    s1 = Int64(cu_q[batch + 1]) - prefix
                else:
                    prefix = batch * q_size
                    s1 = q_size
                if has_q_length != 0:
                    s1 = Int64(q_lengths[batch])
                begin = dyn_select(batch == first_batch, first_query, 0)
                end = dyn_select(batch == last_batch, last_query, s1)
                ori_len = smla_actual_kv_length(
                    ori_lengths, batch, ori_length_mode, ori_page
                )
                cmp_len = smla_actual_kv_length(
                    cmp_lengths, batch, cmp_length_mode, cmp_page
                )
                for s1_idx in range(begin, end):
                    query = prefix + s1_idx
                    ori_start = 0
                    ori_end = ori_len
                    if ori_mask == 3:
                        ori_end = max(0, min(ori_len, ori_len - s1 + s1_idx + 1))
                    if ori_mask == 4:
                        if left != -1:
                            ori_start = max(
                                0, min(ori_len, ori_len - s1 + s1_idx - left)
                            )
                        if right != -1:
                            ori_end = max(
                                0,
                                min(ori_len, ori_len - s1 + s1_idx + right + 1),
                            )
                    ori_valid = max(0, ori_end - ori_start)
                    cmp_limit = min(
                        cmp_len,
                        max(
                            0,
                            (
                                cmp_len * ratio
                                + dyn_select(ratio == 1, 0, Int64(residual[batch]))
                                - s1
                                + s1_idx
                                + 1
                            )
                            // ratio,
                        ),
                    )
                    if cmp_mask == 0:
                        cmp_limit = cmp_len
                    cmp_valid = min(indices.shape[1], cmp_limit)
                    if has_topk != 0:
                        cmp_valid = min(cmp_valid, Int64(topk_length[query]))
                    ori_tiles = (ori_valid + 127) // 128
                    cmp_tiles = (cmp_valid + 127) // 128
                    total = ori_tiles + cmp_tiles
                    for n in range(total):
                        is_cmp = n >= ori_tiles
                        tile_index = dyn_select(is_cmp, n - ori_tiles, n)
                        tile_start = tile_index * 128
                        start = dyn_select(is_cmp, tile_start, ori_start + tile_start)
                        valid = dyn_select(is_cmp, cmp_valid, ori_valid)
                        rows = _rt_min(128, valid - tile_start)
                        self.run_info.push(
                            batch=batch,
                            query=query,
                            m=m_seq,
                            n=n,
                            last=Int64(n == total - 1),
                            start=start,
                            rows=rows,
                            cmp=Int64(is_cmp),
                        )
                        if is_cmp:
                            self.vec0.gather(
                                tile_slice(
                                    ws,
                                    (128, 512),
                                    (group * 3 + tick % 3, 0),
                                ),
                                cmp,
                                physical,
                                query,
                                start,
                                rows,
                                tick % 3,
                            )
                        kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                            tick,
                            tick + 1,
                            out,
                            q,
                            ws,
                            ori,
                            ori_table,
                            sinks,
                            scale,
                            ori_stride,
                            ori_page,
                            group,
                            g_half,
                            kv_qk, kv_pv, q_half0, q_half1,
                        )
                        tick = tick + 1
                    m_seq = m_seq + 1
            for drain in range(3):
                kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                    tick + drain,
                    tick,
                    out,
                    q,
                    ws,
                    ori,
                    ori_table,
                    sinks,
                    scale,
                    ori_stride,
                    ori_page,
                    group,
                    g_half,
                    kv_qk, kv_pv, q_half0, q_half1,
                )
        self.vec0.finish()


@kernel
class SparseFlashMlaSwaKernel:
    """A5 SWA/HCA: direct PA Load(t), QK/Vec1(t-1), PV/Vec2(t-2)."""

    def __init__(self, n1, block_dim, mode, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.mode = mode
        self.zero_output = SmlaZeroOutput(n1, dtype, mode)
        self.qk_ub = Channel(
            MemLoc.UB, (32, 128), dtypes.float32, depth=2, kind=ChannelKind.CrossCore
        )
        self.pv_ub = Channel(
            MemLoc.UB, (32, 512), dtypes.float32, depth=1, kind=ChannelKind.CrossCore
        )
        self.p_l1 = Channel(
            MemLoc.L1, (64, 128), dtype, depth=2, kind=ChannelKind.CrossCore, addr=0
        )
        self.matmul = SmlaMatmul(64, 32, 128, dtype)
        self.vector = MqsmlaVector(
            32,
            128,
            512,
            get_subblock_id(),
            n_heads=n1,
            sinks_span=32,
            use_sinks=True,
            rt_split=True,
            dtype=dtype,
        )
        self.run_info = DelayLineGroup(
            3, "batch", "query", "m", "n", "last", "start", "rows", "cmp"
        )

    @jit
    def stage_softmax(
        self,
        tick: Int64,
        m_seq: Int64,
        n: Int64,
        rows: Int64,
        sinks: Tensor,
        scale: dtypes.float32,
        g_half: Int64,
    ):
        # Reuse MQSMLA's query-axis max/sum and task-axis exp banking.
        m_bank = m_seq % 2
        exp_bank = tick % 2
        slot = self.qk_ub.consume()
        qk = reinterpret(slot, shape=(32, 128), stride=(128, 1))
        if n == 0:
            dst = self.vector.sinks_ub.produce()
            mem_copy(dst, tile_slice(sinks, (32,), (g_half * 2 + get_subblock_id(),)))
            src = self.vector.sinks_ub.consume()
            self.vector.seed_from_sinks(src, m_bank)
        self.vector.softmax_rest(qk, scale, m_bank, exp_bank, rows)
        self.vector.store_p(self.p_l1)

    @jit
    def stage_update(
        self,
        tick: Int64,
        m_seq: Int64,
        n: Int64,
        last: Int64,
        query: Int64,
        out: Tensor,
        g_half: Int64,
    ):
        pv = self.pv_ub.consume()
        sum_idx = m_seq % 2
        exp_idx = tick % 2
        if n == 0:
            if last != 0:
                self.vector.init_o_last(pv, sum_idx)
            else:
                self.vector.init_o(pv)
        else:
            if last != 0:
                self.vector.update_o_last(pv, exp_idx, sum_idx)
            else:
                self.vector.update_o(pv, exp_idx)
        if last != 0:
            tile = tile_slice(out, (64, 512), (query * self.n1 // 64 + g_half, 0))
            self.vector.finalize_o(tile, sum_idx, div_done=True)

    @jit
    def pipeline(
        self,
        tick: Int64,
        count: Int64,
        out: Tensor,
        q: Tensor,
        ori: Tensor,
        cmp: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        sinks: Tensor,
        scale: dtypes.float32,
        ori_stride: Int64,
        cmp_stride: Int64,
        ori_page: Int64,
        cmp_page: Int64,
        g_half: Int64,
        current_batch: Int64,
        current_query: Int64,
        current_m: Int64,
        current_start: Int64,
        current_rows: Int64,
        current_cmp: Int64,
        kv_qk: Tensor, kv_pv: Tensor, q_half0: Tensor, q_half1: Tensor,
    ):
        next_pv = kv_pv
        if tick < count:
            if tick == 0:
                self.matmul.load_q_wide(
                    tile_slice(
                        q, (64, 512), (current_query * self.n1 // 64 + g_half, 0)
                    ),
                    current_m,
                )
            if current_cmp != 0:
                self.matmul.load_kv(
                    cmp,
                    cmp,
                    cmp_table,
                    current_batch,
                    current_start,
                    current_rows,
                    cmp_stride,
                    tick % 3,
                    0,
                    cmp_page,
                )
            else:
                self.matmul.load_kv(
                    ori,
                    ori,
                    ori_table,
                    current_batch,
                    current_start,
                    current_rows,
                    ori_stride,
                    tick % 3,
                    0,
                    ori_page,
                )
        if tick >= 1 and tick < count + 1:
            info = self.run_info.tap(1)
            qk_tick = tick - 1
            self.matmul.bmm1_fanout_q(kv_qk, q_half0, q_half1, actual_n=info.rows)
            next_pv = kv_qk
            kv_qk = self.matmul.kv_ring.consume()
            if info.last != 0:
                q_half0 = self.matmul.q_ring.consume()
                q_half1 = self.matmul.q_ring.consume()
            if info.last != 0 and tick < count:
                self.matmul.load_q_wide(
                    tile_slice(
                        q, (64, 512), (current_query * self.n1 // 64 + g_half, 0)
                    ),
                    current_m,
                )
            self.matmul.store_s(self.qk_ub)
            self.stage_softmax(qk_tick, info.m, info.n, info.rows, sinks, scale, g_half)
        if tick >= 2 and tick < count + 2:
            info = self.run_info.tap(2)
            pv_tick = tick - 2
            self.matmul.compute_pv_fanout(
                self.p_l1, self.pv_ub, kv_pv, actual_n=info.rows
            )
            self.stage_update(
                pv_tick, info.m, info.n, info.last, info.query, out, g_half
            )
        self.run_info.advance()
        return kv_qk, next_pv, q_half0, q_half1

    def __call__(
        self,
        out: Tensor,
        q: Tensor,
        ori: Tensor,
        cmp: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        cu_q: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        scale: dtypes.float32,
        ratio: dtypes.int64,
        left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        kv_qk = self.matmul.kv_ring.consume()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.consume()
        q_half1 = self.matmul.q_ring.consume()
        block = get_block_idx()
        base = block * 9
        g_half = block % 2 if const_expr(self.n1 == 128) else 0
        self.zero_output.initialize(
            out,
            cu_q,
            ori_lengths,
            cmp_lengths,
            residual,
            q_lengths,
            topk_length,
            ori_table,
            batch_count,
            q_size,
            has_q_length,
            has_topk,
            ori_mask,
            cmp_mask,
            left,
            right,
            ratio,
            self.block_dim,
            ori_length_mode,
            cmp_length_mode,
            ori_page,
            cmp_page,
        )
        tick = 0
        m_seq = 0
        if Int32(metadata[base]) != 0:
            first_batch = Int64(metadata[base + 1])
            first_query = Int64(metadata[base + 2])
            last_batch = Int64(metadata[base + 4])
            last_query = Int64(metadata[base + 5])
            for batch in range(first_batch, min(last_batch + 1, batch_count)):
                prefix = 0
                s1 = 0
                if q_size == 0:
                    prefix = Int64(cu_q[batch])
                    s1 = Int64(cu_q[batch + 1]) - prefix
                else:
                    prefix = batch * q_size
                    s1 = q_size
                if has_q_length != 0:
                    s1 = Int64(q_lengths[batch])
                begin = dyn_select(batch == first_batch, first_query, 0)
                end = dyn_select(batch == last_batch, last_query, s1)
                ori_len = smla_actual_kv_length(
                    ori_lengths, batch, ori_length_mode, ori_page
                )
                for s1_idx in range(begin, end):
                    query = prefix + s1_idx
                    ori_start = 0
                    ori_end = ori_len
                    if ori_mask == 3:
                        ori_end = max(0, min(ori_len, ori_len - s1 + s1_idx + 1))
                    if ori_mask == 4:
                        if left != -1:
                            ori_start = max(
                                0, min(ori_len, ori_len - s1 + s1_idx - left)
                            )
                        if right != -1:
                            ori_end = max(
                                0,
                                min(ori_len, ori_len - s1 + s1_idx + right + 1),
                            )
                    ori_valid = max(0, ori_end - ori_start)
                    cmp_valid = 0
                    if const_expr(self.mode == "HCA"):
                        cmp_len = smla_actual_kv_length(
                            cmp_lengths, batch, cmp_length_mode, cmp_page
                        )
                        cmp_valid = cmp_len
                        if cmp_mask == 3:
                            cmp_valid = min(
                                cmp_len,
                                max(
                                    0,
                                    (
                                        cmp_len * ratio
                                        + dyn_select(
                                            ratio == 1,
                                            0,
                                            Int64(residual[batch]),
                                        )
                                        - s1
                                        + s1_idx
                                        + 1
                                    )
                                    // ratio,
                                ),
                            )
                    ori_tiles = (ori_valid + 127) // 128
                    cmp_tiles = (cmp_valid + 127) // 128
                    total = ori_tiles + cmp_tiles
                    for n in range(total):
                        is_cmp = n >= ori_tiles
                        tile_index = dyn_select(is_cmp, n - ori_tiles, n)
                        tile_start = tile_index * 128
                        start = dyn_select(is_cmp, tile_start, ori_start + tile_start)
                        valid = dyn_select(is_cmp, cmp_valid, ori_valid)
                        rows = _rt_min(128, valid - tile_start)
                        self.run_info.push(
                            batch=batch,
                            query=query,
                            m=m_seq,
                            n=n,
                            last=Int64(n == total - 1),
                            start=start,
                            rows=rows,
                            cmp=Int64(is_cmp),
                        )
                        kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                            tick,
                            tick + 1,
                            out,
                            q,
                            ori,
                            cmp,
                            ori_table,
                            cmp_table,
                            sinks,
                            scale,
                            ori_stride,
                            cmp_stride,
                            ori_page,
                            cmp_page,
                            g_half,
                            batch,
                            query,
                            m_seq,
                            start,
                            rows,
                            Int64(is_cmp),
                            kv_qk, kv_pv, q_half0, q_half1,
                        )
                        tick = tick + 1
                    m_seq = m_seq + 1
            for drain in range(2):
                kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                    tick + drain,
                    tick,
                    out,
                    q,
                    ori,
                    cmp,
                    ori_table,
                    cmp_table,
                    sinks,
                    scale,
                    ori_stride,
                    cmp_stride,
                    ori_page,
                    cmp_page,
                    g_half,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    kv_qk, kv_pv, q_half0, q_half1,
                )


class SparseFlashMlaDirectLauncher:
    def __init__(self, n1, block_dim, mode, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.mode = mode

    @jit
    def run(
        self,
        out: Tensor,
        q: Tensor,
        ori: Tensor,
        cmp: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        cu_q: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        scale: dtypes.float32,
        ratio: dtypes.int64,
        left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        SparseFlashMlaSwaKernel(self.n1, self.block_dim, self.mode, self.dtype)[
            self.block_dim
        ](
            out,
            q,
            ori,
            cmp,
            ori_table,
            cmp_table,
            sinks,
            metadata,
            cu_q,
            ori_lengths,
            cmp_lengths,
            residual,
            scale,
            ratio,
            left,
            ori_stride,
            cmp_stride,
            ori_page,
            cmp_page,
            ori_mask,
            cmp_mask,
            right,
            q_size,
            q_lengths,
            topk_length,
            has_q_length,
            has_topk,
            batch_count,
            ori_length_mode,
            cmp_length_mode,
        )


@functools.lru_cache(maxsize=None)
def _compile_smla_direct(n1, block_dim, mode, dtype):
    token_idx = Dim("TN1")
    ori_capacity = Dim("PO")
    cmp_capacity = Dim("PC")
    ori_table_width = Dim("BO")
    cmp_table_width = Dim("BC")
    ori_table_rows = Dim("OB")
    cmp_table_rows = Dim("CB")
    qcu_count = Dim("CQ")
    qlen_count = Dim("LQ")
    topk_count = Dim("TK")
    i32 = dtypes.int32
    specs = (
        TensorSpec((token_idx, 512), dtype),
        TensorSpec((token_idx, 512), dtype),
        TensorSpec((ori_capacity, 512), dtype),
        TensorSpec((cmp_capacity, 512), dtype),
        TensorSpec((ori_table_rows, ori_table_width), i32),
        TensorSpec((cmp_table_rows, cmp_table_width), i32),
        TensorSpec((n1,), dtypes.float32),
        TensorSpec((1024,), i32),
        TensorSpec((qcu_count,), i32),
        TensorSpec((Dim("LO"),), i32),
        TensorSpec((Dim("LC"),), i32),
        TensorSpec((Dim("R"),), i32),
        dtypes.float32,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        TensorSpec((qlen_count,), i32),
        TensorSpec((topk_count,), i32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return SparseFlashMlaDirectLauncher(n1, block_dim, mode, dtype).run.compile(*specs)


class SparseFlashMlaLauncher:
    def __init__(self, n1, block_dim, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim

    @jit
    def run(
        self,
        bmm_result: Tensor,
        query_gm: Tensor,
        v0_res_gm: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        block_table_gm: Tensor,
        cmp_block_table_gm: Tensor,
        sparse_indices_gm: Tensor,
        cmp_kv_phy_addr_gm: Tensor,
        sinks: Tensor,
        metadata_gm: Tensor,
        cu_seqlens_q_gm: Tensor,
        actual_seq_ori_kv_gm: Tensor,
        actual_seq_cmp_kv_gm: Tensor,
        cmp_residual_kv_gm: Tensor,
        softmax_scale: dtypes.float32,
        cmp_ratio: dtypes.int64,
        ori_win_left: dtypes.int64,
        ori_stride_rows: dtypes.int64,
        cmp_stride_rows: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        SparseFlashMlaCsaKernel(self.n1, self.block_dim, self.dtype)[self.block_dim](
            bmm_result,
            query_gm,
            v0_res_gm,
            ori_kv_gm,
            cmp_kv_gm,
            block_table_gm,
            cmp_block_table_gm,
            sparse_indices_gm,
            cmp_kv_phy_addr_gm,
            sinks,
            metadata_gm,
            cu_seqlens_q_gm,
            actual_seq_ori_kv_gm,
            actual_seq_cmp_kv_gm,
            cmp_residual_kv_gm,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_stride_rows,
            cmp_stride_rows,
            ori_page,
            cmp_page,
            ori_mask,
            cmp_mask,
            right,
            q_size,
            q_lengths,
            topk_length,
            has_q_length,
            has_topk,
            batch_count,
            ori_length_mode,
            cmp_length_mode,
        )


@functools.lru_cache(maxsize=None)
def _compile_smla(n1, block_dim, dtype):
    token_idx = Dim("T")
    ori_capacity = Dim("PO")
    cmp_capacity = Dim("PC")
    sparse_slot_idx = Dim("K")
    ori_table_width = Dim("BO")
    cmp_table_width = Dim("BC")
    ori_table_rows = Dim("OB")
    cmp_table_rows = Dim("CB")
    qcu_count = Dim("CQ")
    qlen_count = Dim("LQ")
    topk_count = Dim("TK")
    i32 = dtypes.int32
    specs = (
        TensorSpec((token_idx * n1, 512), dtype),
        TensorSpec((token_idx * n1, 512), dtype),
        TensorSpec((block_dim * 3 * 128, 512), dtype),
        TensorSpec((ori_capacity, 512), dtype),
        TensorSpec((cmp_capacity, 512), dtype),
        TensorSpec((ori_table_rows, ori_table_width), i32),
        TensorSpec((cmp_table_rows, cmp_table_width), i32),
        TensorSpec((token_idx, sparse_slot_idx), i32),
        TensorSpec((token_idx, sparse_slot_idx * 2), i32),
        TensorSpec((n1,), dtypes.float32),
        TensorSpec((1024,), i32),
        TensorSpec((qcu_count,), i32),
        TensorSpec((Dim("LO"),), i32),
        TensorSpec((Dim("LC"),), i32),
        TensorSpec((Dim("R"),), i32),
        dtypes.float32,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        TensorSpec((qlen_count,), i32),
        TensorSpec((topk_count,), i32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return SparseFlashMlaLauncher(n1, block_dim, dtype).run.compile(*specs)


def sparse_flash_mla(
    query_gm,
    *,
    ori_kv,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    if return_softmax_lse:
        raise NotImplementedError("LSE deferred by user")
    if query_gm.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FP16/BF16 required")
    if layout_q not in ("TND", "BSND") or layout_kv not in (
        "TND",
        "BSND",
        "PA",
        "PA_BBND",
    ):
        raise ValueError("Unsupported layout")
    n1 = query_gm.shape[-2]
    if n1 not in (64, 128) or query_gm.shape[-1] != 512:
        raise ValueError("N1=64/128, D=512 required")
    if ori_sparse_indices is not None:
        raise NotImplementedError("SWA/HCA/CSA only")
    if metadata is None or metadata.dtype != torch.int32 or metadata.numel() != 1024:
        raise ValueError("Original int32[1024] metadata required")
    if sinks is None or sinks.dtype != torch.float32 or tuple(sinks.shape) != (n1,):
        raise ValueError("FP32 sinks[N1] required")
    if (
        not math.isfinite(softmax_scale)
        or not 1 <= cmp_ratio <= 128
        or topk_value_mode != 1
    ):
        raise ValueError("Invalid scale/ratio/topk_value_mode")
    if ori_mask_mode not in (0, 3, 4) or cmp_mask_mode not in (0, 3):
        raise ValueError("Unsupported mask mode")
    if (
        ori_win_left < -1
        or ori_win_right < -1
        or (ori_mask_mode != 4 and (ori_win_left, ori_win_right) != (-1, -1))
    ):
        raise ValueError("Invalid window")
    if layout_kv in ("TND", "BSND") and layout_q != layout_kv:
        raise ValueError("Non-PA Q/KV layouts must agree")
    if layout_q == "TND" and cu_seqlens_q is None:
        raise ValueError("TND Q requires cu_seqlens_q")
    if cmp_kv is None and (cmp_ratio != 1 or cmp_mask_mode != 0):
        raise ValueError("SWA requires ratio1 and cmp_mask0")
    if (
        cmp_kv is not None
        and cmp_mask_mode == 3
        and cmp_ratio != 1
        and cmp_residual_kv is None
    ):
        raise ValueError("Compressed causal residual required")
    if cmp_sparse_indices is not None and (
        cmp_sparse_indices.dtype != torch.int32 or cmp_sparse_indices.shape[-1] <= 0
    ):
        raise ValueError("Positive int32 sparse K required")
    if (
        cmp_sparse_indices is not None
        and cmp_mask_mode == 0
        and cmp_topk_length is None
    ):
        raise ValueError("Noncausal CSA requires cmp_topk_length")
    for kv, table, cu, lengths in (
        (ori_kv, ori_block_table, cu_seqlens_ori_kv, seqused_ori_kv),
        (cmp_kv, cmp_block_table, cu_seqlens_cmp_kv, seqused_cmp_kv),
    ):
        if kv is None:
            continue
        if kv.dtype != query_gm.dtype or tuple(kv.shape[-2:]) != (1, 512):
            raise ValueError("KV dtype/N2/D must match specification")
        if layout_kv in ("PA", "PA_BBND") and (
            table is None or lengths is None or not 1 <= kv.shape[1] <= 1024
        ):
            raise ValueError("PA requires page1..1024, table and actual lengths")
        if layout_kv == "TND" and cu is None:
            raise ValueError("TND KV requires storage prefixes")
    batch_count = query_gm.shape[0] if layout_q == "BSND" else cu_seqlens_q.numel() - 1
    q_size = query_gm.shape[1] if layout_q == "BSND" else 0
    cu_q = cu_seqlens_q if cu_seqlens_q is not None else metadata
    q_lengths = seqused_q if seqused_q is not None else metadata
    topk_length = (
        cmp_topk_length.reshape(-1) if cmp_topk_length is not None else metadata
    )
    has_q_length = int(seqused_q is not None)
    has_topk = int(cmp_topk_length is not None)

    def kv_view(kv, table, cu):
        if layout_kv in ("PA", "PA_BBND"):
            page = kv.shape[1]
            stride = kv.stride(0) // 512
            storage = kv.as_strided(((kv.shape[0] - 1) * stride + page, 512), (512, 1))
            return storage, table, stride, page
        page = 0 if layout_kv == "TND" else -kv.shape[1]
        table = cu.reshape(1, -1) if layout_kv == "TND" else metadata.reshape(1, -1)
        return kv.reshape(-1, 512), table, 0, page

    ori_storage, ori_table, ori_stride, ori_page = kv_view(
        ori_kv, ori_block_table, cu_seqlens_ori_kv
    )
    mode = "SWA" if cmp_kv is None else ("HCA" if cmp_sparse_indices is None else "CSA")
    if mode == "SWA":
        cmp_storage = ori_storage
        cmp_table = ori_table
        cmp_stride = ori_stride
        cmp_page = ori_page
        cmp_lengths = seqused_ori_kv
        residual = seqused_ori_kv
    else:
        cmp_storage, cmp_table, cmp_stride, cmp_page = kv_view(
            cmp_kv, cmp_block_table, cu_seqlens_cmp_kv
        )
        cmp_lengths = seqused_cmp_kv
        residual = cmp_residual_kv if cmp_residual_kv is not None else metadata
    ori_length_mode = (
        0 if seqused_ori_kv is not None else (1 if layout_kv == "TND" else 2)
    )
    cmp_length_mode = 0 if cmp_lengths is not None else (1 if layout_kv == "TND" else 2)
    if seqused_ori_kv is None:
        seqused_ori_kv = cu_seqlens_ori_kv if layout_kv == "TND" else metadata
    if cmp_lengths is None:
        cmp_lengths = cu_seqlens_cmp_kv if layout_kv == "TND" else metadata
    if mode == "SWA":
        cmp_lengths = seqused_ori_kv
        residual = seqused_ori_kv
        cmp_length_mode = ori_length_mode

    block_dim = get_platform_info().cube_core_num
    dtype = dtypes.bfloat16 if query_gm.dtype == torch.bfloat16 else dtypes.float16
    out = torch.empty_like(query_gm)
    extra = (
        ori_page,
        cmp_page,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_right,
        q_size,
        q_lengths,
        topk_length,
        has_q_length,
        has_topk,
        batch_count,
        ori_length_mode,
        cmp_length_mode,
    )
    if mode != "CSA":
        _compile_smla_direct(n1, block_dim, mode, dtype)(
            out.view(-1, 512),
            query_gm.view(-1, 512),
            ori_storage,
            cmp_storage,
            ori_table,
            cmp_table,
            sinks,
            metadata,
            cu_q,
            seqused_ori_kv,
            cmp_lengths,
            residual,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_stride,
            cmp_stride,
            *extra,
        )
    else:
        indices = cmp_sparse_indices.reshape(-1, cmp_sparse_indices.shape[-1])
        physical = torch.empty(
            (indices.shape[0], indices.shape[1] * 2),
            device=query_gm.device,
            dtype=torch.int32,
        )
        workspace = torch.zeros(
            (block_dim * 3 * 128, 512),
            device=query_gm.device,
            dtype=query_gm.dtype,
        )
        _compile_smla(n1, block_dim, dtype)(
            out.view(-1, 512),
            query_gm.view(-1, 512),
            workspace,
            ori_storage,
            cmp_storage,
            ori_table,
            cmp_table,
            indices,
            physical,
            sinks,
            metadata,
            cu_q,
            seqused_ori_kv,
            cmp_lengths,
            residual,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_stride,
            cmp_stride,
            *extra,
        )
    return out, torch.empty((0,), device=query_gm.device, dtype=torch.float32)


from typing import Optional

from mojo_opset.kernels.npu_a5_cannbotdsl.sparse_flash_mla_metadata import (
    sparse_flash_mla_metadata,
)


def sparse_flash_mla_infer_fwd(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_ori_kv: torch.Tensor,
    sinks: Optional[torch.Tensor],
    win_left: int,
    win_right: int,
    softmax_scale: float,
    layout_kv: str,
    cu_seqlens_ori_kv: Optional[torch.Tensor],
    cmp_kv: Optional[torch.Tensor],
    cmp_block_table: Optional[torch.Tensor],
    seqused_cmp_kv: Optional[torch.Tensor],
    cu_seqlens_cmp_kv: Optional[torch.Tensor],
    cmp_residual_kv: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    cmp_ratio: int,
    cmp_mask_mode: int,
):
    num_heads_q = q.shape[1]
    head_dim = q.shape[2]
    num_heads_kv = ori_kv.shape[2] if layout_kv == "PA_BBND" else ori_kv.shape[1]
    s1 = q.shape[0]
    s2_cap = ori_kv.shape[0] * ori_kv.shape[1]
    has_cmp = cmp_kv is not None
    cmp_ratio = cmp_ratio if has_cmp else 1
    cmp_mask_mode = cmp_mask_mode if has_cmp else 0
    cmp_cap = cmp_kv.shape[0] * cmp_kv.shape[1] if has_cmp else 0
    if has_cmp and cmp_residual_kv is None:
        cmp_residual_kv = torch.zeros(
            cu_seqlens_q.numel() - 1, dtype=torch.int32, device=q.device
        )
    if sinks is None:
        sinks = torch.zeros(
            num_heads_q, dtype=torch.float32, device=q.device
        )

    metadata = sparse_flash_mla_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        layout_q="TND",
        layout_kv=layout_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv if has_cmp else None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv if has_cmp else None,
        cmp_residual_kv=cmp_residual_kv if has_cmp else None,
        batch_size=cu_seqlens_q.numel() - 1,
        max_seqlen_q=s1,
        max_seqlen_ori_kv=s2_cap,
        max_seqlen_cmp_kv=cmp_cap,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=win_left,
        ori_win_right=win_right,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )

    out, _lse = sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv if has_cmp else None,
        seqused_q=None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv if has_cmp else None,
        cmp_residual_kv=cmp_residual_kv if has_cmp else None,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=win_left,
        ori_win_right=win_right,
        layout_q="TND",
        layout_kv=layout_kv,
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return out