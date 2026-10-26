# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Sparse flash MLA entry and generic kernels.

The public entry routes sparse inputs with 1..64 query heads to the N64 sparse
module and all other inputs to the generic implementation in this file.
"""

import functools
import math

from typing import Optional

import cannbotdsl
import torch

from cannbotdsl import PIPE
from cannbotdsl import Dim
from cannbotdsl import MemLoc
from cannbotdsl import Tensor
from cannbotdsl import TensorSpec
from cannbotdsl import cv_valid_extent
from cannbotdsl import dtypes
from cannbotdsl import get_platform_info
from cannbotdsl import host
from cannbotdsl._mlir.dialects import arith
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import matmul
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.arch import get_subblock_id
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import cube_fill_l1
from cannbotdsl.ops.sync import cube_sync_block_arrive
from cannbotdsl.ops.sync import cube_sync_block_wait
from cannbotdsl.ops.sync import cube_sync_intra_arrive
from cannbotdsl.ops.sync import cube_sync_intra_wait
from cannbotdsl.ops.sync import cube_sync_pipe
from cannbotdsl.ops.sync import vec_sync_intra_arrive
from cannbotdsl.ops.sync import vec_sync_intra_wait
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import reinterpret
from cannbotdsl.tensor import tile_slice
from cannbotdsl.types._integer import Int64
from cannbotdsl.types.delay_line import DelayLineGroup

from .sparse_flash_mla_metadata import sparse_flash_mla_metadata
from .sparse_flash_mla_sparse_n64_kernel import sparse_n64_kernel

_BITS_PER_BYTE = 8
_F32_BITS = 32
_F16_BITS = 16
_I32_BITS = 32
_VECTOR_REGISTER_BITS = 2048
_DATA_BLOCK_BYTES = 32
_D = 512
_TILE = 128
_D_TILES = _D // _TILE
_Q_TILE_ROWS = 64
_GATHER_TILE_ROWS = 16
# Two-row DMA is used only when the source-row distance is strictly below 2^21.
_PAIR_DMA_ROW_DISTANCE_LIMIT = 1 << 21
# A 2048-bit vector holds 64 float32 values.
_F32_VL = _VECTOR_REGISTER_BITS // _F32_BITS
_METADATA_FIELDS_PER_BLOCK = 9
_KV_RING = 3
# Cross-core channels use 0..4; keep the explicit GM workspace protocol
# disjoint: ready flag 9, free-slot flags 10 through 12.
_WORKSPACE_READY_FLAG = 9
_WORKSPACE_FREE_FLAG = 10
_Q_RING = 3
# Each Cube block splits its query tile between two vector subblocks.
_VEC_TILE_M = _Q_TILE_ROWS // 2
# Pipeline stage tick offsets: a stage reading delay-line tap N is active
# while ``tick >= N and tick < count + N``. Kernels that route the KV load
# through the delay line use LOAD/QK/PV = 1/2/3; the SWA/HCA kernel loads
# KV inline, so its QK/PV stages sit one tap earlier. The post-loop drain
# flushes ``range(<deepest stage>)`` ticks.
_STAGE_LOAD = 1
_STAGE_QK = 2
_STAGE_PV = 3
_STAGE_QK_INLINE = 1
_STAGE_PV_INLINE = 2
_NZ_ROW = 16

# Public ABI limits and mask-mode codes, independent of kernel tile sizes.
_MAX_QUERY_HEADS = 128
_MAX_CMP_RATIO = 128
_MAX_PAGE_ROWS = 1024
_METADATA_ELEMENTS = 1024
_MASK_CAUSAL = 3
_MASK_SLIDING_WINDOW = 4
# Metadata ABI offsets within each per-block record.
_META_LAST_BATCH = 4
_META_LAST_QUERY = 5
_META_SYNC_TICKS = 8
# Cross-core synchronization flag encoding and workspace barrier channel.
_SYNC_PEER_FLAG_OFFSET = 16
_WORKSPACE_BARRIER_FLAG = 8
_KV_ROWS_PER_VECTOR = _TILE // 2
_BF16_VL = _VECTOR_REGISTER_BITS // _F16_BITS
_NZ_INNER_COLUMN_AXIS = 3
_SPARSE_PIPELINE_SLOTS = _STAGE_PV + 1
_INLINE_PIPELINE_SLOTS = _STAGE_PV_INLINE + 1
_SOFTMAX_STATE_SLOTS = 3


def dyn_select(predicate, true_value, false_value):
    """Preserve the eager scalar select used by the 0.6.0 implementation."""
    from cannbotdsl.core.frontend.runtime.control_flow import to_mlir_value
    from cannbotdsl.core.frontend.runtime.value_tree_layout import ValueTreeLayout

    pred = to_mlir_value(predicate, context="select predicate")
    layout, lhs, rhs = ValueTreeLayout.reconcile(true_value, false_value)
    return layout.rebuild(tuple(arith.select(pred, a, b) for a, b in zip(lhs, rhs)))


def _clamp_nonneg(value):
    if isinstance(value, int):
        return max(0, value)
    return _clamp_nonneg_dynamic(value)


@jit
def _clamp_nonneg_dynamic(value):
    return dtypes.int64((0 if value < 0 else value))


def _rt_min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return _rt_min_dynamic(a, b)


@jit
def _rt_min_dynamic(a, b):
    a64 = a if not isinstance(a, int) else dtypes.int64(a)
    b64 = b if not isinstance(b, int) else dtypes.int64(b)
    return dtypes.int64((a64 if a64 < b64 else b64))


class SmlaMatmulBase:
    """Three-slot Q/KV rings; selected KV survives both QK and PV reads."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_n, dtype=dtypes.bfloat16):
        self.dtype = dtype
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = _D
        self.q_ring = RingBuffer(MemLoc.L1, (tile_cube_m, _D // 2), dtype, depth=_Q_RING)
        self.kv_ring = RingBuffer(MemLoc.L1, (tile_n, _D), dtype, depth=_KV_RING)
        self.l0a = RingBuffer(MemLoc.L0A, (tile_cube_m, _TILE), dtype, depth=2)
        self.l0a_p = RingBuffer(MemLoc.L0A, (tile_cube_m, tile_n), dtype, depth=2)
        self.l0b = RingBuffer(MemLoc.L0B, (_TILE, _TILE), dtype, depth=2)
        self.l0c_qk = RingBuffer(MemLoc.L0C, (tile_cube_m, _TILE), dtypes.float32, depth=2)
        self.l0c = RingBuffer(MemLoc.L0C, (tile_cube_m, _TILE), dtypes.float32, depth=2)

    @jit
    def load_q_wide(self, q_tile_gm, m_seq: Int64, rows=None):
        # Two independent half-slots, each reused only after its final QK read.
        for half in range_constexpr(2):
            mem_copy(
                self.q_ring.next_write(),
                tile_slice(q_tile_gm, (self.tile_cube_m, _D // 2), (0, half)),
            )

    def bmm1_fanout_q(self, kv_slot, q_half0, q_half1, actual_n=None):
        # The physical NZ frame stays full width. Softmax masks columns
        # beyond actual_n, so narrowing this load must not repack the frame.
        qk_acc = self.l0c_qk.next_write()
        for chunk in range_constexpr(_D_TILES):
            q_half = q_half0 if chunk < 2 else q_half1
            l0_a = self.l0a.next_write()
            l0_b = self.l0b.next_write()
            mem_copy(l0_a, tile_slice(q_half, (self.tile_cube_m, _TILE), (0, chunk % 2)))
            mem_copy(l0_b, self._kv_chunk(kv_slot, chunk))
            matmul(qk_acc, l0_a, l0_b, init=(chunk == 0))

    def store_s(self, qk_ub_ch):
        mem_copy(qk_ub_ch.next_write(), self.l0c_qk.next_read().partition(dim=0))

    def compute_pv_fanout(self, p_l1_ch, pv_ub_ch, kv_slot, actual_n=None):
        valid_kv_rows = actual_n if actual_n is not None else self.tile_n
        p = p_l1_ch.next_read()
        out = pv_ub_ch.next_write()
        l0_a = self.l0a_p.next_write()
        mem_copy(l0_a, p)
        # View P at the parent NZ pitch and round the reduction extent up to
        # the 16-row fractal; P is masked to zero beyond actual_n.
        pv_k = (valid_kv_rows + (_NZ_ROW - 1)) // _NZ_ROW * _NZ_ROW
        l0_a_view = reinterpret(l0_a, shape=(self.tile_cube_m, pv_k), stride=(self.tile_n, 1))
        for chunk in range_constexpr(_D_TILES):
            l0_b = self.l0b.next_write()
            # A ZN destination view makes this L1 -> L0B copy transpose the
            # KV chunk; the matmul reads the same view at the pv_k extent.
            l0_b_view = reinterpret(l0_b, shape=(self.tile_n, _TILE), data_format="zn")
            mem_copy(l0_b_view, self._kv_chunk(kv_slot, chunk))
            pv_accumulator = self.l0c.next_write()
            matmul(
                pv_accumulator,
                l0_a_view,
                reinterpret(l0_b_view, shape=(pv_k, _TILE)),
                init=True,
            )
            mem_copy(
                tile_slice(out, (self.tile_vec_m, _TILE), (0, chunk)),
                pv_accumulator.partition(dim=0),
            )
        # kv_slot remains the same alias used by QK in the previous tick.
        # Automatic synchronization protects reuse after this last L1 read.

    def _kv_chunk(self, kv_slot, chunk):
        return tile_slice(kv_slot, (self.tile_n, _TILE), (0, chunk))


class GeneralSmlaVector:
    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        preload_num=_SOFTMAX_STATE_SLOTS,
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
        if tile_d % _F32_VL != 0:
            raise ValueError(
                f"tile_d ({tile_d}) must be divisible by the float32 vector length "
                f"({_F32_VL}); the output path uses a full mask for every tile."
            )
        self.sm_max_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]
        self.sm_sum_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]
        self.sm_exp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]
        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        self.res_o_b16 = self.res_o.reinterpret(self.dtype, shape=(tile_vec_m, tile_d))
        p_n1_pad = _DATA_BLOCK_BYTES // (_F16_BITS // _BITS_PER_BYTE)
        self.p_ub = RingBuffer(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=self.dtype,
            depth=2,
            data_format="nz",
            n1_pad=p_n1_pad,
        )
        if n_heads and self.use_sinks:
            self.sinks_ub = RingBuffer(MemLoc.UB, shape=(self.sinks_span,), dtype=dtypes.float32, depth=1)

    @staticmethod
    def _half_only(actual_n, vl_lanes):
        return isinstance(actual_n, int) and actual_n <= vl_lanes

    @jit
    def softmax_rest(self, qk_ch, scale, query_state_slot: Int64, tile_state_slot: Int64, actual_n):
        p_slot = self.p_ub.next_write()
        sm_max = self.sm_max_tb[query_state_slot]
        sm_sum = self.sm_sum_tb[query_state_slot]
        sm_exp = self.sm_exp_tb[tile_state_slot]
        kv_tile_cols = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        half_only = self._half_only(actual_n, _F32_VL)
        # VF softmax: mask invalid KV lanes, update row maxima/sums and pack P.
        with vf(mode="simd"):
            query_head_rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            rowmask, _ = rr.update_mask(query_head_rows, elem_bits=_F32_BITS)
            for row in range(query_head_rows):
                full_f32_mask, _ = rr.update_mask(_F32_VL, elem_bits=_F32_BITS)
                full_16bit_mask, _ = rr.update_mask(kv_tile_cols, elem_bits=_F16_BITS)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=_F32_BITS)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - _F32_VL), elem_bits=_F32_BITS)
                self._pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    _F32_VL,
                    half0_mask,
                    half1_mask,
                    full_f32_mask,
                    half_only=half_only,
                )
            rr.vmem_bar("vst_vld")
            updated_max_reg = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, updated_max_reg, rowmask)
            rr.vmem_bar("vst_vld")
            for row in range(query_head_rows):
                full_f32_mask, _ = rr.update_mask(_F32_VL, elem_bits=_F32_BITS)
                full_16bit_mask, _ = rr.update_mask(kv_tile_cols, elem_bits=_F16_BITS)
                even_lane_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=_F32_BITS)
                odd_lane_mask, _ = rr.update_mask(actual_n // 2, elem_bits=_F32_BITS)
                valid_16bit_mask, _ = rr.update_mask(actual_n, elem_bits=_F16_BITS)
                score_row_offset = row * src_row_stride
                p_nz_row_offset = row // m0 * s_m1 + row % m0 * s_m0
                exp_even_reg, exp_odd_reg = self._softmax_fold_row(
                    qk_ch,
                    p_slot,
                    score_row_offset,
                    p_nz_row_offset,
                    self.tmp_new_max,
                    row,
                    even_lane_mask,
                    odd_lane_mask,
                    valid_16bit_mask,
                    full_16bit_mask,
                    full_f32_mask,
                    block_stride,
                )
                row_sum_reg = rr.vreduce_sum(
                    rr.vadd(exp_even_reg, exp_odd_reg, mask=full_f32_mask),
                    mask=even_lane_mask,
                )
                rr.vstore_first(self.tmp_sum, row, row_sum_reg)
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def init_no_sink_buffer(self, sinks_view):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.sinks_span, elem_bits=_F32_BITS)
            negative_inf = rr.vreinterpret(rr.vdups(0xFF800000, dtypes.uint32, mask=mask), dtypes.float32)
            rr.vstore(sinks_view, 0, negative_inf, mask)

    @jit
    def seed_from_sinks(self, sinks_view, query_state_slot: Int64, sink_base=None):
        sm_max = self.sm_max_tb[query_state_slot]
        sm_sum = self.sm_sum_tb[query_state_slot]
        span = self.sinks_span
        with vf(mode="simd"):
            mask, _ = rr.update_mask(_F32_VL, elem_bits=_F32_BITS)
            one = rr.vdups(1.0, dtypes.float32, mask=mask)
            zero = rr.vdups(0.0, dtypes.float32, mask=mask)
            finite_min = rr.vdups(-3.4028234663852886e38, dtypes.float32, mask=mask)
            for row in tuple(range(self.tile_vec_m)):
                if const_expr(sink_base is None):
                    idx = row % span
                else:
                    idx = _rt_min(sink_base + row, self._g_rt - 1)
                sink = rr.vload_broadcast(sinks_view, idx)
                absent = rr.veqs(rr.vreinterpret(sink, dtypes.uint32), 0xFF800000, mask=mask)
                rr.vstore_first(sm_max, row, rr.vselect(finite_min, sink, cond_mask=absent))
                rr.vstore_first(sm_sum, row, rr.vselect(zero, one, cond_mask=absent))

    def store_p(self, p_l1_ch):
        mem_copy(
            p_l1_ch.next_write().partition(dim=0, part_id=self.subblock_idx),
            self.p_ub.next_read(),
        )

    @jit
    def init_o(self, pv_ch):
        mem_copy(self.res_o, pv_ch)

    @jit
    def update_o(self, pv_ch, exp_idx: Int64):
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        # VF output update: rescale the previous accumulator and add this PV tile.
        with vf(mode="simd"):
            full_f32_mask = rr.full_mask()
            for row in range(self.tile_vec_m):
                rescale_factor_reg = rr.vload_broadcast(sm_exp_buf, row)
                output_row_offset = row * self.tile_d
                for d_vector_offset in tuple(range(0, self.tile_d, _F32_VL)):
                    flat_offset = output_row_offset + d_vector_offset
                    previous_output_reg = rr.vload(self.res_o, flat_offset)
                    current_pv_reg = rr.vload(pv_ch, flat_offset)
                    if const_expr(self.ablate_no_pv_acc):
                        rr.vstore(self.res_o, flat_offset, current_pv_reg, full_f32_mask)
                    else:
                        output_reg = rr.vmadd(
                            previous_output_reg,
                            rescale_factor_reg,
                            current_pv_reg,
                            mask=full_f32_mask,
                        )
                        rr.vstore(self.res_o, flat_offset, output_reg, full_f32_mask)

    @jit
    def update_o_last(self, pv_ch, exp_idx: Int64, sum_idx: Int64):
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        # VF final output update: accumulate the last PV tile and normalize by the sum.
        with vf(mode="simd"):
            full_f32_mask = rr.full_mask()
            for row in range(self.tile_vec_m):
                rescale_factor_reg = rr.vload_broadcast(sm_exp_buf, row)
                softmax_sum_reg = rr.vload_broadcast(sm_sum_buf, row)
                output_row_offset = row * self.tile_d
                for d_vector_offset in tuple(range(0, self.tile_d, _F32_VL)):
                    flat_offset = output_row_offset + d_vector_offset
                    previous_output_reg = rr.vload(self.res_o, flat_offset)
                    current_pv_reg = rr.vload(pv_ch, flat_offset)
                    if const_expr(self.ablate_no_pv_acc):
                        output_reg = current_pv_reg
                    else:
                        output_reg = rr.vmadd(
                            previous_output_reg,
                            rescale_factor_reg,
                            current_pv_reg,
                            mask=full_f32_mask,
                        )
                    if const_expr(not self.ablate_no_rescale):
                        output_reg = rr.vdiv(output_reg, softmax_sum_reg, mask=full_f32_mask)
                    rr.vstore(self.res_o, flat_offset, output_reg, full_f32_mask)

    @jit
    def init_o_last(self, pv_ch, sum_idx: Int64):
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        # VF single-tile output: normalize the initial PV result by the softmax sum.
        with vf(mode="simd"):
            full_f32_mask = rr.full_mask()
            for row in range(self.tile_vec_m):
                softmax_sum_reg = rr.vload_broadcast(sm_sum_buf, row)
                output_row_offset = row * self.tile_d
                for d_vector_offset in tuple(range(0, self.tile_d, _F32_VL)):
                    flat_offset = output_row_offset + d_vector_offset
                    current_pv_reg = rr.vload(pv_ch, flat_offset)
                    if const_expr(self.ablate_no_rescale):
                        rr.vstore(self.res_o, flat_offset, current_pv_reg, full_f32_mask)
                    else:
                        rr.vstore(
                            self.res_o,
                            flat_offset,
                            rr.vdiv(current_pv_reg, softmax_sum_reg, mask=full_f32_mask),
                            full_f32_mask,
                        )

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
        # res_o and its dtype view share the framework's Buffer sync identity.
        mem_copy(
            o_tile_gm.partition(dim=0, part_id=self.subblock_idx),
            self.res_o.reinterpret(self.dtype, shape=(self.tile_vec_m, self.tile_d)),
        )

    @jit
    def _finalize_div_vf(self, sum_idx: Int64, actual_vec_m):
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        # VF output normalization: divide valid query-head rows by their softmax sums.
        with vf(mode="simd"):
            full_f32_mask = rr.full_mask()
            for row in range(actual_vec_m):
                softmax_sum_reg = rr.vload_broadcast(sm_sum_buf, row)
                output_row_offset = row * self.tile_d
                for d_vector_offset in tuple(range(0, self.tile_d, _F32_VL)):
                    flat_offset = output_row_offset + d_vector_offset
                    rr.vstore(
                        self.res_o,
                        flat_offset,
                        rr.vdiv(
                            rr.vload(self.res_o, flat_offset),
                            softmax_sum_reg,
                            mask=full_f32_mask,
                        ),
                        full_f32_mask,
                    )

    @jit
    def _cast_o_bf16_vf(self):
        # Match the old tensor-cast lowering: short runtime loop, one contiguous
        # f32 load/cast and one packed bf16 store per 64 source elements.
        # VF output conversion: cast FP32 accumulators and pack them into 16-bit lanes.
        with vf(mode="simd"):
            for flat_offset in range(0, self.tile_vec_m * self.tile_d, _F32_VL):
                f32_mask, _ = rr.update_mask(_F32_VL, elem_bits=_F32_BITS)
                output_f32_reg = rr.vload(self.res_o, flat_offset)
                output_16bit_reg = rr.vcast(
                    output_f32_reg,
                    self.dtype,
                    mask=f32_mask,
                    rounding=rr.RoundingMode.RN,
                )
                rr.vstore_pack(
                    self.res_o_b16,
                    flat_offset,
                    output_16bit_reg,
                    f32_mask,
                    pack_mode="b32_to_b16",
                )

    def _nz_params(self):
        p_nz_strides = self.p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = (
            p_nz_strides[0],
            p_nz_strides[1],
            p_nz_strides[2],
            p_nz_strides[_NZ_INNER_COLUMN_AXIS],
        )
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return (m0, s_m1, s_m0, s_n1 // n0)

    def _softmax_fold_row(
        self,
        qk_ch,
        p_slot,
        score_row_offset,
        p_nz_row_offset,
        max_brc_buf,
        row,
        even_lane_mask,
        odd_lane_mask,
        valid_16bit_mask,
        full_16bit_mask,
        full_f32_mask,
        block_stride,
    ):
        row_max_reg = rr.vload_broadcast(max_brc_buf, row)
        exp_even_reg, exp_odd_reg = rr.vload_deinterleave(qk_ch, score_row_offset, width="b32")
        exp_even_reg = rr.vexp_sub(exp_even_reg, row_max_reg, mask=even_lane_mask)
        exp_odd_reg = rr.vexp_sub(exp_odd_reg, row_max_reg, mask=odd_lane_mask)
        p_even_reg = rr.vcast(exp_even_reg, self.dtype, mask=even_lane_mask, reg_layout=rr.RegLayout.ZERO)
        p_odd_reg = rr.vcast(exp_odd_reg, self.dtype, mask=odd_lane_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(p_even_reg, p_odd_reg, mask=valid_16bit_mask)
        if const_expr(self.rt_split):
            zero_16bit_reg = rr.vdups(0.0, self.dtype, mask=full_16bit_mask)
            invalid_16bit_mask = rr.mask_xor(full_16bit_mask, valid_16bit_mask, exec_mask=full_16bit_mask)
            merged = rr.vselect(zero_16bit_reg, merged, cond_mask=invalid_16bit_mask)
        rr.vstore_strided(
            p_slot,
            p_nz_row_offset,
            merged,
            full_16bit_mask,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return (exp_even_reg, exp_odd_reg)

    def _pass_a_row(
        self,
        qk_ch,
        scale,
        sm_max_dst,
        row,
        row_stride,
        vl_lanes,
        half0_mask,
        half1_mask,
        full_mask,
        half_only=False,
    ):
        score_row_offset = row * row_stride
        if const_expr(half_only):
            scaled_score_lower_reg = rr.vmuls(rr.vload(qk_ch, score_row_offset), scale, mask=half0_mask)
            rr.vstore(qk_ch, score_row_offset, scaled_score_lower_reg, half0_mask)
            row_max_reg = rr.vreduce_max(scaled_score_lower_reg, mask=full_mask)
        else:
            scaled_score_lower_reg = rr.vmuls(rr.vload(qk_ch, score_row_offset), scale, mask=half0_mask)
            scaled_score_upper_reg = rr.vmuls(rr.vload(qk_ch, score_row_offset + vl_lanes), scale, mask=half1_mask)
            rr.vstore(qk_ch, score_row_offset, scaled_score_lower_reg, half0_mask)
            rr.vstore(qk_ch, score_row_offset + vl_lanes, scaled_score_upper_reg, half1_mask)
            if const_expr(self.rt_split):
                lower_max_reg = rr.vreduce_max(scaled_score_lower_reg, mask=half0_mask)
                upper_max_reg = rr.vreduce_max(scaled_score_upper_reg, mask=half1_mask)
                row_max_reg = rr.vmax(lower_max_reg, upper_max_reg, mask=full_mask)
            else:
                row_max_reg = rr.vreduce_max(
                    rr.vmax(scaled_score_lower_reg, scaled_score_upper_reg, mask=full_mask),
                    mask=full_mask,
                )
        rr.vstore_first(sm_max_dst, row, row_max_reg)

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        rescale_factor_reg = rr.vexp_sub(old_max, new_max, mask=rowmask)
        rr.vstore(sm_exp, 0, rescale_factor_reg, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        updated_sum = rr.vmadd(old_sum, rescale_factor_reg, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, updated_sum, rowmask)


# SMLA adaptations: non-quantized PA DMA, sparse gather, original FA metadata.


class SmlaMatmul(SmlaMatmulBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.shared_workspace = False

    @jit
    def load_kv(
        self,
        workspace: Tensor,
        kv_gm: Tensor,
        block_table: Tensor,
        batch: Int64,
        kv_row_start: Int64,
        valid_kv_rows: Int64,
        page_stride: Int64,
        workspace_slot_idx: Int64,
        is_cmp: Int64,
        page_size: Int64,
    ):
        kv_slot = self.kv_ring.next_write()
        if is_cmp != 0:
            mem_copy(
                kv_slot,
                workspace,
            )
            # Both vector writers may reuse this slot only after MTE2 has
            # finished reading it into this Cube's private L1 ring.
            if const_expr(not self.shared_workspace):
                cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + workspace_slot_idx)
                cube_sync_intra_arrive(
                    PIPE.MTE2,
                    _WORKSPACE_FREE_FLAG + _SYNC_PEER_FLAG_OFFSET + workspace_slot_idx,
                )
        else:
            # Assemble the page-table-selected GM segments directly in L1.
            copied_kv_rows = 0
            while copied_kv_rows < valid_kv_rows:
                logical_kv_row_idx = kv_row_start + copied_kv_rows
                page_rows = 0
                physical_row = 0
                if page_size > 0:
                    page = dtypes.int64(block_table[batch, logical_kv_row_idx // page_size])
                    in_page = logical_kv_row_idx % page_size
                    page_rows = min(valid_kv_rows - copied_kv_rows, page_size - in_page)
                    physical_row = page * page_stride + in_page
                else:
                    kv_batch_row_start = 0
                    if page_size == 0:
                        kv_batch_row_start = dtypes.int64(block_table[0, batch])
                    else:
                        kv_batch_row_start = batch * (-page_size)
                    page_rows = valid_kv_rows - copied_kv_rows
                    physical_row = kv_batch_row_start + logical_kv_row_idx
                # Copy the exact page tail into the shared 128-row NZ frame.
                # Each NZ row is one 32-byte DataBlock. The Tensor storage
                # layout pins column pitch independently of segment height.
                src = kv_gm[physical_row : physical_row + page_rows, None]
                if copied_kv_rows == 0:
                    # A tile window preserves the parent's 128-row C0 pitch.
                    mem_copy(tile_slice(kv_slot, (_TILE, _D), (0, 0)), src)
                else:
                    # Preserve the selected NZ slot's physical 128-row C0 pitch
                    # while advancing to a runtime logical row.  The 0.6.0
                    # lowering keeps this as one wide ND2NZ transfer
                    # (nd_num=1, d=512) instead of 32 narrow d=16 stripes.
                    dst = kv_slot[copied_kv_rows : copied_kv_rows + page_rows, None]
                    mem_copy(dst, src)
                copied_kv_rows = copied_kv_rows + page_rows
            # PV consumes complete NZ row fractals; zero its N-axis tail.
            pv_k = (valid_kv_rows + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
            if valid_kv_rows < pv_k:
                cube_fill_l1(
                    kv_slot,
                    0,
                    offset_elems=valid_kv_rows * _NZ_ROW,
                    repeat=_D // _NZ_ROW,
                    blk_num=pv_k - valid_kv_rows,
                    dst_gap=_TILE - (pv_k - valid_kv_rows),
                )
                # Finish both the valid DMA and tail fill before QK/PV reads.
                cube_sync_pipe(PIPE.MTE2)


def copy_sparse_pair_gm_to_ub(dst, src, row, physical0, physical1):
    # Preserve the original pair order and two-copy behavior for this
    # RingBuffer migration; sparse DMA merging can be measured as a separate
    # change.
    mem_copy(tile_slice(dst, (1, _D), (row, 0)), src[physical0 : physical0 + 1, None])
    mem_copy(tile_slice(dst, (1, _D), (row + 1, 0)), src[physical1 : physical1 + 1, None])


class SmlaVec0:
    def __init__(self, n1, dtype):
        self.dtype = dtype
        self.n1 = n1
        self.shared_workspace = False
        self.gather_ping = RingBuffer(MemLoc.UB, (_GATHER_TILE_ROWS, _D), dtype, depth=1)
        self.gather_pong = RingBuffer(MemLoc.UB, (_GATHER_TILE_ROWS, _D), dtype, depth=1)
        self.ready_flag = _WORKSPACE_READY_FLAG

    @jit
    def initialize(self):
        if const_expr(not self.shared_workspace):
            for workspace_slot_idx in range_constexpr(_KV_RING):
                cube_sync_intra_arrive(PIPE.MTE2, _WORKSPACE_FREE_FLAG + workspace_slot_idx)
                cube_sync_intra_arrive(
                    PIPE.MTE2,
                    _WORKSPACE_FREE_FLAG + _SYNC_PEER_FLAG_OFFSET + workspace_slot_idx,
                )

    @jit
    def finish(self):
        if const_expr(not self.shared_workspace):
            for workspace_slot_idx in range_constexpr(_KV_RING):
                vec_sync_intra_wait(PIPE.MTE3, _WORKSPACE_FREE_FLAG + workspace_slot_idx)

    @jit
    def gather_full128(
        self,
        workspace: Tensor,
        kv: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        sparse_index_start: Int64,
        valid_kv_rows: Int64,
        workspace_slot_idx: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        if const_expr(not self.shared_workspace):
            vec_sync_intra_wait(PIPE.MTE3, _WORKSPACE_FREE_FLAG + workspace_slot_idx)
        vector_partition_idx = get_subblock_id()
        rows_per_vec = _KV_ROWS_PER_VECTOR
        if const_expr(self.shared_workspace):
            vector_partition_idx = (get_block_idx() % 2) * 2 + vector_partition_idx
            rows_per_vec = _VEC_TILE_M
        vector_kv_row_start = vector_partition_idx * rows_per_vec
        linear_prefix = 0
        if page_size == 0:
            linear_prefix = dtypes.int64(table[0, batch])
        elif page_size < 0:
            linear_prefix = batch * (-page_size)
        for chunk in range_constexpr(rows_per_vec // _GATHER_TILE_ROWS):
            ring = self.gather_ping if chunk % 2 == 0 else self.gather_pong
            dst = ring.next_write()
            chunk_kv_row_start = vector_kv_row_start + chunk * _GATHER_TILE_ROWS
            self._load_full16(
                dst,
                kv,
                indices,
                table,
                batch,
                query,
                sparse_index_start,
                chunk_kv_row_start,
                page_stride,
                page_size,
                linear_prefix,
            )
            ready = ring.next_read()
            mem_copy(
                tile_slice(
                    workspace,
                    (_GATHER_TILE_ROWS, _D),
                    (chunk_kv_row_start // _GATHER_TILE_ROWS, 0),
                ),
                ready,
            )
        vec_sync_intra_arrive(PIPE.MTE3, self.ready_flag)

    @jit
    def gather_tile(
        self,
        workspace: Tensor,
        kv: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        sparse_index_start: Int64,
        valid_kv_rows: Int64,
        workspace_slot_idx: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        if valid_kv_rows == _MAX_QUERY_HEADS:
            self.gather_full128(
                workspace,
                kv,
                indices,
                table,
                batch,
                query,
                sparse_index_start,
                valid_kv_rows,
                workspace_slot_idx,
                page_stride,
                page_size,
            )
        else:
            self.gather(
                workspace,
                kv,
                indices,
                table,
                batch,
                query,
                sparse_index_start,
                valid_kv_rows,
                workspace_slot_idx,
                page_stride,
                page_size,
            )

    @jit
    def gather(
        self,
        workspace: Tensor,
        kv: Tensor,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        sparse_index_start: Int64,
        valid_kv_rows: Int64,
        workspace_slot_idx: Int64,
        page_stride: Int64,
        page_size: Int64,
    ):
        if const_expr(not self.shared_workspace):
            vec_sync_intra_wait(PIPE.MTE3, _WORKSPACE_FREE_FLAG + workspace_slot_idx)
        vector_partition_idx = get_subblock_id()
        rows_per_vec = _KV_ROWS_PER_VECTOR
        if const_expr(self.shared_workspace):
            vector_partition_idx = (get_block_idx() % 2) * 2 + vector_partition_idx
            rows_per_vec = _VEC_TILE_M
        vector_kv_row_start = vector_partition_idx * rows_per_vec
        linear_prefix = 0
        if page_size == 0:
            linear_prefix = dtypes.int64(table[0, batch])
        elif page_size < 0:
            linear_prefix = batch * (-page_size)
        vector_valid_kv_rows = min(max(valid_kv_rows - vector_kv_row_start, 0), rows_per_vec)
        full_chunks = vector_valid_kv_rows // _GATHER_TILE_ROWS
        tail_rows = vector_valid_kv_rows - full_chunks * _GATHER_TILE_ROWS
        # One RingBuffer write transaction spans all row-pair loads in a chunk.
        # Its consumer MTE3 copy owns the slot until the GM write completes.
        for chunk in range_constexpr(rows_per_vec // _GATHER_TILE_ROWS):
            if chunk < full_chunks:
                ring = self.gather_ping if chunk % 2 == 0 else self.gather_pong
                dst = ring.next_write()
                chunk_kv_row_start = vector_kv_row_start + chunk * _GATHER_TILE_ROWS
                self._load_full16(
                    dst,
                    kv,
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start,
                    chunk_kv_row_start,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                ready = ring.next_read()
                mem_copy(
                    tile_slice(
                        workspace,
                        (_GATHER_TILE_ROWS, _D),
                        (chunk_kv_row_start // _GATHER_TILE_ROWS, 0),
                    ),
                    ready,
                )
        if tail_rows > 0:
            tail_kv_row_start = vector_kv_row_start + full_chunks * _GATHER_TILE_ROWS
            if full_chunks % 2 == 0:
                dst = self.gather_ping.next_write()
                self._load_tail(
                    dst,
                    kv,
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start,
                    tail_kv_row_start,
                    tail_rows,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                ready = self.gather_ping.next_read()
                mem_copy(
                    workspace[tail_kv_row_start : tail_kv_row_start + tail_rows, None],
                    reinterpret(ready, shape=(tail_rows, _D), stride=(_D, 1)),
                )
            else:
                dst = self.gather_pong.next_write()
                self._load_tail(
                    dst,
                    kv,
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start,
                    tail_kv_row_start,
                    tail_rows,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                ready = self.gather_pong.next_read()
                mem_copy(
                    workspace[tail_kv_row_start : tail_kv_row_start + tail_rows, None],
                    reinterpret(ready, shape=(tail_rows, _D), stride=(_D, 1)),
                )
        vec_sync_intra_arrive(PIPE.MTE3, self.ready_flag)

    @jit
    def wait(self):
        cube_sync_intra_wait(PIPE.MTE2, self.ready_flag)
        cube_sync_intra_wait(PIPE.MTE2, self.ready_flag + _SYNC_PEER_FLAG_OFFSET)

    @jit
    def _physical_row(
        self,
        indices: Tensor,
        table: Tensor,
        batch: Int64,
        query: Int64,
        index: Int64,
        page_stride: Int64,
        page_size: Int64,
        linear_prefix: Int64,
    ):
        # Resolve the index in the block that consumes it. No other block
        # needs to publish a physical-address workspace first.
        sparse = max(0, dtypes.int64(indices[query, index]))
        physical = 0
        if page_size > 0:
            physical = dtypes.int64(table[batch, sparse // page_size]) * page_stride + sparse % page_size
        else:
            physical = linear_prefix + sparse
        return physical

    @jit
    def _load_full16(
        self,
        dst,
        kv,
        indices,
        table,
        batch: Int64,
        query: Int64,
        sparse_index_start: Int64,
        chunk_kv_row_start: Int64,
        page_stride: Int64,
        page_size: Int64,
        linear_prefix: Int64,
    ):
        for row in range(0, _GATHER_TILE_ROWS, 2):
            physical0 = self._physical_row(
                indices,
                table,
                batch,
                query,
                sparse_index_start + chunk_kv_row_start + row,
                page_stride,
                page_size,
                linear_prefix,
            )
            physical1 = self._physical_row(
                indices,
                table,
                batch,
                query,
                sparse_index_start + chunk_kv_row_start + row + 1,
                page_stride,
                page_size,
                linear_prefix,
            )
            distance = max(physical0, physical1) - min(physical0, physical1)
            if distance > 0 and distance < _PAIR_DMA_ROW_DISTANCE_LIMIT:
                copy_sparse_pair_gm_to_ub(dst, kv, row, physical0, physical1)
            else:
                mem_copy(
                    tile_slice(dst, (1, _D), (row, 0)),
                    tile_slice(kv, (1, _D), (physical0, 0)),
                )
                mem_copy(
                    tile_slice(dst, (1, _D), (row + 1, 0)),
                    tile_slice(kv, (1, _D), (physical1, 0)),
                )

    @jit
    def _load_tail(
        self,
        dst,
        kv,
        indices,
        table,
        batch: Int64,
        query: Int64,
        sparse_index_start: Int64,
        chunk_kv_row_start: Int64,
        tail_rows: Int64,
        page_stride: Int64,
        page_size: Int64,
        linear_prefix: Int64,
    ):
        # Exact tail: paired valid rows keep the merged DMA; an odd final row
        # uses one row copy. Invalid sparse indices are never read.
        for row in range(0, _GATHER_TILE_ROWS, 2):
            if row + 1 < tail_rows:
                physical0 = self._physical_row(
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start + chunk_kv_row_start + row,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                physical1 = self._physical_row(
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start + chunk_kv_row_start + row + 1,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                distance = max(physical0, physical1) - min(physical0, physical1)
                if distance > 0 and distance < _PAIR_DMA_ROW_DISTANCE_LIMIT:
                    copy_sparse_pair_gm_to_ub(dst, kv, row, physical0, physical1)
                else:
                    mem_copy(
                        tile_slice(dst, (1, _D), (row, 0)),
                        tile_slice(kv, (1, _D), (physical0, 0)),
                    )
                    mem_copy(
                        tile_slice(dst, (1, _D), (row + 1, 0)),
                        tile_slice(kv, (1, _D), (physical1, 0)),
                    )
            elif row < tail_rows:
                physical0 = self._physical_row(
                    indices,
                    table,
                    batch,
                    query,
                    sparse_index_start + chunk_kv_row_start + row,
                    page_stride,
                    page_size,
                    linear_prefix,
                )
                mem_copy(
                    tile_slice(dst, (1, _D), (row, 0)),
                    tile_slice(kv, (1, _D), (physical0, 0)),
                )


@jit
def smla_actual_kv_length(lengths: Tensor, batch: Int64, length_mode: Int64, page_size: Int64):
    length = 0
    if length_mode == 0:
        length = dtypes.int64(lengths[batch])
    elif length_mode == 1:
        length = dtypes.int64(lengths[batch + 1]) - dtypes.int64(lengths[batch])
    else:
        length = -page_size
    return length


class SmlaZeroOutput:
    """Initialize only unused or empty query rows, without an auxiliary kernel."""

    def __init__(self, n1, dtype, mode):
        self.n1 = n1
        self.dtype = dtype
        self.mode = mode
        self.zero_row = Buffer(MemLoc.UB, (1, _D), dtype)

    @jit
    def write(self, out: Tensor, query: Int64):
        # VF zero output: fill one head row before copying it to each query head.
        with vf(mode="simd"):
            mask, _ = rr.update_mask(_BF16_VL, elem_bits=_F16_BITS)
            zero_reg = rr.vdups(0.0, self.dtype)
            for col in range(0, _D, _BF16_VL):
                rr.vstore(self.zero_row, col, zero_reg, mask)
        for head in range(self.n1):
            mem_copy(
                tile_slice(out, (1, _D), (query * self.n1 + head, 0)),
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
        ori_win_left: Int64,
        ori_win_right: Int64,
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
                query_batch_row_start = 0
                storage_length = 0
                if q_size == 0:
                    query_batch_row_start = dtypes.int64(cu[batch])
                    storage_length = dtypes.int64(cu[batch + 1]) - query_batch_row_start
                else:
                    query_batch_row_start = batch * q_size
                    storage_length = q_size
                if query >= query_batch_row_start and query < query_batch_row_start + storage_length:
                    valid_query_length = storage_length
                    if has_q_length != 0:
                        valid_query_length = dtypes.int64(q_lengths[batch])
                    query_pos_in_batch = query - query_batch_row_start
                    ori_len = smla_actual_kv_length(ori_lengths, batch, ori_length_mode, ori_page)
                    ori_kv_begin = 0
                    ori_kv_end = ori_len
                    if ori_mask == _MASK_CAUSAL:
                        ori_kv_end = max(
                            0,
                            min(
                                ori_len,
                                ori_len - valid_query_length + query_pos_in_batch + 1,
                            ),
                        )
                    if ori_mask == _MASK_SLIDING_WINDOW:
                        if ori_win_left != -1:
                            ori_kv_begin = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch - ori_win_left,
                                ),
                            )
                        if ori_win_right != -1:
                            ori_kv_end = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch + ori_win_right + 1,
                                ),
                            )
                    cmp_valid = 0
                    if const_expr(self.mode != "SWA"):
                        cmp_len = smla_actual_kv_length(cmp_lengths, batch, cmp_length_mode, cmp_page)
                        cmp_valid = cmp_len
                        if cmp_mask == _MASK_CAUSAL:
                            cmp_valid = min(
                                cmp_len,
                                max(
                                    0,
                                    (
                                        cmp_len * ratio
                                        + (0 if ratio == 1 else dtypes.int64(residual[batch]))
                                        - valid_query_length
                                        + query_pos_in_batch
                                        + 1
                                    )
                                    // ratio,
                                ),
                            )
                        if const_expr(self.mode == "CSA"):
                            if has_topk != 0:
                                cmp_valid = min(cmp_valid, max(0, dtypes.int64(topk[query])))
                            if dtypes.int32(indices[query, 0]) < 0:
                                cmp_valid = 0
                    if query_pos_in_batch >= valid_query_length or (ori_kv_begin >= ori_kv_end and cmp_valid == 0):
                        self.write(out, query)

    @jit
    def initialize_ori_sparse(
        self,
        out: Tensor,
        cu: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        q_lengths: Tensor,
        topk: Tensor,
        indices: Tensor,
        ori_topk: Tensor,
        ori_indices: Tensor,
        batch_count: Int64,
        q_size: Int64,
        has_q_length: Int64,
        has_topk: Int64,
        has_ori_topk: Int64,
        ori_mask: Int64,
        cmp_mask: Int64,
        ori_win_left: Int64,
        ori_win_right: Int64,
        ratio: Int64,
        block_dim: Int64,
        ori_length_mode: Int64,
        cmp_length_mode: Int64,
        ori_page: Int64,
        cmp_page: Int64,
    ):
        lane = get_block_idx() * 2 + get_subblock_id()
        for query in range(lane, out.shape[0] // dtypes.int64(self.n1), block_dim * 2):
            for batch in range(batch_count):
                query_batch_row_start = 0
                storage_length = 0
                if q_size == 0:
                    query_batch_row_start = dtypes.int64(cu[batch])
                    storage_length = dtypes.int64(cu[batch + 1]) - query_batch_row_start
                else:
                    query_batch_row_start = batch * q_size
                    storage_length = q_size
                if query >= query_batch_row_start and query < query_batch_row_start + storage_length:
                    valid_query_length = storage_length
                    if has_q_length != 0:
                        valid_query_length = dtypes.int64(q_lengths[batch])
                    query_pos_in_batch = query - query_batch_row_start
                    ori_len = smla_actual_kv_length(ori_lengths, batch, ori_length_mode, ori_page)
                    ori_kv_begin = 0
                    ori_kv_end = ori_len
                    if ori_mask == _MASK_CAUSAL:
                        ori_kv_end = max(
                            0,
                            min(
                                ori_len,
                                ori_len - valid_query_length + query_pos_in_batch + 1,
                            ),
                        )
                    if ori_mask == _MASK_SLIDING_WINDOW:
                        if ori_win_left != -1:
                            ori_kv_begin = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch - ori_win_left,
                                ),
                            )
                        if ori_win_right != -1:
                            ori_kv_end = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch + ori_win_right + 1,
                                ),
                            )
                    ori_valid = min(ori_indices.shape[1], max(0, ori_kv_end - ori_kv_begin))
                    if has_ori_topk != 0:
                        ori_valid = min(ori_valid, max(0, dtypes.int64(ori_topk[query])))
                    if dtypes.int32(ori_indices[query, 0]) < 0:
                        ori_valid = 0
                    cmp_valid = 0
                    if const_expr(self.mode == "ORI_CMP_SPARSE"):
                        cmp_len = smla_actual_kv_length(cmp_lengths, batch, cmp_length_mode, cmp_page)
                        cmp_valid = cmp_len
                        if cmp_mask == _MASK_CAUSAL:
                            cmp_valid = min(
                                cmp_len,
                                max(
                                    0,
                                    (
                                        cmp_len * ratio
                                        + dyn_select(ratio == 1, 0, dtypes.int64(residual[batch]))
                                        - valid_query_length
                                        + query_pos_in_batch
                                        + 1
                                    )
                                    // ratio,
                                ),
                            )
                        if has_topk != 0:
                            cmp_valid = min(cmp_valid, max(0, dtypes.int64(topk[query])))
                        if dtypes.int32(indices[query, 0]) < 0:
                            cmp_valid = 0
                    if query_pos_in_batch >= valid_query_length or (ori_valid == 0 and cmp_valid == 0):
                        self.write(out, query)


@kernel
class SparseFlashMlaCsaKernel:
    def __init__(self, n1, block_dim, dtype, has_sink=True):
        self.has_sink = has_sink
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.zero_output = SmlaZeroOutput(n1, dtype, "CSA")
        self.qk_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _TILE), dtypes.float32, depth=2)
        self.pv_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _D), dtypes.float32, depth=1)
        self.p_l1 = RingBuffer(MemLoc.L1, (_Q_TILE_ROWS, _TILE), dtype, depth=2)
        self.matmul = SmlaMatmul(_Q_TILE_ROWS, _VEC_TILE_M, _TILE, dtype)
        # N1=128 uses two Cube blocks per query. Both read the same KV
        # workspace group, so its buffer-release handshake is shared.
        self.matmul.shared_workspace = n1 == _MAX_QUERY_HEADS
        self.vector = GeneralSmlaVector(
            _VEC_TILE_M,
            _TILE,
            _D,
            get_subblock_id(),
            n_heads=n1,
            sinks_span=_VEC_TILE_M,
            use_sinks=True,
            rt_split=True,
            dtype=dtype,
        )
        self.vec0 = SmlaVec0(n1, dtype)
        self.vec0.shared_workspace = n1 == _MAX_QUERY_HEADS
        self.run_info = DelayLineGroup(
            _SPARSE_PIPELINE_SLOTS,
            "batch",
            "query_row_idx",
            "query_job_idx",
            "kv_tile_idx",
            "last_loop",
            "kv_segment_start",
            "valid_kv_rows",
            "cmp",
        )

    def __call__(
        self,
        out: Tensor,
        q: Tensor,
        kv_workspace: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        indices: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        cu_q: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        scale: dtypes.float32,
        ratio: dtypes.int64,
        ori_win_left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        ori_win_right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        load_ori = ori_kv_gm
        load_ori_page = ori_page

        kv_qk = self.matmul.kv_ring.next_read()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.next_read()
        q_half1 = self.matmul.q_ring.next_read()
        block = get_block_idx()
        metadata_base = block * _METADATA_FIELDS_PER_BLOCK
        # Head halves read the same KV values, but own independent workspace
        # slots so an intra-block handshake covers every writer and reader.
        group = block // 2 if const_expr(self.n1 == _MAX_QUERY_HEADS) else block
        query_head_half_idx = block % 2 if const_expr(self.n1 == _MAX_QUERY_HEADS) else 0
        self.vec0.initialize()
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
            ori_win_left,
            ori_win_right,
            ratio,
            self.block_dim,
            ori_length_mode,
            cmp_length_mode,
            ori_page,
            cmp_page,
        )
        tick = 0
        query_job_idx = 0
        if dtypes.int32(metadata[metadata_base]) != 0:
            first_batch = dtypes.int64(metadata[metadata_base + 1])
            first_query = dtypes.int64(metadata[metadata_base + 2])
            last_batch = dtypes.int64(metadata[metadata_base + _META_LAST_BATCH])
            last_query = dtypes.int64(metadata[metadata_base + _META_LAST_QUERY])
            for batch in range(first_batch, min(last_batch + 1, batch_count)):
                query_batch_row_start = 0
                valid_query_length = 0
                if q_size == 0:
                    query_batch_row_start = dtypes.int64(cu_q[batch])
                    valid_query_length = dtypes.int64(cu_q[batch + 1]) - query_batch_row_start
                else:
                    query_batch_row_start = batch * q_size
                    valid_query_length = q_size
                if has_q_length != 0:
                    valid_query_length = dtypes.int64(q_lengths[batch])
                query_begin = first_query if batch == first_batch else 0
                query_end = last_query if batch == last_batch else valid_query_length
                ori_len = smla_actual_kv_length(ori_lengths, batch, ori_length_mode, ori_page)
                cmp_len = smla_actual_kv_length(cmp_lengths, batch, cmp_length_mode, cmp_page)
                for query_pos_in_batch in range(query_begin, query_end):
                    query = query_batch_row_start + query_pos_in_batch
                    ori_start = 0
                    ori_end = ori_len
                    if ori_mask == _MASK_CAUSAL:
                        ori_end = max(
                            0,
                            min(
                                ori_len,
                                ori_len - valid_query_length + query_pos_in_batch + 1,
                            ),
                        )
                    if ori_mask == _MASK_SLIDING_WINDOW:
                        if ori_win_left != -1:
                            ori_start = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch - ori_win_left,
                                ),
                            )
                        if ori_win_right != -1:
                            ori_end = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch + ori_win_right + 1,
                                ),
                            )
                    ori_valid = max(0, ori_end - ori_start)
                    cmp_limit = min(
                        cmp_len,
                        max(
                            0,
                            (
                                cmp_len * ratio
                                + (0 if ratio == 1 else dtypes.int64(residual[batch]))
                                - valid_query_length
                                + query_pos_in_batch
                                + 1
                            )
                            // ratio,
                        ),
                    )
                    if cmp_mask == 0:
                        cmp_limit = cmp_len
                    cmp_valid = min(indices.shape[1], cmp_limit)
                    if has_topk != 0:
                        cmp_valid = min(cmp_valid, max(0, dtypes.int64(topk_length[query])))
                    ori_tiles = (ori_valid + (_TILE - 1)) // _TILE
                    cmp_tiles = (cmp_valid + (_TILE - 1)) // _TILE
                    kv_tile_count = ori_tiles + cmp_tiles
                    for kv_tile_idx in range(kv_tile_count):
                        is_cmp = kv_tile_idx >= ori_tiles
                        tile_index = kv_tile_idx - ori_tiles if is_cmp else kv_tile_idx
                        tile_start = tile_index * _TILE
                        kv_segment_start = tile_start if is_cmp else ori_start + tile_start
                        valid = cmp_valid if is_cmp else ori_valid
                        # Valid KV rows along the N axis in this sparse tile.
                        valid_kv_rows = _rt_min(_TILE, valid - tile_start)
                        self.run_info.push(
                            batch=batch,
                            query_row_idx=query,
                            query_job_idx=query_job_idx,
                            kv_tile_idx=kv_tile_idx,
                            last_loop=dtypes.int64(kv_tile_idx == kv_tile_count - 1),
                            kv_segment_start=kv_segment_start,
                            valid_kv_rows=valid_kv_rows,
                            cmp=dtypes.int64(is_cmp),
                        )
                        if is_cmp:
                            self.vec0.gather(
                                tile_slice(
                                    kv_workspace,
                                    (_TILE, _D),
                                    (group * _KV_RING + tick % _KV_RING, 0),
                                ),
                                cmp_kv_gm,
                                indices,
                                cmp_table,
                                batch,
                                query,
                                kv_segment_start,
                                valid_kv_rows,
                                tick % _KV_RING,
                                cmp_stride,
                                cmp_page,
                            )
                        kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                            tick,
                            tick + 1,
                            out,
                            q,
                            kv_workspace,
                            load_ori,
                            ori_table,
                            sinks,
                            scale,
                            ori_stride,
                            load_ori_page,
                            group,
                            query_head_half_idx,
                            kv_qk,
                            kv_pv,
                            q_half0,
                            q_half1,
                        )
                        tick = tick + 1
                    query_job_idx = query_job_idx + 1
            for drain in range(_STAGE_PV):
                kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                    tick + drain,
                    tick,
                    out,
                    q,
                    kv_workspace,
                    load_ori,
                    ori_table,
                    sinks,
                    scale,
                    ori_stride,
                    load_ori_page,
                    group,
                    query_head_half_idx,
                    kv_qk,
                    kv_pv,
                    q_half0,
                    q_half1,
                )
        if const_expr(self.n1 == _MAX_QUERY_HEADS):
            # Every Cube participates, including empty/short-running cores.
            for _ in range(tick, dtypes.int64(metadata[metadata_base + _META_SYNC_TICKS])):
                self.barrier()
        self.vec0.finish()

    @jit
    def barrier(self):
        # Public global rendezvous protects both Cube readers of shared GM.
        # Flag 8 is disjoint from RingBuffer flags and ready/free GM flags.
        if const_expr(self.n1 == _MAX_QUERY_HEADS):
            cube_sync_block_arrive(PIPE.MTE2, _WORKSPACE_BARRIER_FLAG, mode=0)
            cube_sync_block_wait(PIPE.MTE2, _WORKSPACE_BARRIER_FLAG, mode=0)

    @jit
    def stage_softmax(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        valid_kv_rows: Int64,
        sinks: Tensor,
        scale: dtypes.float32,
        query_head_half_idx: Int64,
    ):
        # Reuse MQSMLA's query-axis max/sum and task-axis exp banking.
        m_bank = query_job_idx % 2
        exp_bank = tick % 2
        score_ub_slot = self.qk_ub.next_read()
        score_ub = reinterpret(score_ub_slot, shape=(_VEC_TILE_M, _TILE), stride=(_TILE, 1))
        if kv_tile_idx == 0:
            sink_ub_write = self.vector.sinks_ub.next_write()
            if const_expr(self.has_sink):
                mem_copy(
                    sink_ub_write,
                    tile_slice(
                        sinks,
                        (_VEC_TILE_M,),
                        (query_head_half_idx * 2 + get_subblock_id(),),
                    ),
                )
            else:
                self.vector.init_no_sink_buffer(sink_ub_write)
            sink_ub_read = self.vector.sinks_ub.next_read()
            self.vector.seed_from_sinks(sink_ub_read, m_bank)
        self.vector.softmax_rest(score_ub, scale, m_bank, exp_bank, valid_kv_rows)
        self.vector.store_p(self.p_l1)

    @jit
    def stage_update(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        last_loop: Int64,
        query: Int64,
        out: Tensor,
        query_head_half_idx: Int64,
    ):
        pv = self.pv_ub.next_read()
        sum_idx = query_job_idx % 2
        exp_idx = tick % 2
        if kv_tile_idx == 0:
            if last_loop != 0:
                self.vector.init_o_last(pv, sum_idx)
            else:
                self.vector.init_o(pv)
        else:
            if last_loop != 0:
                self.vector.update_o_last(pv, exp_idx, sum_idx)
            else:
                self.vector.update_o(pv, exp_idx)
        if last_loop != 0:
            tile = tile_slice(
                out,
                (self.vector.tile_m, self.vector.tile_d),
                (query * (self.n1 // self.vector.tile_m) + query_head_half_idx, 0),
            )
            self.vector.finalize_o(tile, sum_idx, div_done=True)

    @jit
    def pipeline(
        self,
        tick: Int64,
        kv_tile_count: Int64,
        out: Tensor,
        q: Tensor,
        kv_workspace: Tensor,
        ori_kv_gm: Tensor,
        table: Tensor,
        sinks: Tensor,
        scale: dtypes.float32,
        page_stride: Int64,
        ori_page: Int64,
        group: Int64,
        query_head_half_idx: Int64,
        kv_qk: Tensor,
        kv_pv: Tensor,
        q_half0: Tensor,
        q_half1: Tensor,
    ):
        next_pv = kv_pv
        if tick >= _STAGE_LOAD and tick < kv_tile_count + _STAGE_LOAD:
            info = self.run_info.tap(1)
            load_tick = tick - 1
            if load_tick == 0:
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            info.query_row_idx * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    info.query_job_idx,
                )
            if info.cmp != 0:
                self.vec0.wait()
            self.barrier()
            self.matmul.load_kv(
                tile_slice(
                    kv_workspace,
                    (_TILE, _D),
                    (group * _KV_RING + load_tick % _KV_RING, 0),
                ),
                ori_kv_gm,
                table,
                info.batch,
                info.kv_segment_start,
                info.valid_kv_rows,
                page_stride,
                load_tick % _KV_RING,
                info.cmp,
                ori_page,
            )
        if tick >= _STAGE_QK and tick < kv_tile_count + _STAGE_QK:
            info = self.run_info.tap(2)
            qk_tick = tick - 2
            self.matmul.bmm1_fanout_q(kv_qk, q_half0, q_half1, actual_n=info.valid_kv_rows)
            next_pv = kv_qk
            kv_qk = self.matmul.kv_ring.next_read()
            if info.last_loop != 0:
                q_half0 = self.matmul.q_ring.next_read()
                q_half1 = self.matmul.q_ring.next_read()
            if info.last_loop != 0 and tick < kv_tile_count + _STAGE_LOAD:
                next_info = self.run_info.tap(1)
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            next_info.query_row_idx * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    next_info.query_job_idx,
                )
            self.matmul.store_s(self.qk_ub)
            self.stage_softmax(
                qk_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.valid_kv_rows,
                sinks,
                scale,
                query_head_half_idx,
            )
        if tick >= _STAGE_PV and tick < kv_tile_count + _STAGE_PV:
            info = self.run_info.tap(_STAGE_PV)
            pv_tick = tick - _STAGE_PV
            self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub, kv_pv, actual_n=info.valid_kv_rows)
            self.stage_update(
                pv_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.last_loop,
                info.query_row_idx,
                out,
                query_head_half_idx,
            )
        self.run_info.advance()
        return kv_qk, next_pv, q_half0, q_half1


@kernel
class SparseFlashMlaSwaHcaKernel:
    """A5 SWA/HCA: direct PA Load(t), QK/Vec1(t-1), PV/Vec2(t-2)."""

    def __init__(self, n1, block_dim, mode, dtype, has_sink=True):
        self.has_sink = has_sink
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.mode = mode
        self.zero_output = SmlaZeroOutput(n1, dtype, mode)
        self.qk_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _TILE), dtypes.float32, depth=2)
        self.pv_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _D), dtypes.float32, depth=1)
        self.p_l1 = RingBuffer(MemLoc.L1, (_Q_TILE_ROWS, _TILE), dtype, depth=2)
        self.matmul = SmlaMatmul(_Q_TILE_ROWS, _VEC_TILE_M, _TILE, dtype)
        self.vector = GeneralSmlaVector(
            _VEC_TILE_M,
            _TILE,
            _D,
            get_subblock_id(),
            n_heads=n1,
            sinks_span=_VEC_TILE_M,
            use_sinks=True,
            rt_split=True,
            dtype=dtype,
        )
        self.run_info = DelayLineGroup(
            _INLINE_PIPELINE_SLOTS,
            "batch",
            "query_row_idx",
            "query_job_idx",
            "kv_tile_idx",
            "last_loop",
            "kv_segment_start",
            "valid_kv_rows",
            "cmp",
        )

    def __call__(
        self,
        out: Tensor,
        q: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
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
        ori_win_left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        ori_win_right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        load_cmp = cmp_kv_gm
        load_cmp_page = cmp_page
        load_ori = ori_kv_gm
        load_ori_page = ori_page
        kv_qk = self.matmul.kv_ring.next_read()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.next_read()
        q_half1 = self.matmul.q_ring.next_read()
        block = get_block_idx()
        metadata_base = block * _METADATA_FIELDS_PER_BLOCK
        query_head_half_idx = block % 2 if const_expr(self.n1 == _MAX_QUERY_HEADS) else 0
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
            ori_win_left,
            ori_win_right,
            ratio,
            self.block_dim,
            ori_length_mode,
            cmp_length_mode,
            ori_page,
            cmp_page,
        )
        tick = 0
        query_job_idx = 0
        if dtypes.int32(metadata[metadata_base]) != 0:
            first_batch = dtypes.int64(metadata[metadata_base + 1])
            first_query = dtypes.int64(metadata[metadata_base + 2])
            last_batch = dtypes.int64(metadata[metadata_base + _META_LAST_BATCH])
            last_query = dtypes.int64(metadata[metadata_base + _META_LAST_QUERY])
            for batch in range(first_batch, min(last_batch + 1, batch_count)):
                query_batch_row_start = 0
                valid_query_length = 0
                if q_size == 0:
                    query_batch_row_start = dtypes.int64(cu_q[batch])
                    valid_query_length = dtypes.int64(cu_q[batch + 1]) - query_batch_row_start
                else:
                    query_batch_row_start = batch * q_size
                    valid_query_length = q_size
                if has_q_length != 0:
                    valid_query_length = dtypes.int64(q_lengths[batch])
                query_begin = first_query if batch == first_batch else 0
                query_end = last_query if batch == last_batch else valid_query_length
                ori_len = smla_actual_kv_length(ori_lengths, batch, ori_length_mode, ori_page)
                for query_pos_in_batch in range(query_begin, query_end):
                    query = query_batch_row_start + query_pos_in_batch
                    ori_start = 0
                    ori_end = ori_len
                    if ori_mask == _MASK_CAUSAL:
                        ori_end = max(
                            0,
                            min(
                                ori_len,
                                ori_len - valid_query_length + query_pos_in_batch + 1,
                            ),
                        )
                    if ori_mask == _MASK_SLIDING_WINDOW:
                        if ori_win_left != -1:
                            ori_start = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch - ori_win_left,
                                ),
                            )
                        if ori_win_right != -1:
                            ori_end = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch + ori_win_right + 1,
                                ),
                            )
                    ori_valid = max(0, ori_end - ori_start)
                    cmp_valid = 0
                    if const_expr(self.mode == "HCA"):
                        cmp_len = smla_actual_kv_length(cmp_lengths, batch, cmp_length_mode, cmp_page)
                        cmp_valid = cmp_len
                        if cmp_mask == _MASK_CAUSAL:
                            cmp_valid = min(
                                cmp_len,
                                max(
                                    0,
                                    (
                                        cmp_len * ratio
                                        + (0 if ratio == 1 else dtypes.int64(residual[batch]))
                                        - valid_query_length
                                        + query_pos_in_batch
                                        + 1
                                    )
                                    // ratio,
                                ),
                            )
                    ori_tiles = (ori_valid + (_TILE - 1)) // _TILE
                    cmp_tiles = (cmp_valid + (_TILE - 1)) // _TILE
                    kv_tile_count = ori_tiles + cmp_tiles
                    for kv_tile_idx in range(kv_tile_count):
                        is_cmp = kv_tile_idx >= ori_tiles
                        tile_index = kv_tile_idx - ori_tiles if is_cmp else kv_tile_idx
                        tile_start = tile_index * _TILE
                        kv_segment_start = tile_start if is_cmp else ori_start + tile_start
                        valid = cmp_valid if is_cmp else ori_valid
                        # Valid KV rows along the N axis in this sparse tile.
                        valid_kv_rows = _rt_min(_TILE, valid - tile_start)
                        self.run_info.push(
                            batch=batch,
                            query_row_idx=query,
                            query_job_idx=query_job_idx,
                            kv_tile_idx=kv_tile_idx,
                            last_loop=dtypes.int64(kv_tile_idx == kv_tile_count - 1),
                            kv_segment_start=kv_segment_start,
                            valid_kv_rows=valid_kv_rows,
                            cmp=dtypes.int64(is_cmp),
                        )
                        kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                            tick,
                            tick + 1,
                            out,
                            q,
                            load_ori,
                            load_cmp,
                            ori_table,
                            cmp_table,
                            sinks,
                            scale,
                            ori_stride,
                            cmp_stride,
                            load_ori_page,
                            load_cmp_page,
                            query_head_half_idx,
                            batch,
                            query,
                            query_job_idx,
                            kv_segment_start,
                            valid_kv_rows,
                            dtypes.int64(is_cmp),
                            kv_qk,
                            kv_pv,
                            q_half0,
                            q_half1,
                        )
                        tick = tick + 1
                    query_job_idx = query_job_idx + 1
            for drain in range(_STAGE_PV_INLINE):
                kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                    tick + drain,
                    tick,
                    out,
                    q,
                    load_ori,
                    load_cmp,
                    ori_table,
                    cmp_table,
                    sinks,
                    scale,
                    ori_stride,
                    cmp_stride,
                    load_ori_page,
                    load_cmp_page,
                    query_head_half_idx,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    kv_qk,
                    kv_pv,
                    q_half0,
                    q_half1,
                )

    @jit
    def stage_softmax(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        valid_kv_rows: Int64,
        sinks: Tensor,
        scale: dtypes.float32,
        query_head_half_idx: Int64,
    ):
        # Reuse MQSMLA's query-axis max/sum and task-axis exp banking.
        m_bank = query_job_idx % 2
        exp_bank = tick % 2
        score_ub_slot = self.qk_ub.next_read()
        score_ub = reinterpret(score_ub_slot, shape=(_VEC_TILE_M, _TILE), stride=(_TILE, 1))
        if kv_tile_idx == 0:
            sink_ub_write = self.vector.sinks_ub.next_write()
            if const_expr(self.has_sink):
                mem_copy(
                    sink_ub_write,
                    tile_slice(
                        sinks,
                        (_VEC_TILE_M,),
                        (query_head_half_idx * 2 + get_subblock_id(),),
                    ),
                )
            else:
                self.vector.init_no_sink_buffer(sink_ub_write)
            sink_ub_read = self.vector.sinks_ub.next_read()
            self.vector.seed_from_sinks(sink_ub_read, m_bank)
        self.vector.softmax_rest(score_ub, scale, m_bank, exp_bank, valid_kv_rows)
        self.vector.store_p(self.p_l1)

    @jit
    def stage_update(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        last_loop: Int64,
        query: Int64,
        out: Tensor,
        query_head_half_idx: Int64,
    ):
        pv = self.pv_ub.next_read()
        sum_idx = query_job_idx % 2
        exp_idx = tick % 2
        if kv_tile_idx == 0:
            if last_loop != 0:
                self.vector.init_o_last(pv, sum_idx)
            else:
                self.vector.init_o(pv)
        else:
            if last_loop != 0:
                self.vector.update_o_last(pv, exp_idx, sum_idx)
            else:
                self.vector.update_o(pv, exp_idx)
        if last_loop != 0:
            tile = tile_slice(
                out,
                (self.vector.tile_m, self.vector.tile_d),
                (query * (self.n1 // self.vector.tile_m) + query_head_half_idx, 0),
            )
            self.vector.finalize_o(tile, sum_idx, div_done=True)

    @jit
    def pipeline(
        self,
        tick: Int64,
        kv_tile_count: Int64,
        out: Tensor,
        q: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        sinks: Tensor,
        scale: dtypes.float32,
        ori_stride: Int64,
        cmp_stride: Int64,
        ori_page: Int64,
        cmp_page: Int64,
        query_head_half_idx: Int64,
        current_batch: Int64,
        current_query: Int64,
        current_query_job_idx: Int64,
        current_kv_segment_start: Int64,
        current_valid_kv_rows: Int64,
        current_cmp: Int64,
        kv_qk: Tensor,
        kv_pv: Tensor,
        q_half0: Tensor,
        q_half1: Tensor,
    ):
        next_pv = kv_pv
        if tick < kv_tile_count:
            if tick == 0:
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            current_query * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    current_query_job_idx,
                )
            if current_cmp != 0:
                self.matmul.load_kv(
                    cmp_kv_gm,
                    cmp_kv_gm,
                    cmp_table,
                    current_batch,
                    current_kv_segment_start,
                    current_valid_kv_rows,
                    cmp_stride,
                    tick % _KV_RING,
                    0,
                    cmp_page,
                )
            else:
                self.matmul.load_kv(
                    ori_kv_gm,
                    ori_kv_gm,
                    ori_table,
                    current_batch,
                    current_kv_segment_start,
                    current_valid_kv_rows,
                    ori_stride,
                    tick % _KV_RING,
                    0,
                    ori_page,
                )
        if tick >= _STAGE_QK_INLINE and tick < kv_tile_count + _STAGE_QK_INLINE:
            info = self.run_info.tap(1)
            qk_tick = tick - 1
            self.matmul.bmm1_fanout_q(kv_qk, q_half0, q_half1, actual_n=info.valid_kv_rows)
            next_pv = kv_qk
            kv_qk = self.matmul.kv_ring.next_read()
            if info.last_loop != 0:
                q_half0 = self.matmul.q_ring.next_read()
                q_half1 = self.matmul.q_ring.next_read()
            if info.last_loop != 0 and tick < kv_tile_count:
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            current_query * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    current_query_job_idx,
                )
            self.matmul.store_s(self.qk_ub)
            self.stage_softmax(
                qk_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.valid_kv_rows,
                sinks,
                scale,
                query_head_half_idx,
            )
        if tick >= _STAGE_PV_INLINE and tick < kv_tile_count + _STAGE_PV_INLINE:
            info = self.run_info.tap(2)
            pv_tick = tick - 2
            self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub, kv_pv, actual_n=info.valid_kv_rows)
            self.stage_update(
                pv_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.last_loop,
                info.query_row_idx,
                out,
                query_head_half_idx,
            )
        self.run_info.advance()
        return kv_qk, next_pv, q_half0, q_half1


class SparseFlashMlaDirectLauncher:
    def __init__(self, n1, block_dim, mode, dtype, has_sink=True):
        self.has_sink = has_sink
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.mode = mode

    @host
    def run(
        self,
        out: Tensor,
        q: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
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
        ori_win_left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        ori_win_right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        SparseFlashMlaSwaHcaKernel(self.n1, self.block_dim, self.mode, self.dtype, self.has_sink)[self.block_dim](
            out,
            q,
            ori_kv_gm,
            cmp_kv_gm,
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
            ori_win_left,
            ori_stride,
            cmp_stride,
            ori_page,
            cmp_page,
            ori_mask,
            cmp_mask,
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


@functools.cache
def _compile_smla_direct(n1, block_dim, mode, dtype, has_sink=True):
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
        TensorSpec((token_idx, _D), dtype),
        TensorSpec((token_idx, _D), dtype),
        TensorSpec((ori_capacity, _D), dtype),
        TensorSpec((cmp_capacity, _D), dtype),
        TensorSpec((ori_table_rows, ori_table_width), i32),
        TensorSpec((cmp_table_rows, cmp_table_width), i32),
        TensorSpec((n1 if has_sink else 1,), dtypes.float32),
        TensorSpec((_METADATA_ELEMENTS,), i32),
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
    return cannbotdsl.compile(SparseFlashMlaDirectLauncher(n1, block_dim, mode, dtype, has_sink).run, *specs)


class SparseFlashMlaCsaLauncher:
    def __init__(self, n1, block_dim, dtype, has_sink=True):
        self.has_sink = has_sink
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim

    @host
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
        ori_win_right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        SparseFlashMlaCsaKernel(self.n1, self.block_dim, self.dtype, self.has_sink)[self.block_dim](
            bmm_result,
            query_gm,
            v0_res_gm,
            ori_kv_gm,
            cmp_kv_gm,
            block_table_gm,
            cmp_block_table_gm,
            sparse_indices_gm,
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


@functools.cache
def _compile_smla(n1, block_dim, dtype, has_sink=True):
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
        TensorSpec((token_idx * n1, _D), dtype),
        TensorSpec((token_idx * n1, _D), dtype),
        TensorSpec(
            (
                (block_dim // 2 if n1 == _MAX_QUERY_HEADS else block_dim) * _KV_RING * _TILE,
                _D,
            ),
            dtype,
        ),
        TensorSpec((ori_capacity, _D), dtype),
        TensorSpec((cmp_capacity, _D), dtype),
        TensorSpec((ori_table_rows, ori_table_width), i32),
        TensorSpec((cmp_table_rows, cmp_table_width), i32),
        TensorSpec((token_idx, sparse_slot_idx), i32),
        TensorSpec((n1 if has_sink else 1,), dtypes.float32),
        TensorSpec((_METADATA_ELEMENTS,), i32),
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
    return cannbotdsl.compile(SparseFlashMlaCsaLauncher(n1, block_dim, dtype, has_sink).run, *specs)


@kernel
class SparseFlashMlaOriSparseKernel:
    def __init__(self, n1, block_dim, mode, dtype, has_sink=True):
        self.has_sink = has_sink
        self.mode = mode
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.zero_output = SmlaZeroOutput(n1, dtype, mode)
        self.qk_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _TILE), dtypes.float32, depth=2)
        self.pv_ub = RingBuffer(MemLoc.UB, (_VEC_TILE_M, _D), dtypes.float32, depth=1)
        self.p_l1 = RingBuffer(MemLoc.L1, (_Q_TILE_ROWS, _TILE), dtype, depth=2)
        self.matmul = SmlaMatmul(_Q_TILE_ROWS, _VEC_TILE_M, _TILE, dtype)
        # N1=128 uses two Cube blocks per query. Both read the same KV
        # workspace group, so its buffer-release handshake is shared.
        self.matmul.shared_workspace = n1 == _MAX_QUERY_HEADS
        self.vector = GeneralSmlaVector(
            _VEC_TILE_M,
            _TILE,
            _D,
            get_subblock_id(),
            n_heads=n1,
            sinks_span=_VEC_TILE_M,
            use_sinks=True,
            rt_split=True,
            dtype=dtype,
        )
        self.vec0 = SmlaVec0(n1, dtype)
        self.vec0.shared_workspace = n1 == _MAX_QUERY_HEADS
        self.run_info = DelayLineGroup(
            _SPARSE_PIPELINE_SLOTS,
            "batch",
            "query_row_idx",
            "query_job_idx",
            "kv_tile_idx",
            "last_loop",
            "kv_segment_start",
            "valid_kv_rows",
            "cmp",
        )

    def __call__(
        self,
        out,
        q,
        kv_workspace,
        ori_kv_gm,
        cmp_kv_gm,
        ori_table,
        cmp_table,
        ori_indices,
        indices,
        sinks,
        metadata,
        cu_q,
        ori_lengths,
        cmp_lengths,
        residual,
        scale,
        ratio,
        ori_win_left,
        ori_stride,
        cmp_stride,
        ori_page,
        cmp_page,
        ori_mask,
        cmp_mask,
        ori_win_right,
        q_size,
        q_lengths,
        ori_topk_length,
        topk_length,
        has_q_length,
        has_ori_topk,
        has_topk,
        batch_count,
        ori_length_mode,
        cmp_length_mode,
    ):
        kv_qk = self.matmul.kv_ring.next_read()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.next_read()
        q_half1 = self.matmul.q_ring.next_read()
        block = get_block_idx()
        metadata_base = block * _METADATA_FIELDS_PER_BLOCK
        group = block // 2 if const_expr(self.n1 == _MAX_QUERY_HEADS) else block
        query_head_half_idx = block % 2 if const_expr(self.n1 == _MAX_QUERY_HEADS) else 0
        self.vec0.initialize()
        self.zero_output.initialize_ori_sparse(
            out,
            cu_q,
            ori_lengths,
            cmp_lengths,
            residual,
            q_lengths,
            topk_length,
            indices,
            ori_topk_length,
            ori_indices,
            batch_count,
            q_size,
            has_q_length,
            has_topk,
            has_ori_topk,
            ori_mask,
            cmp_mask,
            ori_win_left,
            ori_win_right,
            ratio,
            dtypes.int64(self.block_dim),
            ori_length_mode,
            cmp_length_mode,
            ori_page,
            cmp_page,
        )
        tick = 0
        query_job_idx = 0
        if dtypes.int32(metadata[metadata_base]) != 0:
            first_batch = dtypes.int64(metadata[metadata_base + 1])
            first_query = dtypes.int64(metadata[metadata_base + 2])
            last_batch = dtypes.int64(metadata[metadata_base + _META_LAST_BATCH])
            last_query = dtypes.int64(metadata[metadata_base + _META_LAST_QUERY])
            for batch in range(first_batch, min(last_batch + 1, batch_count)):
                query_batch_row_start = 0
                valid_query_length = 0
                if q_size == 0:
                    query_batch_row_start = dtypes.int64(cu_q[batch])
                    valid_query_length = dtypes.int64(cu_q[batch + 1]) - query_batch_row_start
                else:
                    query_batch_row_start = batch * q_size
                    valid_query_length = q_size
                if has_q_length != 0:
                    valid_query_length = dtypes.int64(q_lengths[batch])
                query_begin = dyn_select(batch == first_batch, first_query, 0)
                query_end = dyn_select(batch == last_batch, last_query, valid_query_length)
                ori_len = smla_actual_kv_length(ori_lengths, batch, ori_length_mode, ori_page)
                cmp_len = smla_actual_kv_length(cmp_lengths, batch, cmp_length_mode, cmp_page)
                for query_pos_in_batch in range(query_begin, query_end):
                    query = query_batch_row_start + query_pos_in_batch
                    ori_begin = 0
                    ori_stop = ori_len
                    if ori_mask == _MASK_CAUSAL:
                        ori_stop = max(
                            0,
                            min(
                                ori_len,
                                ori_len - valid_query_length + query_pos_in_batch + 1,
                            ),
                        )
                    if ori_mask == _MASK_SLIDING_WINDOW:
                        if ori_win_left != -1:
                            ori_begin = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch - ori_win_left,
                                ),
                            )
                        if ori_win_right != -1:
                            ori_stop = max(
                                0,
                                min(
                                    ori_len,
                                    ori_len - valid_query_length + query_pos_in_batch + ori_win_right + 1,
                                ),
                            )
                    ori_valid = min(ori_indices.shape[1], max(0, ori_stop - ori_begin))
                    if has_ori_topk != 0:
                        ori_valid = min(ori_valid, max(0, dtypes.int64(ori_topk_length[query])))
                    if dtypes.int32(ori_indices[query, 0]) < 0:
                        ori_valid = 0
                    cmp_valid = 0
                    if const_expr(self.mode == "ORI_CMP_SPARSE"):
                        cmp_limit = min(
                            cmp_len,
                            max(
                                0,
                                (
                                    cmp_len * ratio
                                    + dyn_select(ratio == 1, 0, dtypes.int64(residual[batch]))
                                    - valid_query_length
                                    + query_pos_in_batch
                                    + 1
                                )
                                // ratio,
                            ),
                        )
                        if cmp_mask == 0:
                            cmp_limit = cmp_len
                        cmp_valid = min(indices.shape[1], cmp_limit)
                        if has_topk != 0:
                            cmp_valid = min(cmp_valid, max(0, dtypes.int64(topk_length[query])))
                        if dtypes.int32(indices[query, 0]) < 0:
                            cmp_valid = 0
                    ori_tiles = (ori_valid + (_TILE - 1)) // _TILE
                    cmp_tiles = (cmp_valid + (_TILE - 1)) // _TILE
                    kv_tile_count = ori_tiles + cmp_tiles
                    for kv_tile_idx in range(kv_tile_count):
                        is_cmp = kv_tile_idx >= ori_tiles
                        kv_segment_start = dyn_select(
                            is_cmp,
                            (kv_tile_idx - ori_tiles) * _TILE,
                            kv_tile_idx * _TILE,
                        )
                        valid_kv_rows = dyn_select(
                            is_cmp,
                            min(_TILE, cmp_valid - (kv_tile_idx - ori_tiles) * _TILE),
                            min(_TILE, ori_valid - kv_tile_idx * _TILE),
                        )
                        self.run_info.push(
                            batch=batch,
                            query_row_idx=query,
                            query_job_idx=query_job_idx,
                            kv_tile_idx=kv_tile_idx,
                            last_loop=dtypes.int64(kv_tile_idx == kv_tile_count - 1),
                            kv_segment_start=kv_segment_start,
                            valid_kv_rows=valid_kv_rows,
                            cmp=dtypes.int64(is_cmp),
                        )
                        if const_expr(self.mode == "ORI_CMP_SPARSE"):
                            if is_cmp:
                                self.vec0.gather_tile(
                                    tile_slice(
                                        kv_workspace,
                                        (_TILE, _D),
                                        (group * _KV_RING + tick % _KV_RING, 0),
                                    ),
                                    cmp_kv_gm,
                                    indices,
                                    cmp_table,
                                    batch,
                                    query,
                                    kv_segment_start,
                                    valid_kv_rows,
                                    tick % _KV_RING,
                                    cmp_stride,
                                    cmp_page,
                                )
                            else:
                                self.vec0.gather_tile(
                                    tile_slice(
                                        kv_workspace,
                                        (_TILE, _D),
                                        (group * _KV_RING + tick % _KV_RING, 0),
                                    ),
                                    ori_kv_gm,
                                    ori_indices,
                                    ori_table,
                                    batch,
                                    query,
                                    kv_segment_start,
                                    valid_kv_rows,
                                    tick % _KV_RING,
                                    ori_stride,
                                    ori_page,
                                )
                        else:
                            self.vec0.gather_tile(
                                tile_slice(
                                    kv_workspace,
                                    (_TILE, _D),
                                    (group * _KV_RING + tick % _KV_RING, 0),
                                ),
                                ori_kv_gm,
                                ori_indices,
                                ori_table,
                                batch,
                                query,
                                kv_segment_start,
                                valid_kv_rows,
                                tick % _KV_RING,
                                ori_stride,
                                ori_page,
                            )
                        kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                            tick,
                            tick + 1,
                            out,
                            q,
                            kv_workspace,
                            ori_kv_gm,
                            ori_table,
                            sinks,
                            scale,
                            0,
                            0,
                            group,
                            query_head_half_idx,
                            kv_qk,
                            kv_pv,
                            q_half0,
                            q_half1,
                        )
                        tick += 1
                    query_job_idx += 1
            for drain in range(_STAGE_PV):
                kv_qk, kv_pv, q_half0, q_half1 = self.pipeline(
                    tick + drain,
                    tick,
                    out,
                    q,
                    kv_workspace,
                    ori_kv_gm,
                    ori_table,
                    sinks,
                    scale,
                    0,
                    0,
                    group,
                    query_head_half_idx,
                    kv_qk,
                    kv_pv,
                    q_half0,
                    q_half1,
                )
        if const_expr(self.n1 == _MAX_QUERY_HEADS):
            # Match CSA: empty and shorter cores must join every rendezvous.
            for _ in range(tick, dtypes.int64(metadata[metadata_base + _META_SYNC_TICKS])):
                self.barrier()
        self.vec0.finish()

    @jit
    def barrier(self):
        # Public global rendezvous protects both Cube readers of shared GM.
        # Flag 8 is disjoint from RingBuffer flags and ready/free GM flags.
        if const_expr(self.n1 == _MAX_QUERY_HEADS):
            cube_sync_block_arrive(PIPE.MTE2, _WORKSPACE_BARRIER_FLAG, mode=0)
            cube_sync_block_wait(PIPE.MTE2, _WORKSPACE_BARRIER_FLAG, mode=0)

    @jit
    def stage_softmax(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        valid_kv_rows: Int64,
        sinks: Tensor,
        scale: dtypes.float32,
        query_head_half_idx: Int64,
    ):
        m_bank = query_job_idx % 2
        exp_bank = tick % 2
        score_ub_slot = self.qk_ub.next_read()
        score_ub = reinterpret(score_ub_slot, shape=(_VEC_TILE_M, _TILE), stride=(_TILE, 1))
        if kv_tile_idx == 0:
            sink_ub_write = self.vector.sinks_ub.next_write()
            if const_expr(self.has_sink):
                mem_copy(
                    sink_ub_write,
                    tile_slice(
                        sinks,
                        (_VEC_TILE_M,),
                        (query_head_half_idx * 2 + get_subblock_id(),),
                    ),
                )
            else:
                self.vector.init_no_sink_buffer(sink_ub_write)
            sink_ub_read = self.vector.sinks_ub.next_read()
            self.vector.seed_from_sinks(sink_ub_read, m_bank)
        self.vector.softmax_rest(score_ub, scale, m_bank, exp_bank, valid_kv_rows)
        self.vector.store_p(self.p_l1)

    @jit
    def stage_update(
        self,
        tick: Int64,
        query_job_idx: Int64,
        kv_tile_idx: Int64,
        last_loop: Int64,
        query: Int64,
        out: Tensor,
        query_head_half_idx: Int64,
    ):
        pv = self.pv_ub.next_read()
        sum_idx = query_job_idx % 2
        exp_idx = tick % 2
        if kv_tile_idx == 0:
            if last_loop != 0:
                self.vector.init_o_last(pv, sum_idx)
            else:
                self.vector.init_o(pv)
        elif last_loop != 0:
            self.vector.update_o_last(pv, exp_idx, sum_idx)
        else:
            self.vector.update_o(pv, exp_idx)
        if last_loop != 0:
            tile = tile_slice(
                out,
                (self.vector.tile_m, self.vector.tile_d),
                (query * (self.n1 // self.vector.tile_m) + query_head_half_idx, 0),
            )
            self.vector.finalize_o(tile, sum_idx, div_done=True)

    @jit
    def pipeline(
        self,
        tick: Int64,
        kv_tile_count: Int64,
        out: Tensor,
        q: Tensor,
        kv_workspace: Tensor,
        ori_kv_gm: Tensor,
        table: Tensor,
        sinks: Tensor,
        scale: dtypes.float32,
        page_stride: Int64,
        ori_page: Int64,
        group: Int64,
        query_head_half_idx: Int64,
        kv_qk: Tensor,
        kv_pv: Tensor,
        q_half0: Tensor,
        q_half1: Tensor,
    ):
        next_pv = kv_pv
        if tick >= _STAGE_LOAD and tick < kv_tile_count + _STAGE_LOAD:
            info = self.run_info.tap(1)
            load_tick = tick - 1
            if load_tick == 0:
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            info.query_row_idx * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    info.query_job_idx,
                )
            self.vec0.wait()
            self.barrier()
            self.matmul.load_kv(
                tile_slice(
                    kv_workspace,
                    (_TILE, _D),
                    (group * _KV_RING + load_tick % _KV_RING, 0),
                ),
                ori_kv_gm,
                table,
                info.batch,
                info.kv_segment_start,
                info.valid_kv_rows,
                page_stride,
                load_tick % _KV_RING,
                1,
                ori_page,
            )
        if tick >= _STAGE_QK and tick < kv_tile_count + _STAGE_QK:
            info = self.run_info.tap(2)
            qk_tick = tick - 2
            self.matmul.bmm1_fanout_q(kv_qk, q_half0, q_half1, actual_n=info.valid_kv_rows)
            next_pv = kv_qk
            kv_qk = self.matmul.kv_ring.next_read()
            if info.last_loop != 0:
                q_half0 = self.matmul.q_ring.next_read()
                q_half1 = self.matmul.q_ring.next_read()
            if info.last_loop != 0 and tick < kv_tile_count + _STAGE_LOAD:
                next_info = self.run_info.tap(1)
                self.matmul.load_q_wide(
                    tile_slice(
                        q,
                        (_Q_TILE_ROWS, _D),
                        (
                            next_info.query_row_idx * self.n1 // _Q_TILE_ROWS + query_head_half_idx,
                            0,
                        ),
                    ),
                    next_info.query_job_idx,
                )
            self.matmul.store_s(self.qk_ub)
            self.stage_softmax(
                qk_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.valid_kv_rows,
                sinks,
                scale,
                query_head_half_idx,
            )
        if tick >= _STAGE_PV and tick < kv_tile_count + _STAGE_PV:
            info = self.run_info.tap(_STAGE_PV)
            pv_tick = tick - _STAGE_PV
            self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub, kv_pv, actual_n=info.valid_kv_rows)
            self.stage_update(
                pv_tick,
                info.query_job_idx,
                info.kv_tile_idx,
                info.last_loop,
                info.query_row_idx,
                out,
                query_head_half_idx,
            )
        self.run_info.advance()
        return (kv_qk, next_pv, q_half0, q_half1)


class SparseFlashMlaOriSparseLauncher:
    def __init__(self, n1, block_dim, mode, dtype, has_sink=True):
        self.has_sink = has_sink
        self.dtype = dtype
        self.n1 = n1
        self.block_dim = block_dim
        self.mode = mode

    @host
    def run(
        self,
        out: Tensor,
        q: Tensor,
        kv_workspace: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        ori_table: Tensor,
        cmp_table: Tensor,
        ori_indices: Tensor,
        indices: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        cu_q: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        residual: Tensor,
        scale: dtypes.float32,
        ratio: dtypes.int64,
        ori_win_left: dtypes.int64,
        ori_stride: dtypes.int64,
        cmp_stride: dtypes.int64,
        ori_page: dtypes.int64,
        cmp_page: dtypes.int64,
        ori_mask: dtypes.int64,
        cmp_mask: dtypes.int64,
        ori_win_right: dtypes.int64,
        q_size: dtypes.int64,
        q_lengths: Tensor,
        ori_topk_length: Tensor,
        topk_length: Tensor,
        has_q_length: dtypes.int64,
        has_ori_topk: dtypes.int64,
        has_topk: dtypes.int64,
        batch_count: dtypes.int64,
        ori_length_mode: dtypes.int64,
        cmp_length_mode: dtypes.int64,
    ):
        SparseFlashMlaOriSparseKernel(self.n1, self.block_dim, self.mode, self.dtype, self.has_sink)[self.block_dim](
            out,
            q,
            kv_workspace,
            ori_kv_gm,
            cmp_kv_gm,
            ori_table,
            cmp_table,
            ori_indices,
            indices,
            sinks,
            metadata,
            cu_q,
            ori_lengths,
            cmp_lengths,
            residual,
            scale,
            ratio,
            ori_win_left,
            ori_stride,
            cmp_stride,
            ori_page,
            cmp_page,
            ori_mask,
            cmp_mask,
            ori_win_right,
            q_size,
            q_lengths,
            ori_topk_length,
            topk_length,
            has_q_length,
            has_ori_topk,
            has_topk,
            batch_count,
            ori_length_mode,
            cmp_length_mode,
        )


@functools.cache
def _compile_smla_ori_sparse(n1, block_dim, mode, dtype, has_sink=True):
    token_idx = Dim("T")
    ori_capacity = Dim("PO")
    cmp_capacity = Dim("PC")
    ori_slot_idx = Dim("K1")
    sparse_slot_idx = Dim("K")
    ori_table_width = Dim("BO")
    cmp_table_width = Dim("BC")
    ori_table_rows = Dim("OB")
    cmp_table_rows = Dim("CB")
    qcu_count = Dim("CQ")
    qlen_count = Dim("LQ")
    ori_topk_count = Dim("TK1")
    topk_count = Dim("TK")
    bf = dtype
    i32 = dtypes.int32
    return cannbotdsl.compile(
        SparseFlashMlaOriSparseLauncher(n1, block_dim, mode, dtype, has_sink).run,
        TensorSpec((token_idx * n1, _D), bf),
        TensorSpec((token_idx * n1, _D), bf),
        TensorSpec(
            (
                (block_dim // 2 if n1 == _MAX_QUERY_HEADS else block_dim) * _KV_RING * _TILE,
                _D,
            ),
            bf,
        ),
        TensorSpec((ori_capacity, _D), bf),
        TensorSpec((cmp_capacity, _D), bf),
        TensorSpec((ori_table_rows, ori_table_width), i32),
        TensorSpec((cmp_table_rows, cmp_table_width), i32),
        TensorSpec((token_idx, ori_slot_idx), i32),
        TensorSpec((token_idx, sparse_slot_idx), i32),
        TensorSpec((n1 if has_sink else 1,), dtypes.float32),
        TensorSpec((_METADATA_ELEMENTS,), i32),
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
        TensorSpec((ori_topk_count,), i32),
        TensorSpec((topk_count,), i32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


def generic_sparse_flash_mla(
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
    """Full-validation generic path.

    Route admission checks descriptors only. This function performs complete
    validation for inputs dispatched to the generic implementation.
    """
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
    if not 1 <= n1 <= _MAX_QUERY_HEADS or query_gm.shape[-1] != _D:
        raise ValueError("N1 in [1,128], D=512 required")
    kernel_n1 = _Q_TILE_ROWS if n1 <= _Q_TILE_ROWS else _MAX_QUERY_HEADS
    if ori_sparse_indices is not None and (
        ori_sparse_indices.dtype != torch.int32 or ori_sparse_indices.shape[-1] <= 0
    ):
        raise ValueError("Positive int32 ori sparse K required")
    if ori_sparse_indices is not None and ori_mask_mode == 0 and ori_topk_length is None:
        raise ValueError("Noncausal ori sparse requires ori_topk_length")
    if metadata is None or metadata.dtype != torch.int32 or metadata.numel() != _METADATA_ELEMENTS:
        raise ValueError("Original int32[1024] metadata required")
    has_sink = sinks is not None
    if has_sink and (sinks.dtype != torch.float32 or tuple(sinks.shape) != (n1,)):
        raise ValueError("Sinks must be FP32[N1] when provided")
    if not math.isfinite(softmax_scale) or not 1 <= cmp_ratio <= _MAX_CMP_RATIO or topk_value_mode != 1:
        raise ValueError("Invalid scale/ratio/topk_value_mode")
    if ori_mask_mode not in (
        0,
        _MASK_CAUSAL,
        _MASK_SLIDING_WINDOW,
    ) or cmp_mask_mode not in (0, _MASK_CAUSAL):
        raise ValueError("Unsupported mask mode")
    if ori_win_left < -1 or ori_win_right < -1:
        raise ValueError("Invalid window")
    if ori_mask_mode != _MASK_SLIDING_WINDOW and (ori_win_left, ori_win_right) != (
        -1,
        -1,
    ):
        raise ValueError("Invalid window")
    if layout_kv in ("TND", "BSND") and layout_q != layout_kv:
        raise ValueError("Non-PA Q/KV layouts must agree")
    if layout_q == "TND" and cu_seqlens_q is None:
        raise ValueError("TND Q requires cu_seqlens_q")
    if cmp_kv is None and (cmp_ratio != 1 or cmp_mask_mode != 0):
        raise ValueError("SWA requires ratio1 and cmp_mask0")
    if cmp_kv is not None and cmp_mask_mode == _MASK_CAUSAL and cmp_ratio != 1:
        if cmp_residual_kv is None:
            raise ValueError("Compressed causal residual required")
    if cmp_sparse_indices is not None and (
        cmp_sparse_indices.dtype != torch.int32 or cmp_sparse_indices.shape[-1] <= 0
    ):
        raise ValueError("Positive int32 sparse K required")
    if cmp_sparse_indices is not None and cmp_mask_mode == 0 and cmp_topk_length is None:
        raise ValueError("Noncausal CSA requires cmp_topk_length")
    for kv, table, cu, lengths in (
        (ori_kv, ori_block_table, cu_seqlens_ori_kv, seqused_ori_kv),
        (cmp_kv, cmp_block_table, cu_seqlens_cmp_kv, seqused_cmp_kv),
    ):
        if kv is None:
            continue
        if kv.dtype != query_gm.dtype or tuple(kv.shape[-2:]) != (1, _D):
            raise ValueError("KV dtype/N2/D must match specification")
        if layout_kv in ("PA", "PA_BBND"):
            if table is None or lengths is None or not 1 <= kv.shape[1] <= _MAX_PAGE_ROWS:
                raise ValueError("PA requires page1..1024, table and actual lengths")
        if layout_kv == "TND" and cu is None:
            raise ValueError("TND KV requires storage prefixes")
    batch_count = query_gm.shape[0] if layout_q == "BSND" else cu_seqlens_q.numel() - 1
    q_size = query_gm.shape[1] if layout_q == "BSND" else 0
    # The fixed kernel signature needs a Tensor in both layouts. BSND uses
    # q_size for offsets and never dereferences this prefix argument.
    cu_q = cu_seqlens_q if layout_q == "TND" else metadata
    q_lengths = seqused_q if seqused_q is not None else metadata
    topk_length = cmp_topk_length.reshape(-1) if cmp_topk_length is not None else metadata
    ori_topk_len = ori_topk_length.reshape(-1) if ori_topk_length is not None else metadata
    has_q_length = int(seqused_q is not None)
    has_topk = int(cmp_topk_length is not None)
    has_ori_topk = int(ori_sparse_indices is not None and ori_topk_length is not None)

    def kv_view(kv, table, cu):
        if layout_kv in ("PA", "PA_BBND"):
            page = kv.shape[1]
            stride = kv.stride(0) // _D
            storage = kv.as_strided(((kv.shape[0] - 1) * stride + page, _D), (_D, 1))
            return storage, table, stride, page
        page = 0 if layout_kv == "TND" else -kv.shape[1]
        table = cu.reshape(1, -1) if layout_kv == "TND" else metadata.reshape(1, -1)
        return kv.reshape(-1, _D), table, 0, page

    ori_storage, ori_table, ori_stride, ori_page = kv_view(ori_kv, ori_block_table, cu_seqlens_ori_kv)
    if ori_sparse_indices is None:
        mode = "SWA" if cmp_kv is None else ("HCA" if cmp_sparse_indices is None else "CSA")
    elif cmp_kv is None:
        mode = "ORI_SPARSE"
    elif cmp_sparse_indices is not None:
        mode = "ORI_CMP_SPARSE"
    else:
        raise NotImplementedError("Ori sparse with dense CMP has no template")
    if mode in ("SWA", "ORI_SPARSE"):
        cmp_storage = ori_storage
        cmp_table = ori_table
        cmp_stride = ori_stride
        cmp_page = ori_page
        cmp_lengths = seqused_ori_kv
        residual = seqused_ori_kv
    else:
        cmp_storage, cmp_table, cmp_stride, cmp_page = kv_view(cmp_kv, cmp_block_table, cu_seqlens_cmp_kv)
        cmp_lengths = seqused_cmp_kv
        residual = cmp_residual_kv if cmp_residual_kv is not None else metadata
    ori_length_mode = 0 if seqused_ori_kv is not None else (1 if layout_kv == "TND" else 2)
    cmp_length_mode = 0 if cmp_lengths is not None else (1 if layout_kv == "TND" else 2)
    if seqused_ori_kv is None:
        seqused_ori_kv = cu_seqlens_ori_kv if layout_kv == "TND" else metadata
    if cmp_lengths is None:
        cmp_lengths = cu_seqlens_cmp_kv if layout_kv == "TND" else metadata
    if mode in ("SWA", "ORI_SPARSE"):
        cmp_lengths = seqused_ori_kv
        residual = seqused_ori_kv
        cmp_length_mode = ori_length_mode

    if kernel_n1 != n1:
        q_pad = torch.zeros(
            (*query_gm.shape[:-2], kernel_n1 - n1, _D),
            device=query_gm.device,
            dtype=query_gm.dtype,
        )
        kernel_query = torch.cat((query_gm, q_pad), dim=-2)
    else:
        kernel_query = query_gm
    if not has_sink:
        kernel_sinks = torch.empty((1,), dtype=torch.float32, device=query_gm.device)
    elif kernel_n1 != n1:
        sink_pad = torch.zeros((kernel_n1 - n1,), device=sinks.device, dtype=sinks.dtype)
        kernel_sinks = torch.cat((sinks, sink_pad), dim=0)
    else:
        kernel_sinks = sinks

    block_dim = get_platform_info().cube_core_num
    dtype = dtypes.bfloat16 if query_gm.dtype == torch.bfloat16 else dtypes.float16
    out = torch.empty_like(kernel_query)
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
    if mode in ("SWA", "HCA"):
        _compile_smla_direct(kernel_n1, block_dim, mode, dtype, has_sink)(
            out.view(-1, _D),
            kernel_query.view(-1, _D),
            ori_storage,
            cmp_storage,
            ori_table,
            cmp_table,
            kernel_sinks,
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
    elif mode == "CSA":
        indices = cmp_sparse_indices.reshape(-1, cmp_sparse_indices.shape[-1])
        workspace = torch.zeros(
            (
                (block_dim // 2 if kernel_n1 == _MAX_QUERY_HEADS else block_dim) * _KV_RING * _TILE,
                _D,
            ),
            device=query_gm.device,
            dtype=query_gm.dtype,
        )
        _compile_smla(kernel_n1, block_dim, dtype, has_sink)(
            out.view(-1, _D),
            kernel_query.view(-1, _D),
            workspace,
            ori_storage,
            cmp_storage,
            ori_table,
            cmp_table,
            indices,
            kernel_sinks,
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
        ori_indices = ori_sparse_indices.reshape(-1, ori_sparse_indices.shape[-1])
        if mode == "ORI_CMP_SPARSE":
            indices = cmp_sparse_indices.reshape(-1, cmp_sparse_indices.shape[-1])
        else:
            indices = ori_indices
        workspace = torch.zeros(
            (
                (block_dim // 2 if kernel_n1 == _MAX_QUERY_HEADS else block_dim) * _KV_RING * _TILE,
                _D,
            ),
            device=query_gm.device,
            dtype=query_gm.dtype,
        )
        _compile_smla_ori_sparse(kernel_n1, block_dim, mode, dtype, has_sink)(
            out.view(-1, _D),
            kernel_query.view(-1, _D),
            workspace,
            ori_storage,
            cmp_storage,
            ori_table,
            cmp_table,
            ori_indices,
            indices,
            kernel_sinks,
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
            ori_page,
            cmp_page,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_right,
            q_size,
            q_lengths,
            ori_topk_len,
            topk_length,
            has_q_length,
            has_ori_topk,
            has_topk,
            batch_count,
            ori_length_mode,
            cmp_length_mode,
        )
    if kernel_n1 != n1:
        out = out.narrow(-2, 0, n1).contiguous()
    return out, torch.empty((0,), device=query_gm.device, dtype=torch.float32)


def _is_sparse_n64_candidate(query_gm, kwargs):
    if not isinstance(query_gm, torch.Tensor) or query_gm.ndim < 2:
        return False
    if (
        not 1 <= query_gm.shape[-2] <= _Q_TILE_ROWS
        or query_gm.shape[-1] != _D
        or query_gm.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    metadata = kwargs.get("metadata")
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.dtype != torch.int32
        or metadata.numel() != _METADATA_ELEMENTS
    ):
        return False
    layout_q = kwargs.get("layout_q", "BSND")
    layout_kv = kwargs.get("layout_kv", "BSND")
    expected_rank = 3 if layout_q == "TND" else 4
    if (
        layout_q not in ("TND", "BSND")
        or query_gm.ndim != expected_rank
        or not query_gm.is_contiguous()
        or not (layout_kv in ("PA", "PA_BBND") or layout_kv == layout_q)
    ):
        return False
    ori_indices = kwargs.get("ori_sparse_indices")
    cmp_kv = kwargs.get("cmp_kv")
    cmp_indices = kwargs.get("cmp_sparse_indices")
    # N64 sparse modes: ORI-only, ORI+CMP, and CSA (dense ORI + sparse CMP).
    return (ori_indices is not None and cmp_kv is None and cmp_indices is None) or (
        cmp_kv is not None and cmp_indices is not None
    )


def route_admission(query_gm, **kwargs):
    """Select the N64 sparse or generic implementation from input descriptors."""
    if _is_sparse_n64_candidate(query_gm, kwargs):
        return {"route": "asc_sparse_n64_kernel", "reasons": []}
    return {"route": "generic", "reasons": ["not_sparse_n64_candidate"]}


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
    kwargs = locals().copy()
    kwargs.pop("query_gm")
    admission = route_admission(query_gm, **kwargs)
    if admission["route"] == "asc_sparse_n64_kernel":
        return sparse_n64_kernel(query_gm, **kwargs)
    return generic_sparse_flash_mla(query_gm, **kwargs)


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
        cmp_residual_kv = torch.zeros(cu_seqlens_q.numel() - 1, dtype=torch.int32, device=q.device)

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
        cmp_topk=cmp_sparse_indices.shape[-1] if cmp_sparse_indices is not None else 0,
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
