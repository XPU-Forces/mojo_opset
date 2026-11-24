#pragma once

#include "kernel/runtime.h"
#include "kernel/task_scheduler.h"
#include "kernel/vector_compute.h"

namespace SWA_v5::Backward {

struct AivStageTask {
    TaskInfo task;
    uint32_t issue_idx;
    uint32_t mm_slot;
    uint32_t p_ds_slot;
};

class BackwardAIV {
    using InputT = SWA_BWD_ENTRY_INPUT_TYPE;

public:
    __aicore__ inline void init(GM_ADDR out, GM_ADDR dout, GM_ADDR softmax_lse,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
        uint64_t dq_workspace_elems, uint64_t dk_workspace_elems,
        uint64_t workspace_elems, uint64_t dkv_scratch_elems,
        uint32_t active_aiv_cores, uint32_t total_q_tokens,
        uint32_t total_kv_tokens, uint32_t q_stride, uint32_t kv_stride,
        uint32_t num_q_heads, uint32_t num_kv_heads,
        bool output_is_f32, bool direct_scaled_dq) {
        out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(out));
        out_f32_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        dout_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dout));
        lse_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmax_lse));
        dq_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dq));
        dk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dk));
        dv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dv));
        workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        color_fp16_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(workspace) +
            2ULL * (workspace_elems + dkv_scratch_elems));
        dq_workspace_elems_ = dq_workspace_elems;
        dk_workspace_elems_ = dk_workspace_elems;
        workspace_elems_ = workspace_elems;
        dkv_scratch_elems_ = dkv_scratch_elems;
        active_aiv_cores_ = active_aiv_cores;
        total_q_tokens_ = total_q_tokens;
        total_kv_tokens_ = total_kv_tokens;
        q_stride_ = q_stride;
        kv_stride_ = kv_stride;
        num_q_heads_ = num_q_heads;
        num_kv_heads_ = num_kv_heads;
        output_is_f32_ = output_is_f32;
        direct_scaled_dq_ = direct_scaled_dq;
        rt.init();
        mte2_to_vec_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        vec_to_mte2_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            fp16_mte2_to_vec_event_[slot] =
                rt.pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
            fp16_vec_to_mte2_event_[slot] =
                rt.pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
        }
        vec_to_mte3_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3_to_vec_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        vec_to_scalar_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::V_S>();
        mte3_to_scalar_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void zero_outputs() {
        zero_gm(dq_gm, total_q_tokens_ * q_stride_);
        zero_gm(dk_gm, total_kv_tokens_ * kv_stride_);
        zero_gm(dv_gm, total_kv_tokens_ * kv_stride_);
    }

    __aicore__ inline void reduce_dkv_color_batch_range(
        uint32_t batch_idx, uint32_t q_len, uint32_t kv_len,
        uint32_t kv_global_begin, uint32_t kv_row_begin,
        uint32_t kv_row_count, uint32_t window_left,
        uint32_t preserve_prefix_rows,
        int32_t min_delta, int32_t max_delta,
        uint64_t color_dk_storage_elems, uint64_t workspace_row_offset,
        uint64_t global_job_base, uint32_t workers, bool use_fp16_partials,
        bool direct_output, float dk_scale) {
        constexpr uint32_t rows_per_job =
            BWD_GRAD_TILE_ELEMS / BWD_DIM / BWD_AIVS_PER_AIC;
        const uint32_t worker = AscendC::GetBlockIdx();
        const uint32_t kv_tile_begin = kv_row_begin / BWD_TILE_K;
        const uint32_t row_job_count =
            ceil_div(kv_row_count, rows_per_job);
        const uint32_t q_tiles = ceil_div(q_len, BWD_TILE_Q);
        const uint32_t kv_prefix = kv_len - q_len;
        const uint32_t prefix_tile = kv_prefix / BWD_TILE_K;
        const uint64_t jobs =
            static_cast<uint64_t>(row_job_count) * num_kv_heads_ *
            BWD_DKV_GRAD_KINDS;
        const uint64_t partial_section_offset =
            workspace_elems_ + dkv_scratch_elems_;
        const uint64_t first_job =
            (static_cast<uint64_t>(worker) + workers -
                global_job_base % workers) %
            workers;

        for (uint64_t job = first_job; job < jobs;
             job += workers) {
            uint64_t remaining = job;
            const uint32_t row_job =
                static_cast<uint32_t>(remaining % row_job_count);
            remaining /= row_job_count;
            const uint32_t kv_head =
                static_cast<uint32_t>(remaining % num_kv_heads_);
            remaining /= num_kv_heads_;
            const uint32_t local_kv_tile =
                row_job / BWD_AIVS_PER_AIC;
            const uint32_t kv_tile =
                kv_tile_begin + local_kv_tile;
            const uint32_t grad_kind =
                static_cast<uint32_t>(remaining);
            const uint32_t tile_rows = min_u32(
                BWD_TILE_K, kv_len - kv_tile * BWD_TILE_K);
            const uint32_t row_half =
                row_job % BWD_AIVS_PER_AIC;
            const uint32_t row_offset = row_half * rows_per_job;
            const uint32_t rows =
                min_u32(rows_per_job, tile_rows - row_offset);
            const uint32_t count = rows * BWD_DIM;
            const uint64_t local_tile_offset =
                (workspace_row_offset +
                    static_cast<uint64_t>(local_kv_tile) * BWD_TILE_K +
                    row_offset) *
                    kv_stride_ +
                static_cast<uint64_t>(kv_head) * BWD_DIM;
            const uint64_t padded_tile_offset =
                (static_cast<uint64_t>(kv_global_begin) +
                    static_cast<uint64_t>(batch_idx) *
                        (BWD_TILE_K - 1U) +
                    static_cast<uint64_t>(kv_tile) * BWD_TILE_K +
                    row_offset) *
                    kv_stride_ +
                static_cast<uint64_t>(kv_head) * BWD_DIM;
            const bool contiguous = kv_stride_ == BWD_DIM;
            AscendC::DataCopyExtParams load{
                static_cast<uint16_t>(contiguous ? 1U : rows),
                (contiguous ? count : BWD_DIM) *
                    static_cast<uint32_t>(sizeof(float)),
                contiguous ? 0 :
                    static_cast<int64_t>(
                        (kv_stride_ - BWD_DIM) * sizeof(float)),
                0, 0U};
            AscendC::DataCopyPadExtParams<float> no_pad{
                false, 0U, 0U, 0.0F};
            AscendC::LocalTensor<float> accum = rt.c1_ub(0U);
            bool has_partial = false;
            bool fp16_slot_busy[2] = {false, false};
            bool fp16_pending = false;
            uint32_t fp16_pending_slot = 0U;
            uint32_t fp16_issue_count = 0U;
            if (use_fp16_partials) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    vec_to_mte2_event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    vec_to_mte2_event_);
            }
            for (int32_t delta = min_delta;
                 delta <= max_delta; ++delta) {
                const int32_t q_tile_signed =
                    delta + static_cast<int32_t>(kv_tile) -
                    static_cast<int32_t>(prefix_tile);
                if (q_tile_signed < 0 ||
                    static_cast<uint32_t>(q_tile_signed) >= q_tiles) {
                    continue;
                }
                const uint32_t q_tile =
                    static_cast<uint32_t>(q_tile_signed);
                const uint32_t q_row = q_tile * BWD_TILE_Q;
                const uint32_t q_rows =
                    min_u32(BWD_TILE_Q, q_len - q_row);
                const KvTileRange range =
                    make_kv_tile_range<
                        BWD_TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                        kv_prefix + q_row, q_rows, kv_len,
                        window_left, 0U, 0U);
                if (kv_tile < range.first_tile ||
                    kv_tile >=
                        range.first_tile + range.local_tile_count) {
                    continue;
                }
                const uint64_t color =
                    static_cast<uint32_t>(delta - min_delta);
                const uint64_t color_count =
                    static_cast<uint32_t>(max_delta - min_delta + 1);
                const uint64_t src_offset = use_fp16_partials ?
                    color * 2U * color_dk_storage_elems +
                        static_cast<uint64_t>(grad_kind) *
                            color_dk_storage_elems +
                        local_tile_offset :
                    partial_section_offset +
                        static_cast<uint64_t>(grad_kind) *
                            color_count * color_dk_storage_elems +
                        color * color_dk_storage_elems +
                        local_tile_offset;
                if (use_fp16_partials) {
                    const uint32_t slot = fp16_issue_count & 1U;
                    if (fp16_slot_busy[slot]) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                            fp16_vec_to_mte2_event_[slot]);
                        fp16_slot_busy[slot] = false;
                    }
                    AscendC::LocalTensor<half> packed =
                        rt.fp16_reduce_ub(slot);
                    AscendC::DataCopyExtParams packed_load{
                        static_cast<uint16_t>(contiguous ? 1U : rows),
                        (contiguous ? count : BWD_DIM) *
                            static_cast<uint32_t>(sizeof(half)),
                        contiguous ? 0 :
                            static_cast<int64_t>(
                                (kv_stride_ - BWD_DIM) * sizeof(half)),
                        0, 0U};
                    AscendC::DataCopyPadExtParams<half> packed_no_pad{
                        false, 0U, 0U, 0.0F};
                    if (contiguous) {
                        AscendC::DataCopy(
                            packed, color_fp16_gm[src_offset], count);
                    } else {
                        AscendC::DataCopyPad(
                            packed, color_fp16_gm[src_offset],
                            packed_load, packed_no_pad);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                        fp16_mte2_to_vec_event_[slot]);

                    if (fp16_pending) {
                        AscendC::LocalTensor<float> input =
                            has_partial ? rt.c2_ub(0U) : accum;
                        AscendC::LocalTensor<half> pending =
                            rt.fp16_reduce_ub(fp16_pending_slot);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                            fp16_mte2_to_vec_event_[fp16_pending_slot]);
                        AscendC::Cast(
                            input, pending, AscendC::RoundMode::CAST_NONE,
                            count);
                        AscendC::PipeBarrier<PIPE_V>();
                        if (has_partial) {
                            AscendC::Add(accum, accum, input, count);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                            fp16_vec_to_mte2_event_[fp16_pending_slot]);
                        fp16_slot_busy[fp16_pending_slot] = true;
                        has_partial = true;
                    }

                    fp16_pending = true;
                    fp16_pending_slot = slot;
                    ++fp16_issue_count;
                    continue;
                }

                AscendC::LocalTensor<float> input =
                    has_partial ? rt.c2_ub(0U) : accum;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    vec_to_mte2_event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    vec_to_mte2_event_);
                if (contiguous) {
                    AscendC::DataCopy(
                        input, workspace_gm[src_offset], count);
                } else {
                    AscendC::DataCopyPad(
                        input, workspace_gm[src_offset], load, no_pad);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                    mte2_to_vec_event_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    mte2_to_vec_event_);
                if (has_partial) {
                    AscendC::Add(accum, accum, input, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                has_partial = true;
            }
            if (fp16_pending) {
                AscendC::LocalTensor<float> input =
                    has_partial ? rt.c2_ub(0U) : accum;
                AscendC::LocalTensor<half> pending =
                    rt.fp16_reduce_ub(fp16_pending_slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    fp16_mte2_to_vec_event_[fp16_pending_slot]);
                AscendC::Cast(
                    input, pending, AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                if (has_partial) {
                    AscendC::Add(accum, accum, input, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    fp16_vec_to_mte2_event_[fp16_pending_slot]);
                fp16_slot_busy[fp16_pending_slot] = true;
                has_partial = true;
            }
            for (uint32_t slot = 0U; slot < 2U; ++slot) {
                if (fp16_slot_busy[slot]) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                        fp16_vec_to_mte2_event_[slot]);
                }
            }
            if (!has_partial) {
                AscendC::Duplicate(
                    accum, 0.0F, static_cast<int32_t>(count));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (direct_output) {
                const uint32_t kv_row =
                    kv_tile * BWD_TILE_K + row_offset;
                const uint32_t preserved_rows =
                    kv_row < preserve_prefix_rows ?
                        min_u32(rows, preserve_prefix_rows - kv_row) : 0U;
                if (preserved_rows > 0U) {
                    const uint64_t preserve_offset =
                        dq_workspace_elems_ +
                        static_cast<uint64_t>(grad_kind) *
                            dk_workspace_elems_ +
                        padded_tile_offset;
                    AscendC::DataCopyExtParams preserve_store{
                        static_cast<uint16_t>(
                            contiguous ? 1U : preserved_rows),
                        (contiguous ? preserved_rows * BWD_DIM : BWD_DIM) *
                            static_cast<uint32_t>(sizeof(float)),
                        0,
                        contiguous ? 0 :
                            static_cast<int64_t>(
                                (kv_stride_ - BWD_DIM) * sizeof(float)),
                        0U};
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        vec_to_mte3_event_);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                        vec_to_mte3_event_);
                    AscendC::DataCopyPad(
                        workspace_gm[preserve_offset], accum,
                        preserve_store);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                        mte3_to_vec_event_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        mte3_to_vec_event_);
                }
                if (grad_kind == 0U && dk_scale != 1.0F) {
                    AscendC::Muls(accum, accum, dk_scale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<InputT> output =
                    rt.score_stage_ub();
                AscendC::Cast(
                    output, accum, AscendC::RoundMode::CAST_RINT,
                    count);
                const uint64_t dst_offset =
                    (static_cast<uint64_t>(kv_global_begin) +
                        static_cast<uint64_t>(kv_tile) * BWD_TILE_K +
                        row_offset) *
                        kv_stride_ +
                    static_cast<uint64_t>(kv_head) * BWD_DIM;
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : BWD_DIM) *
                        static_cast<uint32_t>(sizeof(InputT)),
                    0,
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (kv_stride_ - BWD_DIM) * sizeof(InputT)),
                    0U};
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    vec_to_mte3_event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    vec_to_mte3_event_);
                if (grad_kind == 0U) {
                    if (contiguous) {
                        AscendC::DataCopy(
                            dk_gm[dst_offset], output, count);
                    } else {
                        AscendC::DataCopyPad(
                            dk_gm[dst_offset], output, store);
                    }
                } else {
                    if (contiguous) {
                        AscendC::DataCopy(
                            dv_gm[dst_offset], output, count);
                    } else {
                        AscendC::DataCopyPad(
                            dv_gm[dst_offset], output, store);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    mte3_to_vec_event_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    mte3_to_vec_event_);
            } else {
                const uint64_t dst_offset =
                    dq_workspace_elems_ +
                    static_cast<uint64_t>(grad_kind) *
                        dk_workspace_elems_ +
                    padded_tile_offset;
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : BWD_DIM) *
                        static_cast<uint32_t>(sizeof(float)),
                    0,
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (kv_stride_ - BWD_DIM) * sizeof(float)),
                    0U};
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    vec_to_mte3_event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    vec_to_mte3_event_);
                AscendC::DataCopyPad(
                    workspace_gm[dst_offset], accum, store);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    mte3_to_vec_event_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    mte3_to_vec_event_);
            }
        }
    }

    __aicore__ inline void finalize_dq_ring_range(
        uint32_t q_global_begin, uint32_t ring_row_begin,
        uint32_t q_row_count, float scale) {
        const uint32_t workers = active_aiv_cores_;
        constexpr uint32_t rows_per_job = BWD_LOCAL_ROWS;
        const uint32_t worker = AscendC::GetBlockIdx();
        const uint32_t row_jobs =
            ceil_div(q_row_count, rows_per_job);
        const uint64_t jobs =
            static_cast<uint64_t>(row_jobs) * num_q_heads_;
        for (uint64_t job = worker; job < jobs; job += workers) {
            const uint32_t q_head =
                static_cast<uint32_t>(job % num_q_heads_);
            const uint32_t row_job =
                static_cast<uint32_t>(job / num_q_heads_);
            const uint32_t row_offset = row_job * rows_per_job;
            const uint32_t rows =
                min_u32(rows_per_job, q_row_count - row_offset);
            const uint32_t count = rows * BWD_DIM;
            const uint64_t src_offset =
                (static_cast<uint64_t>(ring_row_begin) + row_offset) *
                    q_stride_ +
                static_cast<uint64_t>(q_head) * BWD_DIM;
            const uint64_t dst_offset =
                (static_cast<uint64_t>(q_global_begin) + row_offset) *
                    q_stride_ +
                static_cast<uint64_t>(q_head) * BWD_DIM;
            const bool contiguous = q_stride_ == BWD_DIM;
            AscendC::LocalTensor<float> fp32 = rt.c1_ub(0U);
            AscendC::LocalTensor<InputT> bf16 = rt.out_ub();
            AscendC::DataCopyExtParams fp32_copy{
                static_cast<uint16_t>(contiguous ? 1U : rows),
                (contiguous ? count : BWD_DIM) *
                    static_cast<uint32_t>(sizeof(float)),
                contiguous ? 0 :
                    static_cast<int64_t>(
                        (q_stride_ - BWD_DIM) * sizeof(float)),
                0U, 0U};
            AscendC::DataCopyPadExtParams<float> no_pad{
                false, 0U, 0U, 0.0F};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                vec_to_mte2_event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                vec_to_mte2_event_);
            AscendC::DataCopyPad(
                fp32, workspace_gm[src_offset], fp32_copy, no_pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                mte2_to_vec_event_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                mte2_to_vec_event_);
            AscendC::Muls(fp32, fp32, scale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                bf16, fp32, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams bf16_copy{
                static_cast<uint16_t>(contiguous ? 1U : rows),
                (contiguous ? count : BWD_DIM) *
                    static_cast<uint32_t>(sizeof(InputT)),
                0U,
                contiguous ? 0 :
                    static_cast<int64_t>(
                        (q_stride_ - BWD_DIM) * sizeof(InputT)),
                0U};
            AscendC::DataCopyExtParams fp32_zero_store{
                static_cast<uint16_t>(contiguous ? 1U : rows),
                (contiguous ? count : BWD_DIM) *
                    static_cast<uint32_t>(sizeof(float)),
                0U,
                contiguous ? 0 :
                    static_cast<int64_t>(
                        (q_stride_ - BWD_DIM) * sizeof(float)),
                0U};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vec_to_mte3_event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vec_to_mte3_event_);
            AscendC::DataCopyPad(dq_gm[dst_offset], bf16, bf16_copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3_to_vec_event_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3_to_vec_event_);
            AscendC::Duplicate(fp32, 0.0F, static_cast<int32_t>(count));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vec_to_mte3_event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vec_to_mte3_event_);
            AscendC::DataCopyPad(
                workspace_gm[src_offset], fp32, fp32_zero_store);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3_to_vec_event_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3_to_vec_event_);
        }
    }

    __aicore__ inline void finish() {
        for (uint32_t slot = 0U; slot < BWD_P_DS_SLOTS; ++slot) {
            rt.wait_aic<PIPE_MTE3>(
                bwd_event_slot(BWD_P_DS_CONSUMED_BASE, slot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar_event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar_event_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vec_to_scalar_event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vec_to_scalar_event_);
        rt.pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(mte3_to_scalar_event_);
        rt.pipe.ReleaseEventID<AscendC::HardEvent::V_S>(vec_to_scalar_event_);
        rt.pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3_to_vec_event_);
        rt.pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vec_to_mte3_event_);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            rt.pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(
                fp16_vec_to_mte2_event_[slot]);
            rt.pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(
                fp16_mte2_to_vec_event_[slot]);
        }
        rt.pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(vec_to_mte2_event_);
        rt.pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2_to_vec_event_);
    }

    __aicore__ inline void destroy() {
        rt.pipe.Destroy();
    }

    __aicore__ inline void prepare_q(const TaskInfo& task) {
        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t local_row_base = subblock * BWD_LOCAL_ROWS;
        const uint32_t active_rows = task.q_rows > local_row_base ?
            min_u32(BWD_LOCAL_ROWS, task.q_rows - local_row_base) : 0U;
        const uint32_t q_token_begin = task.q_global_begin + local_row_base;
        const uint32_t q_offset = q_token_begin * q_stride_ + task.q_head * BWD_DIM;
        const uint32_t lse_offset =
            task.q_head * total_q_tokens_ + q_token_begin;

        copy_v1_inputs(q_offset, lse_offset, active_rows);
        if (output_is_f32_) {
            VectorCompute::row_dot_f32_bf16(
                rt.out_f32_ub(), rt.dout_ub(), rt.d_ub(), active_rows);
        } else {
            VectorCompute::row_dot_bf16(
                rt.out_ub(), rt.dout_ub(), rt.d_ub(), active_rows);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void cache_prepared_d(
        const TaskInfo& task, bool pack_lse) {
        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t local_row_base = subblock * BWD_LOCAL_ROWS;
        const uint32_t active_rows = task.q_rows > local_row_base ?
            min_u32(BWD_LOCAL_ROWS, task.q_rows - local_row_base) : 0U;
        if (active_rows == 0U) {
            return;
        }
        const uint32_t q_token_begin =
            task.q_global_begin + local_row_base;
        uint64_t d_offset = workspace_elems_ +
            static_cast<uint64_t>(task.q_head) * total_q_tokens_ + q_token_begin;
        uint32_t copy_rows = active_rows;
        if (pack_lse) {
            const uint64_t cache_slot =
                static_cast<uint64_t>(task.linear_task) *
                    (num_q_heads_ / num_kv_heads_) +
                task.gqa_group;
            d_offset = workspace_elems_ +
                (cache_slot * BWD_AIVS_PER_AIC + subblock) *
                    2U * BWD_LOCAL_ROWS;
            copy_rows = 2U * BWD_LOCAL_ROWS;
        }
        AscendC::DataCopyExtParams copy_params{
            1U,
            static_cast<uint32_t>(copy_rows * sizeof(float)),
            0U,
            0U,
            0U,
        };
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
            vec_to_mte3_event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            vec_to_mte3_event_);
        AscendC::DataCopyPad(
            workspace_gm[d_offset], rt.d_ub(), copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
            mte3_to_vec_event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
            mte3_to_vec_event_);
    }

    __aicore__ inline void prepare_q_from_cached_d(
        const TaskInfo& task, bool packed_lse) {
        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t local_row_base = subblock * BWD_LOCAL_ROWS;
        const uint32_t active_rows = task.q_rows > local_row_base ?
            min_u32(BWD_LOCAL_ROWS, task.q_rows - local_row_base) : 0U;
        if (active_rows == 0U) {
            return;
        }
        const uint32_t q_token_begin =
            task.q_global_begin + local_row_base;
        const uint64_t q_head_offset =
            static_cast<uint64_t>(task.q_head) * total_q_tokens_ + q_token_begin;
        uint64_t d_offset = workspace_elems_ + q_head_offset;
        uint32_t copy_rows = active_rows;
        if (packed_lse) {
            const uint64_t cache_slot =
                static_cast<uint64_t>(task.linear_task) *
                    (num_q_heads_ / num_kv_heads_) +
                task.gqa_group;
            d_offset = workspace_elems_ +
                (cache_slot * BWD_AIVS_PER_AIC + subblock) *
                    2U * BWD_LOCAL_ROWS;
            copy_rows = 2U * BWD_LOCAL_ROWS;
        }
        AscendC::DataCopyExtParams copy_params{
            1U,
            static_cast<uint32_t>(copy_rows * sizeof(float)),
            0U,
            0U,
            0U,
        };
        AscendC::DataCopyPadExtParams<float> no_pad{
            false, 0U, 0U, 0.0F};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            vec_to_mte2_event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            vec_to_mte2_event_);
        AscendC::DataCopyPad(
            rt.d_ub(), workspace_gm[d_offset],
            copy_params, no_pad);
        if (!packed_lse) {
            AscendC::DataCopyPad(
                rt.lse_ub(), lse_gm[q_head_offset], copy_params, no_pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            mte2_to_vec_event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            mte2_to_vec_event_);
    }

    __aicore__ inline void run_v2_v3_v4(
        const AivStageTask& stage, uint32_t window_left, uint32_t window_right,
        uint32_t global_window_size, float scale) {
        const uint32_t slot = stage.mm_slot;
        const uint32_t score_slot = stage.p_ds_slot;
        const float grad_scale = direct_scaled_dq_ ? scale : 1.0F;
        rt.wait_aic<PIPE_V>(bwd_event_slot(BWD_MM_READY_BASE, slot));
        rt.wait_aic<PIPE_MTE3>(
            bwd_event_slot(BWD_P_DS_CONSUMED_BASE, score_slot));

        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t local_row_base = subblock * BWD_LOCAL_ROWS;
        const uint32_t active_rows = stage.task.q_rows > local_row_base ?
            min_u32(BWD_LOCAL_ROWS, stage.task.q_rows - local_row_base) : 0U;

        const uint32_t q0_abs =
            stage.task.kv_len - stage.task.q_len + stage.task.q_tile * BWD_TILE_Q + local_row_base;
        const uint32_t k0 = stage.task.kv_tile * BWD_TILE_K;
        const int32_t last_k = static_cast<int32_t>(k0 + BWD_TILE_K - 1U);
        const int32_t last_q = static_cast<int32_t>(q0_abs + active_rows) - 1;
        const int32_t earliest_window_end = static_cast<int32_t>(q0_abs) +
            ((!SWA_BWD_ENTRY_IS_CAUSAL && window_right > 0U) ?
                static_cast<int32_t>(window_right) : 0);
        const bool global_only = stage.task.mask_mode == BWD_MASK_GLOBAL_ONLY;
        const bool needs_left_mask =
            static_cast<int32_t>(k0) <
            last_q - static_cast<int32_t>(window_left) + 1;
        const bool needs_right_mask = last_k > earliest_window_end;
        const bool full_local_tile = !needs_left_mask && !needs_right_mask;
        const bool tile_needs_mask = global_only || active_rows < BWD_LOCAL_ROWS ||
            last_k >= stage.task.kv_len || !full_local_tile;
        if (!tile_needs_mask) {
            VectorCompute::make_p_ds_full_tile_and_stage_nz(
                rt.c1_ub(slot), rt.c2_ub(slot), rt.lse_ub(), rt.d_ub(),
                rt.ds_stage_ub(), rt.score_stage_ub(), scale, grad_scale);
            AscendC::PipeBarrier<PIPE_V>();
            copy_stages_to_a1(
                rt.p_a1_for_aiv(score_slot), rt.ds_stage_ub(),
                rt.ds_a1_for_aiv(score_slot), rt.score_stage_ub());
            rt.notify_aic<PIPE_MTE3>(bwd_event_slot(BWD_P_READY_BASE, score_slot));
            rt.notify_aic<PIPE_MTE3>(bwd_event_slot(BWD_DS_READY_BASE, score_slot));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vec_to_scalar_event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vec_to_scalar_event_);
            return;
        }

        if (global_only) {
            VectorCompute::make_p_ds_masked_tile_and_stage_nz<
                (SWA_BWD_ENTRY_IS_CAUSAL != 0), true, true>(
                rt.c1_ub(slot), rt.c2_ub(slot), rt.lse_ub(), rt.d_ub(),
                rt.ds_stage_ub(), rt.score_stage_ub(), q0_abs, k0, active_rows,
                stage.task.kv_len, window_left, window_right, global_window_size, scale, grad_scale);
        } else if (active_rows == BWD_LOCAL_ROWS && last_k < stage.task.kv_len &&
            needs_left_mask && !needs_right_mask) {
            VectorCompute::make_p_ds_masked_tile_and_stage_nz<
                (SWA_BWD_ENTRY_IS_CAUSAL != 0), true, false, true, false, false>(
                rt.c1_ub(slot), rt.c2_ub(slot), rt.lse_ub(), rt.d_ub(),
                rt.ds_stage_ub(), rt.score_stage_ub(), q0_abs, k0, active_rows,
                stage.task.kv_len, window_left, window_right, 0U, scale, grad_scale);
        } else if (active_rows == BWD_LOCAL_ROWS && last_k < stage.task.kv_len &&
            !needs_left_mask && needs_right_mask) {
            VectorCompute::make_p_ds_masked_tile_and_stage_nz<
                (SWA_BWD_ENTRY_IS_CAUSAL != 0), true, false, false, true, false>(
                rt.c1_ub(slot), rt.c2_ub(slot), rt.lse_ub(), rt.d_ub(),
                rt.ds_stage_ub(), rt.score_stage_ub(), q0_abs, k0, active_rows,
                stage.task.kv_len, window_left, window_right, 0U, scale, grad_scale);
        } else {
            VectorCompute::make_p_ds_masked_tile_and_stage_nz<
                (SWA_BWD_ENTRY_IS_CAUSAL != 0), true>(
                rt.c1_ub(slot), rt.c2_ub(slot), rt.lse_ub(), rt.d_ub(),
                rt.ds_stage_ub(), rt.score_stage_ub(), q0_abs, k0, active_rows,
                stage.task.kv_len, window_left, window_right, 0U, scale, grad_scale);
        }
        AscendC::PipeBarrier<PIPE_V>();
        copy_stages_to_a1(
            rt.p_a1_for_aiv(score_slot), rt.ds_stage_ub(),
            rt.ds_a1_for_aiv(score_slot), rt.score_stage_ub());
        rt.notify_aic<PIPE_MTE3>(bwd_event_slot(BWD_P_READY_BASE, score_slot));
        rt.notify_aic<PIPE_MTE3>(bwd_event_slot(BWD_DS_READY_BASE, score_slot));
    }

    __aicore__ inline void release_mm_slot(const AivStageTask& stage) {
        rt.notify_aic<PIPE_MTE3>(
            bwd_event_slot(BWD_MM_CONSUMED_BASE, stage.mm_slot));
    }

private:
    __aicore__ inline void copy_v1_inputs(
        uint32_t q_offset, uint32_t lse_offset, uint32_t active_rows) {
        if (active_rows == 0U) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vec_to_mte2_event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vec_to_mte2_event_);
        AscendC::DataCopyPadExtParams<InputT> no_pad{false, 0U, 0U, static_cast<InputT>(0.0F)};
        if (output_is_f32_) {
            AscendC::DataCopyExtParams output_copy{
                static_cast<uint16_t>(active_rows),
                BWD_DIM * sizeof(float),
                static_cast<int64_t>((q_stride_ - BWD_DIM) * sizeof(float)),
                0U,
                0U,
            };
            AscendC::DataCopyPadExtParams<float> no_pad_f32{false, 0U, 0U, 0.0F};
            AscendC::DataCopyPad(
                rt.out_f32_ub(), out_f32_gm[q_offset], output_copy, no_pad_f32);
        } else {
            AscendC::DataCopyExtParams output_copy{
                static_cast<uint16_t>(active_rows),
                BWD_DIM * sizeof(InputT),
                static_cast<int64_t>((q_stride_ - BWD_DIM) * sizeof(InputT)),
                0U,
                0U,
            };
            AscendC::DataCopyPad(
                rt.out_ub(), out_gm[q_offset], output_copy, no_pad);
        }
        AscendC::DataCopyExtParams dout_copy{
            static_cast<uint16_t>(active_rows),
            BWD_DIM * sizeof(InputT),
            static_cast<int64_t>((q_stride_ - BWD_DIM) * sizeof(InputT)),
            0U,
            0U,
        };
        AscendC::DataCopyPad(rt.dout_ub(), dout_gm[q_offset], dout_copy, no_pad);

        AscendC::DataCopyExtParams lse_copy{
            1U,
            static_cast<uint32_t>(active_rows * sizeof(float)),
            0U,
            0U,
            0U,
        };
        AscendC::DataCopyPadExtParams<float> no_lse_pad{false, 0U, 0U, 0.0F};
        AscendC::DataCopyPad(rt.lse_ub(), lse_gm[lse_offset], lse_copy, no_lse_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2_to_vec_event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2_to_vec_event_);
    }

    __aicore__ inline void copy_stages_to_a1(
        AscendC::LocalTensor<InputT> p_dst_a1, AscendC::LocalTensor<InputT> p_src_ub,
        AscendC::LocalTensor<InputT> ds_dst_a1, AscendC::LocalTensor<InputT> ds_src_ub) {
        constexpr uint32_t kBlocks = BWD_TILE_K / SWA_v5::prim::MMAD_UNIT;
        constexpr uint32_t block_len_32b =
            BWD_LOCAL_ROWS * SWA_v5::prim::MMAD_UNIT * sizeof(bfloat16_t) / 32U;
        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t dst_offset =
            subblock * BWD_LOCAL_ROWS * SWA_v5::prim::MMAD_UNIT;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vec_to_mte3_event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vec_to_mte3_event_);
        AscendC::DataCopyParams copy_params(
            static_cast<uint16_t>(kBlocks), static_cast<uint16_t>(block_len_32b), 1U,
            static_cast<uint16_t>(block_len_32b));
        AscendC::DataCopy(p_dst_a1[dst_offset], p_src_ub, copy_params);
        AscendC::DataCopy(ds_dst_a1[dst_offset], ds_src_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3_to_vec_event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3_to_vec_event_);
    }

    __aicore__ inline void zero_gm(AscendC::GlobalTensor<InputT>& dst_gm, uint32_t total_elems) {
        constexpr uint32_t kZeroChunkElems = BWD_LOCAL_ROWS * BWD_TILE_K;
        const uint32_t worker_idx = AscendC::GetBlockIdx();
        const uint32_t elems_per_worker =
            (total_elems + active_aiv_cores_ - 1U) / active_aiv_cores_;
        const uint32_t begin = min_u32(total_elems, worker_idx * elems_per_worker);
        const uint32_t end = min_u32(total_elems, begin + elems_per_worker);
        if (begin >= end) {
            return;
        }

        AscendC::LocalTensor<InputT> zero_ub = rt.score_stage_ub();
        AscendC::Duplicate<InputT>(
            zero_ub, static_cast<InputT>(0.0F), static_cast<int32_t>(kZeroChunkElems));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t offset = begin; offset < end; offset += kZeroChunkElems) {
            const uint32_t count = min_u32(kZeroChunkElems, end - offset);
            AscendC::DataCopyExtParams copy_params{
                1U,
                static_cast<uint32_t>(count * sizeof(InputT)),
                0U,
                0U,
                0U,
            };
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vec_to_mte3_event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vec_to_mte3_event_);
            AscendC::DataCopyPad(dst_gm[offset], zero_ub, copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3_to_vec_event_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3_to_vec_event_);
        }
    }

    BackwardRuntime rt;
    AscendC::GlobalTensor<InputT> out_gm;
    AscendC::GlobalTensor<float> out_f32_gm;
    AscendC::GlobalTensor<InputT> dout_gm;
    AscendC::GlobalTensor<InputT> dq_gm;
    AscendC::GlobalTensor<InputT> dk_gm;
    AscendC::GlobalTensor<InputT> dv_gm;
    AscendC::GlobalTensor<float> lse_gm;
    AscendC::GlobalTensor<float> workspace_gm;
    AscendC::GlobalTensor<half> color_fp16_gm;
    AscendC::TEventID mte2_to_vec_event_;
    AscendC::TEventID vec_to_mte2_event_;
    AscendC::TEventID fp16_mte2_to_vec_event_[2];
    AscendC::TEventID fp16_vec_to_mte2_event_[2];
    AscendC::TEventID vec_to_mte3_event_;
    AscendC::TEventID mte3_to_vec_event_;
    AscendC::TEventID vec_to_scalar_event_;
    AscendC::TEventID mte3_to_scalar_event_;
    uint64_t dq_workspace_elems_ = 0U;
    uint64_t dk_workspace_elems_ = 0U;
    uint64_t workspace_elems_ = 0U;
    uint64_t dkv_scratch_elems_ = 0U;
    uint32_t total_q_tokens_ = 0U;
    uint32_t total_kv_tokens_ = 0U;
    uint32_t q_stride_ = 0U;
    uint32_t kv_stride_ = 0U;
    uint32_t num_q_heads_ = 0U;
    uint32_t num_kv_heads_ = 0U;
    uint32_t active_aiv_cores_ = 0U;
    bool output_is_f32_ = false;
    bool direct_scaled_dq_ = false;
};

}  // namespace SWA_v5::Backward
