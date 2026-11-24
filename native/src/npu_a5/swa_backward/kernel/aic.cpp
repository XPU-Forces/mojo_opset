#pragma once

#include "kernel/runtime.h"
#include "kernel/task_scheduler.h"

namespace SWA_v5::Backward {

struct AicStageTask {
    TaskInfo task;
    uint32_t issue_idx;
    uint32_t mm_slot;
    uint32_t p_ds_slot;
    uint32_t q_l1_slot;
    bool dq_first;
    bool dq_last;
    bool dq_atomic;
    bool direct_atomic;
    bool skip_dq;
    bool skip_dkv;
    bool dkv_color_partial;
    bool dkv_color_atomic;
    bool dkv_color_bf16;
    bool dkv_color_fp16;
    uint64_t dkv_color_offset;
    uint64_t dkv_color_dst_offset;
    uint64_t dkv_color_storage_elems;
};

class BackwardAIC {
    using InputT = SWA_BWD_ENTRY_INPUT_TYPE;

public:
    __aicore__ inline void init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dout,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, uint32_t batch,
        uint32_t total_q_tokens,
        uint32_t total_kv_tokens, uint32_t q_stride, uint32_t kv_stride,
        uint64_t workspace_elems, uint64_t dkv_scratch_elems,
        uint32_t dq_ring_rows,
        bool direct_scaled_dq) {
        q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(q));
        k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(k));
        v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(v));
        dout_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dout));
        dq_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dq));
        dk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dk));
        dv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(dv));
        workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        color_bf16_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(workspace) +
            2ULL * (workspace_elems + dkv_scratch_elems));
        color_fp16_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(workspace) +
            2ULL * (workspace_elems + dkv_scratch_elems));
        dq_workspace_elems_ =
            (static_cast<uint64_t>(total_q_tokens) +
                static_cast<uint64_t>(batch) * (BWD_TILE_Q - 1U)) * q_stride;
        dk_workspace_elems_ =
            (static_cast<uint64_t>(total_kv_tokens) +
                static_cast<uint64_t>(batch) * (BWD_TILE_K - 1U)) * kv_stride;
        q_stride_ = q_stride;
        kv_stride_ = kv_stride;
        workspace_elems_ = workspace_elems;
        dkv_scratch_elems_ = dkv_scratch_elems;
        dq_ring_rows_ = dq_ring_rows;
        direct_scaled_dq_ = direct_scaled_dq;
        l0_issue_ = 0U;
        has_preissued_c1_c2_ = false;
        rt.init();
        fix_to_scalar_event_ = rt.pipe.AllocEventID<AscendC::HardEvent::FIX_S>();
    }

    __aicore__ inline void prefetch_q(const TaskInfo& task, uint32_t slot) {
        const uint32_t q_offset =
            task.q_global_begin * q_stride_ + task.q_head * BWD_DIM;
        load_l1(rt.q_a1[slot], q_gm, q_stride_, q_offset, task.q_rows);
        load_l1(rt.dout_a1[slot], dout_gm, q_stride_, q_offset, task.q_rows);
    }

    __aicore__ inline void activate_q(uint32_t slot) {
        rt.q_a1[slot].consumer_wait();
        rt.dout_a1[slot].consumer_wait();
    }

    __aicore__ inline void finish_q(uint32_t slot) {
        rt.q_a1[slot].consumer_set();
        rt.dout_a1[slot].consumer_set();
    }

    __aicore__ inline void prefetch_kv(const TaskInfo& task) {
        const uint32_t kv_offset =
            task.kv_global_begin * kv_stride_ + task.kv_head * BWD_DIM;
        load_l1(rt.k_b1[0U], k_gm, kv_stride_, kv_offset, task.kv_rows);
        load_l1(rt.v_b1, v_gm, kv_stride_, kv_offset, task.kv_rows);
    }

    __aicore__ inline void activate_kv() {
        rt.k_b1[0U].consumer_wait();
        rt.v_b1.consumer_wait();
    }

    __aicore__ inline void finish_kv() {
        rt.k_b1[0U].consumer_set();
        rt.v_b1.consumer_set();
    }

    __aicore__ inline void finish() {
        for (uint32_t slot = 0U; slot < BWD_MM_SLOTS; ++slot) {
            rt.wait_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_CONSUMED_BASE, slot));
        }
        fence_grad_writes();
        rt.pipe.ReleaseEventID<AscendC::HardEvent::FIX_S>(fix_to_scalar_event_);
        rt.finish_aic();
    }

    __aicore__ inline void fence_grad_writes() {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(fix_to_scalar_event_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(fix_to_scalar_event_);
    }

    __aicore__ inline void sync_grad_round() {
        constexpr uint16_t GRAD_ROUND_SYNC_EVENT = 15U;
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(GRAD_ROUND_SYNC_EVENT);
        AscendC::CrossCoreWaitFlag<0, PIPE_FIX>(GRAD_ROUND_SYNC_EVENT);
    }

    __aicore__ inline void destroy() {
        rt.pipe.Destroy();
    }

    // C1: dP = dO * V^T. C2: S = Q * K^T.
    __aicore__ inline void run_c1_c2(const AicStageTask& stage) {
        if (has_preissued_c1_c2_ &&
            stage.issue_idx == preissued_issue_idx_) {
            finish_preissued_c1_c2(stage);
            return;
        }
        const uint32_t slot = stage.mm_slot;
        rt.wait_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_CONSUMED_BASE, slot));

        const uint32_t kv_offset =
            stage.task.kv_global_begin * kv_stride_ + stage.task.kv_head * BWD_DIM;
        const uint32_t k_l1_slot = stage.issue_idx % BWD_K_L1_SLOTS;
        uint32_t c1_ab_slot = 0U;
        uint32_t c1_co1_slot = 0U;
        issue_mm_with_resident_lhs<false>(
            rt.dout_a1[stage.q_l1_slot], rt.v_b1, v_gm,
            kv_stride_, kv_offset, BWD_TILE_K,
            c1_ab_slot, c1_co1_slot);
        uint32_t c2_ab_slot = 0U;
        uint32_t c2_co1_slot = 0U;
        issue_mm_with_resident_lhs<true>(
            rt.q_a1[stage.q_l1_slot], rt.k_b1[k_l1_slot], k_gm,
            kv_stride_, kv_offset, BWD_TILE_K,
            c2_ab_slot, c2_co1_slot);
        finish_mm_to_ub(c1_co1_slot, BWD_TILE_K, rt.c1_ub(slot));
        finish_mm_to_ub(c2_co1_slot, BWD_TILE_K, rt.c2_ub(slot));
        rt.notify_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_READY_BASE, slot));
    }

    __aicore__ inline void preissue_c1_c2(const AicStageTask& stage) {
        const uint32_t slot = stage.mm_slot;
        rt.wait_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_CONSUMED_BASE, slot));

        const uint32_t kv_offset =
            stage.task.kv_global_begin * kv_stride_ + stage.task.kv_head * BWD_DIM;
        const uint32_t k_l1_slot = stage.issue_idx % BWD_K_L1_SLOTS;
        uint32_t c1_ab_slot = 0U;
        issue_mm_with_resident_lhs<false>(
            rt.dout_a1[stage.q_l1_slot], rt.v_b1, v_gm,
            kv_stride_, kv_offset, BWD_TILE_K,
            c1_ab_slot, preissued_c1_co1_slot_);
        uint32_t c2_ab_slot = 0U;
        issue_mm_with_resident_lhs<true>(
            rt.q_a1[stage.q_l1_slot], rt.k_b1[k_l1_slot], k_gm,
            kv_stride_, kv_offset, BWD_TILE_K,
            c2_ab_slot, preissued_c2_co1_slot_);
        preissued_issue_idx_ = stage.issue_idx;
        has_preissued_c1_c2_ = true;
    }

    __aicore__ inline void run_kv_owned_c1_c2(
        const AicStageTask& stage) {
        const uint32_t slot = stage.mm_slot;
        rt.wait_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_CONSUMED_BASE, slot));
        run_mm_with_loaded_rhs(
            rt.dout_a1[stage.q_l1_slot], rt.v_b1,
            stage.task.q_rows, BWD_TILE_K, rt.c1_ub(slot));
        run_mm_with_loaded_rhs(
            rt.q_a1[stage.q_l1_slot], rt.k_b1[0U],
            stage.task.q_rows, BWD_TILE_K, rt.c2_ub(slot));
        rt.notify_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_READY_BASE, slot));
    }

    __aicore__ inline void run_kv_owned_grads(
        const AicStageTask& stage, bool first_q, bool last_q,
        bool use_dq_core_planes) {
        const uint32_t score_slot = stage.p_ds_slot;
        rt.wait_aiv<PIPE_MTE1>(bwd_event_slot(BWD_DS_READY_BASE, score_slot));
        rt.wait_aiv<PIPE_MTE1>(bwd_event_slot(BWD_P_READY_BASE, score_slot));

        const uint64_t kv_workspace_offset =
            (static_cast<uint64_t>(stage.task.kv_global_begin) +
                static_cast<uint64_t>(stage.task.batch_idx) *
                    (BWD_TILE_K - 1U)) *
                kv_stride_ +
            static_cast<uint64_t>(stage.task.kv_head) * BWD_DIM;
        const uint64_t kv_output_offset =
            static_cast<uint64_t>(stage.task.kv_global_begin) *
                kv_stride_ +
            static_cast<uint64_t>(stage.task.kv_head) * BWD_DIM;

        if (use_dq_core_planes) {
            run_grad_mm_transient(
                rt.ds_a1[score_slot], false, rt.k_b1[0U],
                BWD_TILE_Q, BWD_DIM, stage.task.kv_rows,
                workspace_gm[
                    workspace_elems_ + dkv_scratch_elems_ +
                    stage.dkv_color_offset + stage.dkv_color_dst_offset],
                q_stride_, stage.dq_atomic);
        } else {
            run_grad_mm_transient(
                rt.ds_a1[score_slot], false, rt.k_b1[0U],
                BWD_TILE_Q, BWD_DIM, stage.task.kv_rows,
                color_bf16_gm[
                    stage.dkv_color_offset + stage.dkv_color_dst_offset],
                q_stride_);
        }
        const bool direct_full_tile =
            stage.task.kv_rows == BWD_TILE_K &&
            kv_stride_ == BWD_DIM;
        if (direct_full_tile) {
            run_grad_mm_accumulate(
                rt.ds_a1[score_slot], rt.q_a1[stage.q_l1_slot],
                stage.task.q_rows, BWD_L0C_PIPELINE_SLOTS - 1U,
                first_q, last_q, dk_gm[kv_output_offset], kv_stride_);
        } else {
            run_grad_mm_accumulate(
                rt.ds_a1[score_slot], rt.q_a1[stage.q_l1_slot],
                stage.task.q_rows, BWD_L0C_PIPELINE_SLOTS - 1U,
                first_q, last_q,
                workspace_gm[dq_workspace_elems_ + kv_workspace_offset],
                kv_stride_);
        }
        if (direct_full_tile && last_q) {
            AscendC::PipeBarrier<PIPE_FIX>();
        }
        if (direct_full_tile) {
            run_grad_mm_accumulate(
                rt.p_a1[score_slot], rt.dout_a1[stage.q_l1_slot],
                stage.task.q_rows, BWD_DQ_ACCUM_L0C_SLOT,
                first_q, last_q, dv_gm[kv_output_offset], kv_stride_);
        } else {
            run_grad_mm_accumulate(
                rt.p_a1[score_slot], rt.dout_a1[stage.q_l1_slot],
                stage.task.q_rows, BWD_DQ_ACCUM_L0C_SLOT,
                first_q, last_q,
                workspace_gm[
                    dq_workspace_elems_ + dk_workspace_elems_ +
                    kv_workspace_offset],
                kv_stride_);
        }
        rt.notify_aiv<PIPE_FIX>(
            bwd_event_slot(BWD_P_DS_CONSUMED_BASE, score_slot));
    }

    // C3/C4 produce unscaled FP32 partials; final reduction applies attention scale.
    // C3: dQ += dS * K. C4: dK += dS^T * Q. C5: dV += P^T * dO.
    __aicore__ inline void run_c3_c4_c5(const AicStageTask& stage) {
        const uint32_t score_slot = stage.p_ds_slot;
        rt.wait_aiv<PIPE_MTE1>(bwd_event_slot(BWD_DS_READY_BASE, score_slot));
        rt.wait_aiv<PIPE_MTE1>(bwd_event_slot(BWD_P_READY_BASE, score_slot));

        const uint64_t q_workspace_row =
            direct_scaled_dq_ && dq_ring_rows_ > 0U ?
                static_cast<uint64_t>(stage.task.batch_idx) *
                    BWD_TILE_Q :
            dq_ring_rows_ > 0U ?
                (static_cast<uint64_t>(stage.task.q_tile) * BWD_TILE_Q) %
                    dq_ring_rows_ :
                stage.task.q_global_begin +
                    stage.task.batch_idx * (BWD_TILE_Q - 1U);
        const uint64_t q_workspace_offset =
            q_workspace_row * q_stride_ +
            stage.task.q_head * BWD_DIM;
        const uint32_t q_output_offset =
            stage.task.q_global_begin * q_stride_ +
            stage.task.q_head * BWD_DIM;
        const uint64_t kv_workspace_offset =
            (static_cast<uint64_t>(stage.task.kv_global_begin) +
                static_cast<uint64_t>(stage.task.batch_idx) *
                    (BWD_TILE_K - 1U)) *
                kv_stride_ +
            static_cast<uint64_t>(stage.task.kv_head) * BWD_DIM;
        const uint64_t scratch_slot =
            static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                BWD_DKV_SCRATCH_SLOTS +
            stage.issue_idx % BWD_DKV_SCRATCH_SLOTS;
        const uint64_t scratch_offset = workspace_elems_ +
            scratch_slot * BWD_DKV_GRAD_KINDS * BWD_GRAD_TILE_ELEMS;
        const uint64_t color_partial_offset =
            workspace_elems_ + dkv_scratch_elems_ +
            stage.dkv_color_offset;

        // C3: [q, kv] x [kv, d] -> [q, d].
        const uint32_t k_l1_slot = stage.issue_idx % BWD_K_L1_SLOTS;
        if (!stage.skip_dq) {
            if (direct_scaled_dq_ && stage.task.q_rows == BWD_TILE_Q) {
                run_dq_grad(rt.ds_a1[score_slot], rt.k_b1[k_l1_slot],
                    stage.task.q_rows, BWD_DIM, stage.task.kv_rows,
                    dq_gm[q_output_offset], q_stride_, stage.dq_first,
                    stage.dq_last, false);
            } else {
                run_dq_grad(rt.ds_a1[score_slot], rt.k_b1[k_l1_slot],
                    stage.task.q_rows, BWD_DIM, stage.task.kv_rows,
                    workspace_gm[q_workspace_offset], q_stride_, stage.dq_first,
                    stage.dq_last, direct_scaled_dq_ ? false : stage.dq_atomic);
            }
        } else {
            rt.k_b1[k_l1_slot].consumer_set();
        }
        // C4: [kv, q] x [q, d] -> [kv, d].
        if (!stage.skip_dkv && stage.dkv_color_fp16) {
            run_grad_mm_with_resident_rhs(
                rt.ds_a1[score_slot], true, rt.q_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                color_fp16_gm[
                    stage.dkv_color_offset + stage.dkv_color_dst_offset],
                kv_stride_, stage.dkv_color_atomic);
        } else if (!stage.skip_dkv && stage.dkv_color_bf16) {
            run_grad_mm_with_resident_rhs(
                rt.ds_a1[score_slot], true, rt.q_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                color_bf16_gm[
                    stage.dkv_color_offset + stage.dkv_color_dst_offset],
                kv_stride_, false);
        } else if (!stage.skip_dkv && stage.dkv_color_partial) {
            run_grad_mm_with_resident_rhs(
                rt.ds_a1[score_slot], true, rt.q_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[
                    color_partial_offset + stage.dkv_color_dst_offset],
                kv_stride_, stage.dkv_color_atomic);
        } else if (!stage.skip_dkv && stage.direct_atomic) {
            run_grad_mm_with_resident_rhs(
                rt.ds_a1[score_slot], true, rt.q_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[dq_workspace_elems_ + kv_workspace_offset],
                kv_stride_, true);
        } else if (!stage.skip_dkv) {
            run_grad_mm_with_resident_rhs(
                rt.ds_a1[score_slot], true, rt.q_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[scratch_offset], BWD_DIM, false);
        }
        // C5: [kv, q] x [q, d] -> [kv, d].
        if (!stage.skip_dkv && stage.dkv_color_fp16) {
            run_grad_mm_with_resident_rhs(
                rt.p_a1[score_slot], true, rt.dout_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                color_fp16_gm[
                    stage.dkv_color_offset +
                    stage.dkv_color_storage_elems +
                    stage.dkv_color_dst_offset],
                kv_stride_, stage.dkv_color_atomic);
        } else if (!stage.skip_dkv && stage.dkv_color_bf16) {
            run_grad_mm_with_resident_rhs(
                rt.p_a1[score_slot], true, rt.dout_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                color_bf16_gm[
                    stage.dkv_color_offset +
                    stage.dkv_color_storage_elems +
                    stage.dkv_color_dst_offset],
                kv_stride_, false);
        } else if (!stage.skip_dkv && stage.dkv_color_partial) {
            run_grad_mm_with_resident_rhs(
                rt.p_a1[score_slot], true, rt.dout_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[
                    color_partial_offset +
                    stage.dkv_color_storage_elems +
                    stage.dkv_color_dst_offset],
                kv_stride_, stage.dkv_color_atomic);
        } else if (!stage.skip_dkv && stage.direct_atomic) {
            run_grad_mm_with_resident_rhs(
                rt.p_a1[score_slot], true, rt.dout_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[
                    dq_workspace_elems_ + dk_workspace_elems_ +
                    kv_workspace_offset],
                kv_stride_, true);
        } else if (!stage.skip_dkv) {
            run_grad_mm_with_resident_rhs(
                rt.p_a1[score_slot], true, rt.dout_a1[stage.q_l1_slot],
                BWD_TILE_K, BWD_DIM, stage.task.q_rows,
                workspace_gm[scratch_offset + BWD_GRAD_TILE_ELEMS],
                BWD_DIM, false);
        }

        rt.notify_aiv<PIPE_FIX>(
            bwd_event_slot(BWD_P_DS_CONSUMED_BASE, score_slot));
        if (!stage.direct_atomic) {
            rt.notify_aiv<PIPE_FIX>(BWD_DKV_SCRATCH_READY_EVENT);
        }
    }

private:
    template<class L1Tensor>
    __aicore__ inline void load_l1(L1Tensor& dst, AscendC::GlobalTensor<InputT>& src,
        uint32_t src_stride, uint32_t src_offset, uint32_t rows) {
        dst.producer_wait();
        if (rows < BWD_TILE_Q) {
            constexpr uint32_t bytes_per_block = 32U;
            AscendC::InitConstValueParams<InputT> params(
                1U, static_cast<uint16_t>(BWD_TILE_Q * BWD_DIM * sizeof(InputT) / bytes_per_block),
                0U, static_cast<InputT>(0.0F));
            AscendC::InitConstValue(dst.tensor, params);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        SWA_v5::prim::copy_gm_to_l1_nz_full_rows(
            dst.tensor, src[src_offset], rows, BWD_DIM, src_stride, BWD_TILE_Q);
        dst.producer_set();
    }

    __aicore__ inline void next_l0_slots(uint32_t& ab_slot, uint32_t& co1_slot) {
        ab_slot = l0_issue_ % BWD_L0_AB_SLOTS;
        co1_slot = l0_issue_ % BWD_L0C_PIPELINE_SLOTS;
        ++l0_issue_;
    }

    __aicore__ inline void next_kv_owned_l0_slots(
        uint32_t& ab_slot, uint32_t& co1_slot) {
        ab_slot = l0_issue_ % BWD_L0_AB_SLOTS;
        co1_slot = l0_issue_ % (BWD_L0C_PIPELINE_SLOTS - 1U);
        ++l0_issue_;
    }

    template<bool RETAIN_RHS, class A1, class B1>
    __aicore__ inline void issue_mm_with_resident_lhs(
        A1& lhs_l1, B1& rhs_l1, AscendC::GlobalTensor<InputT>& rhs_gm,
        uint32_t rhs_stride, uint32_t rhs_offset, uint32_t n,
        uint32_t& ab_slot, uint32_t& co1_slot) {
        next_l0_slots(ab_slot, co1_slot);
        load_l1(rhs_l1, rhs_gm, rhs_stride, rhs_offset, n);
        rhs_l1.consumer_wait();

        rt.a2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_a2_nz<InputT, BWD_TILE_Q, BWD_DIM>(
            rt.a2[ab_slot].tensor, lhs_l1.tensor);
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::split_copy_l1_to_b2_zn_trans<
            InputT, BWD_TILE_K, BWD_DIM, 1U>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, 0U);
        rt.b2[ab_slot].producer_set();

        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();
        if constexpr (!RETAIN_RHS) {
            rhs_l1.consumer_set();
        }

        rt.mm_co1[co1_slot].producer_wait();
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_Q);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(BWD_DIM);
        mm_params.cmatrixInitVal = true;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = 0b11;
        AscendC::Mmad(
            rt.mm_co1[co1_slot].tensor, rt.a2[ab_slot].tensor,
            rt.b2[ab_slot].tensor, mm_params);
        rt.mm_co1[co1_slot].producer_set();
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();
    }

    __aicore__ inline void finish_mm_to_ub(
        uint32_t co1_slot, uint32_t n, AscendC::LocalTensor<float> dst_ub) {
        rt.mm_co1[co1_slot].consumer_wait();
        AscendC::FixpipeParamsC310<
            SWA_v5::prim::CFG_ROW_MAJOR_TO_UB.format> fixpipe_params;
        fixpipe_params.nSize = n;
        fixpipe_params.mSize = BWD_TILE_Q;
        fixpipe_params.srcStride = BWD_TILE_Q;
        fixpipe_params.dstStride = BWD_TILE_K;
        fixpipe_params.unitFlag = 0b11;
        fixpipe_params.dualDstCtl = 1;
        fixpipe_params.params.ndNum = 1;
        fixpipe_params.params.srcNdStride = 0;
        fixpipe_params.params.dstNdStride = 0;
        AscendC::Fixpipe<float, float, SWA_v5::prim::CFG_ROW_MAJOR_TO_UB>(
            dst_ub, rt.mm_co1[co1_slot].tensor, fixpipe_params);
        rt.mm_co1[co1_slot].consumer_set();
    }

    __aicore__ inline void finish_preissued_c1_c2(
        const AicStageTask& stage) {
        const uint32_t slot = stage.mm_slot;
        finish_mm_to_ub(
            preissued_c1_co1_slot_, BWD_TILE_K, rt.c1_ub(slot));
        finish_mm_to_ub(
            preissued_c2_co1_slot_, BWD_TILE_K, rt.c2_ub(slot));
        rt.notify_aiv<PIPE_FIX>(bwd_event_slot(BWD_MM_READY_BASE, slot));
        has_preissued_c1_c2_ = false;
    }

    template<class A1, class B1>
    __aicore__ inline void run_mm_with_loaded_rhs(
        A1& lhs_l1, B1& rhs_l1, uint32_t m, uint32_t n,
        AscendC::LocalTensor<float> dst_ub) {
        uint32_t ab_slot = 0U;
        uint32_t co1_slot = 0U;
        next_kv_owned_l0_slots(ab_slot, co1_slot);
        rt.a2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_a2_nz<InputT, BWD_TILE_Q, BWD_DIM>(
            rt.a2[ab_slot].tensor, lhs_l1.tensor);
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::split_copy_l1_to_b2_zn_trans<
            InputT, BWD_TILE_K, BWD_DIM, 1U>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, 0U);
        rt.b2[ab_slot].producer_set();
        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();

        rt.mm_co1[co1_slot].producer_wait();
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_Q);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(BWD_DIM);
        mm_params.cmatrixInitVal = true;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = 0b11;
        AscendC::Mmad(
            rt.mm_co1[co1_slot].tensor, rt.a2[ab_slot].tensor,
            rt.b2[ab_slot].tensor, mm_params);
        rt.mm_co1[co1_slot].producer_set();
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();

        rt.mm_co1[co1_slot].consumer_wait();
        AscendC::FixpipeParamsC310<
            SWA_v5::prim::CFG_ROW_MAJOR_TO_UB.format> fixpipe_params;
        fixpipe_params.nSize = n;
        fixpipe_params.mSize = BWD_TILE_Q;
        fixpipe_params.srcStride = BWD_TILE_Q;
        fixpipe_params.dstStride = BWD_TILE_K;
        fixpipe_params.unitFlag = 0b11;
        fixpipe_params.dualDstCtl = 1;
        fixpipe_params.params.ndNum = 1;
        fixpipe_params.params.srcNdStride = 0;
        fixpipe_params.params.dstNdStride = 0;
        AscendC::Fixpipe<float, float, SWA_v5::prim::CFG_ROW_MAJOR_TO_UB>(
            dst_ub, rt.mm_co1[co1_slot].tensor, fixpipe_params);
        rt.mm_co1[co1_slot].consumer_set();
    }

    template<bool RETAIN_RHS, class A1, class B1>
    __aicore__ inline void run_mm_with_resident_lhs(A1& lhs_l1, B1& rhs_l1,
        AscendC::GlobalTensor<InputT>& rhs_gm, uint32_t rhs_stride, uint32_t rhs_offset,
        uint32_t m, uint32_t n, AscendC::LocalTensor<float> dst_ub) {
        uint32_t ab_slot = 0U;
        uint32_t co1_slot = 0U;
        next_l0_slots(ab_slot, co1_slot);
        load_l1(rhs_l1, rhs_gm, rhs_stride, rhs_offset, n);
        rhs_l1.consumer_wait();

        rt.a2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_a2_nz<InputT, BWD_TILE_Q, BWD_DIM>(
            rt.a2[ab_slot].tensor, lhs_l1.tensor);
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::split_copy_l1_to_b2_zn_trans<InputT, BWD_TILE_K, BWD_DIM, 1U>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, 0U);
        rt.b2[ab_slot].producer_set();

        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();
        if constexpr (!RETAIN_RHS) {
            rhs_l1.consumer_set();
        }

        rt.mm_co1[co1_slot].producer_wait();
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_Q);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(BWD_DIM);
        mm_params.cmatrixInitVal = true;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = 0b11;
        AscendC::Mmad(
            rt.mm_co1[co1_slot].tensor, rt.a2[ab_slot].tensor, rt.b2[ab_slot].tensor, mm_params);
        rt.mm_co1[co1_slot].producer_set();
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();

        rt.mm_co1[co1_slot].consumer_wait();
        AscendC::FixpipeParamsC310<SWA_v5::prim::CFG_ROW_MAJOR_TO_UB.format> fixpipe_params;
        fixpipe_params.nSize = n;
        fixpipe_params.mSize = BWD_TILE_Q;
        fixpipe_params.srcStride = BWD_TILE_Q;
        fixpipe_params.dstStride = BWD_TILE_K;
        fixpipe_params.unitFlag = 0b11;
        fixpipe_params.dualDstCtl = 1;
        fixpipe_params.params.ndNum = 1;
        fixpipe_params.params.srcNdStride = 0;
        fixpipe_params.params.dstNdStride = 0;
        AscendC::Fixpipe<float, float, SWA_v5::prim::CFG_ROW_MAJOR_TO_UB>(
            dst_ub, rt.mm_co1[co1_slot].tensor, fixpipe_params);
        rt.mm_co1[co1_slot].consumer_set();
    }

    template<class B1, class DstT>
    __aicore__ inline void run_dq_grad(AscendC::LocalTensor<InputT> score_a1,
        B1& rhs_l1, uint32_t m, uint32_t n, uint32_t k,
        AscendC::GlobalTensor<DstT> dst_gm,
        uint32_t dst_stride, bool first, bool last, bool atomic) {
        uint32_t ab_slot = 0U;
        uint32_t unused_co1_slot = 0U;
        next_l0_slots(ab_slot, unused_co1_slot);

        rt.a2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_a2_nz<InputT, BWD_TILE_Q, BWD_TILE_K>(
            rt.a2[ab_slot].tensor, score_a1);
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_b2_zn_rows<InputT, BWD_TILE_Q, BWD_DIM>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, k);
        rt.b2[ab_slot].producer_set();

        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();
        rhs_l1.consumer_set();

        auto& dq_accum = rt.mm_co1[BWD_DQ_ACCUM_L0C_SLOT];
        if (first) {
            dq_accum.producer_wait();
        }
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_Q);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(k);
        mm_params.cmatrixInitVal = first;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = last ? 0b11 : 0U;
        AscendC::Mmad(
            dq_accum.tensor, rt.a2[ab_slot].tensor, rt.b2[ab_slot].tensor, mm_params);
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();

        if (last) {
            dq_accum.producer_set();
            dq_accum.consumer_wait();
            fixpipe_grad(dq_accum.tensor, dst_gm, m, n, dst_stride, atomic, false);
            dq_accum.consumer_set();
        }
    }

    template<class B1, class DstT>
    __aicore__ inline void run_grad_mm_with_resident_rhs(
        AscendC::LocalTensor<InputT> score_a1, bool transpose_score, B1& rhs_l1,
        uint32_t m, uint32_t n, uint32_t k, AscendC::GlobalTensor<DstT> dst_gm,
        uint32_t dst_stride, bool atomic = true) {
        uint32_t ab_slot = 0U;
        uint32_t co1_slot = 0U;
        next_l0_slots(ab_slot, co1_slot);
        rt.a2[ab_slot].producer_wait();
        if (transpose_score) {
            SWA_v5::prim::copy_l1_to_a2_nz_trans_rows<
                InputT, BWD_TILE_Q, BWD_TILE_K>(
                rt.a2[ab_slot].tensor, score_a1, k);
        } else {
            SWA_v5::prim::copy_l1_to_a2_nz<InputT, BWD_TILE_Q, BWD_TILE_K>(
                rt.a2[ab_slot].tensor, score_a1);
        }
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_b2_zn_rows<InputT, BWD_TILE_Q, BWD_DIM>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, k);
        rt.b2[ab_slot].producer_set();

        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();

        rt.mm_co1[co1_slot].producer_wait();
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_Q);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(k);
        mm_params.cmatrixInitVal = true;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = 0b11;
        AscendC::Mmad(
            rt.mm_co1[co1_slot].tensor, rt.a2[ab_slot].tensor, rt.b2[ab_slot].tensor, mm_params);
        rt.mm_co1[co1_slot].producer_set();
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();

        rt.mm_co1[co1_slot].consumer_wait();
        fixpipe_grad(
            rt.mm_co1[co1_slot].tensor, dst_gm, m, n, dst_stride, atomic);
        rt.mm_co1[co1_slot].consumer_set();
    }

    template<class B1, class DstT>
    __aicore__ inline void run_grad_mm_transient(
        AscendC::LocalTensor<InputT> score_a1, bool transpose_score,
        B1& rhs_l1, uint32_t m, uint32_t n, uint32_t k,
        AscendC::GlobalTensor<DstT> dst_gm, uint32_t dst_stride,
        bool atomic = false) {
        uint32_t ab_slot = 0U;
        uint32_t co1_slot = 0U;
        next_kv_owned_l0_slots(ab_slot, co1_slot);
        rt.a2[ab_slot].producer_wait();
        if (transpose_score) {
            SWA_v5::prim::copy_l1_to_a2_nz_trans<
                InputT, BWD_TILE_Q, BWD_TILE_K>(
                rt.a2[ab_slot].tensor, score_a1);
        } else {
            SWA_v5::prim::copy_l1_to_a2_nz<
                InputT, BWD_TILE_Q, BWD_TILE_K>(
                rt.a2[ab_slot].tensor, score_a1);
        }
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_b2_zn_rows<
            InputT, BWD_TILE_Q, BWD_DIM>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, k);
        rt.b2[ab_slot].producer_set();
        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();

        rt.mm_co1[co1_slot].producer_wait();
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(m);
        mm_params.n = static_cast<uint16_t>(n);
        mm_params.k = static_cast<uint16_t>(k);
        mm_params.cmatrixInitVal = true;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = 0b11;
        AscendC::Mmad(
            rt.mm_co1[co1_slot].tensor, rt.a2[ab_slot].tensor,
            rt.b2[ab_slot].tensor, mm_params);
        rt.mm_co1[co1_slot].producer_set();
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();
        rt.mm_co1[co1_slot].consumer_wait();
        fixpipe_grad(
            rt.mm_co1[co1_slot].tensor, dst_gm,
            m, n, dst_stride, atomic);
        rt.mm_co1[co1_slot].consumer_set();
    }

    template<class B1, class DstT>
    __aicore__ inline void run_grad_mm_accumulate(
        AscendC::LocalTensor<InputT> score_a1, B1& rhs_l1,
        uint32_t k, uint32_t accum_slot, bool first, bool last,
        AscendC::GlobalTensor<DstT> dst_gm, uint32_t dst_stride) {
        uint32_t ab_slot = 0U;
        uint32_t unused_co1_slot = 0U;
        next_kv_owned_l0_slots(ab_slot, unused_co1_slot);
        rt.a2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_a2_nz_trans<
            InputT, BWD_TILE_Q, BWD_TILE_K>(
            rt.a2[ab_slot].tensor, score_a1);
        rt.a2[ab_slot].producer_set();
        rt.b2[ab_slot].producer_wait();
        SWA_v5::prim::copy_l1_to_b2_zn_rows<
            InputT, BWD_TILE_Q, BWD_DIM>(
            rt.b2[ab_slot].tensor, rhs_l1.tensor, k);
        rt.b2[ab_slot].producer_set();
        rt.a2[ab_slot].consumer_wait();
        rt.b2[ab_slot].consumer_wait();

        auto& accum = rt.mm_co1[accum_slot];
        if (first) {
            accum.producer_wait();
        }
        AscendC::MmadParams mm_params;
        mm_params.m = static_cast<uint16_t>(BWD_TILE_K);
        mm_params.n = static_cast<uint16_t>(BWD_DIM);
        mm_params.k = static_cast<uint16_t>(k);
        mm_params.cmatrixInitVal = first;
        mm_params.cmatrixSource = false;
        mm_params.unitFlag = last ? 0b11 : 0U;
        AscendC::Mmad(
            accum.tensor, rt.a2[ab_slot].tensor,
            rt.b2[ab_slot].tensor, mm_params);
        rt.a2[ab_slot].consumer_set();
        rt.b2[ab_slot].consumer_set();
        if (last) {
            accum.producer_set();
            accum.consumer_wait();
            fixpipe_grad(
                accum.tensor, dst_gm, BWD_TILE_K, BWD_DIM,
                dst_stride, false);
            accum.consumer_set();
        }
    }

    template<class DstT>
    __aicore__ inline void fixpipe_grad(
        AscendC::LocalTensor<float> src_co1, AscendC::GlobalTensor<DstT> dst_gm,
        uint32_t rows, uint32_t cols, uint32_t dst_stride, bool atomic = true,
        bool scale_output = false) {
        AscendC::FixpipeParamsArch3510<SWA_v5::prim::CFG_ROW_MAJOR_TO_GM.format> params;
        params.nSize = cols;
        params.mSize = BWD_TILE_Q;
        params.srcStride = BWD_TILE_Q;
        params.dstStride = dst_stride;
        if (scale_output) {
            params.quantPre = QuantMode_t::QF322F32_PRE;
            params.deqScalar = 0x3DB504F3U;
        } else if constexpr (AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        } else {
            params.quantPre = QuantMode_t::NoQuant;
        }
        params.reluEn = false;
        params.unitFlag = 0b11;
        params.dualDstCtl = 0;
        params.subBlockId = false;
        if (atomic) {
            if constexpr (AscendC::IsSameType<DstT, float>::value) {
                AscendC::SetAtomicAdd<float>();
            } else if constexpr (AscendC::IsSameType<DstT, half>::value) {
                AscendC::SetAtomicAdd<half>();
            }
        }
        AscendC::Fixpipe<DstT, float, SWA_v5::prim::CFG_ROW_MAJOR_TO_GM>(
            dst_gm, src_co1, params);
        if (atomic) {
            AscendC::DisableDmaAtomic();
        }
    }

    BackwardRuntime rt;
    AscendC::GlobalTensor<InputT> q_gm;
    AscendC::GlobalTensor<InputT> k_gm;
    AscendC::GlobalTensor<InputT> v_gm;
    AscendC::GlobalTensor<InputT> dout_gm;
    AscendC::GlobalTensor<InputT> dq_gm;
    AscendC::GlobalTensor<InputT> dk_gm;
    AscendC::GlobalTensor<InputT> dv_gm;
    AscendC::GlobalTensor<float> workspace_gm;
    AscendC::GlobalTensor<bfloat16_t> color_bf16_gm;
    AscendC::GlobalTensor<half> color_fp16_gm;
    uint64_t dq_workspace_elems_ = 0U;
    uint64_t dk_workspace_elems_ = 0U;
    uint64_t workspace_elems_ = 0U;
    uint64_t dkv_scratch_elems_ = 0U;
    uint32_t dq_ring_rows_ = 0U;
    uint32_t q_stride_ = 0U;
    uint32_t kv_stride_ = 0U;
    uint32_t l0_issue_ = 0U;
    uint32_t preissued_issue_idx_ = 0U;
    uint32_t preissued_c1_co1_slot_ = 0U;
    uint32_t preissued_c2_co1_slot_ = 0U;
    bool direct_scaled_dq_ = false;
    bool has_preissued_c1_c2_ = false;
    AscendC::TEventID fix_to_scalar_event_;
};

}  // namespace SWA_v5::Backward
