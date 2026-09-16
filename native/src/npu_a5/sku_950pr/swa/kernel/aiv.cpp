#pragma once

#include "kernel/compute.h"
#include "kernel/copy.h"
#include "kernel/mask_helper.h"
#include "kernel/runtime.cpp"
#include "kernel/aic.cpp"
#include "basic_api/kernel_operator_dump_tensor_intf.h"

namespace SWA_v5::Pipeline::FullPath {

using SWA_v5::GMTensor_ND;
using SWA_v5::UBTensor_ND;

static constexpr uint32_t AIV_SUBCORES_PER_AIC = 2;

#ifndef SWA_V5_VEC1_STAGE_DEBUG_DUMP
#define SWA_V5_VEC1_STAGE_DEBUG_DUMP 0
#endif

class FullPathAIV {
public:
    template<typename RT>
    __aicore__ inline void init_backward_tokens(RT& rt) {
        for (uint32_t slot = 0U; slot < CUBE2_MM_SLOTS; ++slot) {
            rt.template notify_aic<PIPE_V>(event_slot(CUBE2_QK_CONSUMED_BASE, slot));
            rt.template notify_aic<PIPE_V>(event_slot(CUBE2_BMM2_CONSUMED_BASE, slot));
        }
    }

private:

    template<bool FIRST_TILE, bool IS_CAUSAL, typename RT, uint32_t TILE_Q, uint32_t TILE_L0, uint32_t DIM,
        uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U>
    __aicore__ inline void run_softmax(RT& rt,
        uint32_t qk_slot, uint32_t p_stage_slot, uint32_t summax_slot, uint32_t exp_slot,
        uint32_t q0_abs, uint32_t k0_local, uint32_t active_rows, uint32_t kv_len, float scale) {
        constexpr uint32_t LOCAL_ROWS = TILE_Q / AIV_SUBCORES_PER_AIC;
        const bool all_valid =
            MaskHelper::is_l0_tile_all_valid<TILE_L0, LOCAL_ROWS, WINDOW_LEFT, WINDOW_RIGHT, IS_CAUSAL>(
            q0_abs, k0_local, active_rows, kv_len);
        const SWA_v5::Compute::online_softmax::Buffers<typename RT::InputType> softmax_buffers =
            rt.vec.VEC1_SOFTMAX_BUFFERS.get(qk_slot, p_stage_slot, summax_slot, exp_slot);
        if (all_valid) {
            SWA_v5::Compute::online_softmax::compute_all_valid<typename RT::InputType, FIRST_TILE>(softmax_buffers, scale);
        } else {
            if (active_rows == LOCAL_ROWS && k0_local + TILE_L0 <= kv_len) {
                SWA_v5::Compute::online_softmax::compute_full_tile<typename RT::InputType, FIRST_TILE, WINDOW_LEFT, WINDOW_RIGHT,
                    GLOBAL_WINDOW_SIZE, IS_CAUSAL>(softmax_buffers, q0_abs, k0_local, scale);
            } else {
                SWA_v5::Compute::online_softmax::compute_partial_tile<typename RT::InputType, FIRST_TILE, WINDOW_LEFT, WINDOW_RIGHT,
                    GLOBAL_WINDOW_SIZE, IS_CAUSAL>(
                    softmax_buffers, q0_abs, k0_local, active_rows, kv_len, scale);
            }
        }
    }

    template<typename RT, bool IS_FORWARD, uint32_t TILE_Q>
    __aicore__ inline void maybe_write_softmax_lse(RT& rt, GMTensor_ND<float>& softmax_lse_gm,
        uint32_t q_global_begin, uint32_t q_head, uint32_t q_rows, uint32_t total_q_tokens) {
        if constexpr (IS_FORWARD) {
            constexpr uint32_t LOCAL_ROWS = TILE_Q / AIV_SUBCORES_PER_AIC;
            const uint32_t local_q_row_base = AscendC::GetSubBlockIdx() * LOCAL_ROWS;
            if (q_rows > local_q_row_base) {
                uint32_t active_rows = q_rows - local_q_row_base;
                active_rows = active_rows > LOCAL_ROWS ? LOCAL_ROWS : active_rows;
                const uint32_t lse_offset = q_head * total_q_tokens + q_global_begin + local_q_row_base;
                AscendC::DataCopyExtParams copy_params{1U, static_cast<uint32_t>(active_rows * sizeof(float)), 0, 0, 0};
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
                AscendC::DataCopyPad(softmax_lse_gm.tensor[lse_offset], rt.vec.LSE_UB.tensor, copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
            }
        }
    }

    template<typename RT, uint32_t TILE_Q, uint32_t TILE_L0, uint32_t DIM>
    __aicore__ inline void copy_p_to_a1(RT& rt,
        uint32_t p_stage_slot, uint32_t p_slot) {
        static_assert(TILE_Q == 128 && TILE_L0 == 128 && DIM == 128,
            "Cube2 P staging currently uses 3 slots of 128x128 P");
        constexpr uint32_t LOCAL_ROWS = TILE_Q / AIV_SUBCORES_PER_AIC;
        constexpr uint32_t MMAD_UNIT = SWA_v5::prim::MMAD_UNIT;
        constexpr uint32_t K_BLOCKS = TILE_L0 / MMAD_UNIT;
        constexpr uint32_t P_STAGE_STRIDE_BLOCKS = LOCAL_ROWS + 1;
        constexpr uint32_t P_A1_SLOT_STRIDE =
            RT::CUBE_RT::P_A1_SLOT_BYTES;
        const uint32_t p_a1_addr = p_slot * P_A1_SLOT_STRIDE;
        const uint32_t subblock = AscendC::GetSubBlockIdx();
        const uint32_t dst_offset = subblock * MMAD_UNIT * (TILE_Q - LOCAL_ROWS);
        AscendC::LocalTensor<typename RT::InputType> p_a1(AscendC::TPosition::A1, p_a1_addr, TILE_Q * TILE_L0);
        AscendC::DataCopyParams copy_params(
            static_cast<uint16_t>(K_BLOCKS), static_cast<uint16_t>(LOCAL_ROWS),
            static_cast<uint16_t>(P_STAGE_STRIDE_BLOCKS - LOCAL_ROWS),
            static_cast<uint16_t>(TILE_Q - LOCAL_ROWS));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
        AscendC::DataCopy(p_a1[dst_offset], rt.vec.P_STAGE_UB_DB[p_stage_slot].tensor, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
        rt.template notify_aic<PIPE_MTE3>(event_slot(CUBE2_P_READY_BASE, p_slot));
    }

public:
    static constexpr bool SupportsVec1SoftmaxMaskKind = false;

    __aicore__ static inline uint32_t aic_index(uint32_t aiv_idx, uint32_t num_aic_cores) {
        return (aiv_idx / AIV_SUBCORES_PER_AIC) % num_aic_cores;
    }


    template<typename RT, uint32_t TILE_Q, uint32_t TILE_L0, uint32_t DIM, bool IS_CAUSAL,
        uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U>
    __aicore__ inline void run_vec1(RT& rt,
        GMTensor_ND<uint8_t>& atten_mask_gm, uint32_t ordinal, uint32_t tile_idx,
        uint32_t q_abs_tile_base, uint32_t q_rows, uint32_t kv_len, uint32_t issue_idx, uint32_t q_owner_idx) {
        (void)atten_mask_gm;
        static_assert(TILE_Q == 128 && TILE_L0 == 128 && DIM == 128,
            "FullPathAIV is currently specialized for 128x128 Q/L0/D tiles");
        constexpr float SOFTMAX_SCALE = 0.08838834764831845F;
        const uint32_t qk_slot = issue_idx % CUBE2_MM_SLOTS;
        const uint32_t p_slot = issue_idx % CUBE2_P_SLOTS;
        const uint32_t p_stage_slot = issue_idx % RT::VEC_RT::P_STAGE_SLOTS;
        const uint32_t exp_slot = issue_idx % RT::VEC_RT::SOFTMAX_STATE_SLOTS;
        const uint32_t summax_slot = q_owner_idx % RT::VEC_RT::SOFTMAX_STATE_SLOTS;
        const uint32_t local_q_row_base = AscendC::GetSubBlockIdx() * (TILE_Q / AIV_SUBCORES_PER_AIC);
        const uint32_t q0_abs = q_abs_tile_base + local_q_row_base;
        const uint32_t k0_local = tile_idx * TILE_L0;
        uint32_t active_rows = 0U;
        if (q_rows > local_q_row_base) {
            active_rows = q_rows - local_q_row_base;
            constexpr uint32_t LOCAL_ROWS = TILE_Q / AIV_SUBCORES_PER_AIC;
            active_rows = active_rows > LOCAL_ROWS ? LOCAL_ROWS : active_rows;
        }
        rt.template wait_aic<PIPE_V>(event_slot(CUBE2_QK_READY_BASE, qk_slot));
        if (ordinal == 0) {
            run_softmax<true, IS_CAUSAL, RT, TILE_Q, TILE_L0, DIM, WINDOW_LEFT, WINDOW_RIGHT, GLOBAL_WINDOW_SIZE>(
                rt, qk_slot, p_stage_slot, summax_slot, exp_slot, q0_abs, k0_local, active_rows, kv_len,
                SOFTMAX_SCALE);
        } else {
            run_softmax<false, IS_CAUSAL, RT, TILE_Q, TILE_L0, DIM, WINDOW_LEFT, WINDOW_RIGHT, GLOBAL_WINDOW_SIZE>(
                rt, qk_slot, p_stage_slot, summax_slot, exp_slot, q0_abs, k0_local, active_rows, kv_len,
                SOFTMAX_SCALE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        rt.template notify_aic<PIPE_V>(event_slot(CUBE2_QK_CONSUMED_BASE, qk_slot));
        copy_p_to_a1<RT, TILE_Q, TILE_L0, DIM>(rt, p_stage_slot, p_slot);
    }

    template<typename RT, uint32_t TILE_Q, uint32_t TILE_L0, uint32_t DIM,
        bool IS_FORWARD, bool OUTPUT_FP16, bool OUTPUT_BF16, typename OutputT>
    __aicore__ inline void run_vec2(RT& rt,
        GMTensor_ND<OutputT>& out_gm, GMTensor_ND<float>& softmax_lse_gm,
        uint32_t q_global_begin, uint32_t q_head, uint32_t q_rows, uint32_t total_q_tokens,
        uint32_t ordinal, uint32_t valid_l0_count, uint32_t output_stride, uint32_t issue_idx,
        uint32_t q_owner_idx) {
        static_assert(TILE_Q == 128 && TILE_L0 == 128 && DIM == 128,
            "FullPathAIV is currently specialized for 128x128 Q/L0/D tiles");
        const uint32_t bmm2_slot = issue_idx % CUBE2_MM_SLOTS;
        const uint32_t exp_slot = issue_idx % RT::VEC_RT::SOFTMAX_STATE_SLOTS;
        const uint32_t out_slot = q_owner_idx % RT::VEC_RT::VEC2_OUT_SLOTS;
        const uint32_t summax_slot = q_owner_idx % RT::VEC_RT::SOFTMAX_STATE_SLOTS;
        const uint32_t local_q_row_base = AscendC::GetSubBlockIdx() * (TILE_Q / AIV_SUBCORES_PER_AIC);

        rt.template wait_aic<PIPE_V>(event_slot(CUBE2_BMM2_READY_BASE, bmm2_slot));
        const bool first_tile = ordinal == 0U;
        const bool last_tile = ordinal + 1U == valid_l0_count;
        __ubuf__ OutputT *out_cast_ub = reinterpret_cast<__ubuf__ OutputT *>(
            rt.vec.VEC2_OUT_UB[out_slot].tensor.GetPhyAddr());
        if constexpr (OUTPUT_BF16) {
            out_cast_ub = reinterpret_cast<__ubuf__ OutputT *>(rt.vec.VEC2_OUT_CAST_BF16_UB.tensor.GetPhyAddr());
        } else if constexpr (OUTPUT_FP16) {
            out_cast_ub = reinterpret_cast<__ubuf__ OutputT *>(rt.vec.VEC2_OUT_CAST_UB.tensor.GetPhyAddr());
        }
        constexpr bool WRITE_CAST = OUTPUT_FP16 || OUTPUT_BF16;
        if (first_tile && last_tile) {
            SWA_v5::Compute::online_softmax::finalize<OutputT, true, true, WRITE_CAST, IS_FORWARD>(
                rt.vec.VEC2_OUT_UB[out_slot], out_cast_ub, rt.vec.LSE_UB, rt.vec.BMM2_UB[bmm2_slot],
                rt.vec.EXPMAX_UB_DB[exp_slot], rt.vec.SUM_UB_DB[summax_slot], rt.vec.MAX_UB_DB[summax_slot]);
        } else if (first_tile) {
            SWA_v5::Compute::online_softmax::finalize<OutputT, true, false, WRITE_CAST, false>(
                rt.vec.VEC2_OUT_UB[out_slot], out_cast_ub, rt.vec.LSE_UB, rt.vec.BMM2_UB[bmm2_slot],
                rt.vec.EXPMAX_UB_DB[exp_slot], rt.vec.SUM_UB_DB[summax_slot], rt.vec.MAX_UB_DB[summax_slot]);
        } else if (last_tile) {
            SWA_v5::Compute::online_softmax::finalize<OutputT, false, true, WRITE_CAST, IS_FORWARD>(
                rt.vec.VEC2_OUT_UB[out_slot], out_cast_ub, rt.vec.LSE_UB, rt.vec.BMM2_UB[bmm2_slot],
                rt.vec.EXPMAX_UB_DB[exp_slot], rt.vec.SUM_UB_DB[summax_slot], rt.vec.MAX_UB_DB[summax_slot]);
        } else {
            SWA_v5::Compute::online_softmax::finalize<OutputT, false, false, WRITE_CAST, false>(
                rt.vec.VEC2_OUT_UB[out_slot], out_cast_ub, rt.vec.LSE_UB, rt.vec.BMM2_UB[bmm2_slot],
                rt.vec.EXPMAX_UB_DB[exp_slot], rt.vec.SUM_UB_DB[summax_slot], rt.vec.MAX_UB_DB[summax_slot]);
        }
        AscendC::PipeBarrier<PIPE_V>();

        if (last_tile && q_rows > local_q_row_base) {
            uint32_t active_rows = q_rows - local_q_row_base;
            constexpr uint32_t LOCAL_ROWS = TILE_Q / AIV_SUBCORES_PER_AIC;
            active_rows = active_rows > LOCAL_ROWS ? LOCAL_ROWS : active_rows;
            const uint32_t out_offset = (q_global_begin + local_q_row_base) * output_stride + q_head * DIM;
            AscendC::DataCopyParams copy_params(
                static_cast<uint16_t>(active_rows), static_cast<uint16_t>(DIM * sizeof(OutputT) / 32U),
                0, static_cast<uint16_t>((output_stride - DIM) * sizeof(OutputT) / 32U));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(rt.vec.p_stage_ready);
            if constexpr (OUTPUT_BF16) {
                AscendC::DataCopy(out_gm.tensor[out_offset], rt.vec.VEC2_OUT_CAST_BF16_UB.tensor, copy_params);
            } else if constexpr (OUTPUT_FP16) {
                AscendC::DataCopy(out_gm.tensor[out_offset], rt.vec.VEC2_OUT_CAST_UB.tensor, copy_params);
            } else {
                AscendC::DataCopy(out_gm.tensor[out_offset], rt.vec.VEC2_OUT_UB[out_slot].tensor, copy_params);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(rt.vec.qk_dump_done);
        }
        if (last_tile) {
            maybe_write_softmax_lse<RT, IS_FORWARD, TILE_Q>(
                rt, softmax_lse_gm, q_global_begin, q_head, q_rows, total_q_tokens);
        }
        rt.template notify_aic<PIPE_V>(event_slot(CUBE2_BMM2_CONSUMED_BASE, bmm2_slot));
    }
};

}  // namespace SWA_v5::Pipeline::FullPath
