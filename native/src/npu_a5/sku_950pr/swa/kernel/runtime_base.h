#pragma once

#include "kernel/compute.h"
#include "kernel/tensor.h"

namespace SWA_v5 {

template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class PStageT>
struct AICoreVec_RT {
    static constexpr int QK_SLOTS = 2;
    static constexpr int BMM2_SLOTS = 2;
    static constexpr int SOFTMAX_STATE_SLOTS = 3;
    static constexpr int VEC2_OUT_SLOTS = 2;
    static constexpr int P_STAGE_SLOTS = 2;
    static constexpr uint32_t LOCAL_ROWS = TILE_Q / 2;
    static constexpr uint32_t QK_UB_ADDR = 0;
    static constexpr uint32_t QK_UB_SLOT_ELEMS = LOCAL_ROWS * TILE_L0;
    static constexpr uint32_t BMM2_UB_SLOT_ELEMS = LOCAL_ROWS * DIM;
    static constexpr uint32_t VEC2_OUT_SLOT_ELEMS = LOCAL_ROWS * DIM;
    static constexpr uint32_t P_STAGE_STRIDE_BLOCKS = LOCAL_ROWS + 1;
    static constexpr uint32_t P_STAGE_BLOCKS = (TILE_L0 / 16) * P_STAGE_STRIDE_BLOCKS;
    static constexpr uint32_t P_STAGE_UB_ELEMS = P_STAGE_BLOCKS * 16;
    static constexpr uint32_t P_STAGE_UB_ADDR = QK_UB_ADDR + QK_SLOTS * QK_UB_SLOT_ELEMS * sizeof(float);
    static constexpr uint32_t SOFTMAX_STATE_ELEMS = LOCAL_ROWS;
    static constexpr uint32_t SUM_UB_ADDR = P_STAGE_UB_ADDR + P_STAGE_SLOTS * P_STAGE_UB_ELEMS * sizeof(PStageT);
    static constexpr uint32_t MAX_UB_ADDR = SUM_UB_ADDR + SOFTMAX_STATE_SLOTS * SOFTMAX_STATE_ELEMS * sizeof(float);
    static constexpr uint32_t EXPMAX_UB_ADDR = MAX_UB_ADDR + SOFTMAX_STATE_SLOTS * SOFTMAX_STATE_ELEMS * sizeof(float);
    static constexpr uint32_t COMMON_UB_ADDR = EXPMAX_UB_ADDR + SOFTMAX_STATE_SLOTS * SOFTMAX_STATE_ELEMS * sizeof(float);
    static constexpr uint32_t COMMON_UB_BYTES = 2 * SOFTMAX_STATE_ELEMS * sizeof(float);
    static constexpr uint32_t MASK_UB_ADDR = COMMON_UB_ADDR + COMMON_UB_BYTES;
    static constexpr uint32_t MASK_UB_ELEMS = LOCAL_ROWS * TILE_L0;
    static constexpr uint32_t BMM2_UB_ADDR = MASK_UB_ADDR + MASK_UB_ELEMS * sizeof(uint8_t);
    static constexpr uint32_t VEC2_OUT_UB_ADDR = BMM2_UB_ADDR + BMM2_SLOTS * BMM2_UB_SLOT_ELEMS * sizeof(float);
    static constexpr uint32_t VEC2_OUT_CAST_UB_ADDR =
        VEC2_OUT_UB_ADDR + VEC2_OUT_SLOTS * VEC2_OUT_SLOT_ELEMS * sizeof(float);
    static constexpr uint32_t LSE_UB_ELEMS = LOCAL_ROWS;
    static constexpr uint32_t LSE_UB_ADDR = VEC2_OUT_CAST_UB_ADDR + VEC2_OUT_SLOT_ELEMS * sizeof(half);

    UBTensor_ND<float, LOCAL_ROWS, TILE_L0> QK_UB[QK_SLOTS];
    UBTensor_ND<float, LOCAL_ROWS, DIM> BMM2_UB[BMM2_SLOTS];
    UBTensor_ND<float, LOCAL_ROWS, DIM> VEC2_OUT_UB[VEC2_OUT_SLOTS];
    UBTensor_ND<half, LOCAL_ROWS, DIM> VEC2_OUT_CAST_UB;
    UBTensor_ND<bfloat16_t, LOCAL_ROWS, DIM> VEC2_OUT_CAST_BF16_UB;
    UBTensor_ND<float, 1, LSE_UB_ELEMS> LSE_UB;
    UBTensor_ND<PStageT, 1, P_STAGE_UB_ELEMS> P_STAGE_UB_DB[P_STAGE_SLOTS];
    UBTensor_ND<float, 1, SOFTMAX_STATE_ELEMS> SUM_UB_DB[SOFTMAX_STATE_SLOTS];
    UBTensor_ND<float, 1, SOFTMAX_STATE_ELEMS> MAX_UB_DB[SOFTMAX_STATE_SLOTS];
    UBTensor_ND<float, 1, SOFTMAX_STATE_ELEMS> EXPMAX_UB_DB[SOFTMAX_STATE_SLOTS];
    UBTensor_ND<uint8_t, LOCAL_ROWS, TILE_L0> MASK_UB;
    UBTensor_ND<uint8_t, 1, COMMON_UB_BYTES> COMMON_UB;
    SWA_v5::Compute::online_softmax::BufferCache<PStageT, QK_SLOTS, P_STAGE_SLOTS, SOFTMAX_STATE_SLOTS> VEC1_SOFTMAX_BUFFERS;
    AscendC::TEventID p_stage_ready = 0;
    AscendC::TEventID qk_dump_done = 0;

    __aicore__ inline uint32_t qk_ub_offset(uint32_t slot) {
        return QK_UB_ADDR + slot * QK_UB_SLOT_ELEMS * sizeof(float);
    }

    __aicore__ inline UBTensor_ND<float, LOCAL_ROWS, TILE_L0> qk_ub_abs(
        AscendC::TPipe& pipe, uint32_t slot) {
        UBTensor_ND<float, LOCAL_ROWS, TILE_L0> ub;
        ub.tensor = pipe.template GetAbsAddr<AscendC::TPosition::VECCALC, float>(
            qk_ub_offset(slot), QK_UB_SLOT_ELEMS);
        return ub;
    }

    __aicore__ inline uint32_t bmm2_ub_offset(uint32_t slot) {
        return BMM2_UB_ADDR + slot * BMM2_UB_SLOT_ELEMS * sizeof(float);
    }

    __aicore__ inline UBTensor_ND<float, LOCAL_ROWS, DIM> bmm2_ub_abs(
        AscendC::TPipe& pipe, uint32_t slot) {
        UBTensor_ND<float, LOCAL_ROWS, DIM> ub;
        ub.tensor = pipe.template GetAbsAddr<AscendC::TPosition::VECCALC, float>(
            bmm2_ub_offset(slot), BMM2_UB_SLOT_ELEMS);
        return ub;
    }

    __aicore__ inline uint32_t p_stage_ub_offset(uint32_t slot) {
        return P_STAGE_UB_ADDR + slot * P_STAGE_UB_ELEMS * sizeof(PStageT);
    }

    __aicore__ inline uint32_t state_ub_offset(uint32_t base, uint32_t slot) {
        return base + slot * SOFTMAX_STATE_ELEMS * sizeof(float);
    }

    __aicore__ inline uint32_t vec2_out_ub_offset(uint32_t slot) {
        return VEC2_OUT_UB_ADDR + slot * VEC2_OUT_SLOT_ELEMS * sizeof(float);
    }

    __aicore__ inline uint32_t vec2_out_cast_ub_offset() {
        return VEC2_OUT_CAST_UB_ADDR;
    }

    __aicore__ inline uint32_t lse_ub_offset() {
        return LSE_UB_ADDR;
    }

    __aicore__ inline void init(AscendC::TPipe& pipe) {
        for (uint32_t i = 0; i < QK_SLOTS; ++i) {
            QK_UB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, qk_ub_offset(i), QK_UB_SLOT_ELEMS);
            VEC1_SOFTMAX_BUFFERS.qk[i] = reinterpret_cast<__ubuf__ float *>(QK_UB[i].tensor.GetPhyAddr());
            AscendC::Duplicate<float>(QK_UB[i].tensor, 0.0F, static_cast<int32_t>(QK_UB_SLOT_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t i = 0; i < BMM2_SLOTS; ++i) {
            BMM2_UB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, bmm2_ub_offset(i), BMM2_UB_SLOT_ELEMS);
            AscendC::Duplicate<float>(BMM2_UB[i].tensor, 0.0F, static_cast<int32_t>(BMM2_UB_SLOT_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t i = 0; i < VEC2_OUT_SLOTS; ++i) {
            VEC2_OUT_UB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, vec2_out_ub_offset(i), VEC2_OUT_SLOT_ELEMS);
            AscendC::Duplicate<float>(VEC2_OUT_UB[i].tensor, 0.0F, static_cast<int32_t>(VEC2_OUT_SLOT_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        VEC2_OUT_CAST_UB.tensor = AscendC::LocalTensor<half>(
            AscendC::TPosition::VECCALC, vec2_out_cast_ub_offset(), VEC2_OUT_SLOT_ELEMS);
        VEC2_OUT_CAST_BF16_UB.tensor = AscendC::LocalTensor<bfloat16_t>(
            AscendC::TPosition::VECCALC, vec2_out_cast_ub_offset(), VEC2_OUT_SLOT_ELEMS);
        AscendC::Duplicate<half>(VEC2_OUT_CAST_UB.tensor, static_cast<half>(0.0F),
            static_cast<int32_t>(VEC2_OUT_SLOT_ELEMS));
        AscendC::PipeBarrier<PIPE_V>();
        LSE_UB.tensor = AscendC::LocalTensor<float>(
            AscendC::TPosition::VECCALC, lse_ub_offset(), LSE_UB_ELEMS);
        AscendC::Duplicate<float>(LSE_UB.tensor, 0.0F, static_cast<int32_t>(LSE_UB_ELEMS));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < P_STAGE_SLOTS; ++i) {
            P_STAGE_UB_DB[i].tensor = AscendC::LocalTensor<PStageT>(
                AscendC::TPosition::VECCALC, p_stage_ub_offset(i), P_STAGE_UB_ELEMS);
            VEC1_SOFTMAX_BUFFERS.p_stage[i] =
                reinterpret_cast<__ubuf__ PStageT *>(P_STAGE_UB_DB[i].tensor.GetPhyAddr());
            AscendC::Duplicate<PStageT>(P_STAGE_UB_DB[i].tensor,
                static_cast<PStageT>(0.0F), static_cast<int32_t>(P_STAGE_UB_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t i = 0; i < SOFTMAX_STATE_SLOTS; ++i) {
            SUM_UB_DB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, state_ub_offset(SUM_UB_ADDR, i), SOFTMAX_STATE_ELEMS);
            MAX_UB_DB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, state_ub_offset(MAX_UB_ADDR, i), SOFTMAX_STATE_ELEMS);
            EXPMAX_UB_DB[i].tensor = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECCALC, state_ub_offset(EXPMAX_UB_ADDR, i), SOFTMAX_STATE_ELEMS);
            VEC1_SOFTMAX_BUFFERS.sum[i] = reinterpret_cast<__ubuf__ float *>(SUM_UB_DB[i].tensor.GetPhyAddr());
            VEC1_SOFTMAX_BUFFERS.max[i] = reinterpret_cast<__ubuf__ float *>(MAX_UB_DB[i].tensor.GetPhyAddr());
            VEC1_SOFTMAX_BUFFERS.expmax[i] =
                reinterpret_cast<__ubuf__ float *>(EXPMAX_UB_DB[i].tensor.GetPhyAddr());
            AscendC::Duplicate<float>(SUM_UB_DB[i].tensor, 0.0F, static_cast<int32_t>(SOFTMAX_STATE_ELEMS));
            AscendC::Duplicate<float>(MAX_UB_DB[i].tensor, 0.0F, static_cast<int32_t>(SOFTMAX_STATE_ELEMS));
            AscendC::Duplicate<float>(EXPMAX_UB_DB[i].tensor, 0.0F, static_cast<int32_t>(SOFTMAX_STATE_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        COMMON_UB.tensor = AscendC::LocalTensor<uint8_t>(
            AscendC::TPosition::VECCALC, COMMON_UB_ADDR, COMMON_UB_BYTES);
        VEC1_SOFTMAX_BUFFERS.common = reinterpret_cast<__ubuf__ uint8_t *>(COMMON_UB.tensor.GetPhyAddr());
        AscendC::Duplicate<uint8_t>(COMMON_UB.tensor, 0, static_cast<int32_t>(COMMON_UB_BYTES));
        AscendC::PipeBarrier<PIPE_V>();
        MASK_UB.tensor = AscendC::LocalTensor<uint8_t>(
            AscendC::TPosition::VECCALC, MASK_UB_ADDR, MASK_UB_ELEMS);
        VEC1_SOFTMAX_BUFFERS.mask = reinterpret_cast<__ubuf__ uint8_t *>(MASK_UB.tensor.GetPhyAddr());
        AscendC::Duplicate<uint8_t>(MASK_UB.tensor, 0, static_cast<int32_t>(MASK_UB_ELEMS));
        AscendC::PipeBarrier<PIPE_V>();
        p_stage_ready = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        qk_dump_done = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
    }
};

}  // namespace SWA_v5
