#pragma once

#include "kernel_operator.h"
#include "primitive.h"
#include "../../swa/kernel/tensor.h"

#ifndef SWA_BWD_ENTRY_INPUT_TYPE
#define SWA_BWD_ENTRY_INPUT_TYPE bfloat16_t
#endif

namespace SWA_v5::Backward {

static constexpr uint32_t BWD_TILE_Q = 128U;
static constexpr uint32_t BWD_TILE_K = 128U;
static constexpr uint32_t BWD_DIM = 128U;
static constexpr uint32_t BWD_LOCAL_ROWS = BWD_TILE_Q / 2U;
static constexpr uint32_t BWD_MM_SLOTS = 2U;
static constexpr uint32_t BWD_P_DS_SLOTS = 2U;
static constexpr uint32_t BWD_L0_AB_SLOTS = 2U;
static constexpr uint32_t BWD_Q_L1_SLOTS = 2U;
static constexpr uint32_t BWD_K_L1_SLOTS = 3U;
static constexpr uint32_t BWD_L0C_PIPELINE_SLOTS = 3U;
static constexpr uint32_t BWD_DQ_ACCUM_L0C_SLOT = BWD_L0C_PIPELINE_SLOTS;
static constexpr uint32_t BWD_L0C_SLOTS = BWD_L0C_PIPELINE_SLOTS + 1U;
static constexpr uint32_t BWD_AIVS_PER_AIC = 2U;
static constexpr uint32_t BWD_DKV_SCRATCH_SLOTS = 2U;
static constexpr uint32_t BWD_DKV_GRAD_KINDS = 2U;
static constexpr uint32_t BWD_GRAD_TILE_ELEMS = BWD_TILE_Q * BWD_DIM;

__aicore__ inline uint64_t bwd_dkv_scratch_elems(uint32_t aic_cores) {
    return static_cast<uint64_t>(aic_cores) * BWD_DKV_SCRATCH_SLOTS *
        BWD_DKV_GRAD_KINDS * BWD_GRAD_TILE_ELEMS;
}

static constexpr uint16_t BWD_MM_CONSUMED_BASE = 0U;
static constexpr uint16_t BWD_MM_READY_BASE = 2U;
static constexpr uint16_t BWD_P_DS_CONSUMED_BASE = 4U;
static constexpr uint16_t BWD_DS_READY_BASE = 6U;
static constexpr uint16_t BWD_P_READY_BASE = 8U;
static constexpr uint16_t BWD_DKV_SCRATCH_READY_EVENT = 10U;

__aicore__ inline uint16_t bwd_event_slot(uint16_t base, uint32_t slot) {
    return static_cast<uint16_t>(base + slot);
}

class BackwardRuntime {
    using InputT = SWA_BWD_ENTRY_INPUT_TYPE;
    static constexpr uint32_t TILE_ELEMS = BWD_TILE_Q * BWD_DIM;
    static constexpr uint32_t LOCAL_MM_ELEMS = BWD_LOCAL_ROWS * BWD_TILE_K;
    static constexpr uint32_t C1_UB_ADDR = 0U;
    static constexpr uint32_t C2_UB_ADDR =
        C1_UB_ADDR + BWD_MM_SLOTS * LOCAL_MM_ELEMS * sizeof(float);
    static constexpr uint32_t SCORE_STAGE_UB_ADDR =
        C2_UB_ADDR + BWD_MM_SLOTS * LOCAL_MM_ELEMS * sizeof(float);
    static constexpr uint32_t SCORE_STAGE_ROWS = BWD_LOCAL_ROWS + 1U;
    static constexpr uint32_t SCORE_STAGE_ELEMS = SCORE_STAGE_ROWS * BWD_TILE_K;
    static constexpr uint32_t OUT_UB_ADDR =
        SCORE_STAGE_UB_ADDR + SCORE_STAGE_ELEMS * sizeof(InputT);
    static constexpr uint32_t DOUT_UB_ADDR =
        OUT_UB_ADDR + LOCAL_MM_ELEMS * sizeof(InputT);
    static constexpr uint32_t D_UB_ADDR =
        DOUT_UB_ADDR + LOCAL_MM_ELEMS * sizeof(InputT);
    static constexpr uint32_t LSE_UB_ADDR =
        D_UB_ADDR + BWD_LOCAL_ROWS * sizeof(float);
    static constexpr uint32_t UB_BYTES =
        LSE_UB_ADDR + BWD_LOCAL_ROWS * sizeof(float);
    static constexpr uint32_t Q_A1_ADDR = 0U;
    static constexpr uint32_t DOUT_A1_ADDR =
        Q_A1_ADDR + BWD_Q_L1_SLOTS * TILE_ELEMS * sizeof(InputT);
    static constexpr uint32_t DS_A1_ADDR =
        DOUT_A1_ADDR + BWD_Q_L1_SLOTS * TILE_ELEMS * sizeof(InputT);
    static constexpr uint32_t P_A1_ADDR =
        DS_A1_ADDR + BWD_P_DS_SLOTS * TILE_ELEMS * sizeof(InputT);

public:
    static_assert(
        UB_BYTES <= 192U * 1024U,
        "backward vector buffers exceed UB capacity");

    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::A1> q_a1_tbuf[BWD_Q_L1_SLOTS];
    AscendC::TBuf<AscendC::TPosition::A1> dout_a1_tbuf[BWD_Q_L1_SLOTS];
    AscendC::TBuf<AscendC::TPosition::A1> ds_a1_tbuf[BWD_P_DS_SLOTS];
    AscendC::TBuf<AscendC::TPosition::A1> p_a1_tbuf[BWD_P_DS_SLOTS];
    AscendC::TBuf<AscendC::TPosition::B1> k_b1_tbuf[BWD_K_L1_SLOTS];
    AscendC::TBuf<AscendC::TPosition::B1> v_b1_tbuf;
    AscendC::TBuf<AscendC::TPosition::A2> a2_tbuf;
    AscendC::TBuf<AscendC::TPosition::B2> b2_tbuf;
    AscendC::TBuf<AscendC::TPosition::CO1> mm_co1_tbuf;

    SWA_v5::A1Tensor<InputT, BWD_TILE_Q, BWD_DIM> q_a1[BWD_Q_L1_SLOTS];
    SWA_v5::A1Tensor<InputT, BWD_TILE_Q, BWD_DIM> dout_a1[BWD_Q_L1_SLOTS];
    AscendC::LocalTensor<InputT> ds_a1[BWD_P_DS_SLOTS];
    AscendC::LocalTensor<InputT> p_a1[BWD_P_DS_SLOTS];
    SWA_v5::B1Tensor<InputT, BWD_TILE_K, BWD_DIM> k_b1[BWD_K_L1_SLOTS];
    SWA_v5::B1Tensor<InputT, BWD_TILE_K, BWD_DIM> v_b1;
    SWA_v5::A2Tensor<InputT, BWD_TILE_Q, BWD_DIM> a2[BWD_L0_AB_SLOTS];
    SWA_v5::B2Tensor<InputT, BWD_DIM, BWD_TILE_K> b2[BWD_L0_AB_SLOTS];
    SWA_v5::CO1Tensor<float, BWD_TILE_Q, BWD_TILE_K> mm_co1[BWD_L0C_SLOTS];

    __aicore__ inline void init() {
        if constexpr (g_coreType == AscendC::AIC) {
            init_aic();
        } else if constexpr (g_coreType == AscendC::AIV) {
            init_aiv();
        }
    }

    __aicore__ inline void finish_aic() {
        for (uint32_t slot = 0U; slot < BWD_L0C_SLOTS; ++slot) {
            mm_co1[slot].release_id();
        }
        for (uint32_t slot = 0U; slot < BWD_L0_AB_SLOTS; ++slot) {
            b2[slot].release_id();
            a2[slot].release_id();
        }
        v_b1.release_id();
        for (uint32_t slot = 0U; slot < BWD_K_L1_SLOTS; ++slot) {
            k_b1[slot].release_id();
        }
        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            dout_a1[slot].release_id();
            q_a1[slot].release_id();
        }
    }

    __aicore__ inline AscendC::LocalTensor<float> c1_ub(uint32_t slot) {
        return ub_at(C1_UB_ADDR + slot * LOCAL_MM_ELEMS * sizeof(float));
    }

    __aicore__ inline AscendC::LocalTensor<float> c2_ub(uint32_t slot) {
        return ub_at(C2_UB_ADDR + slot * LOCAL_MM_ELEMS * sizeof(float));
    }

    __aicore__ inline AscendC::LocalTensor<InputT> ds_stage_ub() {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::VECCALC, OUT_UB_ADDR, SCORE_STAGE_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<InputT> score_stage_ub() {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::VECCALC, SCORE_STAGE_UB_ADDR, SCORE_STAGE_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<half> fp16_reduce_ub(
        uint32_t slot) {
        const uint32_t byte_offset = slot == 0U ?
            SCORE_STAGE_UB_ADDR : OUT_UB_ADDR;
        return AscendC::LocalTensor<half>(
            AscendC::TPosition::VECCALC, byte_offset, LOCAL_MM_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<InputT> out_ub() {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::VECCALC, OUT_UB_ADDR, LOCAL_MM_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<float> out_f32_ub() {
        return AscendC::LocalTensor<float>(
            AscendC::TPosition::VECCALC, SCORE_STAGE_UB_ADDR, LOCAL_MM_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<InputT> dout_ub() {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::VECCALC, DOUT_UB_ADDR, LOCAL_MM_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<float> d_ub() {
        return AscendC::LocalTensor<float>(
            AscendC::TPosition::VECCALC, D_UB_ADDR, BWD_LOCAL_ROWS);
    }

    __aicore__ inline AscendC::LocalTensor<float> lse_ub() {
        return AscendC::LocalTensor<float>(
            AscendC::TPosition::VECCALC, LSE_UB_ADDR, BWD_LOCAL_ROWS);
    }

    __aicore__ inline AscendC::LocalTensor<InputT> ds_a1_for_aiv(uint32_t slot) {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::A1, DS_A1_ADDR + slot * TILE_ELEMS * sizeof(InputT), TILE_ELEMS);
    }

    __aicore__ inline AscendC::LocalTensor<InputT> p_a1_for_aiv(uint32_t slot) {
        return AscendC::LocalTensor<InputT>(
            AscendC::TPosition::A1, P_A1_ADDR + slot * TILE_ELEMS * sizeof(InputT), TILE_ELEMS);
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void notify_aiv(uint16_t event_id) {
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(event_id);
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(static_cast<uint16_t>(event_id + 16U));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void wait_aiv(uint16_t event_id) {
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(event_id);
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(static_cast<uint16_t>(event_id + 16U));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void notify_aic(uint16_t event_id) {
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(local_subblock_event(event_id));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void notify_aic_pair(uint16_t event_id) {
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(event_id);
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(
            static_cast<uint16_t>(event_id + 16U));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void wait_aic(uint16_t event_id) {
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(local_subblock_event(event_id));
    }

private:
    __aicore__ inline AscendC::LocalTensor<float> ub_at(uint32_t byte_offset) {
        if constexpr (g_coreType == AscendC::AIC) {
            return pipe.GetAbsAddr<AscendC::TPosition::VECCALC, float>(
                byte_offset, LOCAL_MM_ELEMS);
        }
        return AscendC::LocalTensor<float>(
            AscendC::TPosition::VECCALC, byte_offset, LOCAL_MM_ELEMS);
    }

    __aicore__ static inline uint16_t local_subblock_event(uint16_t event_id) {
        return static_cast<uint16_t>(event_id + AscendC::GetSubBlockIdx() * 16U);
    }

    __aicore__ inline void init_aic() {
        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            pipe.InitBuffer(q_a1_tbuf[slot], TILE_ELEMS * sizeof(InputT));
        }
        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            pipe.InitBuffer(dout_a1_tbuf[slot], TILE_ELEMS * sizeof(InputT));
        }
        for (uint32_t slot = 0U; slot < BWD_P_DS_SLOTS; ++slot) {
            pipe.InitBuffer(ds_a1_tbuf[slot], TILE_ELEMS * sizeof(InputT));
        }
        for (uint32_t slot = 0U; slot < BWD_P_DS_SLOTS; ++slot) {
            pipe.InitBuffer(p_a1_tbuf[slot], TILE_ELEMS * sizeof(InputT));
        }
        for (uint32_t slot = 0U; slot < BWD_K_L1_SLOTS; ++slot) {
            pipe.InitBuffer(k_b1_tbuf[slot], TILE_ELEMS * sizeof(InputT));
        }
        pipe.InitBuffer(v_b1_tbuf, TILE_ELEMS * sizeof(InputT));
        pipe.InitBuffer(a2_tbuf, BWD_L0_AB_SLOTS * TILE_ELEMS * sizeof(InputT));
        pipe.InitBuffer(b2_tbuf, BWD_L0_AB_SLOTS * TILE_ELEMS * sizeof(InputT));
        pipe.InitBuffer(mm_co1_tbuf, BWD_L0C_SLOTS * TILE_ELEMS * sizeof(float));

        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            q_a1[slot].tensor = q_a1_tbuf[slot].Get<InputT>();
            dout_a1[slot].tensor = dout_a1_tbuf[slot].Get<InputT>();
        }
        for (uint32_t slot = 0U; slot < BWD_P_DS_SLOTS; ++slot) {
            ds_a1[slot] = ds_a1_tbuf[slot].Get<InputT>();
            p_a1[slot] = p_a1_tbuf[slot].Get<InputT>();
        }
        for (uint32_t slot = 0U; slot < BWD_K_L1_SLOTS; ++slot) {
            k_b1[slot].tensor = k_b1_tbuf[slot].Get<InputT>();
        }
        v_b1.tensor = v_b1_tbuf.Get<InputT>();
        AscendC::LocalTensor<InputT> a2_base = a2_tbuf.Get<InputT>();
        AscendC::LocalTensor<InputT> b2_base = b2_tbuf.Get<InputT>();
        AscendC::LocalTensor<float> mm_co1_base = mm_co1_tbuf.Get<float>();
        for (uint32_t slot = 0U; slot < BWD_L0_AB_SLOTS; ++slot) {
            a2[slot].tensor = a2_base[slot * TILE_ELEMS];
            b2[slot].tensor = b2_base[slot * TILE_ELEMS];
        }
        for (uint32_t slot = 0U; slot < BWD_L0C_SLOTS; ++slot) {
            mm_co1[slot].tensor = mm_co1_base[slot * TILE_ELEMS];
        }

        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            q_a1[slot].alloc_id();
            dout_a1[slot].alloc_id();
        }
        for (uint32_t slot = 0U; slot < BWD_K_L1_SLOTS; ++slot) {
            k_b1[slot].alloc_id();
        }
        v_b1.alloc_id();
        for (uint32_t slot = 0U; slot < BWD_L0_AB_SLOTS; ++slot) {
            a2[slot].alloc_id();
            b2[slot].alloc_id();
        }
        for (uint32_t slot = 0U; slot < BWD_L0C_SLOTS; ++slot) {
            mm_co1[slot].alloc_id();
        }
        for (uint32_t slot = 0U; slot < BWD_Q_L1_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(q_a1[slot].free_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dout_a1[slot].free_id);
        }
        for (uint32_t slot = 0U; slot < BWD_K_L1_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(k_b1[slot].free_id);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(v_b1.free_id);
        for (uint32_t slot = 0U; slot < BWD_L0_AB_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(a2[slot].free_id);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(b2[slot].free_id);
        }
        for (uint32_t slot = 0U; slot < BWD_L0C_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(mm_co1[slot].free_id);
        }
        for (uint32_t slot = 0U; slot < BWD_P_DS_SLOTS; ++slot) {
            notify_aiv<PIPE_MTE1>(bwd_event_slot(BWD_P_DS_CONSUMED_BASE, slot));
        }
    }

    __aicore__ inline void init_aiv() {
        for (uint32_t slot = 0U; slot < BWD_MM_SLOTS; ++slot) {
            AscendC::Duplicate<float>(c1_ub(slot), 0.0F, static_cast<int32_t>(LOCAL_MM_ELEMS));
            AscendC::Duplicate<float>(c2_ub(slot), 0.0F, static_cast<int32_t>(LOCAL_MM_ELEMS));
            AscendC::PipeBarrier<PIPE_V>();
            notify_aic<PIPE_V>(bwd_event_slot(BWD_MM_CONSUMED_BASE, slot));
        }
    }
};

}  // namespace SWA_v5::Backward
