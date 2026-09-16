#pragma once

#include "kernel/runtime_base.h"

namespace SWA_v5::Pipeline::FullPath {

template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
struct AICore_RT {
    using VEC_RT = SWA_v5::AICoreVec_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>;
    using InputType = InputT;

    struct CUBE_RT {
        static constexpr int K_L1_SLOTS = 2;
        static constexpr int V_L1_SLOTS = 2;
        static constexpr int P_L1_SLOTS = 3;
        static constexpr int QK_CO1_SLOTS = 1;
        static constexpr int PV_CO1_SLOTS = 1;
        static constexpr uint32_t P_A1_SLOT_ELEMS = TILE_Q * TILE_L0;
        static constexpr uint32_t P_A1_SLOT_BYTES = P_A1_SLOT_ELEMS * sizeof(InputT);
        static constexpr uint32_t Q_A1_OFFSET_ELEMS = P_L1_SLOTS * P_A1_SLOT_ELEMS;
        static constexpr uint32_t A1_TOTAL_ELEMS = Q_A1_OFFSET_ELEMS + TILE_Q * DIM;
        static constexpr uint32_t L1_TOTAL_BYTES =
            A1_TOTAL_ELEMS * sizeof(InputT) +
            (K_L1_SLOTS + V_L1_SLOTS) * TILE_L1 * DIM * sizeof(InputT);

        AscendC::TBuf<AscendC::TPosition::A1> A1_tbuf;
        AscendC::TBuf<AscendC::TPosition::B1> K_B1_tbuf[K_L1_SLOTS];
        AscendC::TBuf<AscendC::TPosition::B1> V_B1_tbuf[V_L1_SLOTS];
        AscendC::TBuf<AscendC::TPosition::A2> Q_A2_tbuf;
        AscendC::TBuf<AscendC::TPosition::A2> P_A2_tbuf;
        AscendC::TBuf<AscendC::TPosition::B2> K_B2_tbuf;
        AscendC::TBuf<AscendC::TPosition::B2> V_B2_tbuf;
        AscendC::TBuf<AscendC::TPosition::CO1> QK_CO1_tbuf[QK_CO1_SLOTS];
        AscendC::TBuf<AscendC::TPosition::CO1> PV_CO1_tbuf[PV_CO1_SLOTS];

        SWA_v5::A1Tensor<InputT, TILE_Q, DIM> Q_A1;
        SWA_v5::B1Tensor<InputT, TILE_L1, DIM> K_B1[K_L1_SLOTS];
        SWA_v5::A2Tensor<InputT, TILE_Q, DIM> Q_A2;
        SWA_v5::B2Tensor<InputT, DIM, TILE_L0> K_B2;
        SWA_v5::CO1Tensor<float, TILE_Q, TILE_L0> QK_CO1;

        SWA_v5::A1Tensor<InputT, TILE_Q, TILE_L0> P_A1[P_L1_SLOTS];
        SWA_v5::B1Tensor<InputT, TILE_L1, DIM> V_B1[V_L1_SLOTS];
        SWA_v5::A2Tensor<InputT, TILE_Q, TILE_L0> P_A2;
        SWA_v5::B2Tensor<InputT, DIM, TILE_L0> V_B2;
        SWA_v5::CO1Tensor<float, TILE_Q, DIM> PV_CO1;

        __aicore__ inline void init(AscendC::TPipe& pipe) {
            static_assert(DIM == 128, "v5 full path fixes head_dim=128");
            static_assert(TILE_Q == 128, "v5 full path currently uses TILE_Q=128");
            static_assert(TILE_L0 == 128, "v5 full path currently uses TILE_L0=128");
            static_assert(TILE_L0 >= DIM);
            static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0);
            static_assert(2U * TILE_Q * TILE_L0 * sizeof(InputT) <= 64U * 1024U,
                "FullPath Q/P L0A storage exceeds the 64 KiB L0A capacity");
            static_assert(2U * DIM * TILE_L0 * sizeof(InputT) <= 64U * 1024U,
                "FullPath K/V L0B storage exceeds the 64 KiB L0B capacity");
            static_assert(L1_TOTAL_BYTES <= 512U * 1024U,
                "P/Q/shared-KV L1 buffers exceed the 512 KiB L1 capacity");

            pipe.InitBuffer(A1_tbuf, A1_TOTAL_ELEMS * sizeof(InputT));
            AscendC::LocalTensor<InputT> a1_base = A1_tbuf.template Get<InputT>();

            for (uint32_t i = 0; i < P_L1_SLOTS; ++i) {
                P_A1[i].tensor = a1_base[i * P_A1_SLOT_ELEMS];
            }

            Q_A1.tensor = a1_base[Q_A1_OFFSET_ELEMS];
            Q_A1.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_A1.free_id);

            for (uint32_t i = 0; i < K_L1_SLOTS; ++i) {
                pipe.InitBuffer(K_B1_tbuf[i], TILE_L1 * DIM * sizeof(InputT));
                K_B1[i].tensor = K_B1_tbuf[i].template Get<InputT>();
                K_B1[i].alloc_id();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(K_B1[i].free_id);
            }

            for (uint32_t i = 0; i < V_L1_SLOTS; ++i) {
                pipe.InitBuffer(V_B1_tbuf[i], TILE_L1 * DIM * sizeof(InputT));
                V_B1[i].tensor = V_B1_tbuf[i].template Get<InputT>();
                V_B1[i].alloc_id();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(V_B1[i].free_id);
            }

            pipe.InitBuffer(Q_A2_tbuf, TILE_Q * TILE_L0 * sizeof(InputT));
            Q_A2.tensor = Q_A2_tbuf.template Get<InputT>();
            Q_A2.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Q_A2.free_id);

            pipe.InitBuffer(P_A2_tbuf, TILE_Q * TILE_L0 * sizeof(InputT));
            P_A2.tensor = P_A2_tbuf.template Get<InputT>();
            P_A2.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(P_A2.free_id);

            pipe.InitBuffer(K_B2_tbuf, DIM * TILE_L0 * sizeof(InputT));
            K_B2.tensor = K_B2_tbuf.template Get<InputT>();
            K_B2.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(K_B2.free_id);

            pipe.InitBuffer(V_B2_tbuf, DIM * TILE_L0 * sizeof(InputT));
            V_B2.tensor = V_B2_tbuf.template Get<InputT>();
            V_B2.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(V_B2.free_id);

            pipe.InitBuffer(QK_CO1_tbuf[0], TILE_Q * TILE_L0 * sizeof(float));
            QK_CO1.tensor = QK_CO1_tbuf[0].template Get<float>();
            QK_CO1.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_CO1.free_id);

            pipe.InitBuffer(PV_CO1_tbuf[0], TILE_Q * DIM * sizeof(float));
            PV_CO1.tensor = PV_CO1_tbuf[0].template Get<float>();
            PV_CO1.alloc_id();
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PV_CO1.free_id);
        }
    };

    AscendC::TPipe pipe;
    VEC_RT vec;
    CUBE_RT cube;

    template<pipe_t PIPE_KIND>
    __aicore__ inline void notify_aiv(uint16_t event_id) {
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(event_id);
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(static_cast<uint16_t>(event_id + 16));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void wait_aiv(uint16_t event_id) {
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(event_id);
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(static_cast<uint16_t>(event_id + 16));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void notify_aic(uint16_t event_id) {
        AscendC::CrossCoreSetFlag<4, PIPE_KIND>(local_subblock_event(event_id));
    }

    template<pipe_t PIPE_KIND>
    __aicore__ inline void wait_aic(uint16_t event_id) {
        AscendC::CrossCoreWaitFlag<4, PIPE_KIND>(local_subblock_event(event_id));
    }

private:
    __aicore__ static inline uint16_t local_subblock_event(uint16_t event_id) {
        return static_cast<uint16_t>(event_id + AscendC::GetSubBlockIdx() * 16);
    }

public:
    __aicore__ inline void init() {
        if constexpr (g_coreType == AscendC::AIC) {
            cube.init(pipe);
        } else if constexpr (g_coreType == AscendC::AIV) {
            vec.init(pipe);
        }
    }
};

}  // namespace SWA_v5::Pipeline::FullPath
