#pragma once

#include "kernel_operator.h"

namespace SWA_v5 {

enum class TensorLayout : int32_t {
    ZZ,
    NZ,
    ZN,
    NN,
    ROW_MAJOR,
    COL_MAJOR,
};

template<class T, TensorLayout L>
struct Tensor {
    T tensor;
};

template<class T, int R, int C, AscendC::HardEvent READY_EVENT, AscendC::HardEvent FREE_EVENT, TensorLayout L>
struct SyncTensor {
    T tensor;

    static constexpr int rows = R;
    static constexpr int cols = C;

    AscendC::TEventID ready_id;
    AscendC::TEventID free_id;

    __aicore__ inline void alloc_id() {
        ready_id = GetTPipePtr()->template AllocEventID<READY_EVENT>();
        free_id = GetTPipePtr()->template AllocEventID<FREE_EVENT>();
    }

    __aicore__ inline void release_id() {
        GetTPipePtr()->template ReleaseEventID<READY_EVENT>(ready_id);
        GetTPipePtr()->template ReleaseEventID<FREE_EVENT>(free_id);
    }

    __aicore__ inline void producer_wait() {
        AscendC::WaitFlag<FREE_EVENT>(free_id);
    }

    __aicore__ inline void producer_set() {
        AscendC::SetFlag<READY_EVENT>(ready_id);
    }

    __aicore__ inline void consumer_wait() {
        AscendC::WaitFlag<READY_EVENT>(ready_id);
    }

    __aicore__ inline void consumer_set() {
        AscendC::SetFlag<FREE_EVENT>(free_id);
    }
};

template<class T>
using GMTensor_ND = Tensor<AscendC::GlobalTensor<T>, TensorLayout::ROW_MAJOR>;

template<class T, int R, int C>
struct UBTensor_ND : Tensor<AscendC::LocalTensor<T>, TensorLayout::ROW_MAJOR> {
    static constexpr int rows = R;
    static constexpr int cols = C;
};

template<class T, int R, int C>
using A1Tensor = SyncTensor<AscendC::LocalTensor<T>, R, C, AscendC::HardEvent::MTE2_MTE1,
    AscendC::HardEvent::MTE1_MTE2, TensorLayout::NZ>;

template<class T, int R, int C>
using B1Tensor = SyncTensor<AscendC::LocalTensor<T>, R, C, AscendC::HardEvent::MTE2_MTE1,
    AscendC::HardEvent::MTE1_MTE2, TensorLayout::NZ>;

template<class T, int R, int C>
using A2Tensor = SyncTensor<AscendC::LocalTensor<T>, R, C, AscendC::HardEvent::MTE1_M,
    AscendC::HardEvent::M_MTE1, TensorLayout::NZ>;

template<class T, int R, int C>
using B2Tensor = SyncTensor<AscendC::LocalTensor<T>, R, C, AscendC::HardEvent::MTE1_M,
    AscendC::HardEvent::M_MTE1, TensorLayout::ZN>;

template<class T, int R, int C>
using CO1Tensor = SyncTensor<AscendC::LocalTensor<T>, R, C, AscendC::HardEvent::M_FIX,
    AscendC::HardEvent::FIX_M, TensorLayout::NZ>;

}  // namespace SWA_v5
