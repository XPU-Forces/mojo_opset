#pragma once
#include "kernel_operator.h"

namespace mojo::native::a2::kernels {
using namespace AscendC;
template <class T>
__aicore__ inline void swiglu_load(LocalTensor<float> dst, GlobalTensor<T> src, uint64_t offset,
                                   uint32_t count, LocalTensor<T> raw) {
  if constexpr (std::is_same<T, float>::value)
    raw = dst;
  DataCopyPad(raw, src[offset], {1, count * uint32_t(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  if constexpr (!std::is_same<T, float>::value)
    Cast(dst, raw, RoundMode::CAST_NONE, count);
  PipeBarrier<PIPE_ALL>();
}
template <class T>
__aicore__ inline void swiglu_store(GlobalTensor<T> dst, LocalTensor<float> src, uint64_t offset,
                                    uint32_t count, LocalTensor<T> out) {
  if constexpr (std::is_same<T, float>::value)
    out = src;
  else
    Cast(out, src, RoundMode::CAST_RINT, count);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopyPad(dst[offset], out, {1, count * uint32_t(sizeof(T)), 0, 0, 0});
  PipeBarrier<PIPE_ALL>();
}
} // namespace mojo::native::a2::kernels
