// Arithmetic from xpu-ops 1.0.0.91; see SOURCE.json.
#include "../common/swiglu_device.h"

namespace mojo::native::a2::kernels {
using namespace AscendC;
constexpr uint32_t kBlock = 4096;
constexpr uint32_t kVectors = 3;

template <class T> class Swiglu {
public:
  __aicore__ inline void init(GM_ADDR a, GM_ADDR b, GM_ADDR grad, GM_ADDR da, GM_ADDR db) {
    a_.SetGlobalBuffer((__gm__ T *)a);
    b_.SetGlobalBuffer((__gm__ T *)b);
    grad_.SetGlobalBuffer((__gm__ T *)grad);
    da_.SetGlobalBuffer((__gm__ T *)da);
    db_.SetGlobalBuffer((__gm__ T *)db);
    pipe_.InitBuffer(raw_, kBlock * sizeof(T));
    pipe_.InitBuffer(output_, kBlock * sizeof(T));
    pipe_.InitBuffer(values_, kVectors * kBlock * sizeof(float));
    pipe_.InitBuffer(scratch_, kBlock * sizeof(float));
  }
  __aicore__ inline void process(uint64_t elements) {
    auto a = values_.Get<float>();
    auto b = a[kBlock];
    auto result = a[2 * kBlock];
    a.SetSize(kBlock);
    b.SetSize(kBlock);
    result.SetSize(kBlock);
    const uint64_t tiles = (elements + kBlock - 1) / kBlock;
    const uint64_t tiles_per_core = (tiles + GetBlockNum() - 1) / GetBlockNum();
    const uint64_t begin = uint64_t(GetBlockIdx()) * tiles_per_core;
    for (uint64_t tile = begin; tile < begin + tiles_per_core && tile < tiles; ++tile) {
      const uint64_t offset = tile * kBlock;
      const uint32_t count = (elements - offset < kBlock ? elements - offset : kBlock);
      swiglu_load(a, a_, offset, count, raw_.Get<T>());
      swiglu_load(b, b_, offset, count, raw_.Get<T>());
      SwiGLU(result, b, a, 1.0f, scratch_.Get<uint8_t>());
      PipeBarrier<PIPE_V>();
      swiglu_store(da_, result, offset, count, output_.Get<T>());
    }
  }

private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> raw_, output_, values_, scratch_;
  GlobalTensor<T> a_, b_, grad_, da_, db_;
};
} // namespace mojo::native::a2::kernels

extern "C" __global__ __aicore__ void MOJO_SWIGLU_KERNEL(GM_ADDR a, GM_ADDR b, GM_ADDR grad, GM_ADDR da,
                                                         GM_ADDR db, uint64_t elements) {
  mojo::native::a2::kernels::Swiglu<DTYPE_INPUT> op;
  op.init(a, b, grad, da, db);
  op.process(elements);
}
