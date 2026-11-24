#pragma once

// Adapted from data/xpu_ops commit 8452b29b; see SOURCE.json.
#include "../common/rope_tiling.h"
#include "kernel_operator.h"

namespace mojo::native::a2 {
namespace kernels {

template <typename T, typename EmbedT> class RopeForward {
public:
  __aicore__ RopeForward() {};

  __aicore__ inline void init(GM_ADDR input, GM_ADDR sin_embeds, GM_ADDR cos_embeds,
                              const RopeForwardTiling &tiling);

  __aicore__ inline void process();

  __aicore__ inline void do_rope_mul(AscendC::LocalTensor<float> &input_float_local,
                                     AscendC::LocalTensor<float> &sin_embeds_float_local,
                                     AscendC::LocalTensor<float> &cos_embeds_float_local,
                                     AscendC::LocalTensor<T> &output_local, int32_t valid_copy_rows);

private:
  AscendC::GlobalTensor<T> input_global_; // input
  AscendC::GlobalTensor<EmbedT> sin_embeds_global_;
  AscendC::GlobalTensor<EmbedT> cos_embeds_global_;

  uint32_t skip_head_dim_ = 0;
  uint32_t input_buffer_size_;

private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> sin_embeds_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> cos_embeds_queue_;

  AscendC::TQue<AscendC::TPosition::VECCALC, 1> input_float_queue_;
  AscendC::TQue<AscendC::TPosition::VECCALC, 1> sin_embeds_float_queue_;
  AscendC::TQue<AscendC::TPosition::VECCALC, 1> cos_embeds_float_queue_;

  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue_;
  RopeForwardTiling tiling_;
};
} // namespace kernels
} // namespace mojo::native::a2
