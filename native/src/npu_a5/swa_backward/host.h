#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <tuple>

namespace mojo::native {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> LaunchSwaBackwardCurrentStream(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out,
    const at::Tensor &dout, const at::Tensor &softmax_lse, const at::Tensor &cu_q_lens,
    const at::Tensor &cu_kv_lens, int64_t window_left, int64_t gqa_layout,
    int64_t global_window, double scale);

}  // namespace mojo::native
