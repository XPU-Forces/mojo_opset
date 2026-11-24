#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>

namespace mojo::native {

at::Tensor LaunchVarlenFaCurrentStream(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &cu_q_lens, const at::Tensor &cu_k_lens, bool is_causal,
    double scale, int64_t gqa_layout, int64_t block_dim_arg);

}  // namespace mojo::native
