#pragma once
#include "torch_utils.h"

namespace mojo::native::a2 {
inline void validate_swiglu(const at::Tensor &a, const at::Tensor &b) {
  tensor(a, "a");
  tensor(b, "b");
  TORCH_CHECK(a.sizes() == b.sizes() && a.device() == b.device() && a.scalar_type() == b.scalar_type(),
              "SwiGLU inputs must share shape, device and dtype");
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16 ||
                  a.scalar_type() == at::kFloat,
              "SwiGLU requires FP16/BF16/FP32");
}
} // namespace mojo::native::a2
