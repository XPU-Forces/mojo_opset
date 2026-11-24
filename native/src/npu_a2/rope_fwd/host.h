#pragma once
#include <ATen/ATen.h>
namespace mojo::native::a2 {
void RopeForward(const at::Tensor &input, const at::Tensor &sin, const at::Tensor &cos, int64_t rotary_dim);
}
