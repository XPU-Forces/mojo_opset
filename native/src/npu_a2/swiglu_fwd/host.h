#pragma once
#include <ATen/ATen.h>
namespace mojo::native::a2 {
at::Tensor SwigluForward(const at::Tensor &a, const at::Tensor &b);
}
