#pragma once
#include <ATen/ATen.h>
namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor> SwigluBackward(const at::Tensor &grad, const at::Tensor &a,
                                                  const at::Tensor &b);
}
