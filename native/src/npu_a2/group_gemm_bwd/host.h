#pragma once
#include <ATen/ATen.h>
namespace mojo::native::a2 {
at::Tensor GroupGemmWgrad(const at::Tensor &x, const at::Tensor &w, const at::Tensor &counts);
}
