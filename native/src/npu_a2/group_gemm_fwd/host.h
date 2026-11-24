#pragma once
#include <ATen/ATen.h>
namespace mojo::native::a2 {
at::Tensor GroupGemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &counts,
                     bool transpose_weight);
}
