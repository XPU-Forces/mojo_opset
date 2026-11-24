#pragma once
#include <ATen/ATen.h>
#include <tuple>
namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
FlashAttentionBackward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &dy,
                       const at::Tensor &mask, const at::Tensor &out, const at::Tensor &mx,
                       const at::Tensor &sm, const at::Tensor &cuq, const at::Tensor &cuk, double scale,
                       bool causal);
}
