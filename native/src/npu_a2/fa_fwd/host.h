#pragma once
#include <ATen/ATen.h>
#include <tuple>
namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
FlashAttentionForward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &mask,
                      const at::Tensor &cuq, const at::Tensor &cuk, int64_t maxq, int64_t maxk, double scale,
                      bool causal);
}
