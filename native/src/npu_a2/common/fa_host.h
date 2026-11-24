#pragma once
#include "torch_utils.h"
#include <climits>
namespace mojo::native::a2 {
inline void validate_fa(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &mask,
                        const at::Tensor &cuq, const at::Tensor &cuk, bool causal) {
  for (const auto &x : {q, k, v, mask, cuq, cuk}) {
    tensor(x, "FA input");
    TORCH_CHECK(x.device() == q.device(), "FA inputs must share a device");
  }
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(),
              "FA requires packed q[Tq,Hq,D] and k/v[Tkv,Hkv,D]");
  TORCH_CHECK(q.size(0) > 0 && k.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0,
              "FA requires nonempty inputs and divisible GQA heads");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  (q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf),
              "FA requires matching BF16/FP16 inputs");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) >= 64 && q.size(2) <= 128 && q.size(2) % 16 == 0,
              "FA native supports head_dim 64/80/96/112/128");
  TORCH_CHECK(cuq.dim() == 1 && cuq.numel() > 0 && cuq.sizes() == cuk.sizes() &&
                  cuq.scalar_type() == at::kInt && cuk.scalar_type() == at::kInt,
              "FA native sequence ends must be equally sized nonempty int32 vectors without a leading zero");
  TORCH_CHECK(mask.scalar_type() == at::kBool &&
                  (!causal || (mask.dim() == 2 && mask.size(0) == 2048 && mask.size(1) == 2048)),
              "causal FA requires the packed 2048x2048 bool mask");
  for (auto size : q.sizes())
    TORCH_CHECK(size <= INT_MAX, "FA dimension exceeds int32");
  for (auto size : k.sizes())
    TORCH_CHECK(size <= INT_MAX, "FA dimension exceeds int32");
}
} // namespace mojo::native::a2
