#pragma once
#include "rope_tiling.h"
#include "torch_utils.h"
#include <limits>

namespace mojo::native::a2 {
inline int32_t checked_rope_dim(int64_t value) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int32_t>::max(), "A2 rope dimension exceeds int32");
  return static_cast<int32_t>(value);
}
inline void validate_rope(const at::Tensor &input, const at::Tensor &sin, const at::Tensor &cos,
                          int64_t rotary_dim) {
  tensor(input, "input");
  tensor(sin, "sin");
  tensor(cos, "cos");
  TORCH_CHECK(input.dim() == 3 && sin.dim() == 2 && cos.sizes() == sin.sizes(),
              "A2 rope expects input [T,H,D], tables [S,R]");
  TORCH_CHECK(input.device() == sin.device() && sin.device() == cos.device(),
              "A2 rope tensors must share device");
  TORCH_CHECK(sin.scalar_type() == cos.scalar_type(), "cos/sin must share dtype");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 16 == 0 && rotary_dim <= input.size(2) &&
                  rotary_dim == sin.size(1),
              "rotary_dim must match tables, be a positive multiple of 16, and fit D");
  TORCH_CHECK(input.size(1) > 0 && input.size(2) > 0, "A2 rope requires nonempty heads");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kFloat,
              "A2 rope supports BF16/FP16/FP32");
  TORCH_CHECK(sin.scalar_type() == input.scalar_type() || sin.scalar_type() == at::kFloat,
              "A2 rope tables must have input dtype or FP32");
  checked_rope_dim(input.size(0));
  checked_rope_dim(input.size(1));
  checked_rope_dim(input.size(2));
  checked_rope_dim(input.size(1) * input.size(2));
}
inline RopeForwardTiling make_rope_tiling(const at::Tensor &input, int64_t rotary_dim, uint32_t &cores) {
  const int64_t ratio = (input.size(2) + rotary_dim - 1) / rotary_dim;
  const int64_t bytes_per_element = input.scalar_type() == at::kFloat ? 20 : 18;
  const int32_t valid_chunk = std::min<int64_t>(input.scalar_type() == at::kFloat ? 64 : 72,
                                                192 * 1024 / (bytes_per_element * rotary_dim * ratio));
  TORCH_CHECK(valid_chunk > 0 && rotary_dim <= 2032, "RoPE dimensions exceed the A2 UB/repeat-stride limits");
  RopeForwardTiling tiling{};
  tiling.token_num = checked_rope_dim(input.size(0));
  tiling.head_num = checked_rope_dim(input.size(1));
  tiling.head_dim = checked_rope_dim(input.size(2));
  tiling.seqlen_per_block = valid_chunk * 12;
  tiling.batch_size = (tiling.token_num + tiling.seqlen_per_block - 1) / tiling.seqlen_per_block;
  tiling.token_num_stride = tiling.head_num * tiling.head_dim;
  tiling.head_num_stride = tiling.head_dim;
  tiling.cut_head_dim = checked_rope_dim(rotary_dim);
  const int32_t total = checked_rope_dim(static_cast<int64_t>(tiling.batch_size) * tiling.head_num);
  cores = std::min<uint32_t>(core_counts(input.get_device()).vector, total);
  tiling.blocks_per_core = (total + cores - 1) / cores;
  tiling.blocks_per_core_tail = total / cores;
  tiling.normal_cores = total - tiling.blocks_per_core_tail * cores;
  tiling.tail_cores = cores - tiling.normal_cores;
  tiling.chunk_size =
      (tiling.token_num > valid_chunk * tiling.batch_size ? valid_chunk
                                                          : tiling.token_num / tiling.batch_size) *
      ((tiling.head_dim + tiling.cut_head_dim - 1) / tiling.cut_head_dim);
  // The explicit-stride vector instructions encode repeat count in uint8_t.
  tiling.chunk_size = std::min<int32_t>(tiling.chunk_size, 255);
  return tiling;
}
} // namespace mojo::native::a2
