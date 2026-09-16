#include "host.h"
#include "common/npu.h"

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "aclrtlaunch_swa_v5_mojo_h16_kv4_aabb_bf16.h"
#include "aclrtlaunch_swa_v5_mojo_h16_kv4_abab_bf16.h"
#include "aclrtlaunch_swa_v5_mojo_h8_kv1_aabb_bf16.h"
#include "aclrtlaunch_swa_v5_mojo_h8_kv1_abab_bf16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace mojo::native {
namespace {

constexpr uint32_t kDim = 128;
constexpr uint32_t kMaxQLen = 1024;
constexpr uint32_t kMaxKvLen = 9216;
constexpr uint32_t kDefaultBlockDim = 28;
constexpr uint32_t kGqaLayoutAabb = 0;
constexpr uint32_t kGqaLayoutAbab = 1;
constexpr uint32_t kWindowRight = 0;
constexpr double kDefaultSoftmaxScale = 0.08838834764831845;

enum class SwaKernelKind {
  H16Kv4AabbBf16,
  H16Kv4AbabBf16,
  H8Kv1AabbBf16,
  H8Kv1AbabBf16,
};

void ValidateInputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &cu_q_lens,
                    const at::Tensor &cu_kv_lens, bool is_causal, int64_t local_window_size,
                    int64_t global_window_size, int64_t gqa_layout) {
  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1 &&
                  key.device().type() == c10::DeviceType::PrivateUse1 &&
                  value.device().type() == c10::DeviceType::PrivateUse1,
              "query/key/value must be NPU tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "query/key/value must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query/key/value must be packed TND tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16, "SWA AscendC path currently supports BF16 only");
  TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
              "query/key/value dtype must match");
  TORCH_CHECK(query.size(2) == kDim && key.size(2) == kDim && value.size(2) == kDim, "head_dim must be 128");
  TORCH_CHECK(key.size(0) == value.size(0) && key.size(1) == value.size(1), "key/value shapes are inconsistent");
  TORCH_CHECK(key.size(1) > 0 && query.size(1) > 0 && query.size(1) % key.size(1) == 0,
              "Hq must be positive and divisible by Hkv > 0");
  TORCH_CHECK(query.device() == key.device() && query.device() == value.device() &&
                  query.device() == cu_q_lens.device() && query.device() == cu_kv_lens.device(),
              "all SWA tensors must be on the same NPU device");
  TORCH_CHECK(cu_q_lens.is_contiguous() && cu_kv_lens.is_contiguous(), "cu_lens must be contiguous");
  TORCH_CHECK(cu_q_lens.scalar_type() == at::kInt && cu_kv_lens.scalar_type() == at::kInt,
              "cu_q_lens/cu_total_seq_lens must be int32");
  TORCH_CHECK(cu_q_lens.dim() == 1 && cu_kv_lens.dim() == 1 && cu_q_lens.numel() == cu_kv_lens.numel(),
              "cu_q_lens/cu_total_seq_lens must be 1-D tensors with the same length");
  TORCH_CHECK(is_causal, "SWA AscendC path currently supports causal mode only");
  TORCH_CHECK(cu_q_lens.numel() >= 2, "cu_lens must describe at least one sequence");
  TORCH_CHECK(local_window_size > 0, "SWA AscendC path requires a positive local_window_size");
  TORCH_CHECK(global_window_size >= 0, "global_window_size must be non-negative");
  TORCH_CHECK(gqa_layout == kGqaLayoutAabb || gqa_layout == kGqaLayoutAbab, "invalid gqa_layout");
}

SwaKernelKind SelectKernel(uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
                           uint32_t global_window_size, int64_t gqa_layout) {
  TORCH_CHECK(global_window_size == 4, "SWA AscendC path currently supports global_window_size=4");
  if (num_q_heads == 16 && num_kv_heads == 4 && window_left == 1023 && gqa_layout == kGqaLayoutAabb) {
    return SwaKernelKind::H16Kv4AabbBf16;
  }
  if (num_q_heads == 16 && num_kv_heads == 4 && window_left == 255 && gqa_layout == kGqaLayoutAbab) {
    return SwaKernelKind::H16Kv4AbabBf16;
  }
  if (num_q_heads == 8 && num_kv_heads == 1 && window_left == 1023 && gqa_layout == kGqaLayoutAabb) {
    return SwaKernelKind::H8Kv1AabbBf16;
  }
  if (num_q_heads == 8 && num_kv_heads == 1 && window_left == 255 && gqa_layout == kGqaLayoutAbab) {
    return SwaKernelKind::H8Kv1AbabBf16;
  }
  TORCH_CHECK(false, "unsupported SWA AscendC specialization");
  return SwaKernelKind::H16Kv4AabbBf16;
}

}  // namespace

at::Tensor LaunchSwaCurrentStream(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                  const at::Tensor &cu_q_lens, const at::Tensor &cu_kv_lens, bool is_causal,
                                  int64_t local_window_size_arg, int64_t global_window_size_arg, double scale,
                                  int64_t gqa_layout, int64_t block_dim_arg) {
  ValidateInputs(query, key, value, cu_q_lens, cu_kv_lens, is_causal, local_window_size_arg, global_window_size_arg,
                 gqa_layout);
  const c10::DeviceGuard device_guard(query.device());
  TORCH_CHECK(block_dim_arg == 0 || block_dim_arg == kDefaultBlockDim,
              "block_dim must match the compiled 28-core specialization");
  TORCH_CHECK(std::abs(scale - kDefaultSoftmaxScale) < 1e-12,
              "SWA AscendC kernel currently requires softmax_scale=1/sqrt(128)");

  const auto host_cu_q = ReadCuLens(cu_q_lens);
  const auto host_cu_kv = ReadCuLens(cu_kv_lens);
  const uint32_t batch = static_cast<uint32_t>(cu_q_lens.numel() - 1);
  uint32_t max_q_len = 0;
  uint32_t max_kv_len = 0;
  for (uint32_t i = 0; i < batch; ++i) {
    TORCH_CHECK(host_cu_q[i + 1] > host_cu_q[i] && host_cu_kv[i + 1] > host_cu_kv[i],
                "each sequence must have positive q and kv lengths");
    TORCH_CHECK(host_cu_kv[i + 1] - host_cu_kv[i] >= host_cu_q[i + 1] - host_cu_q[i],
                "SWA requires kv length >= q length for each sequence");
    max_q_len = std::max(max_q_len, static_cast<uint32_t>(host_cu_q[i + 1] - host_cu_q[i]));
    max_kv_len = std::max(max_kv_len, static_cast<uint32_t>(host_cu_kv[i + 1] - host_cu_kv[i]));
  }
  TORCH_CHECK(host_cu_q.front() == 0 && host_cu_kv.front() == 0, "cu_lens must start at zero");
  TORCH_CHECK(host_cu_q.back() == query.size(0) && host_cu_kv.back() == key.size(0), "cu_lens do not match tokens");
  TORCH_CHECK(value.size(0) == key.size(0), "value token count must match key token count");
  TORCH_CHECK(max_q_len > 0 && max_kv_len > 0, "empty SWA input is not supported");
  TORCH_CHECK(max_q_len <= kMaxQLen && max_kv_len <= kMaxKvLen, "SWA sequence length exceeds compiled limits");
  TORCH_CHECK(host_cu_q.back() <= std::numeric_limits<uint32_t>::max() &&
                  host_cu_kv.back() <= std::numeric_limits<uint32_t>::max(),
              "total token count exceeds uint32_t range");

  const uint32_t num_q_heads = static_cast<uint32_t>(query.size(1));
  const uint32_t num_kv_heads = static_cast<uint32_t>(key.size(1));
  const uint32_t window_left = static_cast<uint32_t>(local_window_size_arg);
  const uint32_t global_window_size = static_cast<uint32_t>(global_window_size_arg);
  const uint32_t block_dim = block_dim_arg == 0 ? kDefaultBlockDim : static_cast<uint32_t>(block_dim_arg);
  const auto kernel_kind = SelectKernel(num_q_heads, num_kv_heads, window_left, global_window_size, gqa_layout);

  auto output = at::empty_like(query);
  auto lse = at::empty({8}, query.options().dtype(at::kFloat));
  auto stream = c10_npu::getCurrentNPUStream().stream(false);

  switch (kernel_kind) {
    case SwaKernelKind::H16Kv4AabbBf16:
      CheckAcl(ACLRT_LAUNCH_KERNEL(swa_v5_mojo_h16_kv4_aabb_bf16)(
                   block_dim, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
                   lse.data_ptr(), cu_q_lens.data_ptr(), cu_kv_lens.data_ptr(), batch, num_q_heads, num_kv_heads,
                   static_cast<uint32_t>(host_cu_q.back()), static_cast<uint32_t>(host_cu_kv.back()), kMaxQLen,
                   kMaxKvLen, window_left, kWindowRight, kDim),
               "launch swa_v5_mojo_h16_kv4_aabb_bf16");
      break;
    case SwaKernelKind::H16Kv4AbabBf16:
      CheckAcl(ACLRT_LAUNCH_KERNEL(swa_v5_mojo_h16_kv4_abab_bf16)(
                   block_dim, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
                   lse.data_ptr(), cu_q_lens.data_ptr(), cu_kv_lens.data_ptr(), batch, num_q_heads, num_kv_heads,
                   static_cast<uint32_t>(host_cu_q.back()), static_cast<uint32_t>(host_cu_kv.back()), kMaxQLen,
                   kMaxKvLen, window_left, kWindowRight, kDim),
               "launch swa_v5_mojo_h16_kv4_abab_bf16");
      break;
    case SwaKernelKind::H8Kv1AabbBf16:
      CheckAcl(ACLRT_LAUNCH_KERNEL(swa_v5_mojo_h8_kv1_aabb_bf16)(
                   block_dim, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
                   lse.data_ptr(), cu_q_lens.data_ptr(), cu_kv_lens.data_ptr(), batch, num_q_heads, num_kv_heads,
                   static_cast<uint32_t>(host_cu_q.back()), static_cast<uint32_t>(host_cu_kv.back()), kMaxQLen,
                   kMaxKvLen, window_left, kWindowRight, kDim),
               "launch swa_v5_mojo_h8_kv1_aabb_bf16");
      break;
    case SwaKernelKind::H8Kv1AbabBf16:
      CheckAcl(ACLRT_LAUNCH_KERNEL(swa_v5_mojo_h8_kv1_abab_bf16)(
                   block_dim, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
                   lse.data_ptr(), cu_q_lens.data_ptr(), cu_kv_lens.data_ptr(), batch, num_q_heads, num_kv_heads,
                   static_cast<uint32_t>(host_cu_q.back()), static_cast<uint32_t>(host_cu_kv.back()), kMaxQLen,
                   kMaxKvLen, window_left, kWindowRight, kDim),
               "launch swa_v5_mojo_h8_kv1_abab_bf16");
      break;
  }
  return output;
}

}  // namespace mojo::native
