#include "host.h"
#include "common/npu.h"

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "aclrtlaunch_varlen_fa.h"
#include "aclrtlaunch_varlen_fa_full.h"
#include "aclrtlaunch_varlen_fa_full_fp16.h"
#include "aclrtlaunch_varlen_fa_fp16.h"
#include "varlen_fa_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mojo::native {
namespace {

constexpr uint32_t kDim = 128;
constexpr uint32_t kTileQ = 128;
constexpr uint32_t kTileKv = 1024;
constexpr uint32_t kPipelineStages = 3;
constexpr uint32_t kPackedMaskTilesPerQTask = 2;
constexpr uint32_t kMaskLayoutPackedQTask = 1;
constexpr uint32_t kMaskLayoutCausalTemplate = 2;
constexpr uint32_t kGqaLayoutAabb = 0;
constexpr uint32_t kGqaLayoutAbab = 1;
constexpr uint32_t kDefaultBlockDim = 24;
constexpr size_t kDummyMaskBytes = 1;
constexpr size_t kReservedWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;

size_t WorkspaceBytes(uint32_t aic_num) {
  const uint32_t qk = kTileQ * kTileKv * sizeof(float);
  const uint32_t score = kTileQ * kTileKv * sizeof(uint16_t);
  const uint32_t pv = kTileQ * kDim * sizeof(float);
  return kReservedWorkspaceBytes + static_cast<size_t>(aic_num) * kPipelineStages * (qk + score + pv);
}

void FillPackedMask(std::vector<uint8_t> &mask, const std::vector<int64_t> &cu_q, const std::vector<int64_t> &cu_k,
                    uint32_t max_seq_q, uint32_t max_seq_k, int32_t mask_type) {
  const uint32_t batch_size = static_cast<uint32_t>(cu_q.size() - 1);
  const uint32_t num_q_tiles = (max_seq_q + kTileQ - 1) / kTileQ;
  std::fill(mask.begin(), mask.end(), 0);
  if (mask_type == 1) {
    std::fill(mask.begin(), mask.end(), 1);
    return;
  }
  for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const uint32_t q_len = static_cast<uint32_t>(cu_q[batch_idx + 1] - cu_q[batch_idx]);
    const uint32_t k_len = static_cast<uint32_t>(cu_k[batch_idx + 1] - cu_k[batch_idx]);
    const int64_t diff = static_cast<int64_t>(k_len) - static_cast<int64_t>(q_len);
    if (q_len == 0 || k_len == 0) {
      continue;
    }
    for (uint32_t q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
      const uint32_t start_q = q_tile_idx * kTileQ;
      if (start_q >= q_len) {
        continue;
      }
      const uint32_t q_end = std::min(start_q + kTileQ, q_len);
      const int64_t first_tile_signed = (static_cast<int64_t>(start_q) + diff) / static_cast<int64_t>(kTileKv);
      const int64_t last_tile_signed = (static_cast<int64_t>(q_end) - 1 + diff) / static_cast<int64_t>(kTileKv);
      const uint32_t first_tile = first_tile_signed > 0 ? static_cast<uint32_t>(first_tile_signed) : 0;
      const uint32_t last_tile = last_tile_signed > 0 ? static_cast<uint32_t>(last_tile_signed) : 0;
      for (uint32_t slot = 0; slot < kPackedMaskTilesPerQTask; ++slot) {
        const uint32_t tile_idx = (slot == 0) ? first_tile : last_tile;
        if (slot == 1 && tile_idx == first_tile) {
          continue;
        }
        const uint32_t boundary_start = tile_idx * kTileKv;
        if (boundary_start >= k_len) {
          continue;
        }
        const size_t slot_idx =
            (static_cast<size_t>(batch_idx) * num_q_tiles + q_tile_idx) * kPackedMaskTilesPerQTask + slot;
        uint8_t *slot_base = mask.data() + slot_idx * kTileQ * kTileKv;
        for (uint32_t row = 0; row < kTileQ; ++row) {
          const uint32_t global_q = start_q + row;
          const int64_t causal_limit = static_cast<int64_t>(global_q) + diff;
          if (global_q < q_end && causal_limit >= static_cast<int64_t>(boundary_start)) {
            const uint32_t valid_cols =
                std::min(static_cast<uint32_t>(causal_limit - boundary_start + 1), kTileKv);
            std::memset(slot_base + row * kTileKv, 1, std::min(valid_cols, k_len - boundary_start));
          }
        }
      }
    }
  }
}

bool CanUseCausalTemplateMask(const std::vector<int64_t> &cu_q, const std::vector<int64_t> &cu_k, bool is_causal) {
  if (!is_causal) {
    return false;
  }
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(cu_q.size()); ++i) {
    if (cu_q[i + 1] - cu_q[i] != cu_k[i + 1] - cu_k[i]) {
      return false;
    }
  }
  return true;
}

void FillCausalTemplateMask(std::vector<uint8_t> &mask) {
  std::fill(mask.begin(), mask.end(), 0);
  for (uint32_t row = 0; row < kTileKv; ++row) {
    std::memset(mask.data() + static_cast<size_t>(row) * kTileKv, 1, row + 1);
  }
}

struct CausalTemplateCache {
  std::mutex mutex;
  std::unordered_map<int, at::Tensor> masks;
};

at::Tensor GetCausalTemplateMask(const at::Tensor &query) {
  static CausalTemplateCache cache;
  const int device_index = static_cast<int>(query.device().index());
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto found = cache.masks.find(device_index);
  if (found != cache.masks.end()) {
    return found->second;
  }

  const size_t mask_bytes = static_cast<size_t>(kTileKv) * kTileKv;
  std::vector<uint8_t> host_mask(mask_bytes);
  FillCausalTemplateMask(host_mask);
  auto mask = at::empty({static_cast<int64_t>(mask_bytes)}, query.options().dtype(at::kByte));
  CheckAcl(aclrtMemcpy(mask.data_ptr(), mask_bytes, host_mask.data(), mask_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
           "copy causal template mask");
  cache.masks.emplace(device_index, mask);
  return mask;
}

void ValidateInputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &cu_q_lens,
                    const at::Tensor &cu_k_lens, int64_t gqa_layout) {
  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1 &&
                  key.device().type() == c10::DeviceType::PrivateUse1 &&
                  value.device().type() == c10::DeviceType::PrivateUse1,
              "query/key/value must be NPU tensors");
  TORCH_CHECK(key.device() == query.device() && value.device() == query.device(),
              "query/key/value must be on the same device");
  TORCH_CHECK(cu_q_lens.device() == query.device() && cu_k_lens.device() == query.device(),
              "cu_lens must be on the input NPU device");
  TORCH_CHECK(cu_q_lens.numel() >= 2 && cu_k_lens.numel() >= 2, "cu_lens must contain at least two offsets");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "query/key/value must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query/key/value must be packed TND tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
              "query dtype must be BF16 or FP16");
  TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
              "query/key/value dtype must match");
  TORCH_CHECK(query.size(2) == kDim && key.size(2) == kDim && value.size(2) == kDim, "head_dim must be 128");
  TORCH_CHECK(key.size(0) == value.size(0) && key.size(1) == value.size(1), "key/value shapes are inconsistent");
  TORCH_CHECK(key.size(1) > 0 && query.size(1) >= key.size(1), "GQA requires Hq >= Hkv > 0");
  TORCH_CHECK(cu_q_lens.scalar_type() == at::kInt && cu_k_lens.scalar_type() == at::kInt,
              "cu_q_lens/cu_k_lens must be int32");
  TORCH_CHECK(cu_q_lens.dim() == 1 && cu_k_lens.dim() == 1 && cu_q_lens.numel() == cu_k_lens.numel(),
              "cu_q_lens/cu_k_lens must be 1-D tensors with the same length");
  TORCH_CHECK(gqa_layout == kGqaLayoutAabb || gqa_layout == kGqaLayoutAbab, "invalid gqa_layout");
}

}  // namespace

at::Tensor LaunchVarlenFaCurrentStream(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                       const at::Tensor &cu_q_lens, const at::Tensor &cu_k_lens, bool is_causal,
                                       double scale, int64_t gqa_layout, int64_t block_dim_arg) {
  ValidateInputs(query, key, value, cu_q_lens, cu_k_lens, gqa_layout);

  const c10::DeviceGuard device_guard(query.device());
  TORCH_CHECK(block_dim_arg == 0, "block_dim override is not supported");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "softmax_scale must be positive and finite");
  const uint32_t block_dim = block_dim_arg == 0 ? kDefaultBlockDim : static_cast<uint32_t>(block_dim_arg);
  const uint32_t batch_size = static_cast<uint32_t>(cu_q_lens.numel() - 1);
  const uint32_t num_heads = static_cast<uint32_t>(query.size(1));
  const uint32_t num_kv_heads = static_cast<uint32_t>(key.size(1));
  auto host_cu_q = ReadCuLens(cu_q_lens);
  auto host_cu_k = ReadCuLens(cu_k_lens);
  TORCH_CHECK(host_cu_q.front() == 0 && host_cu_k.front() == 0, "cu_lens must start at zero");
  TORCH_CHECK(host_cu_q.back() == query.size(0) && host_cu_k.back() == key.size(0), "cu_lens do not match tokens");

  uint32_t max_seq_q = 0;
  uint32_t max_seq_k = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    TORCH_CHECK(host_cu_q[i + 1] >= host_cu_q[i] && host_cu_k[i + 1] >= host_cu_k[i],
                "cu_lens must be monotonic");
    TORCH_CHECK(host_cu_q[i + 1] > host_cu_q[i] && host_cu_k[i + 1] > host_cu_k[i],
                "empty sequences are not supported");
    TORCH_CHECK(!is_causal || host_cu_k[i + 1] - host_cu_k[i] >= host_cu_q[i + 1] - host_cu_q[i],
                "causal attention requires Q length <= KV length");
    max_seq_q = std::max(max_seq_q, static_cast<uint32_t>(host_cu_q[i + 1] - host_cu_q[i]));
    max_seq_k = std::max(max_seq_k, static_cast<uint32_t>(host_cu_k[i + 1] - host_cu_k[i]));
  }
  TORCH_CHECK(max_seq_q > 0 && max_seq_k > 0, "empty varlen FA input is not supported");
  TORCH_CHECK(host_cu_q.back() <= std::numeric_limits<uint32_t>::max() &&
                  host_cu_k.back() <= std::numeric_limits<uint32_t>::max(),
              "total token count exceeds uint32_t range");

  const uint32_t total_q_tokens = static_cast<uint32_t>(host_cu_q.back());
  const uint32_t total_k_tokens = static_cast<uint32_t>(host_cu_k.back());
  const uint32_t num_q_tiles = (max_seq_q + kTileQ - 1) / kTileQ;
  const bool needs_device_mask = is_causal;
  const bool use_template_mask = CanUseCausalTemplateMask(host_cu_q, host_cu_k, is_causal);
  const size_t mask_bytes =
      needs_device_mask ? (use_template_mask ? static_cast<size_t>(kTileKv) * kTileKv
                                             : static_cast<size_t>(batch_size) * num_q_tiles *
                                                   kPackedMaskTilesPerQTask * kTileQ * kTileKv)
                        : kDummyMaskBytes;
  const size_t cu_bytes = static_cast<size_t>(batch_size + 1) * sizeof(int64_t);
  const size_t workspace_bytes = WorkspaceBytes(block_dim);

  std::vector<uint8_t> host_mask;
  if (needs_device_mask && !use_template_mask) {
    host_mask.resize(mask_bytes);
    FillPackedMask(host_mask, host_cu_q, host_cu_k, max_seq_q, max_seq_k, 0);
  }

  VarlenFaTiling host_tiling{};
  host_tiling.batch_size = batch_size;
  host_tiling.total_q_tokens = total_q_tokens;
  host_tiling.total_k_tokens = total_k_tokens;
  host_tiling.head_qo = num_heads;
  host_tiling.head_kv = num_kv_heads;
  host_tiling.gqa_layout = static_cast<uint32_t>(gqa_layout);
  host_tiling.seq_len_q = max_seq_q;
  host_tiling.seq_len_k = max_seq_k;
  host_tiling.max_seq_len_q = max_seq_q;
  host_tiling.max_seq_len_k = max_seq_k;
  host_tiling.head_dim = kDim;
  host_tiling.mask_type = is_causal ? 0 : 1;
  host_tiling.mask_layout = use_template_mask ? kMaskLayoutCausalTemplate : kMaskLayoutPackedQTask;
  host_tiling.num_mask_q_tiles = use_template_mask ? (kTileKv / kTileQ) : num_q_tiles;
  host_tiling.scale = static_cast<float>(scale);
  host_tiling.aligned_seq_len_k = kTileKv;
  host_tiling.tile_qo = kTileQ;
  host_tiling.tile_kv = kTileKv;
  host_tiling.pipeline_stages = kPipelineStages;

  auto byte_options = query.options().dtype(at::kByte);
  auto long_options = query.options().dtype(at::kLong);
  auto output = at::empty_like(query);
  auto mask = use_template_mask ? GetCausalTemplateMask(query)
                                : at::empty({static_cast<int64_t>(mask_bytes)}, byte_options);
  auto cu_q = at::empty({static_cast<int64_t>(batch_size + 1)}, long_options);
  auto cu_k = at::empty({static_cast<int64_t>(batch_size + 1)}, long_options);
  auto workspace = at::empty({static_cast<int64_t>(workspace_bytes)}, byte_options);
  auto tiling = at::empty({static_cast<int64_t>(sizeof(host_tiling))}, byte_options);

  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  if (needs_device_mask && !use_template_mask) {
    CheckAcl(aclrtMemcpy(mask.data_ptr(), mask_bytes, host_mask.data(), mask_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "copy mask");
  }
  CheckAcl(aclrtMemcpy(cu_q.data_ptr(), cu_bytes, host_cu_q.data(), cu_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy cu_q");
  CheckAcl(aclrtMemcpy(cu_k.data_ptr(), cu_bytes, host_cu_k.data(), cu_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy cu_k");
  CheckAcl(aclrtMemcpy(tiling.data_ptr(), sizeof(host_tiling), &host_tiling, sizeof(host_tiling),
                       ACL_MEMCPY_HOST_TO_DEVICE),
           "copy tiling");

  if (query.scalar_type() == at::kHalf) {
    if (is_causal) {
      CheckAcl(ACLRT_LAUNCH_KERNEL(varlen_fa_fp16)(block_dim, stream, query.data_ptr(), key.data_ptr(),
                                                   value.data_ptr(), mask.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(),
                                                   output.data_ptr(), workspace.data_ptr(), tiling.data_ptr()),
               "launch varlen_fa_fp16");
    } else {
      CheckAcl(ACLRT_LAUNCH_KERNEL(varlen_fa_full_fp16)(block_dim, stream, query.data_ptr(), key.data_ptr(),
                                                        value.data_ptr(), mask.data_ptr(), cu_q.data_ptr(),
                                                        cu_k.data_ptr(), output.data_ptr(), workspace.data_ptr(),
                                                        tiling.data_ptr()),
               "launch varlen_fa_full_fp16");
    }
  } else {
    if (is_causal) {
      CheckAcl(ACLRT_LAUNCH_KERNEL(varlen_fa)(block_dim, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(),
                                              mask.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(), output.data_ptr(),
                                              workspace.data_ptr(), tiling.data_ptr()),
               "launch varlen_fa");
    } else {
      CheckAcl(ACLRT_LAUNCH_KERNEL(varlen_fa_full)(block_dim, stream, query.data_ptr(), key.data_ptr(),
                                                   value.data_ptr(), mask.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(),
                                                   output.data_ptr(), workspace.data_ptr(), tiling.data_ptr()),
               "launch varlen_fa_full");
    }
  }
  return output;
}

}  // namespace mojo::native
