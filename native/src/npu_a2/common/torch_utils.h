#pragma once
#include "common/npu.h"
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <utility>

namespace mojo::native::a2 {
struct CoreCounts {
  uint32_t cube, vector;
};
inline CoreCounts core_counts(c10::DeviceIndex device) {
  thread_local c10::DeviceIndex cached_device = -1;
  thread_local CoreCounts counts{};
  if (cached_device != device) {
    int64_t cube = 0, vector = 0;
    CheckAcl(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cube), "query cube cores");
    CheckAcl(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector), "query vector cores");
    TORCH_CHECK(cube > 0 && vector > 0, "invalid A2 core counts");
    counts = {static_cast<uint32_t>(cube), static_cast<uint32_t>(vector)};
    cached_device = device;
  }
  return counts;
}
inline void tensor(const at::Tensor &value, const char *name) {
  TORCH_CHECK(value.defined() && value.is_privateuseone(), name, " must be an NPU tensor");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}
inline void record(const at::Tensor &value, c10_npu::NPUStream stream) {
  if (value.defined() && value.numel())
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
}
template <class Launch> inline void submit(const char *name, Launch launch) {
  // Caller captures stream, inputs/outputs and workspaces by value. Tensor
  // storage is also recorded until the device finishes after queued submission.
  // The NPU queue may keep a completed handler in its reusable command slot.
  // Release tensor captures when the handler runs, even if copies of the
  // handler remain in that slot; allocator recordStream owns the subsequent
  // device lifetime.
  auto payload = std::make_shared<std::optional<Launch>>(std::in_place, std::move(launch));
  at_npu::native::OpCommand::RunOpApiV2(name, [payload]() mutable -> int {
    TORCH_CHECK(payload->has_value(), "native kernel submission handler executed twice");
    auto execute = std::move(payload->value());
    payload->reset();
    execute();
    return ACL_SUCCESS;
  });
}
} // namespace mojo::native::a2
