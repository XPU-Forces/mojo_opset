#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "common/npu.h"

extern "C" void LaunchAttentionDropout(uint32_t, void*, void*, void*, uint64_t, uint32_t);

namespace {
void Mask(const at::Tensor& seed, at::Tensor output, int64_t threshold) {
    TORCH_CHECK(output.is_privateuseone() && output.scalar_type() == at::kInt && output.is_contiguous(),
                "attention dropout requires contiguous int32 NPU output");
    TORCH_CHECK(seed.device() == output.device() && seed.scalar_type() == at::kInt &&
                seed.is_contiguous() && seed.numel() == 2, "expected two same-device int32 Philox key words");
    TORCH_CHECK(threshold >= 0 && threshold <= 255, "invalid 8-bit keep threshold");
    const uint64_t bytes = uint64_t(output.numel()) * 4;
    TORCH_CHECK(bytes <= (uint64_t(1) << 32), "attention dropout exceeds the 32-bit counter range");
    if (!bytes) return;
    const c10::DeviceGuard guard(output.device());
    thread_local c10::DeviceIndex cached_device = -1;
    thread_local int64_t cores = 0;
    if (cached_device != output.get_device()) {
        mojo::native::CheckAcl(aclrtGetDeviceInfo(output.get_device(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores),
                              "query attention dropout vector cores");
        TORCH_CHECK(cores > 0, "invalid attention dropout vector core count");
        cached_device = output.get_device();
    }
    const auto stream = c10_npu::getCurrentNPUStream(output.get_device());
    for (const auto& tensor : {seed, output})
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    const std::array<c10::Storage, 2> keep{seed.storage(), output.storage()};
    const std::array<void*, 2> args{seed.data_ptr(), output.data_ptr()};
    auto launch = [keep, args, bytes, threshold, blocks = uint32_t(cores), raw_stream = stream.stream(false)]() {
        LaunchAttentionDropout(blocks, raw_stream, args[0], args[1], bytes, uint32_t(threshold));
    };
    auto payload = std::make_shared<std::optional<decltype(launch)>>(std::in_place, std::move(launch));
    at_npu::native::OpCommand::RunOpApiV2("mojo_attention_dropout", [payload]() mutable -> int {
        TORCH_CHECK(payload->has_value(), "attention dropout submission executed twice");
        auto execute = std::move(payload->value());
        payload->reset();
        execute();
        return ACL_SUCCESS;
    });
}
} // namespace

TORCH_LIBRARY(mojo_native_attention_dropout, m) {
    m.def("mask(Tensor seed, Tensor(a!) output, int threshold) -> ()", &Mask);
}
