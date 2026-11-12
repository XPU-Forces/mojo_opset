#include "host.h"
#include "kernel/launch.h"
#include "common/npu.h"

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace mojo::native::a5 {
namespace {
uint32_t AivCores(int device) {
    static thread_local std::unordered_map<int, uint32_t> cache;
    auto& cores = cache[device];
    if (!cores) {
        int64_t count = 0;
        CheckAcl(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &count), "query AIV core count");
        TORCH_CHECK(count > 0, "AIV core count must be positive");
        cores = static_cast<uint32_t>(count);
    }
    return cores;
}
}

std::tuple<std::optional<at::Tensor>, at::Tensor> TopKForward(
    const at::Tensor& input, int64_t k, const std::optional<at::Tensor>& end,
    bool sorted_index, const std::optional<at::Tensor>& output_idx,
    const std::optional<at::Tensor>& output_idx_offset, int64_t idx_fill,
    double value_fill, bool return_value, bool abort_when_nan_found) {
    const bool fp32 = input.scalar_type() == at::kFloat;
    TORCH_CHECK(input.is_privateuseone() && (input.scalar_type() == at::kBFloat16 || fp32) && input.dim() == 2,
                "native topk_infer requires a 2D BF16 or FP32 NPU tensor");
    const c10::DeviceGuard guard(input.device());
    const auto batch = input.size(0), width = input.size(1);
    TORCH_CHECK(batch <= std::numeric_limits<uint32_t>::max() && width < MAX_VOCAB_SIZE,
                "native topk_infer requires B <= 2^32-1 and N < 2^21");
    TORCH_CHECK(k > 0 && k <= 4096, "topk must be in [1,4096]");
    if (fp32) TORCH_CHECK(width >= 1 && width <= 1024 && k <= width && k <= 32 &&
                          return_value && !sorted_index && !end && !output_idx_offset,
                          "FP32 requires N=1..1024, K<=min(N,32), values, without end, offsets or index sorting");
    TORCH_CHECK(input.stride(1) == 1 && (fp32 || input.stride(0) * input.element_size() % 32 == 0),
                "input must have stride(1)=1; BF16 row stride must be aligned to 32 bytes");
    TORCH_CHECK(idx_fill >= INT32_MIN && idx_fill <= INT32_MAX, "idx_oob_fill_value must fit int32");
    auto check_vector = [&](const std::optional<at::Tensor>& t, const char* name) {
        if (t) TORCH_CHECK(t->device() == input.device() && t->scalar_type() == at::kInt &&
                          t->dim() == 1 && t->size(0) == batch && t->is_contiguous(),
                          name, " must be contiguous int32 [B] on the input NPU");
    };
    check_vector(end, "end");
    check_vector(output_idx_offset, "output_idx_offset");
    auto aligned_output = [&](at::ScalarType dtype, int64_t elements_per_block) {
        const auto stride = (k + elements_per_block - 1) / elements_per_block * elements_per_block;
        return at::empty_strided({batch, k}, {stride, 1}, input.options().dtype(dtype));
    };
    const auto index_type = fp32 ? at::kLong : at::kInt;
    auto indices = output_idx ? *output_idx : aligned_output(index_type, fp32 ? 4 : 8);
    TORCH_CHECK(indices.device() == input.device() && indices.scalar_type() == index_type &&
                indices.sizes() == at::IntArrayRef({batch, k}) && indices.stride(1) == 1 &&
                indices.stride(0) >= k && indices.stride(0) * indices.element_size() % 32 == 0,
                "output_idx must match the index dtype and [B,K] shape with a non-overlapping 32-byte row stride");
    for (const auto& t : {input, indices}) {
        const auto format = at_npu::native::get_npu_format(t);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, "topk_infer requires ND storage");
    }
    std::optional<at::Tensor> values;
    if (return_value) values = aligned_output(input.scalar_type(), 32 / input.element_size());
    if (batch == 0) return {values, indices};

    auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    auto kernel_stream = stream.stream(false);
    const auto cores = fp32 ? AivCores(input.device().index()) : 0;
    for (const auto& t : {std::optional<at::Tensor>(input), end, output_idx_offset, values,
                          std::optional<at::Tensor>(indices)})
        if (t) c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(), stream);
    auto launch = [=]() {
        TopkSelectArgs args{
            static_cast<uint32_t>(batch), static_cast<uint32_t>(width), static_cast<uint32_t>(k),
            input.data_ptr(), values ? values->data_ptr() : nullptr, indices.data_ptr(), nullptr,
            end ? end->data_ptr<int32_t>() : nullptr,
            output_idx_offset ? output_idx_offset->data_ptr<int32_t>() : nullptr,
            static_cast<uint64_t>(input.stride(0)), values ? static_cast<uint64_t>(values->stride(0)) : 0,
            static_cast<uint64_t>(indices.stride(0)), fp32, sorted_index, return_value,
            static_cast<int32_t>(idx_fill), static_cast<float>(value_fill), abort_when_nan_found,
            c10::BFloat16(static_cast<float>(value_fill)).x, kernel_stream};
        if (fp32) LaunchRescanTopK(args, cores);
        else LaunchTopK(args);
        CheckAcl(aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL), "DeepSelect TopK launch");
        return ACL_SUCCESS;
    };
    // Release captures after submission; recordStream keeps storage alive on the device.
    auto payload = std::make_shared<std::optional<decltype(launch)>>(std::in_place, std::move(launch));
    at_npu::native::OpCommand::RunOpApiV2("mojo_a5_topk", [payload]() mutable -> int {
        auto execute = std::move(payload->value());
        payload->reset();
        return execute();
    });
    return {values, indices};
}
}
