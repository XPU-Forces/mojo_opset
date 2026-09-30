// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE in the root of the repository.

#include <limits>
#include <mutex>
#include <stdexcept>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace py = pybind11;

py::tuple reserve_philox_state(int device_index, uint64_t increment) {
    if (device_index < 0 || increment == 0 || increment % 4 != 0) {
        throw std::invalid_argument("resolved device and positive four-aligned increment required");
    }
    at_npu::PhiloxNpuState state;
    at::Tensor captured_seed, captured_offset;
    {
        py::gil_scoped_release release;
        c10::OptionalDeviceGuard device_guard(
            c10::Device(c10::DeviceType::PrivateUse1, device_index));
        auto* generator = at_npu::detail::getDefaultNPUGenerator(device_index)
                              .get<at_npu::NPUGeneratorImpl>();
        std::lock_guard<std::mutex> lock(generator->mutex_);
        // The Python caller has already allocated output on this device.
        // Use the exported context-ready API: the inline no-init wrapper
        // depends on IsContextInitialized, hidden in the installed wheel.
        if (c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None) {
            if (generator->get_offset() > std::numeric_limits<uint64_t>::max() - increment) {
                throw std::invalid_argument("framework offset + increment would overflow uint64");
            }
        }
        state = generator->philox_npu_state(increment);
        if (state.captured_) {
            // These are host Tensor-object pointers, not GM device addresses.
            // Take owning handles while the generator state is still locked.
            captured_seed = *state.seed_.ptr;
            captured_offset = *state.offset_.ptr;
        }
    }
    if (state.captured_) {
        return py::make_tuple(true, state.secondary_stream_capture_state_,
                              captured_seed, captured_offset, state.offset_intragraph_);
    }
    return py::make_tuple(false, false, state.seed_.val, state.offset_.val, uint64_t(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("reserve_philox_state", &reserve_philox_state,
               py::arg("device_index"), py::arg("increment"));
}
