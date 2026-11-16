/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/apis/layout.hpp
Original SHA256: 6f0de50e035764fb5a0f4ee3beadf9c423b5d724a0f53c1b0ab897f59b58308d
Source: csrc/jit_kernels/transform_sf.hpp
Original SHA256: de0049de38af56247b645b83560169ab679b6ffa22ec458840dea2350af31f5f
Adapted for Mojo A5 AOT dispatch.

MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <torch/torch.h>
#include <kernel/common.h>
#include <host/checks.h>
#include "../aot_kernel.h"
#include "../aot_runtime.h"
#include "math.h"

namespace deep_gemm {
// Dense scales describe logical [MN, K], independently of operand transpose.
// Return packed UE8M0x2 in MN-major order, expanding the MN granularity.
static torch::Tensor prepare_scale(const torch::Tensor& sf, uint32_t mn, uint32_t k, uint32_t gran_mn) {
    DG_HOST_ASSERT(sf.scalar_type() == at::kFloat || sf.scalar_type() == at::kShort);
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(sf.stride(0) == 1 || sf.stride(1) == 1,
                   "SF input must be contiguous in one matrix axis");
    DG_HOST_ASSERT(gran_mn >= 1 && gran_mn <= 128 && (gran_mn & (gran_mn - 1)) == 0,
                   "gran_mn only supports a power of two in [1, 128]");
    const bool is_float = sf.scalar_type() == at::kFloat;
    DG_HOST_ASSERT(sf.size(0) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(1) == ceil_div(k, is_float ? 32u : 64u));
    if (!is_float && gran_mn == 1 && sf.stride(0) == 1 && sf.stride(1) == sf.size(0))
        return sf;

    const uint32_t src_m = sf.size(0), src_k = sf.size(1);
    const uint32_t dst_k = ceil_div(src_k, is_float ? 2u : 1u);
    auto output = torch::empty({dst_k, mn}, sf.options().dtype(at::kShort)).t();
    if (output.numel() == 0) return output;
    const bool mn_major = sf.stride(1) != 1;
    uint32_t loggran = 0;
    for (uint32_t g = gran_mn; g > 1; g >>= 1) ++loggran;
    mojo_gemm::ScaleArgs args{
        reinterpret_cast<uintptr_t>(sf.data_ptr()), reinterpret_cast<uintptr_t>(output.data_ptr()),
        static_cast<uint64_t>(mn_major ? sf.stride(1) : sf.stride(0)),
        static_cast<uint64_t>(sf.size(0) * sf.stride(0)), src_m, src_k, mn, dst_k,
        static_cast<int>(is_float) * 16 + static_cast<int>(mn_major) * 8 + static_cast<int>(loggran)};
    mojo_gemm_transform_sf(args, 2 * get_num_cube_cores(), get_current_npu_stream());
    return output;
}
} // namespace deep_gemm
