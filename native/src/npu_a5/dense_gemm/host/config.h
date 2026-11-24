/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/jit_kernels/gemm_config.hpp
Original SHA256: d369e671b2abd86e7bbc40597aea5d9fec907a1ad64f94c94dee908ffdefed12
Source: csrc/utils/dtype.hpp
Original SHA256: 5690ec567489ad5db3a96fbe1c03b6a45b186afa9ba7c6fc75c7cf9c57d68146
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

#include <cstdint>
#include <memory>
#include <fmt/format.h>
#include <host/checks.h>
#include <ATen/core/Tensor.h>

#include <kernel/common.h>  // Major, host L2 cache types, capacities
#include "epilogue.h"

namespace deep_gemm {

inline constexpr auto kPackedFP4 = at::kChar;  // Two E2M1 values per byte.

// Physical dim0 (non-contiguous axis) row stride in ELEMENTS, for the kernel's gm_ptr.
// K-major: inner axis is K (stride(-1) must be 1), outer stride is along MN = stride(-2).
// MN-major: inner axis is MN (stride(-2) must be 1), outer stride is along K = stride(-1).
// Strictly assert the axis we treat as contiguous really has unit stride.
static uint64_t get_outer_stride(const at::Tensor& t, Major major) {
    if (major == Major::K) {
        DG_HOST_ASSERT(t.stride(-1) == 1, "K-major operand requires unit stride on the inner (K) axis");
        return static_cast<uint64_t>(t.stride(-2));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1, "MN-major operand requires unit stride on the inner (MN) axis");
    return static_cast<uint64_t>(t.stride(-1));
}

// Problem descriptor: what to compute (dimensions, dtypes, pointers, layouts).
// major_a/major_b are the GM major axis of each operand (Major::K / Major::MN); D is
// always N-major and SF is always MN-major.
struct GemmDesc {
    void* a_ptr;
    void* b_ptr;
    void* d_ptr;
    at::ScalarType a_dtype, b_dtype;
    void* sfa_ptr;              // Low precision only; nullptr for BF16
    void* sfb_ptr;              // Low precision only; nullptr for BF16
    uint32_t m, n, k;
    uint32_t num_cores;
    at::ScalarType cd_dtype;
    Major major_a, major_b;
    bool acc;                   // accumulate into D (D += A@B)
    std::shared_ptr<EpilogueClass> epilogue_class;
    // Physical dim0 (non-contiguous axis) row strides, in ELEMENTS, for the kernel's
    // gm_ptr. Tight inputs make these equal to k / k / n (A/B/D), reproducing the old
    // dense layout; strided or permuted operands carry their real strides. The
    // inner axis is always contiguous.
    //   outer_stride_a: A's stride along MN (K-major) or along K (MN-major)
    //   outer_stride_b: B's stride along MN (K-major) or along K (MN-major)
    //   outer_stride_d: D's stride along M (D is always N-major)
    uint64_t outer_stride_a = 0, outer_stride_b = 0, outer_stride_d = 0;
};

// Execution configuration: how to compute (tiling, pipeline, launch).
struct GemmConfig {
    // Tile sizes
    uint32_t block_m, block_n, block_k;
    uint32_t mad_m, mad_n, mad_k;

    // Pipeline depths
    uint32_t num_l1_stages;        // L1 data buffer stages (MTE2 <-> MTE1)
    uint32_t num_l0_stages;        // L0 AB buffer stages (MTE1 <-> Cube)
    uint32_t num_epilogue_stages;  // UB epilogue stages per AIV (AIC -> AIV)
    uint32_t num_l1_sf_stages;     // L1 SF buffer stages (0 for BF16)
    uint32_t sf_k_blocks;          // K-blocks per SF chunk (0 for BF16)

    bool direct_store = false;
    bool dual_aiv_dequant = false;
    // Zero dimensions/core count mean dynamic. Checked before every launch,
    // including forced configurations and measured tuning entries.
    uint32_t shape_n = 0, shape_k = 0, num_cores = 0;
    bool single_m_block = false;
    bool resident_a = false, resident_b = false, resident_sf = false;

    // L2 cache control
    asc_load_l2_cache_mode l2_ctrl_a;
    asc_load_l2_cache_mode l2_ctrl_b;
    asc_store_l2_cache_mode l2_ctrl_store_cd;
};

} // namespace deep_gemm

namespace fmt {

template <> struct formatter<deep_gemm::GemmConfig> : formatter<string_view> {
    static auto format(deep_gemm::GemmConfig v, format_context& context)  {
        return fmt::format_to(context.out(),
                              "GemmConfig(block={}x{}x{}, mad={}x{}x{}, l1_stages={}, l0_stages={}, "
                              "epi_stages={}, l1_sf_stages={}, sf_k_blocks={}, l2_ctrl={}/{}/{}, direct_store={}, resident={}/{}/{}, dual_aiv_dequant={})",
                              v.block_m, v.block_n, v.block_k, v.mad_m, v.mad_n, v.mad_k, v.num_l1_stages,
                              v.num_l0_stages, v.num_epilogue_stages, v.num_l1_sf_stages, v.sf_k_blocks, static_cast<int>(v.l2_ctrl_a),
                              static_cast<int>(v.l2_ctrl_b), static_cast<int>(v.l2_ctrl_store_cd), v.direct_store, v.resident_a, v.resident_b, v.resident_sf, v.dual_aiv_dequant);
    }
};

} // namespace fmt
