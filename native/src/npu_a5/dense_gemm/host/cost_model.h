/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/jit_kernels/heuristics.hpp
Original SHA256: 288129cd1a7d315218c37703b87ea0f5848a169c6fb95a82e8691ccf06a68040
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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>

#include <host/checks.h>

#include "math.h"
#include "config.h"

namespace deep_gemm {

struct HWDesc {
    int gm_bw_gbps, l2_bw_gbps;
    int aic_clock_mhz;
    int bf16_flops_per_cycle;
    int gm_to_l1_bytes_per_cycle;
    int l1_to_l0_bytes_per_cycle;
    int l1_to_l0_sf_bytes_per_cycle;
    int l0c_to_ub_bytes_per_cycle;
    int ub_to_l1_bytes_per_cycle;
    int vector_reg_bytes, fp4_decode_bytes_per_cycle;
    int ub_bank_bytes, l1_bank_bytes;
    int num_pipe_events, num_intra_block_events;
    int num_ai_cores;
    int64_t l2_size_bytes;
    int l2_sector_bytes, sf_elements_per_pair, fractal_row_bytes;

    constexpr int get_mad_alignment_mn() const {
        return FRAC_MN;
    }

    constexpr int get_mad_alignment_k(int elem_bits) const {
        return fractal_row_bytes * 8 / elem_bits;
    }

    constexpr int get_mad_flops_per_cycle(int elem_bits) const {
        return bf16_flops_per_cycle * (sizeof(uint16_t) * 8) / elem_bits;
    }

};

static constexpr HWDesc get_hw_desc() {
    // Ascend 950: effective chip rates and CANN's per-core UB -> L1 rate.
    return {
        .gm_bw_gbps=3500, .l2_bw_gbps=4500,
        .aic_clock_mhz=1650,
        .bf16_flops_per_cycle=8192,  // 4096 multiply-adds; each counts as two FLOPs.
        .gm_to_l1_bytes_per_cycle=256,
        .l1_to_l0_bytes_per_cycle=256,
        .l1_to_l0_sf_bytes_per_cycle=32,  // Measured SF transfer rate.
        .l0c_to_ub_bytes_per_cycle=128,
        .ub_to_l1_bytes_per_cycle=128,
        .vector_reg_bytes=256, .fp4_decode_bytes_per_cycle=64,  // 512 decoded bytes per eight vector cycles.
        .ub_bank_bytes=32,
        .l1_bank_bytes=32,
        .num_pipe_events=kNumPipeEvents, .num_intra_block_events=16,
        .num_ai_cores=32,
        .l2_size_bytes=128 * 1024 * 1024,
        .l2_sector_bytes=kL2SectorBytes, .sf_elements_per_pair=64, .fractal_row_bytes=32,
    };
}

struct GemmDtype {
    int elem_bits;  // Operand width in L1; dequantized B is FP8 here.
    int epilogue_elem_bits;
    bool has_sf = false;
    bool dequant_b = false;
};

static GemmDtype get_gemm_dtype(const GemmDesc& d) {
    const bool with_transform = dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) == nullptr;
    int epilogue_elem_bits = (d.cd_dtype == at::kBFloat16 && !with_transform) ? 16 : 32;
    if (d.a_dtype == at::kBFloat16 && d.b_dtype == at::kBFloat16)
        return {.elem_bits=16, .epilogue_elem_bits=epilogue_elem_bits};
    if (d.a_dtype == kPackedFP4 && d.b_dtype == kPackedFP4)
        return {.elem_bits=4, .epilogue_elem_bits=epilogue_elem_bits, .has_sf=true};
    DG_HOST_ASSERT(d.a_dtype == at::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.b_dtype == at::kFloat8_e4m3fn || d.b_dtype == kPackedFP4);
    return {.elem_bits=8, .epilogue_elem_bits=epilogue_elem_bits, .has_sf=true, .dequant_b=d.b_dtype == kPackedFP4};
}

struct GemmComputeCost {
    int64_t mflops, time_ns, num_tail_blocks;
};

static GemmComputeCost get_gemm_compute_cost(const GemmDesc& d, const GemmDtype& dtype, int block_m, int block_n) {
    constexpr int kBlockSwizzleM = 4;
    const auto hw = get_hw_desc();
    const int m = d.m, n = d.n;
    const int64_t k = d.k;
    const int num_m_blocks = ceil_div(m, block_m);
    const int num_n_blocks = ceil_div(n, block_n);
    const int64_t num_blocks = static_cast<int64_t>(num_m_blocks) * num_n_blocks;
    const int num_batches = 1;
    const int64_t num_waves = ceil_div(num_blocks * num_batches, d.num_cores);

    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();

    // Derive Actual MatMul Shape
    const int compute_m = align(std::min(m, block_m), align_m);
    const int compute_n = align(std::min(n, block_n), align_n);

    // Derive MatMul FLOPS and MatMul Time
    int64_t compute_mn = num_waves * compute_m * compute_n;
    int64_t num_tail_blocks = 0;
    const int tail_stride = num_m_blocks % kBlockSwizzleM ? 1 : kBlockSwizzleM;
    // Adjust compute MN for the tail blocks
    if (std::gcd(tail_stride, d.num_cores) == 1) {
        const int num_core_classes = std::gcd(num_blocks, d.num_cores);
        const int num_groups_per_period = d.num_cores / num_core_classes;
        const int num_remaining_groups = num_batches % num_groups_per_period;
        const int64_t num_full_per_period = ceil_div(num_blocks - num_n_blocks, num_core_classes);
        const int tail_remainder = num_n_blocks % d.num_cores;
        const int tail_suffix = tail_stride == 1 ? tail_remainder : std::min(tail_remainder, 1);
        const int64_t num_full_per_group = ceil_div(num_blocks - tail_suffix, d.num_cores) - num_n_blocks / d.num_cores;
        // Bound full blocks on the busiest core over complete periods and the remaining groups.
        const int64_t num_full_remaining = std::min(num_remaining_groups * num_full_per_group, num_full_per_period);
        const int64_t num_full_blocks = num_batches / num_groups_per_period * num_full_per_period + num_full_remaining;
        num_tail_blocks = std::max<int64_t>(0, num_waves - num_full_blocks);
        const int tail_m = align(m - (num_m_blocks - 1) * block_m, align_m);
        compute_mn -= num_tail_blocks * (compute_m - tail_m) * compute_n;
    }
    const int64_t compute_flops = compute_mn * 2 * k;
    const int64_t compute_mflops = static_cast<int64_t>(hw.get_mad_flops_per_cycle(dtype.elem_bits)) * hw.aic_clock_mhz;
    const int64_t compute_time_ns = ceil_div(compute_flops * 1000, compute_mflops);
    return {.mflops=compute_mflops, .time_ns=compute_time_ns, .num_tail_blocks=num_tail_blocks};
}

static auto get_gemm_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c, const GemmComputeCost& compute_cost, int64_t best_cost = std::numeric_limits<int64_t>::max()) {
    constexpr int kBlockSwizzleM = 4;
    const auto hw = get_hw_desc();
    // Single-M scheduling keeps actual extents and traffic; no scalar-cycle discount is modeled.
    const int m = d.m, n = d.n;
    const int64_t k = d.k;
    const int block_m = c.block_m, block_n = c.block_n;
    const int num_m_blocks = ceil_div(m, block_m);
    const int num_n_blocks = ceil_div(n, block_n);
    const int64_t num_blocks = static_cast<int64_t>(num_m_blocks) * num_n_blocks;
    const int num_batches = 1;
    const int64_t num_waves = ceil_div(num_blocks * num_batches, d.num_cores);
    const int b_elem_bits = dtype.dequant_b ? dtype.elem_bits / 2 : dtype.elem_bits;
    const int64_t sf_row_bytes = dtype.has_sf ? ceil_div(k, hw.sf_elements_per_pair) * sizeof(int16_t) : 0;

    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();

    // Derive Actual MatMul Shape
    const int compute_m = align(std::min(m, block_m), align_m);
    const int compute_n = align(std::min(n, block_n), align_n);

    const auto [compute_mflops, compute_time_ns, num_tail_blocks] = compute_cost;

    // Derive the memory loads in a block (BLOCK_M, BLOCK_N)
    const auto gm_load_bytes = [&](int mn, int64_t k, int total_mn, int elem_bits, Major major, int64_t outer_stride) -> int64_t {
        const bool mn_major = is_mn_major(major);
        const int64_t row_bytes = ceil_div((mn_major ? total_mn : k) * elem_bits, 8);
        const int64_t block_bytes = ceil_div((mn_major ? mn : c.block_k) * elem_bits, 8);
        // Packed FP4 strides count bytes, while its logical extents count nibbles.
        const int64_t stride_bytes = outer_stride ? outer_stride * std::max(elem_bits, 8) / 8 : row_bytes;
        const int64_t num_row_blocks = ceil_div(row_bytes, block_bytes);
        const int64_t stride_gcd = std::gcd(stride_bytes, int64_t(hw.l2_sector_bytes));
        const int64_t boundary_period = stride_gcd / std::gcd(block_bytes, stride_gcd);
        // Average covered sectors over row-address phases, including shared block boundaries.
        const int64_t row_load_bytes = align(row_bytes, stride_gcd) + num_row_blocks * hw.l2_sector_bytes
                                    - stride_gcd * (1 + (num_row_blocks - 1) / boundary_period);
        return mn_major ? ceil_div(k * row_load_bytes, num_row_blocks) : mn * row_load_bytes;
    };
    const auto sf_load_bytes = [&](int mn, int64_t k, int total_mn) -> int64_t {
        return dtype.has_sf ? gm_load_bytes(mn, ceil_div(k, hw.sf_elements_per_pair), total_mn, sizeof(int16_t) * 8, Major::MN, total_mn) : 0;
    };
    const int load_m = m;
    const int64_t sfa_block_bytes = sf_load_bytes(block_m, k, load_m);
    const int64_t sfb_block_bytes = sf_load_bytes(block_n, k, n);
    int64_t wa_block_bytes = gm_load_bytes(block_m, k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a);
    const int64_t wb_block_bytes = gm_load_bytes(block_n, k, n, b_elem_bits, d.major_b, d.outer_stride_b);
    if (dtype.elem_bits == 16 && c.resident_a && is_k_major(d.major_a))
        wa_block_bytes = int64_t(block_m) * align(k * sizeof(uint16_t), hw.l2_sector_bytes);
    const int64_t a_block_bytes = wa_block_bytes + sfa_block_bytes;
    const int64_t b_block_bytes = wb_block_bytes + sfb_block_bytes;

    // L2 residency uses logical bytes within one group; SF inherits its operand's hint.
    const int64_t l2_a_row_bytes = c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP ? 0 : sf_row_bytes + ceil_div(k * dtype.elem_bits, 8);
    const int64_t l2_b_row_bytes = c.l2_ctrl_b == asc_load_l2_cache_mode::NOTALLOC_KEEP ? 0 : sf_row_bytes + ceil_div(k * b_elem_bits, 8);

    // Derive the total cold GM loads for A/B and SF of all blocks
    auto [total_gm_a_bytes, total_gm_b_bytes] = [&](){
        const int block_swizzle_m = std::min(num_m_blocks, kBlockSwizzleM);
        const int block_swizzle_n = std::min(num_n_blocks, hw.num_ai_cores / block_swizzle_m);
        const int64_t l2_block_swizzle_a_bytes = std::min(m, block_swizzle_m * block_m) * l2_a_row_bytes;
        const int64_t l2_block_swizzle_b_bytes = std::min(n, block_swizzle_n * block_n) * l2_b_row_bytes;

        // First use loads each A/B block from GM; eviction reloads it at each swizzle boundary.
        int64_t num_a_gm_blocks = static_cast<int64_t>(num_m_blocks) * num_batches;
        int64_t num_b_gm_blocks = static_cast<int64_t>(num_n_blocks) * num_batches;
        if (l2_block_swizzle_a_bytes + l2_block_swizzle_b_bytes > hw.l2_size_bytes) {
            num_a_gm_blocks *= ceil_div(num_n_blocks, block_swizzle_n);
        }
        if (l2_block_swizzle_a_bytes + n * l2_b_row_bytes > hw.l2_size_bytes) {
            num_b_gm_blocks *= ceil_div(num_m_blocks, block_swizzle_m);
        }
        if (c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP) num_a_gm_blocks = num_blocks * num_batches;
        if (c.l2_ctrl_b == asc_load_l2_cache_mode::NOTALLOC_KEEP) num_b_gm_blocks = num_blocks * num_batches;
        const int64_t gm_a_bytes = num_a_gm_blocks * a_block_bytes;
        const int64_t gm_b_bytes = num_b_gm_blocks * b_block_bytes;

        return std::tuple{gm_a_bytes, gm_b_bytes};
    }();

    // Derive the total load time for A/B matrix
    const auto get_load_time_ns = [&](int64_t total_gm_bytes, int64_t total_load_bytes) {
        total_gm_bytes = std::min(total_gm_bytes, total_load_bytes);
        const int64_t l2_bytes = total_load_bytes - total_gm_bytes;
        // Not all aicores are active to access GM, we use the wave utilization to estimate the effective GM bandwidth
        //   active_gm_gbps = gm_bw_gbps * (num_blocks * num_batches / (num_waves * num_ai_cores))
        //   gm_time_ns = gm_bytes / active_gm_gbps
        // The above formula is equivalent to the following, which avoids float arithmetic
        const int64_t gm_time_ns = ceil_div(static_cast<__int128>(total_gm_bytes) * num_waves * hw.num_ai_cores, num_blocks * num_batches * hw.gm_bw_gbps);
        const int64_t l2_time_ns = ceil_div(static_cast<__int128>(l2_bytes) * num_waves * hw.num_ai_cores, num_blocks * num_batches * hw.l2_bw_gbps);
        return gm_time_ns + l2_time_ns;
    };
    int64_t a_load_blocks = c.resident_a && num_m_blocks == 1 ? std::min(num_blocks, int64_t(d.num_cores)) * num_batches : num_blocks * num_batches;
    int64_t b_load_blocks = c.resident_b && num_n_blocks == 1 ? std::min(num_blocks, int64_t(d.num_cores)) * num_batches : num_blocks * num_batches;
    const int64_t sfa_load_blocks = c.resident_sf ? a_load_blocks : num_blocks * num_batches;
    const int64_t sfb_load_blocks = c.resident_sf ? b_load_blocks : num_blocks * num_batches;
    const int64_t load_a_time_ns = get_load_time_ns(total_gm_a_bytes, a_load_blocks * wa_block_bytes + sfa_load_blocks * sfa_block_bytes);
    const int64_t load_b_time_ns = get_load_time_ns(total_gm_b_bytes, b_load_blocks * wb_block_bytes + sfb_load_blocks * sfb_block_bytes);

    // UB -> L1 copies half-row B slices.
    auto [dequant_bytes, dequant_time_ns] = [&](){
        const int b_rows = is_k_major(d.major_b) ? block_n : c.block_k;
        const int contiguous_bytes = b_rows / 2 * hw.fractal_row_bytes;
        const int64_t dequant_bytes = dtype.dequant_b ? ceil_div(b_load_blocks, d.num_cores) * block_n * k * dtype.elem_bits / 8
                                   * align(contiguous_bytes, hw.ub_to_l1_bytes_per_cycle) / contiguous_bytes : 0;
        const int64_t dequant_time_ns = ceil_div(dequant_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        return std::tuple{dequant_bytes, dequant_time_ns};
    }();

    const int64_t memory_time_ns = load_a_time_ns + load_b_time_ns + dequant_time_ns;
    if (memory_time_ns > best_cost) return memory_time_ns;

    // Limit the compute discount by the first tail wave's cold-A service.
    const int64_t tail_memory_delay_ns = [&](){
        const int tail_m = m - (num_m_blocks - 1) * block_m;
        const int tail_compute_m = align(tail_m, align_m);
        const int64_t discount_ns = ceil_div(num_tail_blocks * (compute_m - tail_compute_m) * compute_n * 2 * k * 1000, compute_mflops);
        if (!discount_ns) return int64_t(0);

        const auto mn_tail_bytes = [&](int64_t num_rows, int elem_bits, int64_t stride_bytes) {
            const int64_t stride_gcd = std::gcd(stride_bytes, int64_t(hw.l2_sector_bytes));
            const int64_t offset_bytes = int64_t(m - tail_m) * elem_bits / 8;
            const int64_t row_bytes = ceil_div(tail_m * elem_bits, 8);
            return num_rows * (align(offset_bytes % stride_gcd + row_bytes, stride_gcd) + hw.l2_sector_bytes - stride_gcd);
        };
        const int64_t a_stride_bytes = d.outer_stride_a ? d.outer_stride_a * std::max(dtype.elem_bits, 8) / 8 : ceil_div(m * dtype.elem_bits, 8);
        const int64_t tail_a_bytes = is_mn_major(d.major_a)
            ? mn_tail_bytes(k, dtype.elem_bits, a_stride_bytes)
            : gm_load_bytes(tail_m, k, m, dtype.elem_bits, d.major_a, d.outer_stride_a);
        const int64_t tail_sfa_bytes = dtype.has_sf ? mn_tail_bytes(ceil_div(k, hw.sf_elements_per_pair), sizeof(int16_t) * 8, m * sizeof(int16_t)) : 0;
        const bool reuse_b = num_m_blocks > 1 && c.l2_ctrl_b != asc_load_l2_cache_mode::NOTALLOC_KEEP
                          && std::min(m, kBlockSwizzleM * block_m) * l2_a_row_bytes + n * l2_b_row_bytes <= hw.l2_size_bytes;
        const int num_tail_wave_blocks = std::min<int>(num_n_blocks, d.num_cores);
        const int64_t tail_memory_ns = ceil_div((tail_a_bytes + tail_sfa_bytes) * num_tail_wave_blocks, hw.gm_bw_gbps)
                                     + ceil_div(b_block_bytes * num_tail_wave_blocks, reuse_b ? hw.l2_bw_gbps : hw.gm_bw_gbps)
                                     + ceil_div(dequant_time_ns, num_waves);
        const int64_t tail_compute_ns = ceil_div(int64_t(tail_compute_m) * compute_n * 2 * k * 1000, compute_mflops);
        const int64_t num_cold_tail_waves = c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP ? num_tail_blocks : std::min<int64_t>(num_batches, num_tail_blocks);
        return std::min(discount_ns, num_cold_tail_waves * std::max<int64_t>(0, tail_memory_ns - tail_compute_ns));
    }();

    // First fill includes the full resident-A load before the K loop; it cannot overlap compute.
    auto first_load_time_ns = [&](){
        const int64_t first_k = std::min<int64_t>(k, c.block_k);
        const int64_t first_sf_k = std::min<int64_t>(k, c.block_k * c.sf_k_blocks);
        const int64_t first_sfa_bytes = sf_load_bytes(block_m, first_sf_k, load_m);
        const int64_t first_sfb_bytes = sf_load_bytes(block_n, first_sf_k, n);
        const int64_t first_a_bytes = dtype.elem_bits == 16 && c.resident_a ? a_block_bytes : gm_load_bytes(block_m, first_k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a) + first_sfa_bytes;
        const int64_t first_b_bytes = gm_load_bytes(block_n, first_k, n, b_elem_bits, d.major_b, d.outer_stride_b) + first_sfb_bytes;

        // Estimate A, B, Dequant, SF time by the total stage time with the following formula:
        //    first-stage-time = total-stage-time * (first-stage-bytes / total-bytes)
        const int64_t first_dequant_ns = ceil_div(dequant_bytes * first_k / k * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t first_stage_time_ns = (
            ceil_div(load_a_time_ns * first_a_bytes, a_block_bytes) +
            ceil_div(load_b_time_ns * first_b_bytes, b_block_bytes) +
            first_dequant_ns
        );
        const int64_t sf_startup_time_ns = (
            ceil_div(load_a_time_ns * first_sfa_bytes, a_block_bytes) +
            ceil_div(load_b_time_ns * first_sfb_bytes, b_block_bytes)
        );

        const int64_t prefetch_k = std::min<int64_t>(k, (c.num_l1_stages - 1) * c.block_k);
        const int64_t prefetch_compute_ns = ceil_div(prefetch_k * 2 * compute_m * compute_n * 1000, compute_mflops);
        const int64_t sf_overlap_time_ns = c.num_l1_sf_stages > 1 ? std::min(sf_startup_time_ns / num_waves, prefetch_compute_ns) : 0;
        const int64_t first_load_time_ns = first_stage_time_ns - (num_waves - 1) * sf_overlap_time_ns;
        return first_load_time_ns;
    }();

    // Derive the time to load A/B/SF from L1 to L0, this is overlapped with compute
    const int64_t l0_time_ns = [&](){
        // Measure l0 reuse: when m,k or n,k is small, they can reside in L0
        const int num_l0a_loads = c.block_k <= c.mad_k * c.num_l0_stages ? 1 : block_n / c.mad_n;
        const int num_l0b_loads = block_n == c.mad_n && c.block_k <= c.mad_k * c.num_l0_stages ? 1 : block_m / c.mad_m;
        const int64_t num_l0_rows = num_waves * (block_m * num_l0a_loads + block_n * num_l0b_loads);
        const int64_t a_bytes = num_waves * block_m * num_l0a_loads * k * dtype.elem_bits / 8;
        const int64_t b_bytes = (dtype.elem_bits == 16 && c.resident_b ? 1 : num_waves) * block_n * num_l0b_loads * k * dtype.elem_bits / 8;

        // Measure L0 copy amplification, if the copy is not aligned to the transfer bytes, the bandwidth is wasted
        const int a_contiguous_bytes = (is_k_major(d.major_a) ? c.mad_m : c.mad_k) * hw.fractal_row_bytes;
        const int b_contiguous_bytes = (is_k_major(d.major_b) ? c.mad_n : c.mad_k) * hw.fractal_row_bytes;
        const int64_t ab_bytes = a_bytes * align(a_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) / a_contiguous_bytes
                               + b_bytes * align(b_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) / b_contiguous_bytes;
        const int64_t sf_bytes = num_l0_rows * sf_row_bytes;

        const int64_t l0_time_ns = (
            ceil_div(ab_bytes * 1000, hw.l1_to_l0_bytes_per_cycle * hw.aic_clock_mhz) +
            ceil_div(sf_bytes * 1000, hw.l1_to_l0_sf_bytes_per_cycle * hw.aic_clock_mhz)
        );
        return l0_time_ns;
    }();

    // Derive the compute-bound total time
    // TODO: add first_l0_load_ns and last_mte3_copy_ns
    const int64_t compute_bound_time_ns = first_load_time_ns + std::max(l0_time_ns, compute_time_ns + tail_memory_delay_ns);

    // Total loads include the first fill; the last K block still needs its MADs.
    // TODO: add last_mte3_copy_ns
    const int64_t last_k = k % c.block_k ? k % c.block_k : c.block_k;
    const int64_t last_compute_time_ns = ceil_div(last_k * compute_m * compute_n * 2 * 1000, compute_mflops);
    const int64_t memory_bound_time_ns = memory_time_ns + last_compute_time_ns;

    return std::max(compute_bound_time_ns, memory_bound_time_ns);
}

static auto get_gemm_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    return get_gemm_cost(d, dtype, c, get_gemm_compute_cost(d, dtype, c.block_m, c.block_n));
}

static int64_t get_gemm_output_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const int64_t num_waves = ceil_div(int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n), d.num_cores);
    const int64_t num_values = int64_t(std::min(d.m, c.block_m)) * std::min(d.n, c.block_n);
    const int64_t bytes = num_values * dtype.epilogue_elem_bits / 8;
    const bool with_bias = dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) != nullptr;
    // AddBias owns one BLOCK_N row per AIV drain; its hooks preserve the existing overlap.
    const int64_t bias_bytes = with_bias ? (dtype.dequant_b ? 1 : 2) *
                               std::min(d.n, c.block_n) * c10::elementSize(d.cd_dtype) : 0;
    const int64_t bias_ns = ceil_div(bias_bytes * hw.num_ai_cores, hw.gm_bw_gbps);
    const int64_t gm_bytes = num_values * c10::elementSize(d.cd_dtype) * (1 + d.acc);
    const int64_t fix_ns = ceil_div(bytes * 1000, hw.l0c_to_ub_bytes_per_cycle * hw.aic_clock_mhz);
    const int64_t gm_ns = ceil_div(gm_bytes * hw.num_ai_cores, hw.gm_bw_gbps);
    const int64_t vector_values = int64_t(align(std::min(d.m, c.block_m), c.mad_m)) * align(std::min(d.n, c.block_n), c.mad_n);
    // Pure AddBias performs one vector add per FP32 vector; BF16 packing is fused into its store.
    const int64_t vector_ns = with_bias ? ceil_div(ceil_div(vector_values * sizeof(float), hw.vector_reg_bytes) * 1000, hw.aic_clock_mhz) : 0;
    if (with_bias && c.num_epilogue_stages >= 2 && num_waves >= 2)
        return num_waves * std::max(fix_ns, bias_ns + gm_ns + vector_ns);
    return num_waves * (c.direct_store ? std::max(fix_ns, gm_ns) : std::max(fix_ns, bias_ns) + gm_ns + vector_ns);
}

static int64_t get_gemm_refinement_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    int64_t cost = get_gemm_cost(d, dtype, c);
    if (dtype.elem_bits == 16 && d.m < c.block_m) {
        constexpr int kGmLoadIssueCycles = 64;  // Resident-A source calibration, separate from transfer bytes.
        const int64_t num_waves = ceil_div(ceil_div(d.n, c.block_n), d.num_cores);
        const int num_k_blocks = ceil_div(d.k, c.block_k);
        const int64_t num_loads = (c.resident_a ? 1 : num_waves * num_k_blocks) + num_waves * num_k_blocks;
        cost += ceil_div(num_loads * kGmLoadIssueCycles * 1000, hw.aic_clock_mhz);
    }
    if (dtype.dequant_b) {
        const int64_t num_blocks = int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n);
        const int num_batches = 1;
        int64_t b_load_blocks = c.resident_b ? std::min(num_blocks, int64_t(d.num_cores)) * num_batches : num_blocks * num_batches;
        const int num_aivs = c.dual_aiv_dequant ? 2 : 1;
        const int64_t decoded_bytes = ceil_div(b_load_blocks, d.num_cores) * c.block_n * (d.k);
        const int64_t decode_ns = ceil_div(decoded_bytes * 1000, num_aivs * hw.fp4_decode_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t store_l1_ns = ceil_div(decoded_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t compute_ns = get_gemm_compute_cost(d, dtype, c.block_m, c.block_n).time_ns;
        cost += std::max<int64_t>(0, decode_ns - std::max(compute_ns, store_l1_ns));
    }
    return cost + get_gemm_output_cost(d, dtype, c);
}

static bool can_direct_store(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const auto stride_d = d.outer_stride_d ? d.outer_stride_d : d.n;
    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : 1;
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : FRAC_MN;
    return dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) &&
           (dtype.elem_bits != 4 || (is_k_major(d.major_a) && is_k_major(d.major_b))) &&
           (!d.acc || d.cd_dtype == at::kFloat) && d.m % align_m == 0 && d.n % align_n == 0 &&
           reinterpret_cast<uintptr_t>(d.d_ptr) % hw.ub_bank_bytes == 0 &&
           stride_d * dtype.epilogue_elem_bits / 8 % hw.ub_bank_bytes == 0 &&
           (int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n) <= d.num_cores ||
            (d.m <= c.block_m && c.mad_n * dtype.epilogue_elem_bits / 8 % hw.l2_sector_bytes == 0));
}

static int64_t get_gemm_k_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const int64_t num_waves = ceil_div(int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n), d.num_cores);
    const int64_t mad_cycles = ceil_div(2 * int64_t(c.mad_m) * c.mad_n * c.mad_k, hw.get_mad_flops_per_cycle(dtype.elem_bits));
    const int64_t load_cycles = ceil_div(int64_t(c.mad_m + c.mad_n) * c.mad_k * dtype.elem_bits, 8 * hw.l1_to_l0_bytes_per_cycle)
        + (dtype.has_sf ? ceil_div(int64_t(c.mad_m + c.mad_n) * c.mad_k * sizeof(int16_t), hw.sf_elements_per_pair * hw.l1_to_l0_sf_bytes_per_cycle) : 0);
    const int64_t num_mads = int64_t(c.block_m / c.mad_m) * (c.block_n / c.mad_n);
    // Effective L0/SF issue costs from paired BF16 and native FP8/FP4 measurements.
    const int issue_cycles = dtype.elem_bits == 16 ? 64 : dtype.elem_bits == 4 ? 40 : 116;
    const int64_t issue_ns = ceil_div(num_waves * num_mads * ceil_div(d.k, c.mad_k) * issue_cycles * 1000, hw.aic_clock_mhz);
    const int64_t num_turnovers = d.k / c.block_k * std::max<int64_t>(0, int64_t(c.block_k / c.mad_k) - c.num_l0_stages)
        + std::max<int64_t>(0, int64_t(ceil_div(d.k % c.block_k, c.mad_k)) - c.num_l0_stages);
    const int64_t turnover_ns = ceil_div(num_waves * num_mads * num_turnovers * std::max(mad_cycles, load_cycles) * 1000, hw.aic_clock_mhz);
    return get_gemm_refinement_cost(d, dtype, c) + issue_ns + turnover_ns;
}

} // namespace deep_gemm
