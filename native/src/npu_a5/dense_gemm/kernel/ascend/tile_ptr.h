/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: include/deep_gemm/ascend/tile_ptr.hpp
Original SHA256: c161d41dc84356c24f9aad2da77f826d2833245255025389ebf59d2d7a039a4f
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

#include <c_api/asc_simd.h>
#include <kernel/common.h>

namespace deep_gemm {

template <typename T, typename dst_t> struct cvt_pointee { using type = dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__gm__ T*, dst_t> { using type = __gm__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cbuf__ T*, dst_t> { using type = __cbuf__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__ca__ T*, dst_t> { using type = __ca__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cb__ T*, dst_t> { using type = __cb__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cc__ T*, dst_t> { using type = __cc__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__ubuf__ T*, dst_t> { using type = __ubuf__ dst_t*; };
template <typename ptr_t, typename dst_t>
using cvt_pointee_t = typename cvt_pointee<ptr_t, dst_t>::type;

template <typename ptr_t, typename elem_t, Major MAJOR = Major::K>
struct operand_ptr {
    uintptr_t addr;
    const uint64_t shape_mn, shape_k;  // GEMM operand dimensions (logical, layout-invariant)
    static constexpr Major major = MAJOR;

    using dtype_t = elem_t;

    // K-elements per type-slot: fp4 (float4_e2m1x2_t) packs 2, all others 1.
    static constexpr uint32_t k_per_slot = is_fp4<dtype_t>() ? 2u : 1u;

    constexpr __aicore__ operand_ptr(uint64_t shape_mn, uint64_t shape_k, uintptr_t addr = 0) :
        addr(addr), shape_mn(shape_mn), shape_k(shape_k) {}

    template <typename dst_dtype_t = dtype_t>
    __aicore__ cvt_pointee_t<ptr_t, dst_dtype_t> ptr() const {
        return reinterpret_cast<cvt_pointee_t<ptr_t, dst_dtype_t>>(addr);
    }

    constexpr __aicore__ uint64_t size_per_stage() const { return shape_mn * shape_k / k_per_slot * sizeof(dtype_t); }
    constexpr __aicore__ uintptr_t offset(uint32_t n) const { return addr + n * size_per_stage(); }
    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> operator[](uint32_t i) const {
        operand_ptr<ptr_t, elem_t, MAJOR> r = *this;
        r.addr = offset(i);
        return r;
    }

    // Byte-granular addressing for NZ fractal layouts (element/stage strides do not apply).
    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> offset_bytes(uint64_t bytes) const {
        operand_ptr<ptr_t, elem_t, MAJOR> r = *this;
        r.addr += bytes;
        return r;
    }

    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> as_mad_aligned(uint64_t shape_mn, uint64_t shape_k) const {
        constexpr auto FRAC_K = get_frac_k<dtype_t>();
        return operand_ptr<ptr_t, elem_t, MAJOR>(aligned(shape_mn, FRAC_MN), aligned(shape_k, FRAC_K), addr);
    }
};

template <typename ptr_t, typename elem_t>
struct result_ptr {
    uintptr_t addr;
    const uint64_t shape_m, shape_n;  // GEMM output dimensions

    using dtype_t = elem_t;

    constexpr __aicore__ result_ptr(uint64_t shape_m, uint64_t shape_n, uintptr_t addr = 0) :
        addr(addr), shape_m(shape_m), shape_n(shape_n) {}

    template <typename dst_dtype_t = dtype_t>
    __aicore__ cvt_pointee_t<ptr_t, dst_dtype_t> ptr() const {
        return reinterpret_cast<cvt_pointee_t<ptr_t, dst_dtype_t>>(addr);
    }

    constexpr __aicore__ uint64_t size_per_stage() const { return shape_m * shape_n * sizeof(dtype_t); }
    constexpr __aicore__ uintptr_t offset(uint32_t n) const { return addr + n * size_per_stage(); }
    constexpr __aicore__ result_ptr<ptr_t, elem_t> operator[](uint32_t i) const {
        result_ptr<ptr_t, elem_t> r = *this;
        r.addr = offset(i);
        return r;
    }

    constexpr __aicore__ result_ptr<ptr_t, elem_t> offset_nz_m(uint32_t m) const {
        static_assert(std::is_same_v<ptr_t, __cc__ elem_t*>);
        result_ptr<ptr_t, elem_t> r = *this;
        r.addr += m * FRAC_MN * sizeof(dtype_t);
        return r;
    }

    constexpr __aicore__ result_ptr<ptr_t, elem_t> as_mad_aligned(uint64_t shape_m, uint64_t shape_n) const {
        return result_ptr<ptr_t, elem_t>(aligned(shape_m, FRAC_MN), aligned(shape_n, FRAC_MN), addr);
    }
};

template <typename T, Major M = Major::K> using l1_ptr = operand_ptr<__cbuf__ T*, T, M>;
template <typename T> using l0a_ptr = operand_ptr<__ca__ T*, T>;
template <typename T> using l0b_ptr = operand_ptr<__cb__ T*, T>;
template <typename T> using l0c_ptr = result_ptr<__cc__ T*, T>;
template <typename T> using ub_ptr = result_ptr<__ubuf__ T*, T>;

// gm_ptr<T, Major> (the GM operand handle) lives in common.h so the host
// can build one and pass it as a kernel launch argument.

} // namespace deep_gemm
