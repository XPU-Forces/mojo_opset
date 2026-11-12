/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: include/deep_gemm/epilogue/operators.hpp
Original SHA256: 2184df6956c9c4f09c6e002f6aeca15c8ef8d1a5d9863a80b1c3a7cbb75bef09
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

#include <kernel/ascend.h>

namespace deep_gemm::math {

template <typename quant_dtype_t>
__simd_callee__ inline void get_ue8m0_sf_exp(vector_uint16_t& sf_exp) {
    constexpr __aicore__ bool kIsFP8 = std::is_same_v<quant_dtype_t, float8_e4m3_t>;
    static_assert(kIsFP8 || is_fp4<quant_dtype_t>(), "quantized type must be E4M3 or E2M1");
    constexpr uint16_t kMantissaBits = 7;
    constexpr uint16_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint16_t kQuantMaxMantissa = kIsFP8 ? 0x60 : 0x40;
    constexpr uint16_t kQuantMaxExponent = kIsFP8 ? 8 : 2;
    constexpr uint16_t kMinSFExponent = kIsFP8 ? 105 : 1;
    constexpr uint16_t kSFOffset = (kQuantMaxExponent << kMantissaBits) - (kMantissaMask - kQuantMaxMantissa);

    const auto full_mask = asc_create_mask_b16(PAT_ALL);
    // FP8: max((x + 31) >> 7, 113) - 8 = (max(x, 14433) - 993) >> 7, where x is the absolute BF16 bits.
    asc_max_scalar(sf_exp, sf_exp, (kMinSFExponent << kMantissaBits) + kSFOffset, full_mask);
    // Clamp first so uint16_t wraparound gives a nonnegative difference.
    asc_add_scalar(sf_exp, sf_exp, static_cast<uint16_t>(-kSFOffset), full_mask);
    asc_shiftright_scalar(sf_exp, sf_exp, kMantissaBits, full_mask);
}

__simd_callee__ inline void get_ue8m0_sf_inv(vector_bfloat16_t& sf_inv, vector_uint16_t sf_exp) {
    constexpr uint16_t kMantissaBits = 7;
    constexpr uint16_t kReciprocalExponentBias = 254;
    const auto mask = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t inverse_bits;
    asc_duplicate_scalar(inverse_bits, kReciprocalExponentBias, mask);
    asc_sub(inverse_bits, inverse_bits, sf_exp, mask);
    asc_shiftleft_scalar(inverse_bits, inverse_bits, kMantissaBits, mask);
    sf_inv = reinterpret_cast<vector_bfloat16_t&>(inverse_bits);
}

__simd_callee__ inline void cast_bf16(vector_uint8_t& output, vector_bfloat16_t values) {
    const auto value_mask = asc_create_mask_b16(PAT_ALL);

    vector_bfloat16_t zero;
    vector_float values_f32_lo, values_f32_hi;
    asc_duplicate_scalar(zero, static_cast<bfloat16_t>(0), value_mask);
    asc_intlv(reinterpret_cast<vector_bfloat16_t&>(values_f32_lo),
              reinterpret_cast<vector_bfloat16_t&>(values_f32_hi), zero, values);

    vector_fp8_e4m3fn_t values_fp8_lo, values_fp8_hi;
    vector_uint16_t values_fp8_packed, unused;
    asc_float2e4m3_rn_sat(values_fp8_lo, values_f32_lo, asc_create_mask_b32(PAT_ALL), ASC_DISPERSE_FIRST_QUARTER);
    asc_float2e4m3_rn_sat(values_fp8_hi, values_f32_hi, asc_create_mask_b32(PAT_ALL), ASC_DISPERSE_FIRST_QUARTER);
    asc_deintlv(values_fp8_packed, unused, reinterpret_cast<vector_uint16_t&>(values_fp8_lo),
                reinterpret_cast<vector_uint16_t&>(values_fp8_hi));
    asc_pack_to_low(output, values_fp8_packed);
}

} // namespace deep_gemm::math

namespace deep_gemm::epilogue {

template <typename dtype_t, uint32_t kBlockM, uint32_t kBlockN, uint32_t kMadM, uint32_t kMadN,
          uint32_t kStagesPerAIV, bool kAccumulation, asc_store_l2_cache_mode kStore>
struct Params {
    using cd_dtype_t = dtype_t;
    static constexpr uint32_t BLOCK_M = kBlockM, BLOCK_N = kBlockN;
    static constexpr uint32_t MAD_M = kMadM, MAD_N = kMadN;
    static constexpr uint32_t kNumStagesPerAIV = kStagesPerAIV;
    static constexpr bool kWithAccumulation = kAccumulation;
    static constexpr asc_store_l2_cache_mode kL2CacheMode = kStore;
};

template <typename P>
concept ParamsType = requires {
    typename P::cd_dtype_t;
    P::BLOCK_M;
    P::BLOCK_N;
    P::MAD_M;
    P::MAD_N;
    P::kNumStagesPerAIV;
    P::kWithAccumulation;
    P::kL2CacheMode;
};

} // namespace deep_gemm::epilogue

namespace deep_gemm::epilogue::operators {

struct MadBlock {
    uint32_t stage_idx, drain_stage_idx;
    uint32_t m_idx, n_idx, n_mad;
    uint32_t num_rows, num_cols, batch_idx;
};

template <ParamsType P>
struct Operator {
    using cd_dtype_t = typename P::cd_dtype_t;

    // The epilogue operator MUST define `ub_dtype_t` and `kPipe`
    // Which type of data the epilogue operator want from MAD output, (candidate `bfloat16_t` or `float`).
    using ub_dtype_t = cd_dtype_t;
    // The epilogue acquires the MAD tile on this pipe.
    static constexpr pipe_t kPipe = PIPE_MTE3;

    uintptr_t ub_base;

    constexpr __aicore__ explicit Operator(uintptr_t ub_base) : ub_base(ub_base) {}

    constexpr __aicore__ bool is_supported() const {
        return std::is_same_v<cd_dtype_t, bfloat16_t> || std::is_same_v<cd_dtype_t, float>;
    }

    constexpr __aicore__ uintptr_t ub_end() const { return ub_base; }

    __aicore__ void initialize_aiv_drain(Args args, uint32_t stage_idx, uint32_t n_idx, uint32_t num_cols) const {}

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output, Args args, const MadBlock& block) const {}

    __aicore__ void postprocess_mad_block(Args args, const MadBlock& block) const {}
};

template <ParamsType P>
struct Identity : Operator<P> {
    using Operator<P>::Operator;
};

template <ParamsType P>
struct ScaleByAlpha : Operator<P> {
    using Operator<P>::Operator;
    using cd_dtype_t = typename P::cd_dtype_t;
    using ub_dtype_t = float;
    static constexpr pipe_t kPipe = PIPE_V;

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output, Args args, const MadBlock& block) const {
        apply_values<P::MAD_M, P::MAD_N>(values, output, args.alpha);
    }

private:
    template <uint32_t kNumRows, uint32_t kNumCols>
    static __simd_vf__ void apply_values(__ubuf__ float* values, __ubuf__ cd_dtype_t* output, float alpha) {
        constexpr uint32_t kNumFP32PerVector = asc_get_vf_len() / sizeof(float);
        static_assert(kNumRows * kNumCols % kNumFP32PerVector == 0);

        const auto mask = asc_create_mask_b32(PAT_ALL);
        for (uint32_t value_idx = 0; value_idx < kNumRows * kNumCols; value_idx += kNumFP32PerVector) {
            vector_float value;
            asc_loadalign(value, values + value_idx);
            asc_mul_scalar(value, value, alpha, mask);
            if constexpr (std::is_same_v<cd_dtype_t, float>) {
                asc_storealign(output + value_idx, value, mask);
            } else {
                vector_bfloat16_t converted;
                vector_uint16_t packed;
                asc_float2bfloat16_rn(converted, value, mask, ASC_POSITION_EVEN);
                asc_pack_to_low(packed, reinterpret_cast<vector_uint32_t&>(converted));
                asc_storealign(output + value_idx, reinterpret_cast<vector_bfloat16_t&>(packed),
                               asc_create_mask_b16(PAT_VL64));
            }
        }
    }
};

template <ParamsType P>
struct AddBias : Operator<P> {
    using Operator<P>::Operator;
    using cd_dtype_t = typename P::cd_dtype_t;
    using ub_dtype_t = float;
    static constexpr pipe_t kPipe = PIPE_V;

    ub_ptr<cd_dtype_t> bias_ub{1, P::BLOCK_N, this->ub_base};

    constexpr __aicore__ uintptr_t ub_end() const { return bias_ub.offset(P::kNumStagesPerAIV); }

    constexpr __aicore__ bool is_supported() const {
        return Operator<P>::is_supported() && !P::kWithAccumulation;
    }

    __aicore__ void initialize_aiv_drain(Args args, uint32_t stage_idx, uint32_t n_idx, uint32_t num_cols) const {
        const auto load_event = static_cast<event_t>(stage_idx % kNumPipeEvents);
        asc_sync_notify(PIPE_V, PIPE_MTE2, load_event);
        asc_sync_wait(PIPE_V, PIPE_MTE2, load_event);
        gm_ptr<cd_dtype_t> bias{0, args.bias};
        copy_gm_to_ub(bias_ub[stage_idx], bias.index(0, n_idx), 1, num_cols);
        asc_sync_notify(PIPE_MTE2, PIPE_V, load_event);
        asc_sync_wait(PIPE_MTE2, PIPE_V, load_event);
    }

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output, Args args, const MadBlock& block) const {
        apply_values<P::MAD_M, P::MAD_N>(values, output,
                                         bias_ub[block.drain_stage_idx].ptr() + block.n_mad);
    }

private:
    template <uint32_t kNumRows, uint32_t kNumCols>
    static __simd_vf__ void apply_values(__ubuf__ float* values, __ubuf__ cd_dtype_t* output, __ubuf__ cd_dtype_t* bias) {
        constexpr uint32_t kNumFP32PerVector = asc_get_vf_len() / sizeof(float);
        static_assert(kNumCols % kNumFP32PerVector == 0);
        auto packed_output = reinterpret_cast<__ubuf__ uint32_t*>(output);
        const auto mask = asc_create_mask_b32(PAT_ALL);
        for (uint32_t row_idx = 0; row_idx < kNumRows; ++row_idx) {
            auto bias_ptr = bias;
            for (uint32_t col_idx = 0; col_idx < kNumCols; col_idx += kNumFP32PerVector) {
                vector_float value;
                asc_loadalign_postupdate(value, values, kNumFP32PerVector);
                vector_float bias_value;
                if constexpr (std::is_same_v<cd_dtype_t, float>) {
                    asc_loadalign_postupdate(bias_value, bias_ptr, kNumFP32PerVector);
                } else {
                    vector_bfloat16_t bias_bf16;
                    asc_loadalign_unpack_postupdate(bias_bf16, bias_ptr, kNumFP32PerVector);
                    asc_bfloat162float(bias_value, bias_bf16, mask, ASC_POSITION_EVEN);
                }
                asc_add(value, value, bias_value, mask);
                if constexpr (std::is_same_v<cd_dtype_t, float>) {
                    asc_storealign_postupdate(output, value, kNumFP32PerVector, mask);
                } else {
                    vector_bfloat16_t converted;
                    asc_float2bfloat16_rn(converted, value, mask, ASC_POSITION_EVEN);
                    asc_storealign_pack_postupdate(packed_output, reinterpret_cast<vector_uint32_t&>(converted),
                                                   kNumFP32PerVector * sizeof(cd_dtype_t) / sizeof(uint32_t), mask);
                }
            }
        }
    }
};

template <ParamsType P>
struct StochasticRoundToBF16 : Operator<P> {
    using Operator<P>::Operator;
    using cd_dtype_t = typename P::cd_dtype_t;
    using ub_dtype_t = float;
    static constexpr pipe_t kPipe = PIPE_V;

    constexpr __aicore__ bool is_supported() const {
        return std::is_same_v<cd_dtype_t, bfloat16_t> && !P::kWithAccumulation;
    }

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output, Args args, const MadBlock& block) const {
        apply_values<P::MAD_N>(values, block.n_idx, block.num_rows, block.num_cols);
    }

private:
    template <uint32_t kNumCols>
    static __simd_vf__ void apply_values(__ubuf__ float* values, uint32_t n_idx, uint32_t num_rows, uint32_t num_cols) {
        constexpr uint32_t kNumFP32PerVector = asc_get_vf_len() / sizeof(float);
        static_assert(kNumCols % 4 == 0);
        const auto mask = asc_create_mask_b32(PAT_ALL);
        const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
        vector_uint32_t lanes, nan_bits;
        asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
        asc_duplicate_scalar(nan_bits, 0x7fc00000u);

        vector_uint32_t cols = lanes, col_mask;
        asc_duplicate_scalar(col_mask, kNumCols / 4 - 1);
        vector_uint32_t coefficient_1, coefficient_3;
        for (uint32_t value_idx = 0; value_idx < num_rows * kNumCols; value_idx += 4 * kNumFP32PerVector) {
            vector_uint32_t value_0, value_1, value_2, value_3, h, temp;
            vector_uint32_t partial_2, even_0, even_1, odd_0, odd_1;
            vector_bool valid;
            if constexpr ((kNumCols & (kNumCols - 1)) == 0) {
                asc_add_scalar(cols, lanes, (value_idx / 4) & (kNumCols / 4 - 1), mask);
                if constexpr (kNumCols < 4 * kNumFP32PerVector)
                    asc_and(cols, cols, col_mask, mask);
            } else {
                #pragma unroll
                for (uint32_t wrap_idx = 0; wrap_idx < ceil_div(4 * kNumFP32PerVector - 1, kNumCols); ++wrap_idx) {
                    asc_ge_scalar(valid, cols, kNumCols / 4, mask);
                    asc_add_scalar(temp, cols, 0u - kNumCols / 4, mask);
                    asc_select(cols, temp, cols, valid);
                }
            }

            // Each lane receives the four members of one quartet.
            asc_loadalign_deintlv(even_0, odd_0, reinterpret_cast<__ubuf__ uint32_t*>(values) + value_idx);
            asc_loadalign_deintlv(even_1, odd_1, reinterpret_cast<__ubuf__ uint32_t*>(values) + value_idx + 2 * kNumFP32PerVector);
            asc_deintlv(value_0, value_2, even_0, even_1);
            asc_deintlv(value_1, value_3, odd_0, odd_1);
            asc_add_scalar(temp, cols, n_idx / 4, mask);
            asc_lt_scalar(valid, cols, (num_cols + 3) / 4, mask);
            asc_mul_scalar(h, value_0, 0x5671d42bu, valid);
            asc_lt_scalar(valid, cols, (num_cols + 1) / 4, mask);
            asc_mul_scalar(partial_2, value_2, 0xace1b8a5u, valid);
            asc_lt_scalar(valid, cols, (num_cols + 2) / 4, mask);
            asc_duplicate_scalar(coefficient_1, 0x9995e499u, valid);
            asc_lt_scalar(valid, cols, num_cols / 4, mask);
            asc_duplicate_scalar(coefficient_3, 0xe153538du, valid);
            asc_mula(h, value_1, coefficient_1, mask);
            asc_mula(partial_2, value_3, coefficient_3, mask);
            asc_add(h, h, partial_2, mask);
            asc_add(h, h, temp, mask);
            // Canonicalize NaNs after seeding the hash with the original bits.
            asc_ne(valid, reinterpret_cast<vector_float&>(value_0), reinterpret_cast<vector_float&>(value_0), mask);
            asc_select(value_0, nan_bits, value_0, valid);
            asc_ne(valid, reinterpret_cast<vector_float&>(value_1), reinterpret_cast<vector_float&>(value_1), mask);
            asc_select(value_1, nan_bits, value_1, valid);
            asc_ne(valid, reinterpret_cast<vector_float&>(value_2), reinterpret_cast<vector_float&>(value_2), mask);
            asc_select(value_2, nan_bits, value_2, valid);
            asc_ne(valid, reinterpret_cast<vector_float&>(value_3), reinterpret_cast<vector_float&>(value_3), mask);
            asc_select(value_3, nan_bits, value_3, valid);
            // Pack BF16 high halves and complemented low halves.
            vector_uint16_t high_0, low_0, high_1, low_1;
            asc_shiftright_scalar(temp, value_0, 16, mask);
            asc_select(high_0, reinterpret_cast<vector_uint16_t&>(temp), reinterpret_cast<vector_uint16_t&>(value_1), mask);
            asc_shiftleft_scalar(temp, value_1, 16, mask);
            asc_select(low_0, reinterpret_cast<vector_uint16_t&>(value_0), reinterpret_cast<vector_uint16_t&>(temp), mask);
            asc_not(low_0, low_0, mask_b16);
            asc_shiftright_scalar(temp, value_2, 16, mask);
            asc_select(high_1, reinterpret_cast<vector_uint16_t&>(temp), reinterpret_cast<vector_uint16_t&>(value_3), mask);
            asc_shiftleft_scalar(temp, value_3, 16, mask);
            asc_select(low_1, reinterpret_cast<vector_uint16_t&>(value_2), reinterpret_cast<vector_uint16_t&>(temp), mask);
            asc_not(low_1, low_1, mask_b16);
            asc_shiftright_scalar(temp, h, 23, mask);
            asc_xor(h, h, temp, mask);
            asc_mul_scalar(h, h, 0x7feb352du, mask);
            asc_shiftright_scalar(temp, h, 16, mask);
            asc_xor(h, h, temp, mask);
            vector_uint32_t hash_0, hash_1;
            asc_mul_scalar(hash_0, h, 0x846ca68bu, mask);
            asc_mul_scalar(hash_1, h, 0xd35a2d97u, mask);
            asc_shiftright_scalar(temp, hash_0, 11, mask);
            asc_xor(hash_0, hash_0, temp, mask);
            asc_shiftright_scalar(temp, hash_1, 11, mask);
            asc_xor(hash_1, hash_1, temp, mask);

            // Round pairs with 16-bit comparisons.
            asc_gt(valid, reinterpret_cast<vector_uint16_t&>(hash_0), low_0, mask_b16);
            asc_add_scalar(reinterpret_cast<vector_uint16_t&>(hash_0), high_0, uint16_t(1), valid);
            asc_select(high_0, reinterpret_cast<vector_uint16_t&>(hash_0), high_0, valid);
            asc_gt(valid, reinterpret_cast<vector_uint16_t&>(hash_1), low_1, mask_b16);
            asc_add_scalar(reinterpret_cast<vector_uint16_t&>(hash_1), high_1, uint16_t(1), valid);
            asc_select(high_1, reinterpret_cast<vector_uint16_t&>(hash_1), high_1, valid);
            asc_storealign_intlv(reinterpret_cast<__ubuf__ uint32_t*>(values) + value_idx / 2,
                                 reinterpret_cast<vector_uint32_t&>(high_0), reinterpret_cast<vector_uint32_t&>(high_1));
            if constexpr ((kNumCols & (kNumCols - 1)) != 0)
                asc_add_scalar(cols, cols, kNumFP32PerVector, mask);
        }
    }
};

template <ParamsType P>
struct QuantizeToFP8 : Operator<P> {
    using Operator<P>::Operator;
    using cd_dtype_t = typename P::cd_dtype_t;
    using ub_dtype_t = bfloat16_t;
    static constexpr pipe_t kPipe = PIPE_V;
    static constexpr uint32_t kSFPackN = MX_SF_DIVISOR;

    ub_ptr<uint16_t> sf_ub{P::MAD_N / kSFPackN, P::MAD_M, this->ub_base};

    constexpr __aicore__ uintptr_t ub_end() const { return sf_ub.offset(P::kNumStagesPerAIV); }

    constexpr __aicore__ bool is_supported() const {
        return std::is_same_v<cd_dtype_t, float8_e4m3_t> && !P::kWithAccumulation && P::MAD_N % (2 * kSFPackN) == 0;
    }

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output, Args args, const MadBlock& block) const {
        apply_values<P::MAD_M, P::MAD_N>(values, output, sf_ub[block.stage_idx].ptr());
    }

    __aicore__ void postprocess_mad_block(Args args, const MadBlock& block) const {
        auto sfd = reinterpret_cast<__gm__ uint16_t*>(args.sfd);
        const uint32_t sf_pair_idx = (block.batch_idx * args.shape_n + block.n_idx) / kSFPackN;
        copy_ub_to_gm(sfd + sf_pair_idx * args.sfd_stride + block.m_idx, sf_ub[block.stage_idx],
                      block.num_cols / kSFPackN, block.num_rows, args.sfd_stride * sizeof(uint16_t), P::kL2CacheMode);
    }

private:
    static __simd_callee__ void prepare_vector(__ubuf__ bfloat16_t* values, vector_uint16_t& sf_packed) {
        const auto mask_b16 = asc_create_mask_b16(PAT_VL64);
        const auto sf_lo_mask = asc_create_mask_b16(PAT_VL32);
        vector_bool sf_hi_mask;
        asc_not(sf_hi_mask, sf_lo_mask, mask_b16);

        vector_bfloat16_t values_bf16;
        asc_loadalign(values_bf16, values);
        vector_uint16_t sign_mask, abs_bits, amax_lo, amax_hi, unused, sf_exp;
        asc_duplicate_scalar(sign_mask, static_cast<uint16_t>((1u << 15) - 1), mask_b16);
        asc_and(abs_bits, reinterpret_cast<vector_uint16_t&>(values_bf16), sign_mask, mask_b16);
        asc_reduce_max(amax_lo, abs_bits, sf_lo_mask);
        asc_reduce_max(amax_hi, abs_bits, sf_hi_mask);
        asc_intlv(sf_exp, unused, amax_lo, amax_hi);
        math::get_ue8m0_sf_exp<float8_e4m3_t>(sf_exp);
        vector_uint8_t sf_bytes;
        asc_pack_to_low(sf_bytes, sf_exp);
        asc_duplicate(sf_packed, reinterpret_cast<vector_uint16_t&>(sf_bytes), asc_create_mask_b16(PAT_ALL));
    }

    template <uint32_t kSFStride>
    static __simd_callee__ void cast_vector(__ubuf__ bfloat16_t* values, __ubuf__ float8_e4m3_t* output, __ubuf__ uint16_t* output_sf, vector_bool sf_lo_mask) {
        const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
        vector_uint16_t sf_packed, sf_packed_next;
        asc_loadalign_brc(sf_packed, output_sf);
        asc_loadalign_brc(sf_packed_next, output_sf + kSFStride);
        asc_select(sf_packed, sf_packed, sf_packed_next, asc_create_mask_b16(PAT_VL64));
        vector_uint16_t sf_exp_lo, sf_exp_hi, sf_exp_mask;
        asc_shiftright_scalar(sf_exp_hi, sf_packed, 8, mask_b16);
        asc_duplicate_scalar(sf_exp_mask, static_cast<uint16_t>(0xff), mask_b16);
        asc_and(sf_exp_lo, sf_packed, sf_exp_mask, mask_b16);
        vector_bfloat16_t values_bf16;
        asc_loadalign(values_bf16, values);
        vector_bfloat16_t sf_inv;
        vector_uint16_t sf_exp;
        asc_select(sf_exp, sf_exp_lo, sf_exp_hi, sf_lo_mask);
        math::get_ue8m0_sf_inv(sf_inv, sf_exp);
        asc_mul(values_bf16, values_bf16, sf_inv, mask_b16);

        vector_uint8_t values_fp8;
        math::cast_bf16(values_fp8, values_bf16);
        asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output), values_fp8, asc_create_mask_b8(PAT_VL128));
    }

    template <uint32_t kNumRows, uint32_t kNumCols>
    static __simd_vf__ void apply_values(__ubuf__ bfloat16_t* values, __ubuf__ float8_e4m3_t* output, __ubuf__ uint16_t* output_sf) {
        static_assert(kNumCols % (2 * kSFPackN) == 0);
        constexpr uint32_t kNumRowsPerVector = asc_get_vf_len() / sizeof(uint16_t);
        constexpr uint32_t kNumSFGroups = kNumCols / kSFPackN;
        const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

        for (uint32_t m_base = 0; m_base < kNumRows; m_base += kNumRowsPerVector) {
            vector_uint16_t row_indices;
            asc_arange(reinterpret_cast<vector_int16_t&>(row_indices), 0);
            const uint32_t num_rows = min(kNumRows - m_base, kNumRowsPerVector);
            for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                vector_bool row_mask;
                asc_eq_scalar(row_mask, row_indices, row_idx, mask_b16);

                for (uint32_t sf_group_idx = 0; sf_group_idx < kNumSFGroups; ++sf_group_idx) {
                    vector_uint16_t sf_packed;
                    const uint32_t value_idx = (m_base + row_idx) * kNumCols + sf_group_idx * kSFPackN;
                    prepare_vector(values + value_idx, sf_packed);
                    asc_storealign(output_sf + sf_group_idx * kNumRows + m_base, sf_packed, row_mask);
                }
            }
        }
        vector_uint16_t value_indices, sf_group_bit;
        vector_bool sf_lo_mask;
        asc_arange(reinterpret_cast<vector_int16_t&>(value_indices), 0);
        asc_duplicate_scalar(sf_group_bit, static_cast<uint16_t>(kSFPackN / 2), mask_b16);
        asc_and(value_indices, value_indices, sf_group_bit, mask_b16);
        asc_eq_scalar(sf_lo_mask, value_indices, static_cast<uint16_t>(0), mask_b16);
        asc_mem_bar(VST_VLD);
        for (uint32_t row_idx = 0; row_idx < kNumRows; ++row_idx) {
            for (uint32_t sf_group_idx = 0; sf_group_idx < kNumSFGroups; sf_group_idx += 2) {
                const uint32_t value_idx = row_idx * kNumCols + sf_group_idx * kSFPackN;
                cast_vector<kNumRows>(values + value_idx, output + value_idx,
                                      output_sf + sf_group_idx * kNumRows + row_idx, sf_lo_mask);
            }
        }
    }
};

} // namespace deep_gemm::epilogue::operators
