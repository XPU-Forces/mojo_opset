#pragma once

#include "kernel_operator.h"

namespace SWA_v5::Backward::VectorCompute {

#ifdef __CCE_AICORE__
namespace detail {

static constexpr AscendC::MicroAPI::CastTrait B16_TO_B32_EVEN = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
static constexpr AscendC::MicroAPI::CastTrait B16_TO_B32_ODD = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
static constexpr AscendC::MicroAPI::CastTrait B32_TO_B16_EVEN = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
static constexpr AscendC::MicroAPI::CastTrait B32_TO_B16_ODD = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_vf__ inline void row_dot_bf16(
    __ubuf__ bfloat16_t* out, __ubuf__ bfloat16_t* dout, __ubuf__ float* d,
    int32_t active_rows) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t ROW_ELEMS = 128U;
    MaskReg all_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg all_b16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    RegTensor<bfloat16_t> out_b16;
    RegTensor<bfloat16_t> dout_b16;
    RegTensor<float> out_even;
    RegTensor<float> out_odd;
    RegTensor<float> dout_even;
    RegTensor<float> dout_odd;
    RegTensor<float> product_even;
    RegTensor<float> product_odd;
    RegTensor<float> sum;
    RegTensor<float> reduced;
    UnalignRegForStore d_store;

    for (uint16_t row = 0U; row < 64U; ++row) {
        if (static_cast<int32_t>(row) < active_rows) {
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE>(
                out_b16, out, ROW_ELEMS);
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE>(
                dout_b16, dout, ROW_ELEMS);
            Cast<float, bfloat16_t, B16_TO_B32_EVEN>(out_even, out_b16, all_b16);
            Cast<float, bfloat16_t, B16_TO_B32_ODD>(out_odd, out_b16, all_b16);
            Cast<float, bfloat16_t, B16_TO_B32_EVEN>(dout_even, dout_b16, all_b16);
            Cast<float, bfloat16_t, B16_TO_B32_ODD>(dout_odd, dout_b16, all_b16);
            Mul(product_even, out_even, dout_even, all_f32);
            Mul(product_odd, out_odd, dout_odd, all_f32);
            Add(sum, product_even, product_odd, all_f32);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                reduced, sum, all_f32);
        } else {
            Duplicate(reduced, 0.0F);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(d, reduced, d_store, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(d, d_store, 0);
}

__simd_vf__ inline void row_dot_f32_bf16(
    __ubuf__ float* out, __ubuf__ bfloat16_t* dout, __ubuf__ float* d,
    int32_t active_rows) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t ROW_ELEMS = 128U;
    MaskReg all_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg all_b16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    RegTensor<float> out0;
    RegTensor<float> out1;
    RegTensor<float> out_even;
    RegTensor<float> out_odd;
    RegTensor<bfloat16_t> dout_b16;
    RegTensor<float> dout_even;
    RegTensor<float> dout_odd;
    RegTensor<float> product0;
    RegTensor<float> product1;
    RegTensor<float> reduced0;
    RegTensor<float> reduced1;
    RegTensor<float> reduced;
    UnalignRegForStore d_store;

    for (uint16_t row = 0U; row < 64U; ++row) {
        if (static_cast<int32_t>(row) < active_rows) {
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(out0, out, ROW_ELEMS / 2U);
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(out1, out, ROW_ELEMS / 2U);
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE>(
                dout_b16, dout, ROW_ELEMS);
            DeInterleave(out_even, out_odd, out0, out1);
            Cast<float, bfloat16_t, B16_TO_B32_EVEN>(dout_even, dout_b16, all_b16);
            Cast<float, bfloat16_t, B16_TO_B32_ODD>(dout_odd, dout_b16, all_b16);
            Mul(product0, out_even, dout_even, all_f32);
            Mul(product1, out_odd, dout_odd, all_f32);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                reduced0, product0, all_f32);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                reduced1, product1, all_f32);
            Add(reduced, reduced0, reduced1, all_f32);
        } else {
            Duplicate(reduced, 0.0F);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(d, reduced, d_store, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(d, d_store, 0);
}

template<bool IS_CAUSAL, bool TILE_NEEDS_MASK, bool GLOBAL_ONLY = false,
    bool MASK_LEFT = true, bool MASK_RIGHT = true, bool MASK_TAIL = true>
__simd_vf__ inline void make_p_ds_masked_tile_and_stage_nz(
    __ubuf__ float* dp, __ubuf__ float* score, __ubuf__ float* lse,
    __ubuf__ float* d, __ubuf__ bfloat16_t* p_stage,
    __ubuf__ bfloat16_t* ds_stage, int32_t q0_abs, int32_t k0,
    int32_t active_rows, int32_t kv_len, int32_t window_left,
    int32_t window_right, int32_t global_window_size, float scale,
    float grad_scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t ROW_ELEMS = 128U;
    constexpr uint32_t NZ_BLOCK_STRIDE = 65U;
    MaskReg all_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid0;
    MaskReg invalid1;
    MaskReg tail;
    MaskReg outside_local;
    MaskReg outside_global;
    RegTensor<int32_t> k_index0;
    RegTensor<int32_t> k_index1;
    RegTensor<float> score0;
    RegTensor<float> score1;
    RegTensor<float> lse_broadcast;
    RegTensor<float> p0;
    RegTensor<float> p1;
    RegTensor<float> dp0;
    RegTensor<float> dp1;
    RegTensor<float> d_broadcast;
    RegTensor<float> ds0;
    RegTensor<float> ds1;
    RegTensor<float> zero;
    RegTensor<bfloat16_t> cast0;
    RegTensor<bfloat16_t> cast1;
    RegTensor<bfloat16_t> packed;
    Duplicate(zero, 0.0F);
    if constexpr (TILE_NEEDS_MASK) {
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_index0, 0);
        Muls(k_index0, k_index0, static_cast<int32_t>(2), all_f32);
        Adds(k_index1, k_index0, k0 + 1, all_f32);
        Adds(k_index0, k_index0, k0, all_f32);
    }

    for (uint16_t row = 0U; row < 64U; ++row) {
        if (static_cast<int32_t>(row) < active_rows) {
            const int32_t q_abs = q0_abs + static_cast<int32_t>(row);
            const int32_t local_begin =
                window_left == 0 ? q_abs + 1 : q_abs - window_left + 1;
            int32_t window_end = q_abs;
            if constexpr (!IS_CAUSAL) {
                window_end += window_right;
            }

            LoadAlign<float, LoadDist::DIST_BRC_B32>(lse_broadcast, lse + row);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                score0, score1, score + static_cast<uint32_t>(row) * ROW_ELEMS);
            Muls(score0, score0, scale, all_f32);
            ExpSub(p0, score0, lse_broadcast, all_f32);
            Muls(score1, score1, scale, all_f32);
            ExpSub(p1, score1, lse_broadcast, all_f32);

            if constexpr (TILE_NEEDS_MASK) {
                if constexpr (GLOBAL_ONLY) {
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        invalid0, k_index0, window_end, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        tail, k_index0, kv_len - 1, all_f32);
                    Or(invalid0, invalid0, tail, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        outside_global, k_index0,
                        global_window_size - 1, all_f32);
                    Or(invalid0, invalid0, outside_global, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        outside_local, k_index0, local_begin - 1, all_f32);
                    Or(invalid0, invalid0, outside_local, all_f32);
                } else {
                    if constexpr (MASK_RIGHT) {
                        Compares<int32_t, AscendC::CMPMODE::GT>(
                            invalid0, k_index0, window_end, all_f32);
                    }
                    if constexpr (MASK_TAIL) {
                        if constexpr (MASK_RIGHT) {
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                tail, k_index0, kv_len - 1, all_f32);
                            Or(invalid0, invalid0, tail, all_f32);
                        } else {
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                invalid0, k_index0, kv_len - 1, all_f32);
                        }
                    }
                    if constexpr (MASK_LEFT) {
                        if (global_window_size > 0) {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                outside_local, k_index0, local_begin, all_f32);
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                outside_global, k_index0,
                                global_window_size - 1, all_f32);
                            if constexpr (MASK_RIGHT || MASK_TAIL) {
                                And(outside_local, outside_local, outside_global, all_f32);
                                Or(invalid0, invalid0, outside_local, all_f32);
                            } else {
                                And(invalid0, outside_local, outside_global, all_f32);
                            }
                        } else if constexpr (MASK_RIGHT || MASK_TAIL) {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                outside_local, k_index0, local_begin, all_f32);
                            Or(invalid0, invalid0, outside_local, all_f32);
                        } else {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                invalid0, k_index0, local_begin, all_f32);
                        }
                    }
                }
                Select(p0, zero, p0, invalid0);
            }

            if constexpr (TILE_NEEDS_MASK) {
                if constexpr (GLOBAL_ONLY) {
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        invalid1, k_index1, window_end, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        tail, k_index1, kv_len - 1, all_f32);
                    Or(invalid1, invalid1, tail, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        outside_global, k_index1,
                        global_window_size - 1, all_f32);
                    Or(invalid1, invalid1, outside_global, all_f32);
                    Compares<int32_t, AscendC::CMPMODE::GT>(
                        outside_local, k_index1, local_begin - 1, all_f32);
                    Or(invalid1, invalid1, outside_local, all_f32);
                } else {
                    if constexpr (MASK_RIGHT) {
                        Compares<int32_t, AscendC::CMPMODE::GT>(
                            invalid1, k_index1, window_end, all_f32);
                    }
                    if constexpr (MASK_TAIL) {
                        if constexpr (MASK_RIGHT) {
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                tail, k_index1, kv_len - 1, all_f32);
                            Or(invalid1, invalid1, tail, all_f32);
                        } else {
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                invalid1, k_index1, kv_len - 1, all_f32);
                        }
                    }
                    if constexpr (MASK_LEFT) {
                        if (global_window_size > 0) {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                outside_local, k_index1, local_begin, all_f32);
                            Compares<int32_t, AscendC::CMPMODE::GT>(
                                outside_global, k_index1,
                                global_window_size - 1, all_f32);
                            if constexpr (MASK_RIGHT || MASK_TAIL) {
                                And(outside_local, outside_local, outside_global, all_f32);
                                Or(invalid1, invalid1, outside_local, all_f32);
                            } else {
                                And(invalid1, outside_local, outside_global, all_f32);
                            }
                        } else if constexpr (MASK_RIGHT || MASK_TAIL) {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                outside_local, k_index1, local_begin, all_f32);
                            Or(invalid1, invalid1, outside_local, all_f32);
                        } else {
                            Compares<int32_t, AscendC::CMPMODE::LT>(
                                invalid1, k_index1, local_begin, all_f32);
                        }
                    }
                }
                Select(p1, zero, p1, invalid1);
            }

            LoadAlign<float, LoadDist::DIST_BRC_B32>(d_broadcast, d + row);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                dp0, dp1, dp + static_cast<uint32_t>(row) * ROW_ELEMS);
            Sub(ds0, dp0, d_broadcast, all_f32);
            Mul(ds0, ds0, p0, all_f32);
            Sub(ds1, dp1, d_broadcast, all_f32);
            Mul(ds1, ds1, p1, all_f32);
            Muls(ds0, ds0, grad_scale, all_f32);
            Muls(ds1, ds1, grad_scale, all_f32);
        } else {
            Duplicate(p0, 0.0F);
            Duplicate(p1, 0.0F);
            Duplicate(ds0, 0.0F);
            Duplicate(ds1, 0.0F);
        }
        Cast<bfloat16_t, float, B32_TO_B16_EVEN>(cast0, p0, all_f32);
        Cast<bfloat16_t, float, B32_TO_B16_ODD>(cast1, p1, all_f32);
        Or<uint16_t, MaskMergeMode::ZEROING>(
            reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), all_b16);
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY,
            PostLiteral::POST_MODE_UPDATE>(p_stage, packed, NZ_BLOCK_STRIDE, 1U, all_b16);

        Cast<bfloat16_t, float, B32_TO_B16_EVEN>(cast0, ds0, all_f32);
        Cast<bfloat16_t, float, B32_TO_B16_ODD>(cast1, ds1, all_f32);
        Or<uint16_t, MaskMergeMode::ZEROING>(
            reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), all_b16);
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY,
            PostLiteral::POST_MODE_UPDATE>(ds_stage, packed, NZ_BLOCK_STRIDE, 1U, all_b16);
    }
}

__simd_vf__ inline void make_p_ds_full_tile_and_stage_nz(
    __ubuf__ float* dp, __ubuf__ float* score, __ubuf__ float* lse,
    __ubuf__ float* d, __ubuf__ bfloat16_t* p_stage,
    __ubuf__ bfloat16_t* ds_stage, float scale, float grad_scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t ROW_ELEMS = 128U;
    constexpr uint32_t NZ_BLOCK_STRIDE = 65U;
    MaskReg all_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> score0;
    RegTensor<float> score1;
    RegTensor<float> lse_broadcast;
    RegTensor<float> p0;
    RegTensor<float> p1;
    RegTensor<float> dp0;
    RegTensor<float> dp1;
    RegTensor<float> d_broadcast;
    RegTensor<float> ds0;
    RegTensor<float> ds1;
    RegTensor<bfloat16_t> cast0;
    RegTensor<bfloat16_t> cast1;
    RegTensor<bfloat16_t> packed;

    for (uint16_t row = 0U; row < 64U; ++row) {
        const uint32_t row_offset = static_cast<uint32_t>(row) * ROW_ELEMS;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            score0, score1, score + row_offset);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(lse_broadcast, lse + row);
        Muls(score0, score0, scale, all_f32);
        Muls(score1, score1, scale, all_f32);
        ExpSub(p0, score0, lse_broadcast, all_f32);
        ExpSub(p1, score1, lse_broadcast, all_f32);

        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dp0, dp1, dp + row_offset);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(d_broadcast, d + row);
        Sub(ds0, dp0, d_broadcast, all_f32);
        Sub(ds1, dp1, d_broadcast, all_f32);
        Mul(ds0, ds0, p0, all_f32);
        Mul(ds1, ds1, p1, all_f32);
        Muls(ds0, ds0, grad_scale, all_f32);
        Muls(ds1, ds1, grad_scale, all_f32);

        Cast<bfloat16_t, float, B32_TO_B16_EVEN>(cast0, p0, all_f32);
        Cast<bfloat16_t, float, B32_TO_B16_ODD>(cast1, p1, all_f32);
        Or<uint16_t, MaskMergeMode::ZEROING>(
            reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), all_b16);
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY,
            PostLiteral::POST_MODE_UPDATE>(p_stage, packed, NZ_BLOCK_STRIDE, 1U, all_b16);

        Cast<bfloat16_t, float, B32_TO_B16_EVEN>(cast0, ds0, all_f32);
        Cast<bfloat16_t, float, B32_TO_B16_ODD>(cast1, ds1, all_f32);
        Or<uint16_t, MaskMergeMode::ZEROING>(
            reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), all_b16);
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY,
            PostLiteral::POST_MODE_UPDATE>(ds_stage, packed, NZ_BLOCK_STRIDE, 1U, all_b16);
    }
}

}  // namespace detail
#endif

__aicore__ inline void row_dot_bf16(
    AscendC::LocalTensor<bfloat16_t> out, AscendC::LocalTensor<bfloat16_t> dout,
    AscendC::LocalTensor<float> d, uint32_t active_rows) {
#ifdef __CCE_AICORE__
    detail::row_dot_bf16(
        reinterpret_cast<__ubuf__ bfloat16_t*>(out.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(dout.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(d.GetPhyAddr()), static_cast<int32_t>(active_rows));
#endif
}

__aicore__ inline void row_dot_f32_bf16(
    AscendC::LocalTensor<float> out, AscendC::LocalTensor<bfloat16_t> dout,
    AscendC::LocalTensor<float> d, uint32_t active_rows) {
#ifdef __CCE_AICORE__
    detail::row_dot_f32_bf16(
        reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(dout.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(d.GetPhyAddr()), static_cast<int32_t>(active_rows));
#endif
}

template<bool IS_CAUSAL, bool TILE_NEEDS_MASK, bool GLOBAL_ONLY = false,
    bool MASK_LEFT = true, bool MASK_RIGHT = true, bool MASK_TAIL = true>
__aicore__ inline void make_p_ds_masked_tile_and_stage_nz(
    AscendC::LocalTensor<float> dp, AscendC::LocalTensor<float> score,
    AscendC::LocalTensor<float> lse, AscendC::LocalTensor<float> d,
    AscendC::LocalTensor<bfloat16_t> p_stage,
    AscendC::LocalTensor<bfloat16_t> ds_stage, uint32_t q0_abs,
    uint32_t k0, uint32_t active_rows, uint32_t kv_len,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size,
    float scale, float grad_scale) {
#ifdef __CCE_AICORE__
    detail::make_p_ds_masked_tile_and_stage_nz<IS_CAUSAL, TILE_NEEDS_MASK, GLOBAL_ONLY,
        MASK_LEFT, MASK_RIGHT, MASK_TAIL>(
        reinterpret_cast<__ubuf__ float*>(dp.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(score.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(lse.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(d.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(p_stage.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(ds_stage.GetPhyAddr()),
        static_cast<int32_t>(q0_abs), static_cast<int32_t>(k0),
        static_cast<int32_t>(active_rows), static_cast<int32_t>(kv_len),
        static_cast<int32_t>(window_left), static_cast<int32_t>(window_right),
        static_cast<int32_t>(global_window_size), scale, grad_scale);
#endif
}

__aicore__ inline void make_p_ds_full_tile_and_stage_nz(
    AscendC::LocalTensor<float> dp, AscendC::LocalTensor<float> score,
    AscendC::LocalTensor<float> lse, AscendC::LocalTensor<float> d,
    AscendC::LocalTensor<bfloat16_t> p_stage,
    AscendC::LocalTensor<bfloat16_t> ds_stage, float scale,
    float grad_scale) {
#ifdef __CCE_AICORE__
    detail::make_p_ds_full_tile_and_stage_nz(
        reinterpret_cast<__ubuf__ float*>(dp.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(score.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(lse.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(d.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(p_stage.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(ds_stage.GetPhyAddr()), scale,
        grad_scale);
#endif
}

}  // namespace SWA_v5::Backward::VectorCompute
