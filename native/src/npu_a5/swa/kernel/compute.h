#pragma once

#include "primitive.h"
#include "tensor.h"

namespace SWA_v5::Compute {

namespace online_softmax {

template<class PStageT>
struct Buffers {
    __ubuf__ PStageT *p_stage;
    __ubuf__ float *qk;
    __ubuf__ float *sum;
    __ubuf__ float *max;
    __ubuf__ float *expmax;
    __ubuf__ uint8_t *mask;
    __ubuf__ uint8_t *common;
};

template<class PStageT, uint32_t QK_SLOTS, uint32_t P_STAGE_SLOTS,
    uint32_t SOFTMAX_STATE_SLOTS, uint32_t EXPMAX_SLOTS>
struct BufferCache {
    __ubuf__ float *qk[QK_SLOTS];
    __ubuf__ PStageT *p_stage[P_STAGE_SLOTS];
    __ubuf__ float *sum[SOFTMAX_STATE_SLOTS];
    __ubuf__ float *max[SOFTMAX_STATE_SLOTS];
    __ubuf__ float *expmax[EXPMAX_SLOTS];
    __ubuf__ uint8_t *mask;
    __ubuf__ uint8_t *common;

    __aicore__ inline Buffers<PStageT> get(
        uint32_t qk_slot, uint32_t p_stage_slot, uint32_t summax_slot, uint32_t exp_slot) const {
        return Buffers<PStageT>{
            p_stage[p_stage_slot], qk[qk_slot], sum[summax_slot], max[summax_slot], expmax[exp_slot], mask, common};
    }
};


#ifdef __CCE_AICORE__
namespace detail {
static constexpr AscendC::MicroAPI::CastTrait MICRO_CAST_TRAIT_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
static constexpr AscendC::MicroAPI::CastTrait MICRO_CAST_TRAIT_ONE = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
}  // namespace detail
#endif

namespace VFimpl {
template<class PStageT, bool FIRST_TILE, uint16_t LOCAL_ROWS = 64>
__simd_vf__ inline void compute_prefix_tile(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    __ubuf__ uint8_t *common_ub, int32_t valid_cols, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 3;
    constexpr uint32_t REPEAT_STRIDE = 1;
    constexpr float NEG_INF = -3.4028234663852886e38F;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid;

    RegTensor<float> min_vreg;
    RegTensor<float> r0;
    RegTensor<float> r1;
    RegTensor<float> sel0_vreg;
    RegTensor<float> tile_max_vreg;
    RegTensor<float> old_max_vreg;
    RegTensor<float> new_max_vreg;
    RegTensor<float> max_brc_vreg;
    RegTensor<float> exp0_vreg;
    RegTensor<float> exp1_vreg;
    RegTensor<float> tile_sum_vreg;
    RegTensor<float> old_sum_vreg;
    RegTensor<float> exp_old_vreg;
    RegTensor<PStageT> exp0_stage_vreg;
    RegTensor<PStageT> exp1_stage_vreg;
    RegTensor<PStageT> exp_stage_vreg;
    RegTensor<int32_t> lane_reg;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;
    Duplicate(min_vreg, NEG_INF);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(lane_reg, 0);
    Compares<int32_t, AscendC::CMPMODE::GT>(
        invalid, lane_reg, valid_cols - 1, preg_all);

    if constexpr (FIRST_TILE) {
        __ubuf__ PStageT *p_ptr = p_stage_ub;
        for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
            LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
            Muls(r0, r0, scale, preg_all);
            Select(sel0_vreg, min_vreg, r0, invalid);
            Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(
                tile_max_vreg, sel0_vreg, preg_all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
                max_write_ptr, tile_max_vreg, max_store, 1);
            Duplicate(max_brc_vreg, tile_max_vreg, preg_all);
            DeInterleave(r0, r1, sel0_vreg, min_vreg);
            ExpSub(exp0_vreg, r0, max_brc_vreg, preg_all);
            ExpSub(exp1_vreg, r1, max_brc_vreg, preg_all);
            Add(tile_sum_vreg, exp0_vreg, exp1_vreg, preg_all);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                tile_sum_vreg, tile_sum_vreg, preg_all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
                sum_write_ptr, tile_sum_vreg, sum_store, 1);
            Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(
                exp0_stage_vreg, exp0_vreg, preg_all);
            Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(
                exp1_stage_vreg, exp1_vreg, preg_all);
            Or<uint16_t, MaskMergeMode::ZEROING>(
                reinterpret_cast<RegTensor<uint16_t>&>(exp_stage_vreg),
                reinterpret_cast<RegTensor<uint16_t>&>(exp0_stage_vreg),
                reinterpret_cast<RegTensor<uint16_t>&>(exp1_stage_vreg), preg_all_b16);
            StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p_ptr, exp_stage_vreg, BLOCK_STRIDE, REPEAT_STRIDE, preg_all_b16);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, max_store, 0);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, sum_store, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        return;
    }

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Select(sel0_vreg, min_vreg, r0, invalid);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            qk_ub + static_cast<uint32_t>(row) * TILE_K, sel0_vreg, preg_all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(
            tile_max_vreg, sel0_vreg, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
            max_write_ptr, tile_max_vreg, max_store, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, max_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_max_vreg, tmp_max_ub);
        LoadAlign(old_max_vreg, max_ub);
        Max(new_max_vreg, tile_max_vreg, old_max_vreg, preg_all);
        ExpSub(exp_old_vreg, old_max_vreg, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(max_ub, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __ubuf__ PStageT *p_stage_ptr = p_stage_ub;
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(max_brc_vreg, max_ub + row);
        LoadAlign(sel0_vreg, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        DeInterleave(r0, r1, sel0_vreg, min_vreg);
        ExpSub(exp0_vreg, r0, max_brc_vreg, preg_all);
        ExpSub(exp1_vreg, r1, max_brc_vreg, preg_all);
        Add(tile_sum_vreg, exp0_vreg, exp1_vreg, preg_all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
            tile_sum_vreg, tile_sum_vreg, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
            sum_write_ptr, tile_sum_vreg, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(
            exp0_stage_vreg, exp0_vreg, preg_all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(
            exp1_stage_vreg, exp1_vreg, preg_all);
        Or<uint16_t, MaskMergeMode::ZEROING>(
            reinterpret_cast<RegTensor<uint16_t>&>(exp_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp0_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp1_stage_vreg), preg_all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_stage_ptr, exp_stage_vreg, BLOCK_STRIDE, REPEAT_STRIDE, preg_all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, sum_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_sum_vreg, tmp_sum_ub);
        LoadAlign(old_sum_vreg, sum_ub);
        LoadAlign(exp_old_vreg, expmax_ub);
        Mula(tile_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    }
}

template<class PStageT, bool FIRST_TILE, bool IS_CAUSAL = true, uint16_t LOCAL_ROWS = 64,
    bool CHECK_LOWER = true, bool CHECK_UPPER = true,
    bool FILL_FIRST_HALF = false, bool FILL_SECOND_HALF = false>
__simd_vf__ inline void compute_full_tile(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    __ubuf__ uint8_t *common_ub, int32_t q0_local, int32_t k0_local,
    int32_t window_left, int32_t window_right, int32_t global_window_size, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 3;
    constexpr uint32_t REPEAT_STRIDE = 1;
    constexpr float NEG_INF = -3.4028234663852886e38F;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid;
    MaskReg upper;
    MaskReg global0;
    MaskReg global1;

    RegTensor<float> min_vreg;
    RegTensor<float> r0;
    RegTensor<float> r1;
    RegTensor<float> sel0_vreg;
    RegTensor<float> sel1_vreg;
    RegTensor<float> tile_max_vreg;
    RegTensor<float> old_max_vreg;
    RegTensor<float> new_max_vreg;
    RegTensor<float> max_brc_vreg;
    RegTensor<float> exp0_vreg;
    RegTensor<float> exp1_vreg;
    RegTensor<float> tile_sum_vreg;
    RegTensor<float> old_sum_vreg;
    RegTensor<float> exp_old_vreg;
    RegTensor<PStageT> exp0_stage_vreg;
    RegTensor<PStageT> exp1_stage_vreg;
    RegTensor<PStageT> exp_stage_vreg;
    RegTensor<int32_t> k_reg0;
    RegTensor<int32_t> k_reg1;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;
    UnalignRegForStore expmax_store;

    (void)common_ub;
    __ubuf__ float *max_write_ptr = max_ub;
    __ubuf__ float *sum_write_ptr = sum_ub;
    __ubuf__ float *expmax_write_ptr = expmax_ub;
    __ubuf__ PStageT *p_stage_ptr = p_stage_ub;
    Duplicate(min_vreg, NEG_INF);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg0, k0_local);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(
        k_reg1, k0_local + static_cast<int32_t>(HALF_K));
    if constexpr (CHECK_LOWER) {
        if (global_window_size > k0_local) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global0, k_reg0, global_window_size - 1, preg_all);
        }
        if (global_window_size > k0_local + static_cast<int32_t>(HALF_K)) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global1, k_reg1, global_window_size - 1, preg_all);
        }
    }

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        const int32_t q_local = q0_local + static_cast<int32_t>(row);
        const int32_t local_begin =
            window_left == 0 ? q_local + 1 : q_local - window_left + 1;
        int32_t window_end = q_local;
        if constexpr (!IS_CAUSAL) {
            window_end += window_right;
        }
        if constexpr (FILL_FIRST_HALF) {
            sel0_vreg = min_vreg;
        } else {
            LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
            Muls(r0, r0, scale, preg_all);
            if constexpr (CHECK_LOWER) {
                Compares<int32_t, AscendC::CMPMODE::LT>(invalid, k_reg0, local_begin, preg_all);
                if (global_window_size > k0_local) {
                    And(invalid, invalid, global0, preg_all);
                }
            }
            if constexpr (CHECK_UPPER) {
                if constexpr (CHECK_LOWER) {
                    Compares<int32_t, AscendC::CMPMODE::GT>(upper, k_reg0, window_end, preg_all);
                    Or(invalid, invalid, upper, preg_all);
                } else {
                    Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg0, window_end, preg_all);
                }
            }
            Select(sel0_vreg, min_vreg, r0, invalid);
        }
        if constexpr (FILL_SECOND_HALF) {
            sel1_vreg = min_vreg;
        } else {
            LoadAlign(r1, qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K);
            Muls(r1, r1, scale, preg_all);
            if constexpr (CHECK_LOWER) {
                Compares<int32_t, AscendC::CMPMODE::LT>(invalid, k_reg1, local_begin, preg_all);
                if (global_window_size > k0_local + static_cast<int32_t>(HALF_K)) {
                    And(invalid, invalid, global1, preg_all);
                }
            }
            if constexpr (CHECK_UPPER) {
                if constexpr (CHECK_LOWER) {
                    Compares<int32_t, AscendC::CMPMODE::GT>(upper, k_reg1, window_end, preg_all);
                    Or(invalid, invalid, upper, preg_all);
                } else {
                    Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg1, window_end, preg_all);
                }
            }
            Select(sel1_vreg, min_vreg, r1, invalid);
        }
        Max(tile_max_vreg, sel0_vreg, sel1_vreg, preg_all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(tile_max_vreg, tile_max_vreg, preg_all);
        Duplicate(max_brc_vreg, tile_max_vreg, preg_all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_max_vreg, max_ub + row);
            Max(new_max_vreg, max_brc_vreg, old_max_vreg, preg_all);
            ExpSub(exp_old_vreg, old_max_vreg, new_max_vreg, preg_all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
                expmax_write_ptr, exp_old_vreg, expmax_store, 1);
            max_brc_vreg = new_max_vreg;
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
            max_write_ptr, max_brc_vreg, max_store, 1);
        // Match DIST_DINTLV_B32 before reduction and the interleaved cast.
        DeInterleave(r0, r1, sel0_vreg, sel1_vreg);
        ExpSub(exp0_vreg, r0, max_brc_vreg, preg_all);
        ExpSub(exp1_vreg, r1, max_brc_vreg, preg_all);
        Add(tile_sum_vreg, exp0_vreg, exp1_vreg, preg_all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tile_sum_vreg, tile_sum_vreg, preg_all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_sum_vreg, sum_ub + row);
            Mula(tile_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, tile_sum_vreg, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(exp0_stage_vreg, exp0_vreg, preg_all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(exp1_stage_vreg, exp1_vreg, preg_all);
        Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(exp_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp0_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp1_stage_vreg), preg_all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_stage_ptr, exp_stage_vreg, BLOCK_STRIDE, REPEAT_STRIDE, preg_all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, max_store, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, sum_store, 0);
    if constexpr (!FIRST_TILE) {
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(expmax_write_ptr, expmax_store, 0);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}
}  // namespace VFimpl

template<class PStageT, bool FIRST_TILE, uint16_t LOCAL_ROWS = 64>
__aicore__ inline void compute_prefix_tile(
    Buffers<PStageT> bufs, uint32_t valid_cols, float scale) {
    VFimpl::compute_prefix_tile<PStageT, FIRST_TILE, LOCAL_ROWS>(
        bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.common,
        static_cast<int32_t>(valid_cols), scale);
}

template<class PStageT, bool FIRST_TILE, bool IS_CAUSAL = true, uint16_t LOCAL_ROWS = 64,
    bool CHECK_LOWER = true, bool CHECK_UPPER = true,
    bool FILL_FIRST_HALF = false, bool FILL_SECOND_HALF = false>
__aicore__ inline void compute_full_tile(
    Buffers<PStageT> bufs, uint32_t q0_local, uint32_t k0_local,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size, float scale) {
    VFimpl::compute_full_tile<PStageT, FIRST_TILE, IS_CAUSAL, LOCAL_ROWS,
        CHECK_LOWER, CHECK_UPPER, FILL_FIRST_HALF, FILL_SECOND_HALF>(
        bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.common,
        static_cast<int32_t>(q0_local), static_cast<int32_t>(k0_local),
        static_cast<int32_t>(window_left), static_cast<int32_t>(window_right),
        static_cast<int32_t>(global_window_size), scale);
}

#ifdef __CCE_AICORE__
namespace VFimpl {
template<class PStageT, bool FIRST_TILE, bool IS_CAUSAL, uint16_t LOCAL_ROWS,
    bool CHECK_LOWER, bool HAS_GLOBAL = false>
__simd_vf__ inline void compute_partial_active_fused(
    __ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    int32_t q0_abs, int32_t k0, int32_t active_rows, int32_t kv_len,
    int32_t window_left, int32_t window_right, int32_t global_window_size, float scale) {
    using namespace AscendC::MicroAPI;
    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 3;
    constexpr float NEG_INF = -3.4028234663852886e38F;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid;
    MaskReg tail0;
    MaskReg tail1;
    MaskReg lower;
    MaskReg outside_global0;
    MaskReg outside_global1;
    RegTensor<float> min_vreg;
    RegTensor<float> r0;
    RegTensor<float> r1;
    RegTensor<float> sel0;
    RegTensor<float> sel1;
    RegTensor<float> tile_max;
    RegTensor<float> max_brc;
    RegTensor<float> old_max;
    RegTensor<float> exp_old;
    RegTensor<float> exp0;
    RegTensor<float> exp1;
    RegTensor<float> tile_sum;
    RegTensor<float> old_sum;
    RegTensor<PStageT> cast0;
    RegTensor<PStageT> cast1;
    RegTensor<PStageT> packed;
    RegTensor<int32_t> k_reg0;
    RegTensor<int32_t> k_reg1;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;
    UnalignRegForStore expmax_store;
    __ubuf__ float *max_ptr = max_ub;
    __ubuf__ float *sum_ptr = sum_ub;
    __ubuf__ float *expmax_ptr = expmax_ub;
    __ubuf__ PStageT *p_ptr = p_stage_ub;
    Duplicate(min_vreg, NEG_INF);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg0, k0);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg1, k0 + HALF_K);
    Compares<int32_t, AscendC::CMPMODE::GT>(tail0, k_reg0, kv_len - 1, preg_all);
    Compares<int32_t, AscendC::CMPMODE::GT>(tail1, k_reg1, kv_len - 1, preg_all);
    if constexpr (CHECK_LOWER && HAS_GLOBAL) {
        Compares<int32_t, AscendC::CMPMODE::GT>(
            outside_global0, k_reg0, global_window_size - 1, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(
            outside_global1, k_reg1, global_window_size - 1, preg_all);
    }
    for (uint16_t row = 0; row < static_cast<uint16_t>(active_rows); ++row) {
        const int32_t q_abs = q0_abs + row;
        const int32_t local_begin = window_left == 0 ? q_abs + 1 :
            (q_abs >= window_left ? q_abs - window_left + 1 : 0);
        const int32_t window_end = q_abs + (IS_CAUSAL ? 0 : window_right);
        LoadAlign(r0, qk_ub + row * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg0, window_end, preg_all);
        Or(invalid, invalid, tail0, preg_all);
        if constexpr (CHECK_LOWER) {
            Compares<int32_t, AscendC::CMPMODE::LT>(lower, k_reg0, local_begin, preg_all);
            if constexpr (HAS_GLOBAL) {
                And(lower, lower, outside_global0, preg_all);
            }
            Or(invalid, invalid, lower, preg_all);
        }
        Select(sel0, min_vreg, r0, invalid);
        LoadAlign(r1, qk_ub + row * TILE_K + HALF_K);
        Muls(r1, r1, scale, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg1, window_end, preg_all);
        Or(invalid, invalid, tail1, preg_all);
        if constexpr (CHECK_LOWER) {
            Compares<int32_t, AscendC::CMPMODE::LT>(lower, k_reg1, local_begin, preg_all);
            if constexpr (HAS_GLOBAL) {
                And(lower, lower, outside_global1, preg_all);
            }
            Or(invalid, invalid, lower, preg_all);
        }
        Select(sel1, min_vreg, r1, invalid);
        Max(tile_max, sel0, sel1, preg_all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(tile_max, tile_max, preg_all);
        Duplicate(max_brc, tile_max, preg_all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_max, max_ub + row);
            Max(max_brc, max_brc, old_max, preg_all);
            ExpSub(exp_old, old_max, max_brc, preg_all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expmax_ptr, exp_old, expmax_store, 1);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_ptr, max_brc, max_store, 1);
        DeInterleave(r0, r1, sel0, sel1);
        ExpSub(exp0, r0, max_brc, preg_all);
        ExpSub(exp1, r1, max_brc, preg_all);
        Add(tile_sum, exp0, exp1, preg_all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tile_sum, tile_sum, preg_all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_sum, sum_ub + row);
            Mula(tile_sum, old_sum, exp_old, preg_all);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_ptr, tile_sum, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(cast0, exp0, preg_all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(cast1, exp1, preg_all);
        Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), preg_all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_ptr, packed, BLOCK_STRIDE, 1, preg_all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_ptr, max_store, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_ptr, sum_store, 0);
    if constexpr (!FIRST_TILE) {
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(expmax_ptr, expmax_store, 0);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template<class PStageT, bool FIRST_TILE, bool IS_CAUSAL = true, uint16_t LOCAL_ROWS = 64,
    bool CHECK_LOWER = true, bool SKIP_INACTIVE_ROWS = false>
__simd_vf__ inline void compute_partial_tile(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    __ubuf__ uint8_t *common_ub, int32_t q0_abs, int32_t k0, int32_t active_rows, int32_t kv_len,
    int32_t window_left, int32_t window_right, int32_t global_window_size, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 3;
    constexpr uint32_t REPEAT_STRIDE = 1;
    constexpr float NEG_INF = -3.4028234663852886e38F;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid;
    MaskReg tail0;
    MaskReg tail1;
    MaskReg window;
    MaskReg global0;
    MaskReg global1;

    RegTensor<float> min_vreg;
    RegTensor<float> r0;
    RegTensor<float> r1;
    RegTensor<float> sel0_vreg;
    RegTensor<float> sel1_vreg;
    RegTensor<float> tile_max_vreg;
    RegTensor<float> old_max_vreg;
    RegTensor<float> new_max_vreg;
    RegTensor<float> max_brc_vreg;
    RegTensor<float> exp0_vreg;
    RegTensor<float> exp1_vreg;
    RegTensor<float> tile_sum_vreg;
    RegTensor<float> old_sum_vreg;
    RegTensor<float> exp_old_vreg;
    RegTensor<PStageT> exp0_stage_vreg;
    RegTensor<PStageT> exp1_stage_vreg;
    RegTensor<PStageT> exp_stage_vreg;
    RegTensor<int32_t> k_reg0;
    RegTensor<int32_t> k_reg1;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;
    Duplicate(min_vreg, NEG_INF);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg0, k0);
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(
        k_reg1, k0 + static_cast<int32_t>(HALF_K));
    Compares<int32_t, AscendC::CMPMODE::GT>(tail0, k_reg0, kv_len - 1, preg_all);
    Compares<int32_t, AscendC::CMPMODE::GT>(tail1, k_reg1, kv_len - 1, preg_all);
    if constexpr (CHECK_LOWER) {
        if (global_window_size > 0) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global0, k_reg0, global_window_size - 1, preg_all);
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global1, k_reg1, global_window_size - 1, preg_all);
        }
    }

    const uint16_t process_rows = SKIP_INACTIVE_ROWS ?
        static_cast<uint16_t>(active_rows) : LOCAL_ROWS;
    for (uint16_t row = 0; row < process_rows; ++row) {
        if constexpr (!SKIP_INACTIVE_ROWS) {
            if (static_cast<int32_t>(row) >= active_rows) {
                StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K,
                    min_vreg, preg_all);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K,
                    min_vreg, preg_all);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, min_vreg, max_store, 1);
                continue;
            }
        }
        const int32_t q_abs = q0_abs + static_cast<int32_t>(row);
        const int32_t local_begin = window_left == 0 ? q_abs + 1 :
            (q_abs >= window_left ? q_abs - window_left + 1 : 0);
        int32_t window_end = q_abs;
        if constexpr (!IS_CAUSAL) {
            window_end += window_right;
        }

        LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg0, window_end, preg_all);
        Or(invalid, invalid, tail0, preg_all);
        if constexpr (CHECK_LOWER) {
            Compares<int32_t, AscendC::CMPMODE::LT>(window, k_reg0, local_begin, preg_all);
            if (global_window_size > 0) {
                And(window, window, global0, preg_all);
            }
            Or(invalid, invalid, window, preg_all);
        }
        Select(sel0_vreg, min_vreg, r0, invalid);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K,
            sel0_vreg, preg_all);

        LoadAlign(r1, qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K);
        Muls(r1, r1, scale, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid, k_reg1, window_end, preg_all);
        Or(invalid, invalid, tail1, preg_all);
        if constexpr (CHECK_LOWER) {
            Compares<int32_t, AscendC::CMPMODE::LT>(window, k_reg1, local_begin, preg_all);
            if (global_window_size > 0) {
                And(window, window, global1, preg_all);
            }
            Or(invalid, invalid, window, preg_all);
        }
        Select(sel1_vreg, min_vreg, r1, invalid);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K,
            sel1_vreg, preg_all);

        Max(tile_max_vreg, sel0_vreg, sel1_vreg, preg_all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(tile_max_vreg, tile_max_vreg, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, tile_max_vreg, max_store, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, max_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_max_vreg, tmp_max_ub);
        LoadAlign(old_max_vreg, max_ub);
        Max(new_max_vreg, tile_max_vreg, old_max_vreg, preg_all);
        ExpSub(exp_old_vreg, old_max_vreg, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(max_ub, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __ubuf__ PStageT *p_stage_ptr = p_stage_ub;
    for (uint16_t row = 0; row < process_rows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(max_brc_vreg, max_ub + row);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        ExpSub(exp0_vreg, r0, max_brc_vreg, preg_all);
        ExpSub(exp1_vreg, r1, max_brc_vreg, preg_all);
        Add(tile_sum_vreg, exp0_vreg, exp1_vreg, preg_all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tile_sum_vreg, tile_sum_vreg, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, tile_sum_vreg, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(exp0_stage_vreg, exp0_vreg, preg_all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(exp1_stage_vreg, exp1_vreg, preg_all);
        Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(exp_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp0_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp1_stage_vreg), preg_all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_stage_ptr, exp_stage_vreg, BLOCK_STRIDE, REPEAT_STRIDE, preg_all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, sum_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_sum_vreg, tmp_sum_ub);
        LoadAlign(old_sum_vreg, sum_ub);
        LoadAlign(exp_old_vreg, expmax_ub);
        Mula(tile_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    }
}
}  // namespace VFimpl
#endif

template<class PStageT, bool FIRST_TILE, bool IS_CAUSAL = true, uint16_t LOCAL_ROWS = 64,
    bool CHECK_LOWER = true, bool SKIP_INACTIVE_ROWS = false, bool HAS_GLOBAL = false>
__aicore__ inline void compute_partial_tile(
    Buffers<PStageT> bufs, uint32_t q0_abs, uint32_t k0, uint32_t active_rows,
    uint32_t kv_len, uint32_t window_left, uint32_t window_right,
    uint32_t global_window_size, float scale) {
    if constexpr (SKIP_INACTIVE_ROWS) {
        VFimpl::compute_partial_active_fused<
            PStageT, FIRST_TILE, IS_CAUSAL, LOCAL_ROWS, CHECK_LOWER, HAS_GLOBAL>(
            bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax,
            static_cast<int32_t>(q0_abs), static_cast<int32_t>(k0), static_cast<int32_t>(active_rows),
            static_cast<int32_t>(kv_len), static_cast<int32_t>(window_left),
            static_cast<int32_t>(window_right), static_cast<int32_t>(global_window_size), scale);
    } else {
        VFimpl::compute_partial_tile<PStageT, FIRST_TILE, IS_CAUSAL, LOCAL_ROWS, CHECK_LOWER>(
            bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.common,
            static_cast<int32_t>(q0_abs), static_cast<int32_t>(k0), static_cast<int32_t>(active_rows),
            static_cast<int32_t>(kv_len), static_cast<int32_t>(window_left),
            static_cast<int32_t>(window_right), static_cast<int32_t>(global_window_size), scale);
    }
}

namespace VFimpl {
template<class OutputT, bool FIRST_TILE, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE, uint16_t LOCAL_ROWS = 64>
__simd_vf__ inline void finalize(__ubuf__ float *vec2_out_ub, __ubuf__ OutputT *out_cast_ub,
    __ubuf__ float *lse_ub, __ubuf__ float *bmm2_ub, __ubuf__ float *expmax_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, bool write_output_cache,
    uint16_t active_rows = LOCAL_ROWS) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t DIM = 128;
    constexpr uint16_t HALF_DIM = 64;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> old0;
    RegTensor<float> old1;
    RegTensor<float> cur0;
    RegTensor<float> cur1;
    RegTensor<float> scale_vreg;
    RegTensor<float> sum_vreg;
    RegTensor<float> one_vreg;
    RegTensor<float> max_vreg;
    RegTensor<float> out0;
    RegTensor<float> out1;
    RegTensor<float> lse_vreg;
    RegTensor<OutputT> cast0;
    RegTensor<OutputT> cast1;
    RegTensor<OutputT> packed;

    if constexpr (LAST_TILE) {
        Duplicate(one_vreg, 1.0F);
        LoadAlign(sum_vreg, sum_ub);
        Div(scale_vreg, one_vreg, sum_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lse_ub, scale_vreg, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t row = 0; row < active_rows; ++row) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            cur0, cur1, bmm2_ub + static_cast<uint32_t>(row) * DIM);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(old0, old1,
                vec2_out_ub + static_cast<uint32_t>(row) * DIM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale_vreg, expmax_ub + row);
            Mula(cur0, old0, scale_vreg, preg_all);
            Mula(cur1, old1, scale_vreg, preg_all);
        }
        if constexpr (LAST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale_vreg, lse_ub + row);
            Mul(out0, cur0, scale_vreg, preg_all);
            Mul(out1, cur1, scale_vreg, preg_all);
            if constexpr (WRITE_CAST) {
                Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ZERO>(cast0, out0, preg_all);
                Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ONE>(cast1, out1, preg_all);
                Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(packed),
                    reinterpret_cast<RegTensor<uint16_t>&>(cast0),
                    reinterpret_cast<RegTensor<uint16_t>&>(cast1), preg_all_b16);
                StoreAlign<OutputT>(out_cast_ub + static_cast<uint32_t>(row) * DIM, packed, preg_all_b16);
                if (write_output_cache) {
                    StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                        vec2_out_ub + static_cast<uint32_t>(row) * DIM, out0, out1, preg_all);
                }
            } else {
                StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM, out0, out1, preg_all);
            }
        } else {
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                vec2_out_ub + static_cast<uint32_t>(row) * DIM, cur0, cur1, preg_all);
        }
    }

    if constexpr (LAST_TILE && WRITE_LSE) {
        LoadAlign(sum_vreg, sum_ub);
        LoadAlign(max_vreg, max_ub);
        Ln(lse_vreg, sum_vreg, preg_all);
        Add(lse_vreg, lse_vreg, max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lse_ub, lse_vreg, preg_all);
    }
}
}  // namespace VFimpl

namespace VFimpl {
template<class OutputT, bool FIRST_PAIR, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE,
    uint16_t LOCAL_ROWS = 64>
__simd_vf__ inline void finalize_pair(__ubuf__ float *vec2_out_ub, __ubuf__ OutputT *out_cast_ub,
    __ubuf__ float *lse_ub, __ubuf__ float *prev_bmm2_ub, __ubuf__ float *cur_bmm2_ub,
    __ubuf__ float *prev_expmax_ub, __ubuf__ float *cur_expmax_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, bool write_output_cache,
    uint16_t active_rows = LOCAL_ROWS) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t DIM = 128;
    constexpr uint16_t HALF_DIM = 64;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> old0;
    RegTensor<float> old1;
    RegTensor<float> prev0;
    RegTensor<float> prev1;
    RegTensor<float> cur0;
    RegTensor<float> cur1;
    RegTensor<float> prev_scale;
    RegTensor<float> cur_scale;
    RegTensor<float> pair_scale;
    RegTensor<float> sum_vreg;
    RegTensor<float> one_vreg;
    RegTensor<float> max_vreg;
    RegTensor<float> lse_vreg;
    RegTensor<OutputT> cast0;
    RegTensor<OutputT> cast1;
    RegTensor<OutputT> packed;

    if constexpr (LAST_TILE) {
        Duplicate(one_vreg, 1.0F);
        LoadAlign(sum_vreg, sum_ub);
        Div(pair_scale, one_vreg, sum_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lse_ub, pair_scale, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t row = 0; row < active_rows; ++row) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            cur0, cur1, cur_bmm2_ub + static_cast<uint32_t>(row) * DIM);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            prev0, prev1, prev_bmm2_ub + static_cast<uint32_t>(row) * DIM);
        if constexpr (!FIRST_PAIR) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                old0, old1, vec2_out_ub + static_cast<uint32_t>(row) * DIM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(prev_scale, prev_expmax_ub + row);
            Mula(prev0, old0, prev_scale, preg_all);
            Mula(prev1, old1, prev_scale, preg_all);
        }
        LoadAlign<float, LoadDist::DIST_BRC_B32>(cur_scale, cur_expmax_ub + row);
        Mula(cur0, prev0, cur_scale, preg_all);
        Mula(cur1, prev1, cur_scale, preg_all);
        if constexpr (LAST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(pair_scale, lse_ub + row);
            Mul(cur0, cur0, pair_scale, preg_all);
            Mul(cur1, cur1, pair_scale, preg_all);
        }
        if constexpr (LAST_TILE && WRITE_CAST) {
            Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ZERO>(cast0, cur0, preg_all);
            Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ONE>(cast1, cur1, preg_all);
            Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(packed),
                reinterpret_cast<RegTensor<uint16_t>&>(cast0),
                reinterpret_cast<RegTensor<uint16_t>&>(cast1), preg_all_b16);
            StoreAlign<OutputT>(out_cast_ub + static_cast<uint32_t>(row) * DIM, packed, preg_all_b16);
            if (write_output_cache) {
                StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM, cur0, cur1, preg_all);
            }
        } else {
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                vec2_out_ub + static_cast<uint32_t>(row) * DIM, cur0, cur1, preg_all);
        }
    }

    if constexpr (LAST_TILE && WRITE_LSE) {
        LoadAlign(sum_vreg, sum_ub);
        LoadAlign(max_vreg, max_ub);
        Ln(lse_vreg, sum_vreg, preg_all);
        Add(lse_vreg, lse_vreg, max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lse_ub, lse_vreg, preg_all);
    }
}
}  // namespace VFimpl

template<class OutputT, bool FIRST_PAIR, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE,
    uint16_t LOCAL_ROWS = 64>
__aicore__ inline void finalize_pair(UBTensor_ND<float, LOCAL_ROWS, 128>& vec2_out_ub,
    __ubuf__ OutputT *out_cast_ub, UBTensor_ND<float, 1, 64>& lse_ub,
    UBTensor_ND<float, LOCAL_ROWS, 128>& prev_bmm2_ub,
    UBTensor_ND<float, LOCAL_ROWS, 128>& cur_bmm2_ub, UBTensor_ND<float, 1, 64>& prev_expmax_ub,
    UBTensor_ND<float, 1, 64>& cur_expmax_ub, UBTensor_ND<float, 1, 64>& sum_ub,
    UBTensor_ND<float, 1, 64>& max_ub, bool write_output_cache = false,
    uint16_t active_rows = LOCAL_ROWS) {
    VFimpl::finalize_pair<OutputT, FIRST_PAIR, LAST_TILE, WRITE_CAST, WRITE_LSE, LOCAL_ROWS>(
        reinterpret_cast<__ubuf__ float *>(vec2_out_ub.tensor.GetPhyAddr()),
        out_cast_ub,
        reinterpret_cast<__ubuf__ float *>(lse_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(prev_bmm2_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(cur_bmm2_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(prev_expmax_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(cur_expmax_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(max_ub.tensor.GetPhyAddr()), write_output_cache, active_rows);
}

template<class OutputT, bool FIRST_TILE, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE, uint16_t LOCAL_ROWS = 64>
__aicore__ inline void finalize(UBTensor_ND<float, LOCAL_ROWS, 128>& vec2_out_ub, __ubuf__ OutputT *out_cast_ub,
    UBTensor_ND<float, 1, 64>& lse_ub, UBTensor_ND<float, LOCAL_ROWS, 128>& bmm2_ub,
    UBTensor_ND<float, 1, 64>& expmax_ub, UBTensor_ND<float, 1, 64>& sum_ub,
    UBTensor_ND<float, 1, 64>& max_ub, bool write_output_cache = false,
    uint16_t active_rows = LOCAL_ROWS) {
    VFimpl::finalize<OutputT, FIRST_TILE, LAST_TILE, WRITE_CAST, WRITE_LSE, LOCAL_ROWS>(
        reinterpret_cast<__ubuf__ float *>(vec2_out_ub.tensor.GetPhyAddr()),
        out_cast_ub,
        reinterpret_cast<__ubuf__ float *>(lse_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(bmm2_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(expmax_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(max_ub.tensor.GetPhyAddr()), write_output_cache, active_rows);
}

template<bool FIRST_TILE, bool LAST_TILE>
__aicore__ inline void finalize(UBTensor_ND<float, 64, 128>& vec2_out_ub,
    UBTensor_ND<float, 64, 128>& bmm2_ub, UBTensor_ND<float, 1, 64>& expmax_ub,
    UBTensor_ND<float, 1, 64>& sum_ub) {
    VFimpl::finalize<float, FIRST_TILE, LAST_TILE, false, false>(
        reinterpret_cast<__ubuf__ float *>(vec2_out_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(vec2_out_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(bmm2_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(expmax_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()), false);
}

namespace VFimpl {
template<class PStageT, bool FIRST_TILE, uint16_t LOCAL_ROWS = 64>
__simd_vf__ inline void compute_all_valid_fused(__ubuf__ PStageT *p_stage_ub,
    __ubuf__ float *qk_ub, __ubuf__ float *sum_ub, __ubuf__ float *max_ub,
    __ubuf__ float *expmax_ub, float scale) {
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> r0, r1, tile_max, old_max, new_max, exp_old;
    RegTensor<float> exp0, exp1, tile_sum, old_sum;
    RegTensor<PStageT> cast0, cast1, packed;
    UnalignRegForStore max_store, sum_store, expmax_store;
    __ubuf__ float *max_ptr = max_ub;
    __ubuf__ float *sum_ptr = sum_ub;
    __ubuf__ float *expmax_ptr = expmax_ub;
    __ubuf__ PStageT *p_ptr = p_stage_ub;
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, qk_ub + row * 128U);
        Max(tile_max, r0, r1, all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(tile_max, tile_max, all);
        Muls(tile_max, tile_max, scale, all);
        Duplicate(new_max, tile_max, all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_max, max_ub + row);
            Max(new_max, new_max, old_max, all);
            ExpSub(exp_old, old_max, new_max, all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expmax_ptr, exp_old, expmax_store, 1);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_ptr, new_max, max_store, 1);
        Muls(r0, r0, scale, all);
        Muls(r1, r1, scale, all);
        ExpSub(exp0, r0, new_max, all);
        ExpSub(exp1, r1, new_max, all);
        Add(tile_sum, exp0, exp1, all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tile_sum, tile_sum, all);
        if constexpr (!FIRST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(old_sum, sum_ub + row);
            Mula(tile_sum, old_sum, exp_old, all);
        }
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_ptr, tile_sum, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(cast0, exp0, all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(cast1, exp1, all);
        Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(packed),
            reinterpret_cast<RegTensor<uint16_t>&>(cast0),
            reinterpret_cast<RegTensor<uint16_t>&>(cast1), all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_ptr, packed, LOCAL_ROWS + 3U, 1, all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_ptr, max_store, 0);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_ptr, sum_store, 0);
    if constexpr (!FIRST_TILE) {
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(expmax_ptr, expmax_store, 0);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template<class PStageT, bool FIRST_TILE, uint16_t LOCAL_ROWS = 64>
__simd_vf__ inline void compute_all_valid(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub, __ubuf__ uint8_t *mask_ub,
    __ubuf__ uint8_t *common_ub, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 3;
    constexpr uint32_t REPEAT_STRIDE = 1;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<float> r0;
    RegTensor<float> r1;
    RegTensor<float> tile_max_vreg;
    RegTensor<float> old_max_vreg;
    RegTensor<float> new_max_vreg;
    RegTensor<float> max_brc_vreg;
    RegTensor<float> exp0_vreg;
    RegTensor<float> exp1_vreg;
    RegTensor<float> tile_sum_vreg;
    RegTensor<float> old_sum_vreg;
    RegTensor<float> exp_old_vreg;
    RegTensor<PStageT> exp0_stage_vreg;
    RegTensor<PStageT> exp1_stage_vreg;
    RegTensor<PStageT> exp_stage_vreg;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, qk_ub + static_cast<uint32_t>(row) * TILE_K);

        Max(tile_max_vreg, r0, r1, preg_all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(tile_max_vreg, tile_max_vreg, preg_all);
        Muls(tile_max_vreg, tile_max_vreg, scale, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, tile_max_vreg, max_store, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, max_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_max_vreg, tmp_max_ub);
        LoadAlign(old_max_vreg, max_ub);
        Max(new_max_vreg, tile_max_vreg, old_max_vreg, preg_all);
        ExpSub(exp_old_vreg, old_max_vreg, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(max_ub, new_max_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __ubuf__ PStageT *p_stage_ptr = p_stage_ub;
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(max_brc_vreg, max_ub + row);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Muls(r1, r1, scale, preg_all);
        ExpSub(exp0_vreg, r0, max_brc_vreg, preg_all);
        ExpSub(exp1_vreg, r1, max_brc_vreg, preg_all);
        Add(tile_sum_vreg, exp0_vreg, exp1_vreg, preg_all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(tile_sum_vreg, tile_sum_vreg, preg_all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, tile_sum_vreg, sum_store, 1);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ZERO>(exp0_stage_vreg, exp0_vreg, preg_all);
        Cast<PStageT, float, detail::MICRO_CAST_TRAIT_ONE>(exp1_stage_vreg, exp1_vreg, preg_all);
        Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(exp_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp0_stage_vreg),
            reinterpret_cast<RegTensor<uint16_t>&>(exp1_stage_vreg), preg_all_b16);
        StoreAlign<PStageT, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p_stage_ptr, exp_stage_vreg, BLOCK_STRIDE, REPEAT_STRIDE, preg_all_b16);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sum_write_ptr, sum_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if constexpr (!FIRST_TILE) {
        LoadAlign(tile_sum_vreg, tmp_sum_ub);
        LoadAlign(old_sum_vreg, sum_ub);
        LoadAlign(exp_old_vreg, expmax_ub);
        Mula(tile_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    }
}
}  // namespace VFimpl

template<class PStageT, bool FIRST_TILE, uint16_t LOCAL_ROWS = 64>
__aicore__ inline void compute_all_valid(
    Buffers<PStageT> bufs, float scale) {
    VFimpl::compute_all_valid<PStageT, FIRST_TILE, LOCAL_ROWS>(
        bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.mask, bufs.common, scale);
}

}  // namespace online_softmax

template<class T, int DST_R, int DST_C, AscendC::HardEvent READY_EVENT, AscendC::HardEvent FREE_EVENT, TensorLayout L>
__aicore__ inline void copy_sync(
    SyncTensor<AscendC::LocalTensor<T>, DST_R, DST_C, READY_EVENT, FREE_EVENT, L>& dst,
    GMTensor_ND<T>& src, uint32_t src_cols, uint32_t src_offset, uint32_t copy_rows) {
    dst.producer_wait();
    if (copy_rows == static_cast<uint32_t>(DST_R)) {
        prim::copy_gm_to_l1_nz(
            dst.tensor,
            src.tensor[src_offset],
            static_cast<uint32_t>(DST_R),
            static_cast<uint32_t>(DST_C),
            src_cols);
    } else {
        prim::copy_gm_to_l1_nz_full_rows(
            dst.tensor,
            src.tensor[src_offset],
            copy_rows,
            static_cast<uint32_t>(DST_C),
            src_cols,
            static_cast<uint32_t>(DST_R));
    }
    dst.producer_set();
}

template<class AT, class BT, int M, int N, int K>
__aicore__ inline void matmul_sync(CO1Tensor<float, M, N>& dst, A2Tensor<AT, M, K>& a,
    B2Tensor<BT, K, N>& b, bool init_c = true, uint8_t unit_flag = 0) {
    dst.producer_wait();
    prim::mmad<AT, BT, M, N, K>(dst.tensor, a.tensor, b.tensor, init_c, unit_flag);
    dst.producer_set();
}

template<class AT, class BT, int M, int N, int K>
__aicore__ inline void matmul_sync_dynamic_k(CO1Tensor<float, M, N>& dst, A2Tensor<AT, M, K>& a,
    B2Tensor<BT, K, N>& b, uint32_t runtime_k, bool init_c = true, uint8_t unit_flag = 0) {
    dst.producer_wait();
    prim::dynamic_k_mmad<AT, BT, M, N>(
        dst.tensor, a.tensor, b.tensor, runtime_k, init_c, unit_flag);
    dst.producer_set();
}


}  // namespace SWA_v5::Compute
