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

template<class PStageT, uint32_t QK_SLOTS, uint32_t P_STAGE_SLOTS, uint32_t SOFTMAX_STATE_SLOTS>
struct BufferCache {
    __ubuf__ float *qk[QK_SLOTS];
    __ubuf__ PStageT *p_stage[P_STAGE_SLOTS];
    __ubuf__ float *sum[SOFTMAX_STATE_SLOTS];
    __ubuf__ float *max[SOFTMAX_STATE_SLOTS];
    __ubuf__ float *expmax[SOFTMAX_STATE_SLOTS];
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
template<class PStageT, bool FIRST_TILE, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U,
    bool IS_CAUSAL = true>
__simd_vf__ inline void compute_full_tile(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    __ubuf__ uint8_t *common_ub, int32_t q0_local, int32_t k0_local, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t LOCAL_ROWS = 64;
    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 1;
    constexpr uint32_t REPEAT_STRIDE = 1;
    constexpr float NEG_INF = -3.4028234663852886e38F;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_compare;
    MaskReg preg_compare_unroll;
    MaskReg mask_gt_for_swa;
    MaskReg mask_ge_global;

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
    RegTensor<int32_t> k_reg;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;
    Duplicate(min_vreg, NEG_INF);

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        LoadAlign(r1, qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K);
        Muls(r0, r0, scale, preg_all);
        Muls(r1, r1, scale, preg_all);
        const int32_t q_local = q0_local + static_cast<int32_t>(row);
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg, k0_local);
        Compares<int32_t, AscendC::CMPMODE::LT>(
            preg_compare, k_reg, q_local - static_cast<int32_t>(WINDOW_LEFT), preg_all);
        if constexpr (GLOBAL_WINDOW_SIZE > 0U) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                mask_ge_global, k_reg, static_cast<int32_t>(GLOBAL_WINDOW_SIZE - 1U), preg_all);
            And(preg_compare, preg_compare, mask_ge_global, preg_all);
        }
        int32_t window_end = q_local;
        if constexpr (!IS_CAUSAL && WINDOW_RIGHT > 0U) {
            window_end += static_cast<int32_t>(WINDOW_RIGHT);
        }
        Compares<int32_t, AscendC::CMPMODE::GT>(
            preg_compare_unroll, k_reg, window_end, preg_all);
        Or(preg_compare, preg_compare, preg_compare_unroll, preg_all);
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg, k0_local + static_cast<int32_t>(HALF_K));
        Compares<int32_t, AscendC::CMPMODE::LT>(
            preg_compare_unroll, k_reg, q_local - static_cast<int32_t>(WINDOW_LEFT), preg_all);
        if constexpr (GLOBAL_WINDOW_SIZE > 0U) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                mask_ge_global, k_reg, static_cast<int32_t>(GLOBAL_WINDOW_SIZE - 1U), preg_all);
            And(preg_compare_unroll, preg_compare_unroll, mask_ge_global, preg_all);
        }
        Compares<int32_t, AscendC::CMPMODE::GT>(
            mask_gt_for_swa, k_reg, window_end, preg_all);
        Or(preg_compare_unroll, preg_compare_unroll, mask_gt_for_swa, preg_all);
        Select(sel0_vreg, min_vreg, r0, preg_compare);
        Select(sel1_vreg, min_vreg, r1, preg_compare_unroll);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K, sel0_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K, sel1_vreg, preg_all);
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
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
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
        Mul(old_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        Add(tile_sum_vreg, tile_sum_vreg, old_sum_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    } else {
        Duplicate(exp_old_vreg, 0.0F);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
    }
}
}  // namespace VFimpl

template<class PStageT, bool FIRST_TILE, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U,
    bool IS_CAUSAL = true>
__aicore__ inline void compute_full_tile(
    Buffers<PStageT> bufs, uint32_t q0_local, uint32_t k0_local, float scale) {
    VFimpl::compute_full_tile<PStageT, FIRST_TILE, WINDOW_LEFT, WINDOW_RIGHT, GLOBAL_WINDOW_SIZE, IS_CAUSAL>(
        bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.common,
        static_cast<int32_t>(q0_local), static_cast<int32_t>(k0_local), scale);
}

#ifdef __CCE_AICORE__
namespace VFimpl {
template<class PStageT, bool FIRST_TILE, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U,
    bool IS_CAUSAL = true>
__simd_vf__ inline void compute_partial_tile(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub,
    __ubuf__ uint8_t *common_ub, int32_t q0_abs, int32_t k0, int32_t active_rows, int32_t kv_len,
    float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t LOCAL_ROWS = 64;
    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 1;
    constexpr uint32_t REPEAT_STRIDE = 1;
    constexpr float NEG_INF = -3.4028234663852886e38F;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg invalid0;
    MaskReg invalid1;
    MaskReg tail0;
    MaskReg tail1;
    MaskReg window0;
    MaskReg window1;
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
    RegTensor<int32_t> k_reg;
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;
    Duplicate(min_vreg, NEG_INF);

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        if (static_cast<int32_t>(row) >= active_rows) {
            StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K,
                min_vreg, preg_all);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K,
                min_vreg, preg_all);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_write_ptr, min_vreg, max_store, 1);
            continue;
        }

        const int32_t q_abs = q0_abs + static_cast<int32_t>(row);
        const int32_t local_begin = q_abs > static_cast<int32_t>(WINDOW_LEFT) ?
            q_abs - static_cast<int32_t>(WINDOW_LEFT) : 0;
        int32_t window_end = q_abs;
        if constexpr (!IS_CAUSAL && WINDOW_RIGHT > 0U) {
            window_end += static_cast<int32_t>(WINDOW_RIGHT);
        }

        LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg, k0);
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid0, k_reg, window_end, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(tail0, k_reg, kv_len - 1, preg_all);
        Or(invalid0, invalid0, tail0, preg_all);
        Compares<int32_t, AscendC::CMPMODE::LT>(window0, k_reg, local_begin, preg_all);
        if constexpr (GLOBAL_WINDOW_SIZE > 0U) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global0, k_reg, static_cast<int32_t>(GLOBAL_WINDOW_SIZE - 1U), preg_all);
            And(window0, window0, global0, preg_all);
        }
        Or(invalid0, invalid0, window0, preg_all);
        Select(sel0_vreg, min_vreg, r0, invalid0);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(qk_ub + static_cast<uint32_t>(row) * TILE_K,
            sel0_vreg, preg_all);

        LoadAlign(r1, qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K);
        Muls(r1, r1, scale, preg_all);
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(k_reg, k0 + static_cast<int32_t>(HALF_K));
        Compares<int32_t, AscendC::CMPMODE::GT>(invalid1, k_reg, window_end, preg_all);
        Compares<int32_t, AscendC::CMPMODE::GT>(tail1, k_reg, kv_len - 1, preg_all);
        Or(invalid1, invalid1, tail1, preg_all);
        Compares<int32_t, AscendC::CMPMODE::LT>(window1, k_reg, local_begin, preg_all);
        if constexpr (GLOBAL_WINDOW_SIZE > 0U) {
            Compares<int32_t, AscendC::CMPMODE::GT>(
                global1, k_reg, static_cast<int32_t>(GLOBAL_WINDOW_SIZE - 1U), preg_all);
            And(window1, window1, global1, preg_all);
        }
        Or(invalid1, invalid1, window1, preg_all);
        Select(sel1_vreg, min_vreg, r1, invalid1);
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
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
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
        Mul(old_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        Add(tile_sum_vreg, tile_sum_vreg, old_sum_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    } else {
        Duplicate(exp_old_vreg, 0.0F);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
    }
}
}  // namespace VFimpl
#endif

template<class PStageT, bool FIRST_TILE, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE = 0U,
    bool IS_CAUSAL = true>
__aicore__ inline void compute_partial_tile(
    Buffers<PStageT> bufs, uint32_t q0_abs, uint32_t k0, uint32_t active_rows,
    uint32_t kv_len, float scale) {
    VFimpl::compute_partial_tile<PStageT, FIRST_TILE, WINDOW_LEFT, WINDOW_RIGHT, GLOBAL_WINDOW_SIZE, IS_CAUSAL>(
        bufs.p_stage, bufs.qk, bufs.sum, bufs.max, bufs.expmax, bufs.common,
        static_cast<int32_t>(q0_abs), static_cast<int32_t>(k0), static_cast<int32_t>(active_rows),
        static_cast<int32_t>(kv_len), scale);
}

namespace VFimpl {
template<class OutputT, bool FIRST_TILE, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE>
__simd_vf__ inline void finalize(__ubuf__ float *vec2_out_ub, __ubuf__ OutputT *out_cast_ub,
    __ubuf__ float *lse_ub, __ubuf__ float *bmm2_ub, __ubuf__ float *expmax_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t LOCAL_ROWS = 64;
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
    RegTensor<float> max_vreg;
    RegTensor<float> out0;
    RegTensor<float> out1;
    RegTensor<float> lse_vreg;
    RegTensor<OutputT> cast0;
    RegTensor<OutputT> cast1;
    RegTensor<OutputT> packed;

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        if constexpr (WRITE_CAST) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(cur0, cur1, bmm2_ub + static_cast<uint32_t>(row) * DIM);
        } else {
            LoadAlign(cur0, bmm2_ub + static_cast<uint32_t>(row) * DIM);
            LoadAlign(cur1, bmm2_ub + static_cast<uint32_t>(row) * DIM + HALF_DIM);
        }
        if constexpr (!FIRST_TILE) {
            if constexpr (WRITE_CAST) {
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(old0, old1,
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM);
            } else {
                LoadAlign(old0, vec2_out_ub + static_cast<uint32_t>(row) * DIM);
                LoadAlign(old1, vec2_out_ub + static_cast<uint32_t>(row) * DIM + HALF_DIM);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale_vreg, expmax_ub + row);
            Mul(old0, old0, scale_vreg, preg_all);
            Mul(old1, old1, scale_vreg, preg_all);
            Add(cur0, old0, cur0, preg_all);
            Add(cur1, old1, cur1, preg_all);
        }
        if constexpr (LAST_TILE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sum_vreg, sum_ub + row);
            Div(out0, cur0, sum_vreg, preg_all);
            Div(out1, cur1, sum_vreg, preg_all);
            if constexpr (WRITE_CAST) {
                Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ZERO>(cast0, out0, preg_all);
                Cast<OutputT, float, detail::MICRO_CAST_TRAIT_ONE>(cast1, out1, preg_all);
                Or<uint16_t, MaskMergeMode::ZEROING>(reinterpret_cast<RegTensor<uint16_t>&>(packed),
                    reinterpret_cast<RegTensor<uint16_t>&>(cast0),
                    reinterpret_cast<RegTensor<uint16_t>&>(cast1), preg_all_b16);
                StoreAlign<OutputT>(out_cast_ub + static_cast<uint32_t>(row) * DIM, packed, preg_all_b16);
            } else {
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM, out0, preg_all);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM + HALF_DIM, out1, preg_all);
            }
        } else {
            if constexpr (WRITE_CAST) {
                StoreAlign<float, StoreDist::DIST_INTLV_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM, cur0, cur1, preg_all);
            } else {
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM, cur0, preg_all);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    vec2_out_ub + static_cast<uint32_t>(row) * DIM + HALF_DIM, cur1, preg_all);
            }
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

template<class OutputT, bool FIRST_TILE, bool LAST_TILE, bool WRITE_CAST, bool WRITE_LSE>
__aicore__ inline void finalize(UBTensor_ND<float, 64, 128>& vec2_out_ub, __ubuf__ OutputT *out_cast_ub,
    UBTensor_ND<float, 1, 64>& lse_ub, UBTensor_ND<float, 64, 128>& bmm2_ub,
    UBTensor_ND<float, 1, 64>& expmax_ub, UBTensor_ND<float, 1, 64>& sum_ub,
    UBTensor_ND<float, 1, 64>& max_ub) {
    VFimpl::finalize<OutputT, FIRST_TILE, LAST_TILE, WRITE_CAST, WRITE_LSE>(
        reinterpret_cast<__ubuf__ float *>(vec2_out_ub.tensor.GetPhyAddr()),
        out_cast_ub,
        reinterpret_cast<__ubuf__ float *>(lse_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(bmm2_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(expmax_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(max_ub.tensor.GetPhyAddr()));
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
        reinterpret_cast<__ubuf__ float *>(sum_ub.tensor.GetPhyAddr()));
}

namespace VFimpl {
template<class PStageT, bool FIRST_TILE>
__simd_vf__ inline void compute_all_valid(__ubuf__ PStageT *p_stage_ub, __ubuf__ float *qk_ub,
    __ubuf__ float *sum_ub, __ubuf__ float *max_ub, __ubuf__ float *expmax_ub, __ubuf__ uint8_t *mask_ub,
    __ubuf__ uint8_t *common_ub, float scale) {
    using namespace AscendC::MicroAPI;

    constexpr uint16_t LOCAL_ROWS = 64;
    constexpr uint32_t TILE_K = 128;
    constexpr uint32_t HALF_K = 64;
    constexpr uint32_t STATE_ELEMS = 64;
    constexpr uint32_t BLOCK_STRIDE = LOCAL_ROWS + 1;
    constexpr uint32_t REPEAT_STRIDE = 1;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

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
    UnalignRegForStore max_store;
    UnalignRegForStore sum_store;

    __ubuf__ float *tmp_sum_ub = reinterpret_cast<__ubuf__ float *>(common_ub);
    __ubuf__ float *tmp_max_ub = tmp_sum_ub + STATE_ELEMS;
    __ubuf__ float *max_write_ptr = FIRST_TILE ? max_ub : tmp_max_ub;
    __ubuf__ float *sum_write_ptr = FIRST_TILE ? sum_ub : tmp_sum_ub;

    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
        LoadAlign(r0, qk_ub + static_cast<uint32_t>(row) * TILE_K);
        Muls(r0, r0, scale, preg_all);
        Muls(sel0_vreg, r0, 1.0F, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            qk_ub + static_cast<uint32_t>(row) * TILE_K, sel0_vreg, preg_all);

        LoadAlign(r1, qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K);
        Muls(r1, r1, scale, preg_all);
        Muls(sel1_vreg, r1, 1.0F, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            qk_ub + static_cast<uint32_t>(row) * TILE_K + HALF_K, sel1_vreg, preg_all);

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
    for (uint16_t row = 0; row < LOCAL_ROWS; ++row) {
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
        Mul(old_sum_vreg, old_sum_vreg, exp_old_vreg, preg_all);
        Add(tile_sum_vreg, tile_sum_vreg, old_sum_vreg, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sum_ub, tile_sum_vreg, preg_all);
    } else {
        Duplicate(exp_old_vreg, 0.0F);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expmax_ub, exp_old_vreg, preg_all);
    }
}
}  // namespace VFimpl

template<class PStageT, bool FIRST_TILE>
__aicore__ inline void compute_all_valid(
    Buffers<PStageT> bufs, float scale) {
    VFimpl::compute_all_valid<PStageT, FIRST_TILE>(
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


}  // namespace SWA_v5::Compute
