#pragma once

#include "kernel/primitive.h"
#include "kernel/tensor.h"

namespace SWA_v5::Copy {

// Synchronized wrappers only handle dst-side producer synchronization.
// The caller owns src-side ready/consumed synchronization.

template<class T, int DST_R, int DST_C>
__aicore__ inline void copy_sync(A1Tensor<T, DST_R, DST_C>& dst, GMTensor_ND<T>& src,
    uint32_t src_cols, uint32_t src_offset) {
    dst.producer_wait();
    prim::copy_gm_to_l1_nz(dst.tensor, src.tensor[src_offset], DST_R, DST_C, src_cols);
    dst.producer_set();
}

template<class T, int R, int C>
__aicore__ inline void copy_sync(A2Tensor<T, R, C>& dst, A1Tensor<T, R, C>& src) {
    dst.producer_wait();
    prim::copy_l1_to_a2_nz<T, R, C>(dst.tensor, src.tensor);
    dst.producer_set();
}

template<int R, int C>
__aicore__ inline void copy(UBTensor_ND<float, R / 2, C>& dst, CO1Tensor<float, R, C>& src,
    uint8_t unit_flag = 0) {
    static_assert(R % 2 == 0);
    prim::fixpipe_c310_full_l0c_to_aiv_ub<R, C>(dst.tensor, src.tensor, C, unit_flag);
}

template<class T, int R, int C>
__aicore__ inline void copy(
    GMTensor_ND<T>& dst, UBTensor_ND<T, R, C> src, uint32_t dst_stride, uint32_t dst_offset) {
    prim::copy_ub_to_gm_rows<T, R, C>(dst.tensor[dst_offset], src.tensor, dst_stride);
}

template<int R, int C>
__aicore__ inline void copy(GMTensor_ND<float>& dst, CO1Tensor<float, R, C>& src,
    uint32_t dst_stride, uint32_t dst_offset, uint8_t unit_flag = 0) {
    prim::fixpipe_l0c_to_gm_col_blocks<R, C>(dst.tensor[dst_offset], src.tensor, dst_stride, unit_flag);
}

template<class T, int R, int C, int ROW_PARTS>
__aicore__ inline void split_copy_sync_trans(
    B2Tensor<T, C, R / ROW_PARTS>& dst, B1Tensor<T, R, C>& src, uint32_t part_idx) {
    dst.producer_wait();
    prim::split_copy_l1_to_b2_zn_trans<T, R, C, ROW_PARTS>(dst.tensor, src.tensor, part_idx);
    dst.producer_set();
}

template<class T, int R, int C, int ROW_PARTS>
__aicore__ inline void split_copy_sync(
    B2Tensor<T, R / ROW_PARTS, C>& dst, B1Tensor<T, R, C>& src, uint32_t part_idx) {
    dst.producer_wait();
    prim::split_copy_l1_to_b2_zn<T, R, C, ROW_PARTS>(dst.tensor, src.tensor, part_idx);
    dst.producer_set();
}

}  // namespace SWA_v5::Copy
