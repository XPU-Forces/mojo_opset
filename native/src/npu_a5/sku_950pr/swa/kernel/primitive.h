#pragma once

#include "kernel_operator.h"

namespace SWA_v5::prim {

static constexpr uint32_t MMAD_UNIT = 16;
static constexpr uint32_t MMAD_SIZE = MMAD_UNIT * MMAD_UNIT;
static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_TO_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_TO_GM = {AscendC::CO2Layout::ROW_MAJOR, false};

__aicore__ inline uint32_t ceil_div(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

template<class T>
__aicore__ inline void copy_gm_to_l1_nz_full_rows(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
    uint32_t rows, uint32_t cols, uint32_t src_cols, uint32_t dst_full_rows) {
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = cols;
    params.srcNdMatrixStride = 0;
    params.srcDValue = src_cols;
    params.dstNzC0Stride = ceil_div(dst_full_rows, MMAD_UNIT) * MMAD_UNIT;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

template<class T>
__aicore__ inline void copy_gm_to_l1_nz(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
    uint32_t rows, uint32_t cols, uint32_t src_cols) {
    copy_gm_to_l1_nz_full_rows(dst, src, rows, cols, src_cols, rows);
}

template<class T, uint32_t rows, uint32_t cols>
__aicore__ inline void copy_l1_to_a2_nz(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src) {
    AscendC::LoadData2DParamsV2 params;
    params.mStartPosition = 0;
    params.kStartPosition = 0;
    params.mStep = ceil_div(rows, MMAD_UNIT);
    params.kStep = ceil_div(cols, MMAD_UNIT);
    params.srcStride = ceil_div(rows, MMAD_UNIT);
    params.dstStride = ceil_div(rows, MMAD_UNIT);
    params.sid = 0;
    params.ifTranspose = false;
    AscendC::LoadData(dst, src, params);
}

template<class T, uint32_t src_rows, uint32_t src_cols, uint32_t row_parts>
__aicore__ inline void split_copy_l1_to_b2_zn_trans(
    AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t part_idx) {
    constexpr uint32_t dst_rows = src_cols;
    constexpr uint32_t dst_cols = src_rows / row_parts;
    constexpr uint32_t unit_rows = dst_rows / MMAD_UNIT;
    constexpr uint32_t unit_cols = dst_cols / MMAD_UNIT;

    AscendC::LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = static_cast<uint8_t>(unit_cols);
    params.srcStride = 1;
    params.sid = 0;
    params.ifTranspose = false;
    params.addrMode = 0;
    params.dstGap = 0;

    for (uint32_t i = 0; i < unit_rows; ++i) {
        AscendC::LoadData(dst[i * unit_cols * MMAD_SIZE],
            src[(i * row_parts + part_idx) * unit_cols * MMAD_SIZE], params);
    }
}

template<class T, uint32_t src_rows, uint32_t src_cols, uint32_t row_parts>
__aicore__ inline void split_copy_l1_to_b2_zn(
    AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t part_idx) {
    constexpr uint32_t dst_rows = src_rows / row_parts;
    constexpr uint32_t dst_cols = src_cols;
    constexpr uint32_t unit_rows = dst_rows / MMAD_UNIT;
    constexpr uint32_t unit_cols = dst_cols / MMAD_UNIT;

    AscendC::LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = static_cast<uint8_t>(unit_cols);
    params.srcStride = static_cast<uint16_t>(unit_rows * row_parts);
    params.sid = 0;
    params.ifTranspose = true;
    params.addrMode = 0;
    params.dstGap = 0;

    for (uint32_t i = 0; i < unit_rows; ++i) {
        AscendC::LoadData(dst[i * unit_cols * MMAD_SIZE],
            src[(part_idx * unit_rows + i) * MMAD_SIZE], params);
    }
}

template<class AT, class BT, uint32_t M, uint32_t N, uint32_t K>
__aicore__ inline void mmad(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<AT> a,
    AscendC::LocalTensor<BT> b, bool init_c, uint8_t unit_flag) {
    AscendC::MmadParams params;
    params.m = static_cast<uint16_t>(M);
    params.n = static_cast<uint16_t>(N);
    params.k = static_cast<uint16_t>(K);
    params.cmatrixInitVal = init_c;
    params.cmatrixSource = false;
    params.unitFlag = unit_flag;
    AscendC::Mmad(dst, a, b, params);
}

template<uint32_t rows, uint32_t cols>
__aicore__ inline void fixpipe_c310_full_l0c_to_aiv_ub(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t dst_cols, uint8_t unit_flag = 0b11) {
    AscendC::FixpipeParamsC310<CFG_ROW_MAJOR_TO_UB.format> params;
    params.nSize = cols;
    params.mSize = rows;
    params.srcStride = ceil_div(rows, MMAD_UNIT) * MMAD_UNIT;
    params.dstStride = dst_cols;
    params.unitFlag = unit_flag;
    params.dualDstCtl = 1;
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_TO_UB>(dst, src, params);
}

template<class T, uint32_t rows, uint32_t cols>
__aicore__ inline void copy_ub_to_gm_rows(
    AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t dst_stride) {
    AscendC::DataCopyParams params(1, static_cast<uint16_t>(cols * sizeof(T) / 32), 0, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        AscendC::DataCopy(dst[row * dst_stride], src[row * cols], params);
    }
}

template<uint32_t rows, uint32_t cols>
__aicore__ inline void fixpipe_l0c_to_gm_col_blocks(AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src, uint32_t dst_stride, uint8_t unit_flag) {
    static_assert(cols % MMAD_UNIT == 0);
    for (uint32_t col_block = 0; col_block < cols / MMAD_UNIT; ++col_block) {
        AscendC::FixpipeParamsArch3510<CFG_ROW_MAJOR_TO_GM.format> params;
        params.nSize = MMAD_UNIT;
        params.mSize = rows;
        params.srcStride = ceil_div(rows, MMAD_UNIT) * MMAD_UNIT;
        params.dstStride = dst_stride;
        params.quantPre = QuantMode_t::NoQuant;
        params.reluEn = false;
        params.unitFlag = unit_flag;
        params.dualDstCtl = 0;
        params.subBlockId = false;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_TO_GM>(
            dst[col_block * MMAD_UNIT], src[col_block * rows * MMAD_UNIT], params);
    }
}

}  // namespace SWA_v5::prim
