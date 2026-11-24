#pragma once

#include <cstdint>

#include "common.h"
#include "kernel_operator.h"
#include "math_utils.h"

namespace xpu_ops::kernels {

template <class T>
__aicore__ inline void aligned_datacopy_gm2ub(const uint32_t &rows, const uint32_t &cols, const uint32_t &src_stride,
                                              const uint32_t &dst_stride, const AscendC::GlobalTensor<T> &src,
                                              const AscendC::LocalTensor<T> &dst, T pad_val = 0) {
  constexpr int32_t elems_per_block = VEC_BLOCK_BYTES / sizeof(T);
  // if cols is not aligned with 32bytes, we may need pad some zeros in dst tensor
  if ((cols % elems_per_block == 0) && (src_stride % elems_per_block == 0) && (dst_stride % elems_per_block == 0)) {
    AscendC::DataCopyParams copy_params;
    copy_params.blockCount = rows;
    copy_params.blockLen = cols / elems_per_block;
    copy_params.srcStride = (src_stride - cols) / elems_per_block;
    copy_params.dstStride = (dst_stride - cols) / elems_per_block;
    DataCopy(dst, src, copy_params);
  } else {
    AscendC::DataCopyExtParams copy_params;
    copy_params.blockCount = rows;
    copy_params.blockLen = cols * sizeof(T);
    copy_params.srcStride = (src_stride - cols) * sizeof(T);
    copy_params.dstStride = (dst_stride - cols) / elems_per_block;

    AscendC::DataCopyPadExtParams<T> pad_params;
    uint32_t padding_size = xpu_ops::kernels::align_to(cols, elems_per_block) - cols;
    pad_params.isPad = padding_size != 0;
    pad_params.paddingValue = pad_val;
    pad_params.leftPadding = 0;
    pad_params.rightPadding = padding_size;
    DataCopyPad(dst, src, copy_params, pad_params);
  }
}

template <class T>
__aicore__ inline void aligned_datacopy_ub2gm(const uint32_t &rows, const uint32_t &cols, const uint32_t &src_stride,
                                              const uint32_t &dst_stride, const AscendC::LocalTensor<T> &src,
                                              const AscendC::GlobalTensor<T> &dst) {
  constexpr int32_t elems_per_block = VEC_BLOCK_BYTES / sizeof(T);

  if ((cols % elems_per_block == 0) && (src_stride % elems_per_block == 0) && (dst_stride % elems_per_block == 0)) {
    AscendC::DataCopyParams copy_params;
    copy_params.blockCount = rows;
    copy_params.blockLen = cols / elems_per_block;
    copy_params.srcStride = (src_stride - cols) / elems_per_block;
    copy_params.dstStride = (dst_stride - cols) / elems_per_block;
    DataCopy(dst, src, copy_params);
  } else {
    AscendC::DataCopyExtParams copy_params;
    copy_params.blockCount = rows;
    copy_params.blockLen = cols * sizeof(T);
    copy_params.srcStride = (src_stride - cols) / elems_per_block;
    copy_params.dstStride = (dst_stride - cols) * sizeof(T);
    copy_params.rsv = 0;
    DataCopyPad(dst, src, copy_params);
  }
}

template <class T>
__aicore__ inline bool align_matmul_input(const uint32_t &row, const uint32_t &src_col, const uint32_t &dst_col,
                                          const uint32_t &ub_size, const uint32_t &core_idx, const uint32_t &num_core,
                                          AscendC::TPipe &pipe, const AscendC::GlobalTensor<T> &src_gm,
                                          AscendC::GlobalTensor<T> &dst_gm) {
  uint32_t blk_row = 128;
  uint32_t blk_col = 512;
  while (blk_row * blk_col * sizeof(T) > ub_size) {
    blk_col >>= 1;
  }
  if (blk_col == 0) {
    return false;
  }

  AscendC::TBuf<> tmp_buf;
  pipe.InitBuffer(tmp_buf, ub_size);
  auto tmp_ub = tmp_buf.template Get<T>();

  uint32_t num_blk_row = xpu_ops::kernels::ceil_div(row, blk_row);
  uint32_t num_blk_col = xpu_ops::kernels::ceil_div(src_col, blk_col);

  uint32_t num_blk = num_blk_row * num_blk_col;
  uint32_t num_blk_per_core = xpu_ops::kernels::ceil_div(num_blk, num_core);

  uint32_t blk_base = core_idx * num_blk_per_core;
  for (uint32_t blk_idx = blk_base; blk_idx < num_blk && blk_idx - blk_base < num_blk_per_core; ++blk_idx) {
    uint32_t row_idx = blk_idx / num_blk_col * blk_row;
    uint32_t col_idx = blk_idx % num_blk_col * blk_col;

    uint32_t valid_blk_row = min(blk_row, row - row_idx);
    uint32_t valid_blk_col = min(blk_col, src_col - col_idx);

    uint32_t src_offset = row_idx * src_col + col_idx;
    uint32_t dst_offset = row_idx * dst_col + col_idx;
    auto curr_src_gm = src_gm[src_offset];
    auto curr_dst_gm = dst_gm[dst_offset];

    aligned_datacopy_gm2ub(valid_blk_row, valid_blk_col, src_col, blk_col, curr_src_gm, tmp_ub);

    {
      auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
      set_flag(PIPE_MTE2, PIPE_MTE3, event);
      wait_flag(PIPE_MTE2, PIPE_MTE3, event);
    }

    aligned_datacopy_ub2gm(valid_blk_row, valid_blk_col, blk_col, dst_col, tmp_ub, curr_dst_gm);
    {
      auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
      set_flag(PIPE_MTE3, PIPE_MTE2, event);
      wait_flag(PIPE_MTE3, PIPE_MTE2, event);
    }
  }

  return true;
}

}  // namespace xpu_ops::kernels
