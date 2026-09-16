#pragma once

#include <cstdint>
#include <type_traits>

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#if __has_include("adv_api/activation/softmaxflashv2.h")
#include "adv_api/activation/softmaxflashv2.h"
#else
#include "lib/activation/softmaxflashv2.h"
#endif
#include "varlen_fa_config.h"
#include "data_copy_utils.h"
#include "debug_utils.h"

namespace xpu_ops::kernels {

#define CUSTOM_FA_SOFTMAX_IMPL_CUSTOM 0
#define CUSTOM_FA_SOFTMAX_IMPL_CUSTOM_FUSED 1
#define CUSTOM_FA_SOFTMAX_IMPL_FLASH_V2 2

#define CUSTOM_FA_AGGREGATE_IMPL_CUSTOM 0
#define CUSTOM_FA_AGGREGATE_IMPL_CUSTOM_FUSED 1

#ifndef CUSTOM_FA_SOFTMAX_IMPL
#if CUSTOM_FA_ENABLE_SOFTMAX_FLASH_V2
#define CUSTOM_FA_SOFTMAX_IMPL CUSTOM_FA_SOFTMAX_IMPL_FLASH_V2
#elif CUSTOM_FA_ENABLE_SOFTMAX_FUSED_VEC_OPS
#define CUSTOM_FA_SOFTMAX_IMPL CUSTOM_FA_SOFTMAX_IMPL_CUSTOM_FUSED
#else
#define CUSTOM_FA_SOFTMAX_IMPL CUSTOM_FA_SOFTMAX_IMPL_CUSTOM
#endif
#endif

#ifndef CUSTOM_FA_SOFTMAX_FLASH_V2_FALLBACK_IMPL
#define CUSTOM_FA_SOFTMAX_FLASH_V2_FALLBACK_IMPL CUSTOM_FA_SOFTMAX_IMPL_CUSTOM_FUSED
#endif

#define CUSTOM_FA_SOFTMAX_BODY_IMPL CUSTOM_FA_SOFTMAX_IMPL
#if CUSTOM_FA_SOFTMAX_IMPL == CUSTOM_FA_SOFTMAX_IMPL_FLASH_V2
#undef CUSTOM_FA_SOFTMAX_BODY_IMPL
#define CUSTOM_FA_SOFTMAX_BODY_IMPL CUSTOM_FA_SOFTMAX_FLASH_V2_FALLBACK_IMPL
#endif

#ifndef CUSTOM_FA_AGGREGATE_IMPL
#if CUSTOM_FA_ENABLE_AGGREGATE_FUSED_VEC_OPS
#define CUSTOM_FA_AGGREGATE_IMPL CUSTOM_FA_AGGREGATE_IMPL_CUSTOM_FUSED
#else
#define CUSTOM_FA_AGGREGATE_IMPL CUSTOM_FA_AGGREGATE_IMPL_CUSTOM
#endif
#endif

template <uint32_t BLOCK_QO_, uint32_t HEAD_DIM_, bool IS_CAUSAL_ = true, uint32_t MAX_PIPELINE_STAGE_ = 3>
struct SoftmaxAndAggregateTileSize {
  static constexpr uint32_t BLOCK_QO = BLOCK_QO_;
  static constexpr uint32_t HEAD_DIM = HEAD_DIM_;
  static constexpr bool IS_CAUSAL = IS_CAUSAL_;
  static constexpr uint32_t MAX_PIPELINE_STAGE = MAX_PIPELINE_STAGE_;
};

namespace details {

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
constexpr uint32_t HALF_BLOCK_SIZE = 16;
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t UINT8_BLOCK_SIZE = 32;
constexpr uint32_t VECTOR_SIZE = 128;
constexpr AscendC::SoftmaxConfig CUSTOM_FA_SOFTMAX_REDUCE_CFG = {false, 0, 0};

CATLASS_DEVICE
void SetVecMask(int32_t len) {
  uint64_t mask = 0;
  uint64_t one = 1;
  uint64_t temp = len % FLOAT_VECTOR_SIZE;
  // for (int64_t i = 0; i < temp; i++) {
  //   mask |= one << i;
  // }
  mask = (one << temp) - 1;

  if (len == VECTOR_SIZE || len == 0) {
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
  } else if (len >= FLOAT_VECTOR_SIZE) {
    AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
  } else {
    AscendC::SetVectorMask<int8_t>(0x0, mask);
  }
}

CATLASS_DEVICE
void SetBlockReduceMask(int32_t len) {
  if (len > 8 || len < 1) {
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    return;
  }
  uint64_t sub_mask = ((uint64_t)1 << len) - 1;
  uint64_t mask_value = (sub_mask << 48) + (sub_mask << 32) + (sub_mask << 16) + sub_mask + (sub_mask << 56) +
                        (sub_mask << 40) + (sub_mask << 24) + (sub_mask << 8);
  AscendC::SetVectorMask<int8_t>(mask_value, mask_value);
}

template <class Element>
CATLASS_DEVICE void SetCounterMask(uint32_t element_count) {
  AscendC::SetMaskCount();
  AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(0, element_count);
}

CATLASS_DEVICE
void ResetToNormMask() {
  AscendC::SetMaskNorm();
  AscendC::ResetMask();
  AscendC::SetVectorMask<int8_t, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
}

CATLASS_DEVICE void Rowsum1024(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                               const AscendC::LocalTensor<float> &output,
                               const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  uint32_t aligned_rows = ((rows + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE) * FLOAT_BLOCK_SIZE;
  AscendC::BlockReduceSum<float, false>(workspace, input, aligned_rows * 16, AscendC::MASK_PLACEHOLDER, 1, 1,
                                        8);  // [r, 1024] -> [r, 128]
  AscendC::BlockReduceSum<float, false>(workspace, workspace, aligned_rows * 2, AscendC::MASK_PLACEHOLDER, 1, 1,
                                        8);  // [r, 128] -> [r, 16]
  AscendC::Add<float, false>(workspace, workspace, workspace[FLOAT_BLOCK_SIZE], AscendC::MASK_PLACEHOLDER,
                             aligned_rows / FLOAT_BLOCK_SIZE,
                             {1, 2, 2, 8, 16, 16});  // [r, 16] -> [r, 8]
  AscendC::BlockReduceSum<float, false>(output, workspace, aligned_rows / FLOAT_BLOCK_SIZE,
                                        AscendC::MASK_PLACEHOLDER, 1, 1, 8);  // [r, 8] -> [r, 1]
}

// 3 blockreduces: 512 -> 64 -> 8 -> 1
CATLASS_DEVICE
void Rowsum512Tree(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                   const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::BlockReduceSum<float, false>(workspace, input, rows * 8, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum<float, false>(workspace[rows * FLOAT_VECTOR_SIZE], workspace, rows, AscendC::MASK_PLACEHOLDER,
                                        1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum<float, false>(output, workspace[rows * FLOAT_VECTOR_SIZE], (rows + 7) / 8,
                                        AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
}

// one blockreduce and one wholereduce: 512 -> 64 -> 1
CATLASS_DEVICE
void Rowsum512(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::BlockReduceSum<float, false>(workspace, input, rows * 8, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceSum<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
}

// two adds and one wholereduce: 256 -> 128 -> 64 -> 1
CATLASS_DEVICE
void Rowsum256(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::Add<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows * 2,
                             AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::Add<float, false>(workspace, workspace, workspace[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                             AscendC::BinaryRepeatParams(1, 1, 1, 16, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceSum<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 16);
  // AscendC::PipeBarrier<PIPE_V>();
}

// one add and one wholereduce: 128 -> 64 -> 1
CATLASS_DEVICE
void Rowsum128(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::Add<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                             AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceSum<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void RowsumTail(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  uint32_t cols = layout.shape(1);
  uint32_t ldm = layout.stride(0);

  XPU_OPS_DEBUG_PRINT("    RowsumTail called with rows=%u, cols=%u, ldm=%u.\n", rows, cols, ldm);
  if (cols <= FLOAT_VECTOR_SIZE) {
    SetVecMask(cols);
    AscendC::WholeReduceSum<float, false>(output, input, AscendC::MASK_PLACEHOLDER, rows, 1, 1, ldm / FLOAT_BLOCK_SIZE);
    // AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    XPU_OPS_DEBUG_PRINT("      used single wholereduce.\n");
  } else if (cols <= 2 * FLOAT_VECTOR_SIZE) {
    AscendC::Adds<float, false>(workspace, input, 0.0f, AscendC::MASK_PLACEHOLDER, rows,
                                AscendC::UnaryRepeatParams(1, 1, 8, ldm / FLOAT_BLOCK_SIZE));
    // AscendC::PipeBarrier<PIPE_V>();
    SetVecMask(cols - FLOAT_VECTOR_SIZE);
    AscendC::Add<float, false>(workspace, workspace, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                               AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, ldm / FLOAT_BLOCK_SIZE));
    // AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    AscendC::WholeReduceSum<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8);
    // AscendC::PipeBarrier<PIPE_V>();
    XPU_OPS_DEBUG_PRINT("      used one add and one wholereduce.\n");
  } else {
    uint32_t block_ldm = ldm / FLOAT_BLOCK_SIZE;
    uint32_t repeat_num = cols / FLOAT_VECTOR_SIZE;
    AscendC::Add<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                               AscendC::BinaryRepeatParams(1, 1, 1, 8, block_ldm, block_ldm));
    // AscendC::PipeBarrier<PIPE_V>();
    XPU_OPS_DEBUG_PRINT("      used first add of %u adds.\n", repeat_num - 1);
    for (uint32_t i = 2; i < repeat_num; i++) {
      AscendC::Add<float, false>(workspace, workspace, input[i * FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, block_ldm));
      // AscendC::PipeBarrier<PIPE_V>();
      XPU_OPS_DEBUG_PRINT("      used add %u of %u adds.\n", i - 1, repeat_num - 1);
    }
    uint32_t tail = cols % FLOAT_VECTOR_SIZE;
    if (tail > 0) {
      SetVecMask(tail);
      AscendC::Add<float, false>(workspace, workspace, input[repeat_num * FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER,
                                 rows, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, block_ldm));
      // AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
      XPU_OPS_DEBUG_PRINT("      used last add with tail %u of %u adds.\n", tail, repeat_num - 1);
    }

    AscendC::WholeReduceSum<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8);
    // AscendC::PipeBarrier<PIPE_V>();
    XPU_OPS_DEBUG_PRINT("      used one wholereduce.\n");
  }
}

CATLASS_DEVICE
void Rowsum(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
            const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t cols = layout.shape(1);
  if (cols == 1024) {
    Rowsum1024(input, layout, output, workspace);
  } else if (cols == 512) {
    Rowsum512Tree(input, layout, output, workspace);
  } else if (cols == 256) {
    Rowsum256(input, layout, output, workspace);
  } else if (cols == 128) {
    Rowsum128(input, layout, output, workspace);
  } else {
    RowsumTail(input, layout, output, workspace);
  }
}

CATLASS_DEVICE void Rowmax1024(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                               const AscendC::LocalTensor<float> &output,
                               const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  uint32_t aligned_rows = ((rows + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE) * FLOAT_BLOCK_SIZE;
  AscendC::BlockReduceMax<float, false>(workspace, input, aligned_rows * 16, AscendC::MASK_PLACEHOLDER, 1, 1,
                                        8);  // [r, 1024] -> [r, 128]
  AscendC::BlockReduceMax<float, false>(workspace, workspace, aligned_rows * 2, AscendC::MASK_PLACEHOLDER, 1, 1,
                                        8);  // [r, 128] -> [r, 16]
  AscendC::Max<float, false>(workspace, workspace, workspace[FLOAT_BLOCK_SIZE], AscendC::MASK_PLACEHOLDER,
                             aligned_rows / FLOAT_BLOCK_SIZE,
                             {1, 2, 2, 8, 16, 16});  // [r, 16] -> [r, 8]
  AscendC::BlockReduceMax<float, false>(output, workspace, aligned_rows / FLOAT_BLOCK_SIZE,
                                        AscendC::MASK_PLACEHOLDER, 1, 1, 8);  // [r, 8] -> [r, 1]
}

// 3 blockreduces: 512 -> 64 -> 8 -> 1
CATLASS_DEVICE
void Rowmax512Tree(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                   const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::BlockReduceMax<float, false>(workspace, input, rows * 8, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceMax<float, false>(workspace[rows * FLOAT_VECTOR_SIZE], workspace, rows, AscendC::MASK_PLACEHOLDER,
                                        1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceMax<float, false>(output, workspace[rows * FLOAT_VECTOR_SIZE], (rows + 7) / 8,
                                        AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void Rowmax512(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::BlockReduceMax<float, false>(workspace, input, rows * 8, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceMax<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8,
                                        AscendC::ReduceOrder::ORDER_ONLY_VALUE);
  // AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void Rowmax256(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::Max<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows * 2,
                             AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::Max<float, false>(workspace, workspace, workspace[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                             AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceMax<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 16,
                                        AscendC::ReduceOrder::ORDER_ONLY_VALUE);
  // AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void Rowmax128(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
               const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  AscendC::Max<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                             AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
  // AscendC::PipeBarrier<PIPE_V>();
  AscendC::WholeReduceMax<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8,
                                        AscendC::ReduceOrder::ORDER_ONLY_VALUE);
  // AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void RowmaxTail(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
                const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t rows = layout.shape(0);
  uint32_t cols = layout.shape(1);
  uint32_t ldm = layout.stride(0);

  if (cols <= FLOAT_VECTOR_SIZE) {
    SetVecMask(cols);
    AscendC::WholeReduceMax<float, false>(output, input, AscendC::MASK_PLACEHOLDER, rows, 1, 1, ldm / FLOAT_BLOCK_SIZE,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    // AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
  } else if (cols <= 2 * FLOAT_VECTOR_SIZE) {
    AscendC::Adds<float, false>(workspace, input, 0.0f, AscendC::MASK_PLACEHOLDER, rows,
                                AscendC::UnaryRepeatParams(1, 1, 8, ldm / FLOAT_BLOCK_SIZE));
    // AscendC::PipeBarrier<PIPE_V>();
    SetVecMask(cols - FLOAT_VECTOR_SIZE);
    AscendC::Max<float, false>(workspace, workspace, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                               AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, ldm / FLOAT_BLOCK_SIZE));
    // AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    AscendC::WholeReduceMax<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    // AscendC::PipeBarrier<PIPE_V>();
  } else {
    uint32_t block_ldm = ldm / FLOAT_BLOCK_SIZE;
    uint32_t repeat_num = cols / FLOAT_VECTOR_SIZE;
    AscendC::Max<float, false>(workspace, input, input[FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                               AscendC::BinaryRepeatParams(1, 1, 1, 8, block_ldm, block_ldm));
    // AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 2; i < repeat_num; i++) {
      AscendC::Max<float, false>(workspace, workspace, input[i * FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER, rows,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, block_ldm));
      // AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t tail = cols % FLOAT_VECTOR_SIZE;
    if (tail > 0) {
      SetVecMask(tail);
      AscendC::Max<float, false>(workspace, workspace, input[repeat_num * FLOAT_VECTOR_SIZE], AscendC::MASK_PLACEHOLDER,
                                 rows, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, block_ldm));
      // AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    AscendC::WholeReduceMax<float, false>(output, workspace, AscendC::MASK_PLACEHOLDER, rows, 1, 1, 8,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    // AscendC::PipeBarrier<PIPE_V>();
  }
}

CATLASS_DEVICE
void Rowmax(const AscendC::LocalTensor<float> &input, const Catlass::layout::RowMajor &layout,
            const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &workspace) {
  uint32_t cols = layout.shape(1);
  if (cols == 1024) {
    Rowmax1024(input, layout, output, workspace);
  } else if (cols == 512) {
    Rowmax512Tree(input, layout, output, workspace);
  } else if (cols == 256) {
    Rowmax256(input, layout, output, workspace);
  } else if (cols == 128) {
    Rowmax128(input, layout, output, workspace);
  } else {
    RowmaxTail(input, layout, output, workspace);
  }
}

}  // namespace details

template <class ArchTag_, class ElementInOut_, class ElementCalc_, class ElementMask_, class TileSize_, class EventSet_>
class SoftmaxAndAggregate {
public:
  using ArchTag = ArchTag_;
  using ElementInOut = ElementInOut_;
  using ElementCalc = ElementCalc_;
  using ElementMask = ElementMask_;
  using TileSize = TileSize_;
  using EventSet = EventSet_;
  using Layout = Catlass::layout::RowMajor;

  static constexpr bool IS_CAUSAL = TileSize::IS_CAUSAL;

  static constexpr uint32_t QK_READY = EventSet::QK_READY;
  static constexpr uint32_t SCORE_READY = EventSet::SCORE_READY;
  static constexpr uint32_t PV_READY = EventSet::PV_READY;

  static constexpr uint32_t UB_STAGE_NUM = 2;
  static constexpr uint32_t NUM_VEC_CORES = 2;
  static constexpr uint32_t NUM_ROWS_ALIGNED = 8;
  static constexpr uint32_t NUM_COLS_ALIGNED = 16;
  static constexpr uint32_t NUM_COLS_ALIGNED_MASK = details::UINT8_BLOCK_SIZE;
  static constexpr uint32_t BLOCK_QO = TileSize::BLOCK_QO;
  static constexpr uint32_t VEC_BLOCK_QO = BLOCK_QO / NUM_VEC_CORES;
  static constexpr uint32_t HEAD_DIM = TileSize::HEAD_DIM;
  static constexpr uint32_t MAX_PIPELINE_STAGE = TileSize::MAX_PIPELINE_STAGE;
  static constexpr bool SOFTMAX_USE_COUNTER_MASK = CUSTOM_FA_SOFTMAX_USE_COUNTER_MASK != 0;
  static constexpr bool AGGREGATE_USE_COUNTER_MASK = CUSTOM_FA_AGGREGATE_USE_COUNTER_MASK != 0;
  static constexpr bool SOFTMAX_FLASH_V2_BASIC_BLOCK = CUSTOM_FA_SOFTMAX_FLASH_V2_BASIC_BLOCK != 0;

  // static_assert(MAX_PIPELINE_STAGE == 1, "Pipeline stage must be 1");

  static constexpr uint32_t DOUBLE_INPUT_BUF_MAX_SIZE = 32 * 1024;
  static constexpr uint32_t SINGLE_INPUT_BUF_MAX_SIZE = 16 * 1024;
  static constexpr uint32_t O_BUF_MAX_SIZE = 32 * 1024;
  static constexpr uint32_t WORKSPACE_MAX_SIZE = 32 * 1024;
  static constexpr uint32_t OUTPUT_BUF_MAX_SIZE = 16 * 1024;
  static constexpr uint32_t FA_STATE_MAX_SIZE = 1024;

  static_assert(DOUBLE_INPUT_BUF_MAX_SIZE * UB_STAGE_NUM + SINGLE_INPUT_BUF_MAX_SIZE + O_BUF_MAX_SIZE +
                        WORKSPACE_MAX_SIZE + OUTPUT_BUF_MAX_SIZE + FA_STATE_MAX_SIZE * (4 + 3 * MAX_PIPELINE_STAGE) <=
                    ArchTag::UB_SIZE,
                "UB size is not enough");

  static constexpr uint32_t INPUT_QK_MAX_ELEMENTS = DOUBLE_INPUT_BUF_MAX_SIZE / sizeof(ElementCalc);
  static constexpr uint32_t INPUT_PV_MAX_ELEMENTS = DOUBLE_INPUT_BUF_MAX_SIZE / sizeof(ElementCalc);
  static constexpr uint32_t INPUT_MASK_MAX_ELEMENTS = SINGLE_INPUT_BUF_MAX_SIZE / sizeof(ElementMask);
  static constexpr uint32_t O_MAX_ELEMENTS = VEC_BLOCK_QO * HEAD_DIM;

  static_assert(INPUT_MASK_MAX_ELEMENTS >= INPUT_QK_MAX_ELEMENTS, "mask buffer size is not enough");
  static_assert(INPUT_MASK_MAX_ELEMENTS * sizeof(half) <= WORKSPACE_MAX_SIZE,
                "workspace size is not enough for casted mask");
  static_assert(O_MAX_ELEMENTS * sizeof(ElementCalc) <= O_BUF_MAX_SIZE, "o buffer size is not enough");
  static_assert(O_MAX_ELEMENTS * sizeof(ElementInOut) <= OUTPUT_BUF_MAX_SIZE,
                "workspace size is not enough for casted o");
  static_assert(INPUT_QK_MAX_ELEMENTS * sizeof(ElementInOut) <= OUTPUT_BUF_MAX_SIZE,
                "workspace split 0 size is not enough for casted input");
  static_assert(VEC_BLOCK_QO * sizeof(ElementCalc) <= FA_STATE_MAX_SIZE, "FA state size is not enough");

  struct Params {
    uint32_t tile_kv, pipeline_stages;
    float scale;
  };

  CATLASS_DEVICE
  SoftmaxAndAggregate(const Params &params, Catlass::Arch::Resource<ArchTag> &resource) {
    constexpr uint32_t SINGLE_INPUT_BUF_OFFSET = DOUBLE_INPUT_BUF_MAX_SIZE * UB_STAGE_NUM;  // 64K
    constexpr uint32_t O_BUF_OFFSET = SINGLE_INPUT_BUF_OFFSET + SINGLE_INPUT_BUF_MAX_SIZE;  // 80K
    constexpr uint32_t WORKSPACE_OFFSET = O_BUF_OFFSET + O_BUF_MAX_SIZE;                    // 112K
    constexpr uint32_t OUTPUT_BUF_OFFSET = WORKSPACE_OFFSET + WORKSPACE_MAX_SIZE;           // 144K
    constexpr uint32_t FA_STATE_OFFSET = OUTPUT_BUF_OFFSET + OUTPUT_BUF_MAX_SIZE;           // 160K

    tile_kv = params.tile_kv;
    pipeline_stages = params.pipeline_stages;
    single_qk_size = BLOCK_QO * tile_kv;
    single_score_size = BLOCK_QO * tile_kv;
    single_pv_size = BLOCK_QO * HEAD_DIM;
    scale = params.scale;
    for (int i = 0; i < UB_STAGE_NUM; ++i) {
      input_bufs[i] = resource.ubBuf.template GetBufferByByte<ElementCalc>(i * DOUBLE_INPUT_BUF_MAX_SIZE);
      input_events[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[i]);
    }

    input_mask_buf = resource.ubBuf.template GetBufferByByte<ElementMask>(SINGLE_INPUT_BUF_OFFSET);
    input_mask_event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_mask_event);

    // they share the same workspace
    casted_mask_buf = resource.ubBuf.template GetBufferByByte<ElementCalc>(WORKSPACE_OFFSET);
    reduce_workspace_buf = resource.ubBuf.template GetBufferByByte<ElementCalc>(WORKSPACE_OFFSET);
    flatten_workspace_buf = resource.ubBuf.template GetBufferByByte<ElementCalc>(WORKSPACE_OFFSET);

    contiguous_mask_buf0 = resource.ubBuf.template GetBufferByByte<ElementMask>(WORKSPACE_OFFSET);
    casted_mask_buf1 = resource.ubBuf.template GetBufferByByte<half>(WORKSPACE_OFFSET + WORKSPACE_MAX_SIZE / 2);
    bit_mask_buf0 = resource.ubBuf.template GetBufferByByte<uint8_t>(WORKSPACE_OFFSET);

    local_max = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET);
    // global_max = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE);
    local_exp_sum = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * 1);
    // global_exp_sum = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * 3);
    tmp_max = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * 2);
    scale_dup_buf = resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * 3);
    for (uint32_t i = 0; i < MAX_PIPELINE_STAGE; ++i) {
      global_scaler_bufs[i] =
          resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * (4 + 3 * i));
      global_exp_sum_bufs[i] =
          resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * (5 + 3 * i));
      global_max_bufs[i] =
          resource.ubBuf.template GetBufferByByte<ElementCalc>(FA_STATE_OFFSET + FA_STATE_MAX_SIZE * (6 + 3 * i));
    }

    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
#if CUSTOM_FA_HOIST_SOFTMAX_SCALE_DUP
    AscendC::Duplicate<ElementCalc, false>(scale_dup_buf, scale, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
#endif

    o_state_buf = resource.ubBuf.template GetBufferByByte<ElementCalc>(O_BUF_OFFSET);
    output_buf = resource.ubBuf.template GetBufferByByte<ElementInOut>(OUTPUT_BUF_OFFSET);
    output_event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(output_event);

    XPU_OPS_DEBUG_PRINT("SoftmaxAndAggregate: initialized with tile_kv=%u, pipeline_stages=%u, scale=%f.\n", tile_kv,
                        pipeline_stages, scale);
  }

  CATLASS_DEVICE
  ~SoftmaxAndAggregate() {
    for (int i = 0; i < UB_STAGE_NUM; ++i) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_events[i]);
      GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(input_events[i]);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_mask_event);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(input_mask_event);

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(output_event);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(output_event);

    XPU_OPS_DEBUG_PRINT("SoftmaxAndAggregate: destroyed.\n");
  }

  CATLASS_DEVICE
  void operator()(const AscendC::GlobalTensor<ElementCalc> &qk_pingpong_gm,
                  const AscendC::GlobalTensor<ElementMask> &mask_gm, const Layout &layout_mask,
                  const AscendC::GlobalTensor<ElementInOut> &score_pingpong_gm,
                  const AscendC::GlobalTensor<ElementCalc> &pv_pingpong_gm,
                  const AscendC::GlobalTensor<ElementInOut> &o_gm, const Layout &layout_o, uint32_t start_qo_pos) {
    uint32_t valid_len_qo = layout_mask.shape(0), valid_len_kv = layout_mask.shape(1);

    // split task to vector cores
    uint32_t sub_core_num = AscendC::GetSubBlockNum();
    uint32_t core_id = AscendC::GetSubBlockIdx();
    uint32_t avg_len_qo = valid_len_qo / sub_core_num;
    uint32_t sub_core_start_qo_pos = core_id * avg_len_qo;
    uint32_t sub_core_len_qo = (core_id == sub_core_num - 1) ? (valid_len_qo - sub_core_start_qo_pos) : avg_len_qo;

    uint32_t num_tile_kv = (valid_len_kv + tile_kv - 1) / tile_kv;

    XPU_OPS_DEBUG_PRINT(
        "SoftmaxAndAggregate: start processing with valid_len_qo=%u, valid_len_kv=%u, sub_core_num=%u, core_id=%u, "
        "sub_core_start_qo_pos=%u, sub_core_len_qo=%u, num_tile_kv=%u.\n",
        valid_len_qo, valid_len_kv, sub_core_num, core_id, sub_core_start_qo_pos, sub_core_len_qo, num_tile_kv);

    uint32_t preload_num = pipeline_stages - 1;
    preload_num = preload_num < num_tile_kv ? preload_num : num_tile_kv;
    // for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; ++tile_kv_idx) {
    //   bool first_tile_kv = tile_kv_idx == 0;
    //   bool last_tile_kv = tile_kv_idx == num_tile_kv - 1;
    //   uint32_t start_kv_pos = tile_kv_idx * tile_kv;
    //   uint32_t valid_tile_kv = last_tile_kv ? valid_len_kv - tile_kv_idx * tile_kv : tile_kv;

    //   auto layout_qk =
    //       Layout::template MakeLayoutInUb<ElementCalc>(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));
    //   XPU_OPS_DEBUG_PRINT("    layout_qk = (%u, %u): (%u, 1).\n", layout_qk.shape(0), layout_qk.shape(1),
    //                       layout_qk.stride(0));
    //   auto layout_tile_mask = layout_mask.GetTileLayout(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));
    //   auto layout_tile_score =
    //       Layout::template MakeLayoutInUb<ElementInOut>(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));

    //   AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(QK_READY);

    //   XPU_OPS_DEBUG_PRINT("  QK[%u] ready.\n", tile_kv_idx);

    //   uint32_t pingpong_id = tile_kv_idx % pipeline_stages;
    //   uint32_t qk_pingpong_offset = pingpong_id * single_qk_size + sub_core_start_qo_pos * layout_qk.stride(0);
    //   uint32_t score_pingpong_offset =
    //       pingpong_id * single_score_size + sub_core_start_qo_pos * layout_tile_score.stride(0);
    //   uint32_t mask_offset = layout_mask.GetOffset(Catlass::MatrixCoord(sub_core_start_qo_pos, start_kv_pos));

    //   if (sub_core_len_qo > 0) {
    //     XPU_OPS_DEBUG_PRINT(
    //         "  processing Softmax[%u], pingpong_id = %u, qk_pingpong_offset = %u, score_pingpong_offset = %u.\n",
    //         tile_kv_idx, pingpong_id, qk_pingpong_offset, score_pingpong_offset);
    //     subCoreOnlineSoftmax(qk_pingpong_gm[qk_pingpong_offset], layout_qk, mask_gm[mask_offset], layout_tile_mask,
    //                          score_pingpong_gm[score_pingpong_offset], layout_tile_score,
    //                          start_qo_pos + sub_core_start_qo_pos, start_kv_pos, pingpong_id, first_tile_kv,
    //                          last_tile_kv);
    //     XPU_OPS_DEBUG_PRINT("  Softmax[%u] done.\n", tile_kv_idx);
    //   }

    //   AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SCORE_READY);

    //   XPU_OPS_DEBUG_PRINT("  Score[%u] ready.\n", tile_kv_idx);

    //   AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(PV_READY);

    //   XPU_OPS_DEBUG_PRINT("  PV[%u] ready.\n", tile_kv_idx);

    //   auto layout_pv = Layout::template MakeLayoutInUb<ElementCalc>(Catlass::MatrixCoord(sub_core_len_qo, HEAD_DIM));
    //   auto layout_tile_o = layout_o.GetTileLayout(Catlass::MatrixCoord(sub_core_len_qo, HEAD_DIM));
    //   uint32_t pv_pingpong_offset = pingpong_id * single_pv_size + sub_core_start_qo_pos * layout_pv.stride(0);

    //   if (sub_core_len_qo > 0) {
    //     XPU_OPS_DEBUG_PRINT("  processing Aggregate[%u].\n", tile_kv_idx);
    //     subCoreAggregate(pv_pingpong_gm[pv_pingpong_offset], layout_pv,
    //                      o_gm[sub_core_start_qo_pos * layout_o.stride(0)], layout_tile_o, pingpong_id, first_tile_kv,
    //                      last_tile_kv);
    //     XPU_OPS_DEBUG_PRINT("  Aggregate[%u] done.\n", tile_kv_idx);
    //   }
    // }
    for (uint32_t iter = 0; iter < num_tile_kv + preload_num; ++iter) {
      if (iter < num_tile_kv) {
        uint32_t tile_kv_idx = iter;
        bool first_tile_kv = tile_kv_idx == 0;
        bool last_tile_kv = tile_kv_idx == num_tile_kv - 1;
        uint32_t start_kv_pos = tile_kv_idx * tile_kv;
        uint32_t valid_tile_kv = last_tile_kv ? valid_len_kv - tile_kv_idx * tile_kv : tile_kv;

        auto layout_qk =
            Layout::template MakeLayoutInUb<ElementCalc>(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));
        XPU_OPS_DEBUG_PRINT("    layout_qk = (%u, %u): (%u, 1).\n", layout_qk.shape(0), layout_qk.shape(1),
                            layout_qk.stride(0));
        auto layout_tile_mask = layout_mask.GetTileLayout(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));
        auto layout_tile_score =
            Layout::template MakeLayoutInUb<ElementInOut>(Catlass::MatrixCoord(sub_core_len_qo, valid_tile_kv));

        uint32_t pingpong_id = tile_kv_idx % pipeline_stages;
        uint32_t qk_pingpong_offset = pingpong_id * single_qk_size + sub_core_start_qo_pos * layout_qk.stride(0);
        uint32_t score_pingpong_offset =
            pingpong_id * single_score_size + sub_core_start_qo_pos * layout_tile_score.stride(0);
        uint32_t mask_offset = layout_mask.GetOffset(Catlass::MatrixCoord(sub_core_start_qo_pos, start_kv_pos));

        XPU_OPS_DEBUG_PRINT(
            "  processing Softmax[%u], pingpong_id = %u, qk_pingpong_offset = %u, score_pingpong_offset = %u.\n",
            tile_kv_idx, pingpong_id, qk_pingpong_offset, score_pingpong_offset);
        subCoreOnlineSoftmax(qk_pingpong_gm[qk_pingpong_offset], layout_qk, mask_gm[mask_offset], layout_tile_mask,
                             score_pingpong_gm[score_pingpong_offset], layout_tile_score,
                             start_qo_pos + sub_core_start_qo_pos, start_kv_pos, pingpong_id, 0, first_tile_kv,
                             last_tile_kv, true);
        XPU_OPS_DEBUG_PRINT("  Softmax[%u] done.\n", tile_kv_idx);

        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SCORE_READY);
        XPU_OPS_DEBUG_PRINT("  Score[%u] ready.\n", tile_kv_idx);
      }

      if (iter >= preload_num) {
        uint32_t tile_kv_idx = iter - preload_num;
        bool first_tile_kv = tile_kv_idx == 0;
        bool last_tile_kv = tile_kv_idx == num_tile_kv - 1;
        uint32_t start_kv_pos = tile_kv_idx * tile_kv;
        uint32_t valid_tile_kv = last_tile_kv ? valid_len_kv - tile_kv_idx * tile_kv : tile_kv;

        auto layout_pv = Layout::template MakeLayoutInUb<ElementCalc>(Catlass::MatrixCoord(sub_core_len_qo, HEAD_DIM));
        auto layout_tile_o = layout_o.GetTileLayout(Catlass::MatrixCoord(sub_core_len_qo, HEAD_DIM));
        uint32_t pingpong_id = tile_kv_idx % pipeline_stages;
        uint32_t pv_pingpong_offset = pingpong_id * single_pv_size + sub_core_start_qo_pos * layout_pv.stride(0);

        XPU_OPS_DEBUG_PRINT("  processing Aggregate[%u].\n", tile_kv_idx);
        subCoreAggregate(pv_pingpong_gm[pv_pingpong_offset], layout_pv,
                         o_gm[sub_core_start_qo_pos * layout_o.stride(0)], layout_tile_o, pingpong_id, 0, first_tile_kv,
                         last_tile_kv);
        XPU_OPS_DEBUG_PRINT("  Aggregate[%u] done.\n", tile_kv_idx);
      }
    }
  }

  CATLASS_DEVICE
  void maskPaddedColumns(const AscendC::LocalTensor<ElementCalc> &input_qk, uint32_t num_rows,
                         uint32_t valid_cols, uint32_t padded_cols) {
    if (valid_cols >= padded_cols) {
      return;
    }
    constexpr uint32_t VECTOR_COLS = details::FLOAT_VECTOR_SIZE;
    uint32_t first_block = valid_cols / VECTOR_COLS;
    uint32_t first_valid_offset = valid_cols % VECTOR_COLS;
    for (uint32_t row = 0; row < num_rows; ++row) {
      uint32_t block = first_block;
      if (first_valid_offset != 0) {
        uint64_t tail_mask = ~((static_cast<uint64_t>(1) << first_valid_offset) - 1);
        AscendC::SetVectorMask<int8_t>(0, tail_mask);
        AscendC::Duplicate<ElementCalc, false>(input_qk[row * padded_cols + block * VECTOR_COLS],
                                               static_cast<ElementCalc>(-1e20f), AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        ++block;
      }
      AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
      for (; block < padded_cols / VECTOR_COLS; ++block) {
        AscendC::Duplicate<ElementCalc, false>(input_qk[row * padded_cols + block * VECTOR_COLS],
                                               static_cast<ElementCalc>(-1e20f), AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      }
    }
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
  }

  template <bool HANDLE_MASK>
  CATLASS_DEVICE
  void subCoreOnlineSoftmaxImpl(const AscendC::GlobalTensor<ElementCalc> &input_qk_gm, const Layout &layout_qk,
                                const AscendC::GlobalTensor<ElementMask> &input_mask_gm, const Layout &layout_mask,
                                const AscendC::GlobalTensor<ElementInOut> &output_score_gm, const Layout &layout_score,
                                uint32_t start_qo_pos, uint32_t start_kv_pos, uint32_t scaler_pingpong_id,
                                uint32_t state_pingpong_idx, bool first_tile_kv, bool last_tile_kv,
                                bool tile_has_partial_mask, uint32_t valid_cols) {
    uint32_t num_rows = layout_qk.shape(0), num_cols = layout_qk.shape(1);
    uint32_t valid_rows = layout_mask.shape(0);
    uint32_t num_cols_aligned = num_cols;
    uint32_t num_cols_aligned_mask = num_cols;
    uint32_t num_rows_per_iter = INPUT_QK_MAX_ELEMENTS / num_cols_aligned;
    num_rows_per_iter = (num_rows_per_iter / NUM_ROWS_ALIGNED) * NUM_ROWS_ALIGNED;
    uint32_t num_row_iters = (num_rows + num_rows_per_iter - 1) / num_rows_per_iter;

    auto global_max = global_max_bufs[state_pingpong_idx];
    auto global_exp_sum = global_exp_sum_bufs[state_pingpong_idx];
    auto global_scaler = global_scaler_bufs[scaler_pingpong_id];
    // if (num_rows_per_iter == 1) {
    //   num_rows_per_iter = (num_rows / 2 + NUM_ROWS_ALIGNED - 1) / NUM_ROWS_ALIGNED * NUM_ROWS_ALIGNED;
    //   num_row_iters = (num_rows + num_rows_per_iter - 1) / num_rows_per_iter;
    // }

    uint32_t k_end = start_kv_pos + valid_cols;

    XPU_OPS_DEBUG_PRINT(
        "  OnlineSoftmax: num_rows=%u, num_cols=%u, num_cols_aligned=%u, num_rows_per_iter=%u, num_row_iters=%u, "
        "k_end=%u.\n",
        num_rows, num_cols, num_cols_aligned, num_rows_per_iter, num_row_iters, k_end);
    for (uint32_t row_iter = 0; row_iter < num_row_iters; ++row_iter) {
      uint32_t row_base = row_iter * num_rows_per_iter;
      uint32_t valid_num_rows = (row_iter == num_row_iters - 1) ? (num_rows - row_base) : num_rows_per_iter;
      uint32_t num_rows_aligned = (valid_num_rows + NUM_ROWS_ALIGNED - 1) / NUM_ROWS_ALIGNED * NUM_ROWS_ALIGNED;
      uint32_t mask_num_rows =
          row_base < valid_rows ? ((valid_rows - row_base < valid_num_rows) ? valid_rows - row_base : valid_num_rows) : 0;
      uint32_t q_start = row_base + start_qo_pos;
      bool need_mask =
          mask_num_rows > 0 && HANDLE_MASK && tile_has_partial_mask &&
          (CUSTOM_FA_FULL_SKIP_MASK ? IS_CAUSAL : true) && (!IS_CAUSAL || (q_start < k_end));
      if (row_iter == 0) {
        // in first iter, we can load mask first
        if constexpr (HANDLE_MASK) {
          if (need_mask) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_mask_event);
            aligned_datacopy_gm2ub(mask_num_rows, valid_cols, layout_mask.stride(0), num_cols_aligned_mask,
                                   input_mask_gm[row_base * layout_mask.stride(0)], input_mask_buf);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_mask_event);

            XPU_OPS_DEBUG_PRINT("    loaded first mask\n");
          }
        }
        AscendC::CrossCoreWaitFlag(QK_READY);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        aligned_datacopy_gm2ub(valid_num_rows, num_cols, layout_qk.stride(0), num_cols_aligned,
                               input_qk_gm[row_base * layout_qk.stride(0)], input_bufs[input_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);

        XPU_OPS_DEBUG_PRINT("    loaded first qk.\n");
      }

      if (row_iter < num_row_iters - 1) {
        uint32_t next_input_event_id = (input_event_id + 1) % UB_STAGE_NUM;
        uint32_t next_row_base = (row_iter + 1) * num_rows_per_iter;
        uint32_t next_valid_num_rows =
            (row_iter + 1 == num_row_iters - 1) ? (num_rows - next_row_base) : num_rows_per_iter;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_events[next_input_event_id]);
        aligned_datacopy_gm2ub(next_valid_num_rows, num_cols, layout_qk.stride(0), num_cols_aligned,
                               input_qk_gm[next_row_base * layout_qk.stride(0)], input_bufs[next_input_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_events[next_input_event_id]);

        XPU_OPS_DEBUG_PRINT("    loaded next qk.\n");
      }

      auto input_qk = input_bufs[input_event_id];
      if constexpr (HANDLE_MASK) {
        if (need_mask) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_mask_event);
          if (num_cols_aligned_mask == num_cols_aligned) {
            AscendC::Cast<half, ElementMask, false>(
                casted_mask_buf1, input_mask_buf, AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER,
                (mask_num_rows * num_cols_aligned + details::HALF_VECTOR_SIZE - 1) / details::HALF_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 8, 4));
            XPU_OPS_DEBUG_PRINT("    casted mask.\n");
          } else {
            uint64_t rsvd_cnt = 0;
            AscendC::GatherMask<uint16_t>(
                contiguous_mask_buf0.template ReinterpretCast<uint16_t>(),
                input_mask_buf.template ReinterpretCast<uint16_t>(), 7, true,
                num_cols_aligned * sizeof(ElementMask) / sizeof(uint16_t),
                AscendC::GatherMaskParams(1, valid_num_rows, num_cols_aligned_mask / details::UINT8_BLOCK_SIZE, 0),
                rsvd_cnt);
            XPU_OPS_DEBUG_PRINT("    made contiguous mask.\n");
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_mask_event);
        }

        if (row_iter < num_row_iters - 1) {
          uint32_t next_row_base = (row_iter + 1) * num_rows_per_iter;
          uint32_t next_valid_num_rows =
              (row_iter + 1 == num_row_iters - 1) ? (num_rows - next_row_base) : num_rows_per_iter;
          uint32_t next_mask_num_rows =
              next_row_base < valid_rows
                  ? ((valid_rows - next_row_base < next_valid_num_rows) ? valid_rows - next_row_base
                                                                       : next_valid_num_rows)
                  : 0;
          bool next_need_mask = next_mask_num_rows > 0 && tile_has_partial_mask &&
                                (CUSTOM_FA_FULL_SKIP_MASK ? IS_CAUSAL : true) &&
                                (!IS_CAUSAL || (next_row_base + start_qo_pos < k_end));
          if (next_need_mask) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_mask_event);
            aligned_datacopy_gm2ub(next_mask_num_rows, valid_cols, layout_mask.stride(0), num_cols_aligned_mask,
                                   input_mask_gm[next_row_base * layout_mask.stride(0)], input_mask_buf);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_mask_event);
          }

          XPU_OPS_DEBUG_PRINT("    loaded next mask.\n");
        }

        if (need_mask) {
          if (num_cols_aligned_mask != num_cols_aligned) {
            AscendC::Cast<half, ElementMask, false>(
                casted_mask_buf1, contiguous_mask_buf0, AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER,
                (mask_num_rows * num_cols_aligned + details::HALF_VECTOR_SIZE - 1) / details::HALF_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 8, 4));
            XPU_OPS_DEBUG_PRINT("    casted mask.\n");
          }
          AscendC::CompareScalar<half, uint8_t, false>(
              bit_mask_buf0, casted_mask_buf1, static_cast<half>(0), AscendC::CMPMODE::NE,
              AscendC::MASK_PLACEHOLDER,
              (mask_num_rows * num_cols_aligned + details::HALF_VECTOR_SIZE - 1) / details::HALF_VECTOR_SIZE,
              AscendC::UnaryRepeatParams(1, 1, 8, 8));
          XPU_OPS_DEBUG_PRINT("    made bit mask.\n");
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);

          uint64_t tmp_mask[2] = {(uint64_t)-1, (uint64_t)-1};
          AscendC::Select<ElementCalc, uint8_t, true>(
              input_qk, bit_mask_buf0, input_qk, static_cast<ElementCalc>(-1e20f),
              AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, tmp_mask,
              (mask_num_rows * num_cols_aligned + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
              AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
          XPU_OPS_DEBUG_PRINT("    selected by mask.\n");
        } else {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);
        }
      } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);
      }
      maskPaddedColumns(input_qk, valid_num_rows, valid_cols, num_cols_aligned);
#if CUSTOM_FA_SOFTMAX_IMPL == CUSTOM_FA_SOFTMAX_IMPL_FLASH_V2
      AscendC::Muls<ElementCalc, false>(
          input_qk, input_qk, scale, AscendC::MASK_PLACEHOLDER,
          (valid_num_rows * num_cols_aligned + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
          AscendC::UnaryRepeatParams(1, 1, 8, 8));

      uint32_t softmax_input_shape[2] = {valid_num_rows, num_cols_aligned};
      uint32_t softmax_state_shape[2] = {valid_num_rows, 1};
      input_qk.SetShapeInfo(AscendC::ShapeInfo(2, softmax_input_shape, AscendC::DataFormat::ND));
      auto softmax_sum = global_exp_sum[row_base];
      auto softmax_max = global_max[row_base];
      auto softmax_exp_max = global_scaler[row_base];
      softmax_sum.SetShapeInfo(AscendC::ShapeInfo(2, softmax_state_shape, AscendC::DataFormat::ND));
      softmax_max.SetShapeInfo(AscendC::ShapeInfo(2, softmax_state_shape, AscendC::DataFormat::ND));
      softmax_exp_max.SetShapeInfo(AscendC::ShapeInfo(2, softmax_state_shape, AscendC::DataFormat::ND));

      AscendC::SoftMaxShapeInfo softmax_shape_info{valid_num_rows, num_cols_aligned, valid_num_rows, valid_cols};
      auto softmax_api_tmp_buf = reduce_workspace_buf.template ReinterpretCast<uint8_t>();
      if (first_tile_kv) {
        auto softmax_tiling = AscendC::SoftMaxFlashV2TilingFunc(
            softmax_shape_info, sizeof(ElementCalc), sizeof(ElementCalc), WORKSPACE_MAX_SIZE, false,
            SOFTMAX_FLASH_V2_BASIC_BLOCK, false);
        AscendC::SoftmaxFlashV2<ElementCalc, false, true, SOFTMAX_FLASH_V2_BASIC_BLOCK, false,
                                details::CUSTOM_FA_SOFTMAX_REDUCE_CFG>(
            input_qk, softmax_sum, softmax_max, input_qk, softmax_exp_max, softmax_sum, softmax_max,
            softmax_api_tmp_buf, softmax_tiling, softmax_shape_info);
        details::SetVecMask(valid_num_rows);
        AscendC::Duplicate<ElementCalc, false>(softmax_exp_max, static_cast<ElementCalc>(1),
                                               AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
      } else {
        auto softmax_tiling = AscendC::SoftMaxFlashV2TilingFunc(
            softmax_shape_info, sizeof(ElementCalc), sizeof(ElementCalc), WORKSPACE_MAX_SIZE, true,
            SOFTMAX_FLASH_V2_BASIC_BLOCK, false);
        AscendC::SoftmaxFlashV2<ElementCalc, true, true, SOFTMAX_FLASH_V2_BASIC_BLOCK, false,
                                details::CUSTOM_FA_SOFTMAX_REDUCE_CFG>(
            input_qk, softmax_sum, softmax_max, input_qk, softmax_exp_max, softmax_sum, softmax_max,
            softmax_api_tmp_buf, softmax_tiling, softmax_shape_info);
      }

      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(output_event);
      AscendC::Cast<ElementInOut, ElementCalc, false>(
          output_buf, input_qk, AscendC::RoundMode::CAST_ROUND, AscendC::MASK_PLACEHOLDER,
          (valid_num_rows * num_cols_aligned + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
          AscendC::UnaryRepeatParams(1, 1, 4, 8));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(output_event);

      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(output_event);
      aligned_datacopy_ub2gm(valid_num_rows, num_cols, num_cols_aligned, layout_score.stride(0), output_buf,
                             output_score_gm[row_base * layout_score.stride(0)]);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(output_event);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);

      XPU_OPS_DEBUG_PRINT("    calculated softmax via SoftmaxFlashV2.\n");

      input_event_id = (input_event_id + 1) < UB_STAGE_NUM ? input_event_id + 1 : 0;
      continue;
#endif
      uint32_t input_count = valid_num_rows * num_cols_aligned;
#if CUSTOM_FA_SOFTMAX_BODY_IMPL == CUSTOM_FA_SOFTMAX_IMPL_CUSTOM
      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(input_count);
        AscendC::Muls<ElementCalc, false>(input_qk, input_qk, scale, AscendC::MASK_PLACEHOLDER, 1,
                                          AscendC::UnaryRepeatParams(1, 1, 8, 8));
        details::ResetToNormMask();
      } else {
        AscendC::Muls<ElementCalc, false>(
            input_qk, input_qk, scale, AscendC::MASK_PLACEHOLDER,
            (input_count + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
      }

      if (first_tile_kv) {
        details::Rowmax(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), global_max[row_base],
                        reduce_workspace_buf);
        AscendC::DataCopy(tmp_max[row_base], local_max[row_base],
                          AscendC::DataCopyParams(1, num_rows_aligned / details::FLOAT_BLOCK_SIZE, 0, 0));
      } else {
        details::Rowmax(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), local_max[row_base],
                        reduce_workspace_buf);
        details::SetVecMask(valid_num_rows);
        AscendC::Max<ElementCalc, false>(tmp_max[row_base], global_max[row_base], local_max[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::Sub<ElementCalc, false>(global_scaler[row_base], global_max[row_base], tmp_max[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::Exp<ElementCalc, false>(global_scaler[row_base], global_scaler[row_base], AscendC::MASK_PLACEHOLDER, 1,
                                         AscendC::UnaryRepeatParams(1, 1, 8, 8));
        details::ResetToNormMask();

        AscendC::DataCopy(global_max[row_base], tmp_max[row_base],
                          AscendC::DataCopyParams(1, num_rows_aligned / details::FLOAT_BLOCK_SIZE, 0, 0));
      }

      XPU_OPS_DEBUG_PRINT("    calculated rowmax.\n");

      AscendC::Brcb(flatten_workspace_buf.template ReinterpretCast<uint32_t>(),
                    global_max[row_base].template ReinterpretCast<uint32_t>(),
                    num_rows_aligned / details::FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));

      uint32_t repeat_stride = num_cols_aligned / details::FLOAT_BLOCK_SIZE;
      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(valid_num_rows * details::FLOAT_VECTOR_SIZE);
        for (uint32_t sub_idx = 0; sub_idx < num_cols_aligned / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::Sub<ElementCalc, false>(input_qk[col_offset], input_qk[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
        }
      } else {
        for (uint32_t sub_idx = 0; sub_idx < num_cols / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::Sub<ElementCalc, false>(input_qk[col_offset], input_qk[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, valid_num_rows,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
        }
        if (num_cols % details::FLOAT_VECTOR_SIZE != 0) {
          uint32_t col_offset = (num_cols / details::FLOAT_VECTOR_SIZE) * details::FLOAT_VECTOR_SIZE;
          uint32_t tail = num_cols - col_offset;
          details::SetVecMask(tail);
          AscendC::Sub<ElementCalc, false>(input_qk[col_offset], input_qk[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, valid_num_rows,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
          AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
      }
#else
      if (first_tile_kv) {
        details::Rowmax(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), tmp_max[row_base],
                        reduce_workspace_buf);
        details::SetVecMask(valid_num_rows);
        AscendC::Muls<ElementCalc, false>(global_max[row_base], tmp_max[row_base], scale, AscendC::MASK_PLACEHOLDER, 1,
                                          AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::Duplicate<ElementCalc, false>(global_scaler[row_base], static_cast<ElementCalc>(1),
                                               AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        details::ResetToNormMask();
      } else {
        details::Rowmax(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), local_max[row_base],
                        reduce_workspace_buf);
        details::SetVecMask(valid_num_rows);
        AscendC::Muls<ElementCalc, false>(local_max[row_base], local_max[row_base], scale, AscendC::MASK_PLACEHOLDER,
                                          1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::Max<ElementCalc, false>(tmp_max[row_base], global_max[row_base], local_max[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::Sub<ElementCalc, false>(global_scaler[row_base], global_max[row_base], tmp_max[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::Exp<ElementCalc, false>(global_scaler[row_base], global_scaler[row_base], AscendC::MASK_PLACEHOLDER, 1,
                                         AscendC::UnaryRepeatParams(1, 1, 8, 8));
        details::ResetToNormMask();

        AscendC::DataCopy(global_max[row_base], tmp_max[row_base],
                          AscendC::DataCopyParams(1, num_rows_aligned / details::FLOAT_BLOCK_SIZE, 0, 0));
      }

      details::SetVecMask(valid_num_rows);
      AscendC::Muls<ElementCalc, false>(tmp_max[row_base], global_max[row_base], -1.0f, AscendC::MASK_PLACEHOLDER,
                                        1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
      details::ResetToNormMask();

      AscendC::Brcb(flatten_workspace_buf.template ReinterpretCast<uint32_t>(),
                    tmp_max[row_base].template ReinterpretCast<uint32_t>(),
                    num_rows_aligned / details::FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));

      uint32_t fl_ws_offset = (num_rows_aligned / details::FLOAT_BLOCK_SIZE) * details::FLOAT_VECTOR_SIZE;
#if CUSTOM_FA_HOIST_SOFTMAX_SCALE_DUP
      auto tmp_ws = scale_dup_buf;
#else
      auto tmp_ws = flatten_workspace_buf[fl_ws_offset];
      AscendC::Duplicate<ElementCalc, false>(tmp_ws, scale, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
#endif

      uint32_t repeat_stride = num_cols_aligned / details::FLOAT_BLOCK_SIZE;
      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(valid_num_rows * details::FLOAT_VECTOR_SIZE);
        for (uint32_t sub_idx = 0; sub_idx < num_cols_aligned / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::FusedMulAdd<ElementCalc, false>(
              input_qk[col_offset], tmp_ws, flatten_workspace_buf, AscendC::MASK_PLACEHOLDER, 1,
              AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, 0, 1));
        }
      } else {
        for (uint32_t sub_idx = 0; sub_idx < (num_cols + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE;
             ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::FusedMulAdd<ElementCalc, false>(
              input_qk[col_offset], tmp_ws, flatten_workspace_buf, AscendC::MASK_PLACEHOLDER, valid_num_rows,
              AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, 0, 1));
        }
      }
#endif

      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(input_count);
        AscendC::Exp<ElementCalc, false>(input_qk, input_qk, AscendC::MASK_PLACEHOLDER, 1,
                                         AscendC::UnaryRepeatParams(1, 1, 8, 8));
      } else {
        AscendC::Exp<ElementCalc, false>(
            input_qk, input_qk, AscendC::MASK_PLACEHOLDER,
            (input_count + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
      }
      // AscendC::PipeBarrier<PIPE_V>();

      // AscendC::DumpTensor(input_qk[(valid_num_rows - 4) * num_cols_aligned], 4, 4 * num_cols_aligned);

      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(output_event);
      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(input_count);
        AscendC::Cast<ElementInOut, ElementCalc, false>(
            output_buf, input_qk, AscendC::RoundMode::CAST_ROUND, AscendC::MASK_PLACEHOLDER, 1,
            AscendC::UnaryRepeatParams(1, 1, 4, 8));
      } else {
        AscendC::Cast<ElementInOut, ElementCalc, false>(
            output_buf, input_qk, AscendC::RoundMode::CAST_ROUND, AscendC::MASK_PLACEHOLDER,
            (input_count + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 4, 8));
      }
      // AscendC::PipeBarrier<PIPE_V>();
      // AscendC::DumpTensor(output_buf, 5, valid_num_rows * num_cols_aligned);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(output_event);

      XPU_OPS_DEBUG_PRINT("    calculated exp and casted score.\n");

      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(output_event);
      aligned_datacopy_ub2gm(valid_num_rows, num_cols, num_cols_aligned, layout_score.stride(0), output_buf,
                             output_score_gm[row_base * layout_score.stride(0)]);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(output_event);

      XPU_OPS_DEBUG_PRINT("    copied score to gm.\n");

      // details::Rowsum(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), local_exp_sum[row_base],
      //                 reduce_workspace_buf);
      // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
      // XPU_OPS_DEBUG_PRINT("    calculated rowsum.\n");
      if constexpr (SOFTMAX_USE_COUNTER_MASK) {
        details::ResetToNormMask();
      }

      if (first_tile_kv) {
        // AscendC::DataCopy(global_exp_sum[row_base], local_exp_sum[row_base],
        //                   AscendC::DataCopyParams(1, num_rows_aligned / details::FLOAT_BLOCK_SIZE, 0, 0));
        // // AscendC::PipeBarrier<PIPE_V>();
        details::Rowsum(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), global_exp_sum[row_base],
                        reduce_workspace_buf);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        XPU_OPS_DEBUG_PRINT("    calculated rowsum.\n");
      } else {
        details::Rowsum(input_qk, Layout(valid_num_rows, num_cols, num_cols_aligned), local_exp_sum[row_base],
                        reduce_workspace_buf);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        details::SetVecMask(valid_num_rows);
        AscendC::Mul<ElementCalc, false>(global_exp_sum[row_base], global_exp_sum[row_base], global_scaler[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        // AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementCalc, false>(global_exp_sum[row_base], global_exp_sum[row_base], local_exp_sum[row_base],
                                         AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        // AscendC::PipeBarrier<PIPE_V>();
        details::ResetToNormMask();
      }

      XPU_OPS_DEBUG_PRINT("    updated rowsum.\n");

      input_event_id = (input_event_id + 1) < UB_STAGE_NUM ? input_event_id + 1 : 0;
    }

    // AscendC::DumpTensor(local_max, 0, num_rows);
    // AscendC::DumpTensor(local_exp_sum, 1, num_rows);
    // AscendC::DumpTensor(global_max, 2, num_rows);
    // AscendC::DumpTensor(global_scaler_bufs[scaler_pingpong_id], 3, num_rows);
    // AscendC::DumpTensor(global_exp_sum, 4, num_rows);
  }

  CATLASS_DEVICE
  void subCoreOnlineSoftmax(const AscendC::GlobalTensor<ElementCalc> &input_qk_gm, const Layout &layout_qk,
                            const AscendC::GlobalTensor<ElementMask> &input_mask_gm, const Layout &layout_mask,
                            const AscendC::GlobalTensor<ElementInOut> &output_score_gm, const Layout &layout_score,
                            uint32_t start_qo_pos, uint32_t start_kv_pos, uint32_t scaler_pingpong_id,
                            uint32_t state_pingpong_idx, bool first_tile_kv, bool last_tile_kv,
                            bool tile_has_partial_mask) {
    subCoreOnlineSoftmaxImpl<true>(input_qk_gm, layout_qk, input_mask_gm, layout_mask, output_score_gm, layout_score,
                                   start_qo_pos, start_kv_pos, scaler_pingpong_id, state_pingpong_idx, first_tile_kv,
                                   last_tile_kv, tile_has_partial_mask, layout_mask.shape(1));
  }

  CATLASS_DEVICE
  void subCoreOnlineSoftmaxNoMask(const AscendC::GlobalTensor<ElementCalc> &input_qk_gm, const Layout &layout_qk,
                                  const AscendC::GlobalTensor<ElementInOut> &output_score_gm,
                                  const Layout &layout_score, uint32_t start_qo_pos, uint32_t start_kv_pos,
                                  uint32_t scaler_pingpong_id, uint32_t state_pingpong_idx, bool first_tile_kv,
                                  bool last_tile_kv, uint32_t valid_kv_len) {
    AscendC::GlobalTensor<ElementMask> dummy_mask_gm;
    Layout dummy_layout_mask(0, 0, 0);
    // Padding still needs the true column count even when no external mask is used.
    subCoreOnlineSoftmaxImpl<false>(input_qk_gm, layout_qk, dummy_mask_gm, dummy_layout_mask, output_score_gm,
                                    layout_score, start_qo_pos, start_kv_pos, scaler_pingpong_id, state_pingpong_idx,
                                    first_tile_kv, last_tile_kv, false, valid_kv_len);
  }

  CATLASS_DEVICE
  void subCoreAggregate(const AscendC::GlobalTensor<ElementCalc> &input_pv_gm, const Layout &layout_pv,
                        const AscendC::GlobalTensor<ElementInOut> &output_o_gm, const Layout &layout_o,
                        uint32_t scaler_pingpong_id, uint32_t state_pingpong_idx, bool first_tile_kv,
                        bool last_tile_kv) {
    // hsh: we may try to enforce double buffer to overlap mte2 and v
    uint32_t num_rows = layout_pv.shape(0), num_cols = layout_pv.shape(1);
    uint32_t output_rows = layout_o.shape(0);
    uint32_t num_cols_aligned = (num_cols + NUM_COLS_ALIGNED - 1) / NUM_COLS_ALIGNED * NUM_COLS_ALIGNED;
    uint32_t num_rows_per_iter = INPUT_PV_MAX_ELEMENTS / num_cols_aligned;
    num_rows_per_iter = (num_rows_per_iter / NUM_ROWS_ALIGNED) * NUM_ROWS_ALIGNED;
    uint32_t num_row_iters = (num_rows + num_rows_per_iter - 1) / num_rows_per_iter;

    auto global_scaler = global_scaler_bufs[scaler_pingpong_id];
    auto global_exp_sum = global_exp_sum_bufs[state_pingpong_idx];
    // if (num_rows_per_iter == 1) {
    //   num_rows_per_iter = (num_rows / 2 + NUM_ROWS_ALIGNED - 1) / NUM_ROWS_ALIGNED * NUM_ROWS_ALIGNED;
    //   num_row_iters = (num_rows + num_rows_per_iter - 1) / num_rows_per_iter;
    // }

    XPU_OPS_DEBUG_PRINT(
        "  Aggregate: num_rows=%u, num_cols=%u, num_cols_aligned=%u, num_rows_per_iter=%u, num_row_iters=%u.\n",
        num_rows, num_cols, num_cols_aligned, num_rows_per_iter, num_row_iters);
    for (uint32_t row_iter = 0; row_iter < num_row_iters; ++row_iter) {
      uint32_t row_base = row_iter * num_rows_per_iter;
      uint32_t valid_num_rows = (row_iter == num_row_iters - 1) ? (num_rows - row_base) : num_rows_per_iter;
      uint32_t num_rows_aligned = (valid_num_rows + NUM_ROWS_ALIGNED - 1) / NUM_ROWS_ALIGNED * NUM_ROWS_ALIGNED;

      if (row_iter == 0) {
        AscendC::CrossCoreWaitFlag(PV_READY);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        aligned_datacopy_gm2ub(valid_num_rows, num_cols, layout_pv.stride(0), num_cols_aligned,
                               input_pv_gm[row_base * layout_pv.stride(0)], input_bufs[input_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);

        XPU_OPS_DEBUG_PRINT("    loaded first pv.\n");
      }

      if (row_iter < num_row_iters - 1) {
        uint32_t next_input_event_id = (input_event_id + 1) % UB_STAGE_NUM;
        uint32_t next_row_base = (row_iter + 1) * num_rows_per_iter;
        uint32_t next_valid_num_rows =
            (row_iter + 1 == num_row_iters - 1) ? (num_rows - next_row_base) : num_rows_per_iter;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_events[next_input_event_id]);
        aligned_datacopy_gm2ub(next_valid_num_rows, num_cols, layout_pv.stride(0), num_cols_aligned,
                               input_pv_gm[next_row_base * layout_pv.stride(0)], input_bufs[next_input_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_events[next_input_event_id]);

        XPU_OPS_DEBUG_PRINT("    loaded next pv.\n");
      }

      auto input_pv = input_bufs[input_event_id];
      if (first_tile_kv) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);
        AscendC::DataCopy(
            o_state_buf[row_base * num_cols_aligned], input_pv,
            AscendC::DataCopyParams(1, num_rows_aligned * num_cols_aligned / details::FLOAT_BLOCK_SIZE, 0, 0));
        // AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        XPU_OPS_DEBUG_PRINT("    initialized o_state.\n");
      } else {
#if CUSTOM_FA_AGGREGATE_IMPL == CUSTOM_FA_AGGREGATE_IMPL_CUSTOM
        uint32_t repeat_stride = num_cols_aligned / details::FLOAT_BLOCK_SIZE;
        uint32_t row_offset = row_base * num_cols_aligned;
        AscendC::Brcb(flatten_workspace_buf.template ReinterpretCast<uint32_t>(),
                      global_scaler[row_base].template ReinterpretCast<uint32_t>(),
                      num_rows_aligned / details::FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        if constexpr (AGGREGATE_USE_COUNTER_MASK) {
          details::SetCounterMask<ElementCalc>(valid_num_rows * details::FLOAT_VECTOR_SIZE);
          for (uint32_t sub_idx = 0; sub_idx < num_cols_aligned / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
            uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
            AscendC::Mul<ElementCalc, false>(
                o_state_buf[row_offset + col_offset], o_state_buf[row_offset + col_offset], flatten_workspace_buf,
                AscendC::MASK_PLACEHOLDER, 1,
                AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
          }
        } else {
          for (uint32_t sub_idx = 0; sub_idx < num_cols / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
            uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
            AscendC::Mul<ElementCalc, false>(
                o_state_buf[row_offset + col_offset], o_state_buf[row_offset + col_offset], flatten_workspace_buf,
                AscendC::MASK_PLACEHOLDER, valid_num_rows,
                AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
          }
          if (num_cols % details::FLOAT_VECTOR_SIZE != 0) {
            uint32_t col_offset = (num_cols / details::FLOAT_VECTOR_SIZE) * details::FLOAT_VECTOR_SIZE;
            uint32_t tail = num_cols - col_offset;
            details::SetVecMask(tail);
            AscendC::Mul<ElementCalc, false>(
                o_state_buf[row_offset + col_offset], o_state_buf[row_offset + col_offset], flatten_workspace_buf,
                AscendC::MASK_PLACEHOLDER, valid_num_rows,
                AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
            // AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
          }
        }
        XPU_OPS_DEBUG_PRINT("    scaled o_state.\n");
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);
        if constexpr (AGGREGATE_USE_COUNTER_MASK) {
          details::SetCounterMask<ElementCalc>(valid_num_rows * num_cols_aligned);
        }
        AscendC::Add<ElementCalc, false>(
            o_state_buf[row_offset], o_state_buf[row_offset], input_pv,
            AscendC::MASK_PLACEHOLDER,
            AGGREGATE_USE_COUNTER_MASK
                ? 1
                : (num_rows_aligned * num_cols_aligned + details::FLOAT_VECTOR_SIZE - 1) /
                      details::FLOAT_VECTOR_SIZE,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        // AscendC::PipeBarrier<PIPE_V>();
        if constexpr (AGGREGATE_USE_COUNTER_MASK) {
          details::ResetToNormMask();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
        XPU_OPS_DEBUG_PRINT("    updated o_state.\n");
#else
        uint32_t repeat_stride = num_cols_aligned / details::FLOAT_BLOCK_SIZE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_events[input_event_id]);
        AscendC::Brcb(flatten_workspace_buf.template ReinterpretCast<uint32_t>(),
                      global_scaler[row_base].template ReinterpretCast<uint32_t>(),
                      num_rows_aligned / details::FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        if constexpr (AGGREGATE_USE_COUNTER_MASK) {
          details::SetCounterMask<ElementCalc>(valid_num_rows * details::FLOAT_VECTOR_SIZE);
        }
        for (uint32_t sub_idx = 0; sub_idx < num_cols_aligned / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          uint32_t row_offset = row_base * num_cols_aligned;
          AscendC::FusedMulAdd<ElementCalc, false>(
              o_state_buf[col_offset + row_offset], flatten_workspace_buf, input_pv[col_offset],
              AscendC::MASK_PLACEHOLDER, AGGREGATE_USE_COUNTER_MASK ? 1 : valid_num_rows,
              AscendC::BinaryRepeatParams(1, 0, 1, repeat_stride, 1, repeat_stride));
        }
        if constexpr (AGGREGATE_USE_COUNTER_MASK) {
          details::ResetToNormMask();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_events[input_event_id]);
#endif
      }

      input_event_id = (input_event_id + 1) < UB_STAGE_NUM ? input_event_id + 1 : 0;
    }

    if (last_tile_kv) {
      uint32_t repeat_stride = num_cols_aligned / details::FLOAT_BLOCK_SIZE;
      AscendC::Brcb(flatten_workspace_buf.template ReinterpretCast<uint32_t>(),
                    global_exp_sum.template ReinterpretCast<uint32_t>(), num_rows / NUM_ROWS_ALIGNED,
                    AscendC::BrcbRepeatParams(1, 8));
      if constexpr (AGGREGATE_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(num_rows * details::FLOAT_VECTOR_SIZE);
        for (uint32_t sub_idx = 0; sub_idx < num_cols_aligned / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::Div<ElementCalc, false>(o_state_buf[col_offset], o_state_buf[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
        }
      } else {
        for (uint32_t sub_idx = 0; sub_idx < num_cols / details::FLOAT_VECTOR_SIZE; ++sub_idx) {
          uint32_t col_offset = sub_idx * details::FLOAT_VECTOR_SIZE;
          AscendC::Div<ElementCalc, false>(o_state_buf[col_offset], o_state_buf[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, num_rows,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
        }
        if (num_cols % details::FLOAT_VECTOR_SIZE != 0) {
          uint32_t col_offset = (num_cols / details::FLOAT_VECTOR_SIZE) * details::FLOAT_VECTOR_SIZE;
          uint32_t tail = num_cols - col_offset;
          details::SetVecMask(tail);
          AscendC::Div<ElementCalc, false>(o_state_buf[col_offset], o_state_buf[col_offset], flatten_workspace_buf,
                                           AscendC::MASK_PLACEHOLDER, num_rows,
                                           AscendC::BinaryRepeatParams(1, 1, 0, repeat_stride, repeat_stride, 1));
          // AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
      }
      if constexpr (AGGREGATE_USE_COUNTER_MASK) {
        details::SetCounterMask<ElementCalc>(num_rows * num_cols_aligned);
      }
      XPU_OPS_DEBUG_PRINT("    finalized o_state.\n");

      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(output_event);
      AscendC::Cast<ElementInOut, ElementCalc, false>(
          output_buf, o_state_buf, AscendC::RoundMode::CAST_ROUND, AscendC::MASK_PLACEHOLDER,
          AGGREGATE_USE_COUNTER_MASK
              ? 1
              : (num_rows * num_cols_aligned + details::FLOAT_VECTOR_SIZE - 1) / details::FLOAT_VECTOR_SIZE,
          AscendC::UnaryRepeatParams(1, 1, 4, 8));
      // AscendC::PipeBarrier<PIPE_V>();
      if constexpr (AGGREGATE_USE_COUNTER_MASK) {
        details::ResetToNormMask();
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(output_event);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(output_event);
      if (output_rows > 0) {
        aligned_datacopy_ub2gm(output_rows, num_cols, num_cols_aligned, layout_o.stride(0), output_buf, output_o_gm);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(output_event);
      XPU_OPS_DEBUG_PRINT("    copied o to gm.\n");
    }
  }

private:
  uint32_t tile_kv, pipeline_stages;
  uint32_t single_qk_size, single_score_size, single_pv_size;
  float scale;

  AscendC::LocalTensor<ElementCalc> input_bufs[UB_STAGE_NUM];
  AscendC::LocalTensor<ElementMask> input_mask_buf;
  AscendC::LocalTensor<ElementInOut> output_buf;

  // they are all workspace
  AscendC::LocalTensor<ElementCalc> casted_mask_buf;
  AscendC::LocalTensor<ElementCalc> reduce_workspace_buf;
  AscendC::LocalTensor<ElementCalc> flatten_workspace_buf;

  // each buf uses half of workspace
  AscendC::LocalTensor<ElementMask> contiguous_mask_buf0;
  AscendC::LocalTensor<half> casted_mask_buf1;
  AscendC::LocalTensor<uint8_t> bit_mask_buf0;

  AscendC::LocalTensor<ElementCalc> o_state_buf;
  AscendC::LocalTensor<ElementCalc> local_max, global_max_bufs[MAX_PIPELINE_STAGE], local_exp_sum,
      global_exp_sum_bufs[MAX_PIPELINE_STAGE], tmp_max, scale_dup_buf, global_scaler_bufs[MAX_PIPELINE_STAGE];

  uint32_t input_events[UB_STAGE_NUM];
  uint32_t input_event_id = 0;

  uint32_t input_mask_event;
  uint32_t output_event;
};

}  // namespace xpu_ops::kernels
