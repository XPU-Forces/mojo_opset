// Adapted from data/xpu_ops commit 8452b29b; see group_gemm_SOURCE.json.
#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace mojo::native::a2::kernels {

enum class ActiveType { INVALID_TYPE = 0, RELU, GELU, SILU, SWISH_GLU };

struct MNConfig {
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  uint32_t baseM = 0;
  uint32_t baseN = 0;
  uint32_t mIdx = 0;
  uint32_t nIdx = 0;
  uint32_t blockDimM = 0;
  uint32_t blockDimN = 0;
  uint32_t singleM = 0;
  uint32_t singleN = 0;
  uint64_t wBaseOffset = 0;
  uint64_t nAxisBaseOffset = 0;
  uint64_t mAxisBaseOffset = 0;
  uint64_t xBaseOffset = 0;
  uint64_t yBaseOffset = 0;
  uint64_t wOutOffset = 0;
  uint64_t workSpaceOffset = 0;
};

template <bool has_activation, bool transpose_a, bool transpose_b, bool is_weight_nz, bool has_bias>
struct GroupGemmKernelConfig {
  static constexpr bool HAS_ACTIVATION = has_activation;
  static constexpr bool TRANSPOSE_A = transpose_a;
  static constexpr bool TRANSPOSE_B = transpose_b;
  static constexpr bool IS_WEIGHT_NZ = is_weight_nz;
  static constexpr bool HAS_BIAS = has_bias;
};

template <typename T> __aicore__ inline T Ceil(T a, T b) {
  if (b == 0) {
    return 0;
  }
  return (a + b - 1) / b;
}

template <uint32_t base, typename T = uint32_t> __aicore__ inline T AlignUp(T a) {
  return (a + base - 1) / base * base;
}

template <typename T> __aicore__ inline T AlignUp(T a, T base) { return (a + base - 1) / base * base; }

// template <typename T>
// __aicore__ inline void EmptyTensorCompute(GM_ADDR groupListPtr, GM_ADDR y, const GMMTrainingTilingData
// *__restrict tiling) {
//   const GMMTrainBaseParams *__restrict gmmBaseParams = &tiling->gmmBaseParams;
//   const TCubeTiling *__restrict mmTilingData = &tiling->mmTilingData;
//   // V2接口在host分组后进来grouptype为-1，因此这里接受grouptype为-1或2
//   if (groupListPtr == nullptr || gmmBaseParams->groupType == 0) {
//     return;
//   }

//   AscendC::GlobalTensor<T> yGm;
//   AscendC::GlobalTensor<int32_t> groupListGm;
//   yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
//   if (groupListPtr != nullptr) {
//     groupListGm.SetGlobalBuffer((__gm__ int32_t *)groupListPtr);
//   }
//   uint64_t yBaseOffset = 0;
//   int32_t preOffset = 0;

//   int64_t coreIdx = AscendC::GetBlockIdx();
//   int64_t coreRation = AscendC::GetTaskRation();
//   if (coreRation > 1) {
//     coreIdx /= coreRation;
//   }

//   for (uint32_t groupIdx = 0; groupIdx < gmmBaseParams->groupNum; ++groupIdx) {
//     int32_t splitValue = groupListGm.GetValue(groupIdx);
//     uint32_t m = mmTilingData->M;
//     uint32_t k = splitValue;
//     uint32_t n = mmTilingData->N;
//     if (k == 0) {
//       AscendC::InitOutput<T>(yGm[yBaseOffset], m * n, 0);
//     }
//     yBaseOffset += m * n;
//   }
// }

} // namespace mojo::native::a2::kernels
