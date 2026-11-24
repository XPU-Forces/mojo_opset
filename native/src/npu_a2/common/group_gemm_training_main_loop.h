// Adapted from data/xpu_ops commit 8452b29b; see group_gemm_SOURCE.json.
#pragma once
#include "group_gemm_training.h"
#include "group_gemm_training_utils.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "math_utils.h"

namespace mojo::native::a2 {
namespace kernels {

constexpr uint32_t thresholdBlockNum = 8; // 8 is obtained by tests, indicating the threshold of basic block
                                          // numbers in both directions when assigning data blocks to cube
                                          // cores when using diagnal strategy
constexpr uint32_t thresholdDimM = 5; // 5 is obtained by tests, indicating the threshold for distinguishing
                                      // strategies for large/small shapes

template <typename GroupGemm, typename Config, typename Epilogue> class GroupGemmMainLoop {
protected:
  const GMMTrainBaseParams *__restrict gmmBaseParams;
  TCubeTiling mmTilingData;
  uint32_t blockIdx;
  uint32_t subBlockIdx;
  uint32_t coreIdx;
  uint32_t groupNum;
  AscendC::GlobalTensor<int32_t> group_list_gm_;

public:
  /** @brief constructor */
  __aicore__ inline GroupGemmMainLoop() {}

  __aicore__ inline void init(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR group_list, GM_ADDR output,
                              GM_ADDR workspace, const GMMTrainBaseParams *__restrict gmmBaseParamsIn,
                              const TCubeTiling &mmTilingDataIn, GM_ADDR tiling, AscendC::TPipe &pipe);

  __aicore__ inline void process();

  __aicore__ inline void setMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

  __aicore__ inline void updateGroupOffset(MNConfig &mnConfig);

  __aicore__ inline void compDiagonalBlockIdx(MNConfig &mnConfig, const uint32_t curBlock,
                                              const uint32_t count, const uint32_t thresholdM_dimN);
  GroupGemm group_gemm_obj_; // inernal computation operator
};

template <typename GroupGemm, typename Config, typename Epilogue>
__aicore__ inline void GroupGemmMainLoop<GroupGemm, Config, Epilogue>::init(
    GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR group_list, GM_ADDR output, GM_ADDR workspace,
    const GMMTrainBaseParams *__restrict gmmBaseParamsIn, const TCubeTiling &mmTilingDataIn, GM_ADDR tiling,
    AscendC::TPipe &pipe) {
  gmmBaseParams = gmmBaseParamsIn;
  mmTilingData = mmTilingDataIn;
  group_gemm_obj_.init(input, weight, bias, output, workspace, gmmBaseParams, mmTilingData, pipe);

  group_list_gm_.SetGlobalBuffer((__gm__ int32_t *)group_list);

  blockIdx = AscendC::GetBlockIdx();
  subBlockIdx = AscendC::GetSubBlockIdx();
  coreIdx = blockIdx;
  int64_t coreRation = AscendC::GetTaskRation();
  if (coreRation > 1) {
    coreIdx /= coreRation;
  }
  groupNum = gmmBaseParams->groupNum;
}

template <typename GroupGemm, typename Config, typename Epilogue>
__aicore__ inline void GroupGemmMainLoop<GroupGemm, Config, Epilogue>::setMNConfig(const int32_t splitValue,
                                                                                   const uint32_t groupIdx,
                                                                                   MNConfig &mnConfig) {
  // split m
  if (gmmBaseParams->groupType == 0) {
    mnConfig.m = splitValue;
    mnConfig.k = mmTilingData.Ka;
    mnConfig.n = mmTilingData.N;
  } else if (gmmBaseParams->groupType == 2) {
    mnConfig.m = mmTilingData.M;
    mnConfig.k = splitValue;
    mnConfig.n = mmTilingData.N;
  } else {
    mnConfig.m = mmTilingData.M;
    mnConfig.k = mmTilingData.Ka;
    mnConfig.n = mmTilingData.N;
  }
  mnConfig.baseM = mmTilingData.baseM;
  mnConfig.baseN = mmTilingData.baseN;
  mnConfig.singleM = mnConfig.baseM;
  mnConfig.singleN = mnConfig.baseN;
  return;
}

template <typename GroupGemm, typename Config, typename Epilogue>
__aicore__ inline void GroupGemmMainLoop<GroupGemm, Config, Epilogue>::updateGroupOffset(MNConfig &mnConfig) {
  if constexpr (Config::IS_WEIGHT_NZ) {
    mnConfig.wBaseOffset +=
        align_up<16>(mnConfig.k) * align_up<16>(mnConfig.n); // 16: nz format last two dim size
  } else {
    if constexpr (Config::TRANSPOSE_B) {
      if (gmmBaseParams->groupType == 2) {
        mnConfig.wBaseOffset += mnConfig.k;
      } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
      }
    } else {
      mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
  }
  mnConfig.nAxisBaseOffset += mnConfig.n;
  mnConfig.mAxisBaseOffset += mnConfig.m;
  if constexpr (Config::TRANSPOSE_A) {
    if (gmmBaseParams->groupType == 0) {
      mnConfig.xBaseOffset += mnConfig.m;
    } else {
      mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    }
  } else {
    if (gmmBaseParams->groupType == 0) {
      mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    } else {
      mnConfig.xBaseOffset += mnConfig.k;
    }
  }
  mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename GroupGemm, typename Config, typename Epilogue>
__aicore__ inline void GroupGemmMainLoop<GroupGemm, Config, Epilogue>::compDiagonalBlockIdx(
    MNConfig &mnConfig, const uint32_t curBlock, const uint32_t count, const uint32_t thresholdM_dimN) {
  if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
    mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
    mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
  } else {
    uint32_t relativeBlock = curBlock - count;
    uint32_t curThresholdM =
        relativeBlock >= align_down(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN)
            ? mnConfig.blockDimM % thresholdBlockNum
            : thresholdBlockNum;
    uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
    uint32_t curThresholdN = relativeBlock % thresholdM_dimN >=
                                     align_down(curThresholdM * mnConfig.blockDimN, curThresholdM_thresholdN)
                                 ? mnConfig.blockDimN % thresholdBlockNum
                                 : thresholdBlockNum;

    uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
    mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
    mnConfig.nIdx =
        (localRelativeBlock + localRelativeBlock / least_common_multiple(curThresholdM, curThresholdN)) %
            curThresholdN +
        relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
  }
}

template <typename GroupGemm, typename Config, typename Epilogue>
__aicore__ inline void GroupGemmMainLoop<GroupGemm, Config, Epilogue>::process() {
  MNConfig mnConfig;
  MNConfig mnPreConfig;
  for (uint32_t groupIdx = 0, count = 0; groupIdx < groupNum; ++groupIdx) {
    int32_t splitValue = static_cast<int32_t>(group_list_gm_.GetValue(groupIdx));
    setMNConfig(splitValue, groupIdx, mnConfig);
    uint32_t dimM = ceil_div(mnConfig.m, mnConfig.singleM);
    uint32_t dimN = ceil_div(mnConfig.n, mnConfig.singleN);
    mnConfig.blockDimM = dimM;
    mnConfig.blockDimN = dimN;
    if (mnConfig.k == 0) {
      updateGroupOffset(mnConfig);
      continue;
    }

    uint32_t curCount = count + dimM * dimN;
    uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + gmmBaseParams->coreNum;
    uint32_t thresholdM_dimN = thresholdBlockNum * dimN;

    while (curBlock < curCount) {
      compDiagonalBlockIdx(mnConfig, curBlock, count, thresholdM_dimN);
      group_gemm_obj_.compute(groupIdx, mnConfig, subBlockIdx);
      curBlock += gmmBaseParams->coreNum;
      mnPreConfig = mnConfig;
    }
    updateGroupOffset(mnConfig);
    count = curCount % gmmBaseParams->coreNum;
  }
}
} // namespace kernels
} // namespace mojo::native::a2
