/*!
 * \file vec_pre.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_MLA_PRE_KERNEL_H_
#define FLASH_ATTENTION_SCORE_GRAD_MLA_PRE_KERNEL_H_

#include "kernel_operator.h"
using namespace AscendC;

template <class TILING_TYPE, bool DETERMINISTIC_ENABLE> class VectorInitOuput {
public:
  __aicore__ inline VectorInitOuput() {}
  __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                              __gm__ uint8_t *workspace, const TILING_TYPE *orgTilingData);
  __aicore__ inline void Process();

  TPipe *pipe;
  GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

  const TILING_TYPE *tilingData;

  uint32_t cBlockIdx;
  // query
  uint32_t qPreBlockFactor;
  uint32_t qPreBlockTotal;
  uint32_t qPreBlockTail;
  uint32_t kvPreBlockFactor;
  uint32_t kvPreBlockTotal;
  uint32_t kvPreBlockTail;

  int64_t initdqSize;
  int64_t dqOffset;
  int64_t initdkSize;
  int64_t dkvOffset;
  uint32_t dqPostAbsorb;
};

template <class TILING_TYPE, bool DETERMINISTIC_ENABLE>
__aicore__ inline void VectorInitOuput<TILING_TYPE, DETERMINISTIC_ENABLE>::Init(TPipe *pipe_in, __gm__ uint8_t *dq,
                                                                                __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                                                                                __gm__ uint8_t *workspace,
                                                                                const TILING_TYPE *orgTilingData) {
  cBlockIdx = GetBlockIdx();

  tilingData = orgTilingData;
  pipe = pipe_in;

  uint32_t coreNum = tilingData->coreNum;
  dqPostAbsorb = tilingData->dqPostAbsorb;

  // compute tiling params
  qPreBlockFactor = (tilingData->qSize + coreNum - 1) / coreNum;
  qPreBlockTotal = (tilingData->qSize + qPreBlockFactor - 1) / qPreBlockFactor;
  int64_t qPreTailNumTmp = tilingData->qSize % qPreBlockFactor;
  qPreBlockTail = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

  kvPreBlockFactor = (tilingData->kvSize + coreNum - 1) / coreNum;
  kvPreBlockTotal = (tilingData->kvSize + kvPreBlockFactor - 1) / kvPreBlockFactor;
  int64_t kvPreTailNumTmp = tilingData->kvSize % kvPreBlockFactor;
  kvPreBlockTail = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

  dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dqWorkSpaceOffset / sizeof(float));
  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));
  dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dvWorkSpaceOffset / sizeof(float));

  initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
  dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
  initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
  dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
}

template <class TILING_TYPE, bool DETERMINISTIC_ENABLE>
__aicore__ inline void VectorInitOuput<TILING_TYPE, DETERMINISTIC_ENABLE>::Process() {
  // process clear dq dk dv workspace

  if (dqPostAbsorb == 0 && g_coreType == AIV && cBlockIdx < qPreBlockTotal) {
    InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
  }

  if (g_coreType == AIV && cBlockIdx < kvPreBlockTotal) {
    InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
    InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
  }
}

#endif // FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_
