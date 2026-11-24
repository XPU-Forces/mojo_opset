/*!
 * \file vec_post.h
 * \brief common post process
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_MLA_POST_H_
#define _FLASH_ATTENTION_SCORE_GRAD_MLA_POST_H_

#include "kernel_operator.h"
#include "fa_grad_tiling.h"
#include "../common/fa_util.h"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename OUT_TYPE, class TILING_TYPE, bool DETERMINISTIC_ENABLE> class VectorPost {
public:
  __aicore__ inline VectorPost(){};
  __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                              __gm__ uint8_t *workspace, const TILING_TYPE *__restrict ordTilingData);
  __aicore__ inline void Process();

  constexpr static uint32_t POST_BUFFER_NUM = 1;

  TPipe *pipe;
  TQue<QuePosition::VECIN, POST_BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, POST_BUFFER_NUM> outQueue;

  // input
  AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
  // output
  AscendC::GlobalTensor<OUT_TYPE> dqGm, dkGm, dvGm;

  const TILING_TYPE *__restrict tilingData;

  int64_t cBlockIdx;
  // query
  int64_t ubBaseSize;
  int64_t qPostBlockFactor;
  uint64_t qPostBlockTotal;
  int64_t qPostBaseNum;
  int64_t qPostTailNum;
  int64_t kvPostBlockFactor;
  uint64_t kvPostBlockTotal;
  int64_t kvPostBaseNum;
  int64_t kvPostTailNum;
  uint32_t dqPostAbsorb;
};

template <typename OUT_TYPE, class TILING_TYPE, bool DETERMINISTIC_ENABLE>
__aicore__ inline void
VectorPost<OUT_TYPE, TILING_TYPE, DETERMINISTIC_ENABLE>::Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
                                                              __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
                                                              const TILING_TYPE *__restrict ordTilingData) {
  cBlockIdx = GetBlockIdx();

  tilingData = ordTilingData;
  pipe = pipe_in;

  dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
  dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
  dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);

  dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dqWorkSpaceOffset / sizeof(float));
  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));
  dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dvWorkSpaceOffset / sizeof(float));

  uint32_t coreNum = tilingData->coreNum;
  dqPostAbsorb = tilingData->dqPostAbsorb;

  // compute tiling
  constexpr static uint32_t POST_COEX_NODE = 3;
  constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
  uint32_t curPostCoexNode = POST_COEX_NODE;
  uint32_t ubSize = HardwareInfo<ArchType::ASCEND_V220>::ubSize;
  ubBaseSize = ubSize / curPostCoexNode / POST_BUFFER_NUM;
  ubBaseSize = ubBaseSize / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN; // align

  // dq
  qPostBaseNum = ubBaseSize / sizeof(float);
  qPostBlockTotal = tilingData->qSize;

  int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
  int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;

  qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
  qPostBlockFactor = (qPostBlockOuterTotal + coreNum - 1) / coreNum;

  // dkv
  kvPostBaseNum = qPostBaseNum;
  kvPostBlockTotal = tilingData->kvSize;

  int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
  int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;

  kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
  kvPostBlockFactor = (kvPostBlockOuterTotal + coreNum - 1) / coreNum;

  pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
  pipe->InitBuffer(outQueue, 1, ubBaseSize);
}

template <typename OUT_TYPE, class TILING_TYPE, bool DETERMINISTIC_ENABLE>
__aicore__ inline void VectorPost<OUT_TYPE, TILING_TYPE, DETERMINISTIC_ENABLE>::Process() {
  // init q
  uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
  uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

  if (dqPostAbsorb == 0) {
    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
      qEnd = qPostBlockTotal;
    }
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
      AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
      AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
      uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
      DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
      inQueue.EnQue(vecIn);
      inQueue.template DeQue<float>();

      Muls(vecIn, vecIn, (float)tilingData->scaleValue, dataSize);
      pipe_barrier(PIPE_V);
      Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
      outQueue.EnQue(vecOut);
      outQueue.template DeQue<OUT_TYPE>();
      DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B

      inQueue.FreeTensor(vecIn);
      outQueue.FreeTensor(vecOut);
    }
    pipe_barrier(PIPE_ALL);
  }

  // init k
  uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
  uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
  if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
    kvEnd = kvPostBlockTotal;
  }

  for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
    AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
    AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
    uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
    DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
    inQueue.EnQue(vecIn);
    inQueue.template DeQue<float>();
    Muls(vecIn, vecIn, (float)tilingData->scaleValue, dataSize);
    pipe_barrier(PIPE_V);
    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
    outQueue.EnQue(vecOut);
    outQueue.template DeQue<OUT_TYPE>();
    DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
    inQueue.FreeTensor(vecIn);
    outQueue.FreeTensor(vecOut);
  }
  pipe_barrier(PIPE_ALL);

  // init v
  for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
    AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
    AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
    uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
    DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
    inQueue.EnQue(vecIn);
    inQueue.template DeQue<float>();
    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
    outQueue.EnQue(vecOut);
    outQueue.template DeQue<OUT_TYPE>();
    DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
    inQueue.FreeTensor(vecIn);
    outQueue.FreeTensor(vecOut);
  }
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_POST_H_
