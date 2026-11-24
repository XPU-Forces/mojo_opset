/*!
 * \file flash_attention_score_grad_sfmg.h
 * \brief
 */

#ifndef __VECTOR_SFMG_KERNEL_H__
#define __VECTOR_SFMG_KERNEL_H__

#include "kernel_operator.h"

template <typename IN_TYPE, typename SEQLEN_TYPE> class VectorSoftmaxGrad {
public:
  __aicore__ inline VectorSoftmaxGrad(){};
  __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dy, __gm__ uint8_t *attenIn, __gm__ uint8_t *drop_mask,
                              __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *workspace, int64_t batch,
                              const FlashAttentionTrainingGradTilingData *orgTilingData);
  __aicore__ inline void Process();

protected:
  __aicore__ inline void InitIndex(int64_t startIdx, int64_t &curS, GM_ADDR seqS);
  __aicore__ inline void CopyInSfmg(int64_t leftNburst, int64_t &curS, GM_ADDR seqS);
  __aicore__ inline void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS);

protected:
  constexpr static int64_t BLOCK_BYTE_SIZE = 32;
  constexpr static int64_t BLOCK_SIZE = 8;
  constexpr static int64_t SFMG_HIGH_PERF_N_FACTOR = 8;
  constexpr static int64_t SFMG_HIGH_PERF_D_FACTOR = 64;

  TPipe *pipe;
  uint32_t cBlockIdx;

  const FlashAttentionTrainingGradTilingData *tilingData;

  GlobalTensor<float> sfmgWorkspaceGm;
  GlobalTensor<IN_TYPE> dyGm;
  GlobalTensor<float> attenInGmFp32;
  TQue<QuePosition::VECIN, 1> input1Que, input2Que;
  TBuf<> cast1Buf, tmpBuf;
  TQue<QuePosition::VECOUT, 1> out1Que;

  int64_t b;
  int64_t n1;
  int64_t t1;
  int64_t d;
  int64_t dAlign;
  GM_ADDR actual_seq_qlen_addr;

  int64_t bIdx = 0;
  int64_t nIdx = 0;
  int64_t sIdx = 0;

  int64_t dstOffset = 0;
  int64_t n_stride = 0;

  int64_t usedCoreNum;
  int64_t normalCoreSize;
  int64_t singleLoopNBurstNum;
  int64_t normalCoreLoopTimes;
  int64_t normalCoreLastLoopNBurstNum;
  int64_t tailCoreLoopTimes;
  int64_t tailCoreLastLoopNBurstNum;

  LocalTensor<IN_TYPE> input1Buf;
  LocalTensor<float> input2Buf;
  LocalTensor<float> outputBuf;
};

template <typename IN_TYPE, typename SEQLEN_TYPE>
__aicore__ inline void
VectorSoftmaxGrad<IN_TYPE, SEQLEN_TYPE>::Init(TPipe *pipe_in, __gm__ uint8_t *dy, __gm__ uint8_t *attenInFp32,
                                              __gm__ uint8_t *drop_mask, __gm__ uint8_t *actual_seq_qlen,
                                              __gm__ uint8_t *workspace, int64_t batch,
                                              const FlashAttentionTrainingGradTilingData *orgTilingData) {
  cBlockIdx = GetBlockIdx();
  tilingData = orgTilingData;
  pipe = pipe_in;

  b = batch;
  t1 = tilingData->t1;
  n1 = tilingData->n2 * tilingData->g;
  d = tilingData->d;
  dAlign = (d + 15) / 16 * 16;
  actual_seq_qlen_addr = actual_seq_qlen;

  n_stride = (n1 - 1) * d * sizeof(IN_TYPE);

  uint32_t coreNum = tilingData->coreNum;

  // 计算 buffer 大小
  constexpr static uint32_t inputBufferLen = 24 * 1024;                                 // castBuffer 24K*2=48K
  constexpr static uint32_t castBufferLen = 48 * 1024;                                  // castBuffer 48K*2=96K
  uint32_t outputBufferLen = CeilDiv(castBufferLen, static_cast<uint32_t>(dAlign)) * 8; // 输出(s1,8)
  uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

  // 计算单核的计算量
  int64_t normalAxisSize = t1 * n1;
  normalCoreSize = CeilDiv(normalAxisSize, static_cast<int64_t>(coreNum));
  usedCoreNum = CeilDiv(normalAxisSize, normalCoreSize);

  // 计算单loop的计算量及loop次数
  singleLoopNBurstNum = inputBufferLen / sizeof(float) / dAlign;
  normalCoreLoopTimes = CeilDiv(normalCoreSize, singleLoopNBurstNum);
  normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;

  int64_t tailCoreSize = normalAxisSize - (usedCoreNum - 1) * normalCoreSize;
  tailCoreLoopTimes = CeilDiv(tailCoreSize, singleLoopNBurstNum);
  tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

  // 初始化 buffer
  pipe->InitBuffer(input1Que, 1, inputBufferLen); // 24K
  pipe->InitBuffer(input2Que, 1, castBufferLen);  // 48K
  pipe->InitBuffer(cast1Buf, castBufferLen);      // 48K
  pipe->InitBuffer(out1Que, 1, outputBufferLen);
  pipe->InitBuffer(tmpBuf, tempBufferLen); // 40K - outputBufferLen

  // 初始化 GM
  dyGm.SetGlobalBuffer((__gm__ IN_TYPE *)dy);
  attenInGmFp32.SetGlobalBuffer((__gm__ float *)attenInFp32);
  sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->sfmgWorkspaceOffset / sizeof(float));
}

template <typename IN_TYPE, typename SEQLEN_TYPE>
__aicore__ inline void VectorSoftmaxGrad<IN_TYPE, SEQLEN_TYPE>::InitIndex(int64_t startIdx, int64_t &curS,
                                                                          GM_ADDR seqS) {
  int64_t totalLen = 0;
  for (int64_t bDimIdx = bIdx; bDimIdx < b; bDimIdx++) {
    totalLen = n1 * ((__gm__ SEQLEN_TYPE *)seqS)[bDimIdx] * d;
    if (totalLen > startIdx) {
      bIdx = bDimIdx;
      curS = (bIdx == 0) ? ((__gm__ SEQLEN_TYPE *)seqS)[bIdx]
                         : (((__gm__ SEQLEN_TYPE *)seqS)[bIdx] - ((__gm__ SEQLEN_TYPE *)seqS)[bIdx - 1]);
      int64_t bTail = startIdx - (totalLen - n1 * curS * d);
      nIdx = bTail / (curS * d);
      int64_t nTail = bTail % (curS * d);
      sIdx = nTail / d;
      break;
    }
  }
}

template <typename IN_TYPE, typename SEQLEN_TYPE>
__aicore__ inline void VectorSoftmaxGrad<IN_TYPE, SEQLEN_TYPE>::DoCopyIn(int64_t curS, int64_t curNBurst,
                                                                         int64_t dstOffset, GM_ADDR seqS) {
  int64_t srcOffset = 0;
  int64_t bOffset = bIdx == 0 ? 0 : n1 * ((__gm__ SEQLEN_TYPE *)seqS)[bIdx - 1] * d;
  srcOffset = bOffset + (sIdx * n1 + nIdx) * d;

  DataCopyPad(input1Buf[dstOffset], dyGm[srcOffset],
              {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(IN_TYPE)),
               static_cast<uint32_t>(n_stride), 0, 0},
              {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
  DataCopyPad(input2Buf[dstOffset], attenInGmFp32[srcOffset],
              {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(float)),
               static_cast<uint32_t>(n_stride * 2), 0, 0},
              {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
}

template <typename IN_TYPE, typename SEQLEN_TYPE>
__aicore__ inline void VectorSoftmaxGrad<IN_TYPE, SEQLEN_TYPE>::CopyInSfmg(int64_t leftNburst, int64_t &curS,
                                                                           GM_ADDR seqS) {
  int64_t dstOffset = 0;
  while (leftNburst > 0) {
    int64_t curNburst = 0;
    if (curS - sIdx < leftNburst) { // 需要借N或借B
      curNburst = curS - sIdx;
      DoCopyIn(curS, curNburst, dstOffset, seqS);
      leftNburst = leftNburst - curNburst;
      sIdx = 0;
      if (nIdx < n1 - 1) { // 需要借N
        nIdx += 1;
      } else {
        nIdx = 0;
        if (bIdx < b - 1) { // 需要借B
          bIdx += 1;
          curS = ((__gm__ SEQLEN_TYPE *)seqS)[bIdx] - ((__gm__ SEQLEN_TYPE *)seqS)[bIdx - 1];
        } else { // 没有轴可以借了，end
          leftNburst = 0;
        }
      }
    } else { // 当前S够用
      curNburst = leftNburst;
      DoCopyIn(curS, curNburst, dstOffset, seqS);
      sIdx = sIdx + leftNburst;
      leftNburst = 0;
    }
    dstOffset = dstOffset + curNburst * dAlign;
  }
}

template <typename IN_TYPE, typename SEQLEN_TYPE>
__aicore__ inline void VectorSoftmaxGrad<IN_TYPE, SEQLEN_TYPE>::Process() {
  pipe_barrier(PIPE_ALL); // 去掉pre和sfmg之间的SyncALL，这里需要增加pipeALL

  uint32_t usedCoreNums = usedCoreNum;
  if (cBlockIdx < usedCoreNums) {
    LocalTensor<uint8_t> tempBuf = tmpBuf.Get<uint8_t>();
    LocalTensor<float> sfmgClc1 = cast1Buf.Get<float>();

    int64_t singleCoreLoopTimes = normalCoreLoopTimes;
    int64_t singleCoreLastLoopNBurstNum = normalCoreLastLoopNBurstNum; // 普通单核最后一次loop处理多少个D
    if (cBlockIdx == usedCoreNums - 1) {
      singleCoreLoopTimes = tailCoreLoopTimes;
      singleCoreLastLoopNBurstNum = tailCoreLastLoopNBurstNum;
    }

    int64_t startIdx = cBlockIdx * normalCoreSize;
    int64_t nBurst = singleLoopNBurstNum;
    int64_t curS = 0;

    for (int64_t i = 0; i < singleCoreLoopTimes; i++) {
      if (i == singleCoreLoopTimes - 1) {
        nBurst = singleCoreLastLoopNBurstNum;
      }

      // copyIn
      if (i == 0) {
        input1Buf = input1Que.AllocTensor<IN_TYPE>();
        input2Buf = input2Que.AllocTensor<float>();
        InitIndex((startIdx + i * singleLoopNBurstNum) * d, curS, actual_seq_qlen_addr);
        CopyInSfmg(nBurst, curS, actual_seq_qlen_addr);
      }

      // cast 1
      input1Que.EnQue(input1Buf);
      input1Que.DeQue<IN_TYPE>();
      int64_t calcSize = nBurst * dAlign;
      Cast(sfmgClc1, input1Buf, RoundMode::CAST_NONE, calcSize);
      pipe_barrier(PIPE_V);
      input1Que.FreeTensor(input1Buf);

      // cast 2
      input2Que.EnQue(input2Buf);
      input2Que.DeQue<float>();

      // sfmg
      outputBuf = out1Que.AllocTensor<float>();
      Duplicate<float>(outputBuf, 0.0, nBurst * 8);
      pipe_barrier(PIPE_V);

      uint32_t shapeArray[] = {static_cast<uint32_t>(nBurst), static_cast<uint32_t>(dAlign)};
      sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
      input2Buf.SetShapeInfo(ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
      uint32_t shapeArray1[] = {static_cast<uint32_t>(nBurst), BLOCK_BYTE_SIZE / sizeof(float)};
      outputBuf.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));

      bool isBasicBlock = (nBurst % SFMG_HIGH_PERF_N_FACTOR == 0) && (dAlign % SFMG_HIGH_PERF_D_FACTOR == 0);
      if (likely(isBasicBlock)) {
        SoftmaxGradFront<float, true>(outputBuf, sfmgClc1, input2Buf, tempBuf, tilingData->softmaxGradTilingData);
      } else {
        SoftmaxGradFront<float, false>(outputBuf, sfmgClc1, input2Buf, tempBuf, tilingData->softmaxGradTilingData);
      }
      pipe_barrier(PIPE_V);

      // release attenOutFp32 buffer
      input2Que.FreeTensor(input2Buf);
      // pre copyIn next nBurst
      if (i < singleCoreLoopTimes - 1) {
        int64_t nextNBurst = i == singleCoreLoopTimes - 2 ? singleCoreLastLoopNBurstNum : nBurst;
        input1Buf = input1Que.AllocTensor<IN_TYPE>();
        input2Buf = input2Que.AllocTensor<float>();
        InitIndex((startIdx + (i + 1) * singleLoopNBurstNum) * d, curS, actual_seq_qlen_addr);
        CopyInSfmg(nextNBurst, curS, actual_seq_qlen_addr);
      }

      // copyOut
      out1Que.EnQue(outputBuf);
      out1Que.DeQue<float>();
      int64_t sfmgOutputOffset = (startIdx + i * singleLoopNBurstNum) * BLOCK_SIZE;
      DataCopy(sfmgWorkspaceGm[sfmgOutputOffset], outputBuf, nBurst * BLOCK_SIZE);
      out1Que.FreeTensor(outputBuf);
    }
  }
}

#endif // __VECTOR_SFMG_KERNEL_H__
