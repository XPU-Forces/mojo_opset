#ifndef _VEC_OP_DET_H_
#define _VEC_OP_DET_H_

#include "../common/fa_dropmask.h"
#include "kernel_operator.h"

template <typename FAGT> class VecOpDet {
  using TILING_CLASS = typename FAGT::tiling_class;
  using SEQLEN_TYPE = typename FAGT::seqlen_type;
  using INPUT_TYPE = typename FAGT::input_type;
  static constexpr bool DROP_ENABLE = FAGT::drop_enable;
  static constexpr bool DETERMINISTIC_ENABLE = FAGT::deterministic_enable;

public:
  __aicore__ inline VecOpDet(){};
  __aicore__ void Init(GM_ADDR drop_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR atten_mask,
                       GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                       GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData, TPipe *pipe_in);
  __aicore__ inline void DetVector1(const VecAddrInfoDet &addrs);
  __aicore__ inline void DetVector2(const VecAddrInfoDet &addrs);

private:
  __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, SEQLEN_TYPE &actualSeqQlen, SEQLEN_TYPE &actualSeqKvlen);
  __aicore__ inline void CalFullMask(int64_t curIdx, event_t mte2WaitMte3A, event_t mte2WaitMte3B);
  __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                             uint32_t s1Extend, uint32_t s2Extend);
  __aicore__ inline void CalcAttenMaskBool(LocalTensor<float> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                           uint32_t s1Extend, uint32_t s2SrcExtend, uint32_t s2MaskExtend = 128,
                                           uint8_t maskType = 0);
  __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
  __aicore__ inline void CalcSoftMax(LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor,
                                     LocalTensor<float> &src1Tensor, uint32_t s1Extend, uint32_t s2Extend,
                                     uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
  __aicore__ inline void SubGrapA(int64_t curIdx, const VecBlockInfoDet &blockInfo, event_t mte2WaitMte3A,
                                  const bool skipSftMaxSumCopy);
  __aicore__ inline void SubGrapB(int64_t curIdx, const VecBlockInfoDet &blockInfo, event_t mte2WaitMte3B,
                                  const bool skipSftMaxSumCopy);
  __aicore__ inline void DropOutCopy(LocalTensor<uint8_t> &dropMaskTensor, int64_t s2VBegin);
  __aicore__ inline void OrderAccum(GlobalTensor<float> dstTensor, GlobalTensor<float> srcTensor, DetGroup detGrop,
                                    uint32_t pingpongFlag, int32_t headNum, int32_t startBlkIdx, int32_t usedCubeCore);
  __aicore__ inline void OrderAccumDq(GlobalTensor<INPUT_TYPE> dstTensor, GlobalTensor<float> srcTensor,
                                      DetGroup detGrop, uint32_t pingpongFlag, uint32_t startBlkIdx,
                                      uint32_t usedCubeCore);

protected:
  TBuf<> unifiedBuffer;
  const TILING_CLASS *__restrict tilingData;
  GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
  GlobalTensor<float> mm1WorkspaceGm;
  GlobalTensor<float> mm2WorkspaceGm;
  GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
  GlobalTensor<float> sfmgWorkspaceGm;
  GlobalTensor<INPUT_TYPE> dropWorkSpaceGm;
  GlobalTensor<INPUT_TYPE> dsWorkSpaceGm;
  // 确定性
  GlobalTensor<float> dqDetWorkSpaceGm;
  GlobalTensor<float> dkDetWorkSpaceGm;
  GlobalTensor<float> dvDetWorkSpaceGm;
  GlobalTensor<float> dqWorkSpaceGm;
  GlobalTensor<float> dkWorkSpaceGm;
  GlobalTensor<float> dvWorkSpaceGm;
  GlobalTensor<INPUT_TYPE> dqGm;
  constexpr static uint32_t BLOCK_SIZE = 32;
  constexpr static uint32_t TMP_UB_OFFSET = 158 * 1024;
  constexpr static uint32_t TMP_UB_SIZE = 33 * 1024;
  constexpr static uint32_t TOTAL_SIZE = 191 * 1024;
  constexpr static int32_t CUBE_BLOCK_SIZE = 512 * 512;
  uint32_t eventIdAPing = 3;
  uint32_t eventIdAPong = 4;
  uint32_t eventIdBPing = 5;
  uint32_t eventIdBPong = 6;
  // org shape info
  int64_t batch;
  int64_t n2;
  int64_t g;
  int64_t headNum;
  int64_t headDim;
  // core param
  uint32_t vecCoreIdx;
  uint32_t cubeCoreIdx;
  uint32_t subIdx;
  // attr param
  uint32_t sparseMode;
  uint32_t dqPostAbsorb;
  int32_t enableCausalOpt;
  float scaleValue;
  // cal param
  int32_t blockLen = 0;
  int32_t s1VecSize;
  int32_t s1Extend;
  int32_t s2Extend;
  int32_t s2ExtendAlign;
  int32_t s1CubeExtend;
  int32_t s2CubeExtend;
  int32_t attenMaskDimS2;
  // offset
  int64_t sfmgOffset = 0;
  int64_t attnMaskOffset = 0;
  int64_t copyInOffset = 0;
  int64_t copyOutOffset = 0;
  DataCopyParams copyInParam;
  DataCopyParams copyOutParam;
  GM_ADDR actual_seq_qlen_addr;
  GM_ADDR actual_seq_kvlen_addr;
  mojo::kernels::DropMaskInfo dropMaskInfo = {0};
  // LocalTensor
  LocalTensor<uint8_t> attenMaskTensor;
  LocalTensor<uint8_t> dropMaskTensor;
  LocalTensor<uint8_t> dropMaskCopyTensor;
  LocalTensor<float> mm2OutTensor;
  LocalTensor<float> sftMaxSumTensor;
  LocalTensor<float> softmaxResTensor;
  LocalTensor<INPUT_TYPE> softmaxCastTensor;
  LocalTensor<INPUT_TYPE> softmaxGradCastTensor;
  LocalTensor<float> sfmgTensor;
  LocalTensor<float> mm1OutTensor;
  LocalTensor<uint8_t> tmpTensor;
  // 确定性
  LocalTensor<float> addPingTensor;
  LocalTensor<float> addPongTensor;
  LocalTensor<INPUT_TYPE> addCastPingTensor;
  LocalTensor<INPUT_TYPE> addCastPongTensor;
  LocalTensor<float> copyPingTensor;
  LocalTensor<float> copyPongTensor;
};

template <typename FAGT>
__aicore__ void VecOpDet<FAGT>::Init(GM_ADDR drop_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR atten_mask,
                                     GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk,
                                     GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData,
                                     TPipe *pipe) {
  // ub分配
  tilingData = ordTilingData;
  vecCoreIdx = GetBlockIdx();
  cubeCoreIdx = vecCoreIdx / 2;
  subIdx = vecCoreIdx % 2;
  batch = tilingData->b;
  headNum = tilingData->n2 * tilingData->g;
  headDim = tilingData->d;
  n2 = tilingData->n2;
  g = tilingData->g;
  scaleValue = tilingData->scaleValue;
  sparseMode = tilingData->sparseMode;
  dqPostAbsorb = tilingData->dqPostAbsorb;
  actual_seq_qlen_addr = actual_seq_qlen;
  actual_seq_kvlen_addr = actual_seq_kvlen;

  pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);

  // global tensor
  softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
  softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
  attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
  mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->mm1WorkspaceOffset));
  mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->mm2WorkspaceOffset));
  dsWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)(workspace + tilingData->mm1WorkspaceOffset));
  dropWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)(workspace + tilingData->mm2WorkspaceOffset));
  sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sfmgWorkspaceOffset));

  // 确定性
  if constexpr (DETERMINISTIC_ENABLE) {
    dqDetWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqDetWorkspaceOffset));
    dkDetWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkDetWorkspaceOffset));
    dvDetWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvDetWorkspaceOffset));
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqWorkSpaceOffset));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkWorkSpaceOffset));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvWorkSpaceOffset));
    dqGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq);
  }

  if constexpr (DROP_ENABLE == true) {
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    // init dropMaskInfo
    dropMaskInfo.n2G = n2 * g;
    dropMaskInfo.gSize = g;
    dropMaskInfo.s2Idx = 1;
    dropMaskInfo.s1BaseSize = 1;
    dropMaskInfo.s1InnerIdx = subIdx;
    dropMaskInfo.boolMode = false;
    dropMaskInfo.keepProb = tilingData->keepProb;
  }

  uint32_t ubOffset = 0;
  attenMaskTensor = unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), 0);
  ubOffset += 16 * 1024;
  sftMaxSumTensor = unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubOffset);
  ubOffset += 8 * 1024;
  mm2OutTensor = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubOffset);
  softmaxCastTensor = unifiedBuffer.GetWithOffset<INPUT_TYPE>(17 * 1024 / sizeof(INPUT_TYPE), ubOffset);
  ubOffset += 32 * 1024;
  softmaxResTensor = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), ubOffset);
  ubOffset += 33 * 1024;
  sfmgTensor = unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubOffset);
  ubOffset += 8 * 1024;
  mm1OutTensor = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), ubOffset);
  softmaxGradCastTensor = unifiedBuffer.GetWithOffset<INPUT_TYPE>(16 * 1024 / sizeof(INPUT_TYPE), ubOffset);
  ubOffset += 33 * 1024;

  dropMaskTensor = unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubOffset);
  ubOffset += 8 * 1024;
  dropMaskCopyTensor = unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubOffset);
  ubOffset += 8 * 1024;
  tmpTensor = unifiedBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), ubOffset);

  // 确定性
  // 由于attenmask常驻，从attenmask之后开始
  uint32_t detUbOffset = 16 * 1024;
  addPingTensor = unifiedBuffer.GetWithOffset<float>(16 * 1024 / sizeof(float), detUbOffset);
  addCastPingTensor = unifiedBuffer.GetWithOffset<INPUT_TYPE>(8 * 1024 / sizeof(INPUT_TYPE), detUbOffset);
  detUbOffset += 16 * 1024;
  addPongTensor = unifiedBuffer.GetWithOffset<float>(16 * 1024 / sizeof(float), detUbOffset);
  addCastPongTensor = unifiedBuffer.GetWithOffset<INPUT_TYPE>(8 * 1024 / sizeof(INPUT_TYPE), detUbOffset);
  detUbOffset += 16 * 1024;
  copyPingTensor = unifiedBuffer.GetWithOffset<float>(16 * 1024 / sizeof(float), detUbOffset);
  detUbOffset += 16 * 1024;
  copyPongTensor = unifiedBuffer.GetWithOffset<float>(16 * 1024 / sizeof(float), detUbOffset);
  detUbOffset += 16 * 1024;
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::GetSeqQlenKvlenByBidx(int64_t bIdx, SEQLEN_TYPE &actualSeqQlen,
                                                             SEQLEN_TYPE &actualSeqKvlen) {
  if (unlikely(bIdx == 0)) {
    actualSeqQlen = ((__gm__ SEQLEN_TYPE *)actual_seq_qlen_addr)[0];
    actualSeqKvlen = ((__gm__ SEQLEN_TYPE *)actual_seq_kvlen_addr)[0];
  } else {
    actualSeqQlen =
        ((__gm__ SEQLEN_TYPE *)actual_seq_qlen_addr)[bIdx] - ((__gm__ SEQLEN_TYPE *)actual_seq_qlen_addr)[bIdx - 1];
    actualSeqKvlen =
        ((__gm__ SEQLEN_TYPE *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ SEQLEN_TYPE *)actual_seq_kvlen_addr)[bIdx - 1];
  }
  return;
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                                           uint32_t s1Extend, uint32_t s2Extend) {
  AscendC::DataCopyExtParams intriParams;
  intriParams.blockCount = s1Extend;
  intriParams.blockLen = s2Extend * sizeof(uint8_t);
  intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
  intriParams.dstStride = 0;
  intriParams.rsv = 0;

  DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::CalcAttenMaskBool(LocalTensor<float> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                                         uint32_t s1Extend, uint32_t s2SrcExtend, uint32_t s2MaskExtend,
                                                         uint8_t maskType) {
  LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
  uint32_t tmp = 0xFF7FFFFF;
  float scalar = *((float *)&tmp);
  SelectWithBytesMaskShapeInfo info;
  info.firstAxis = s1Extend;
  info.srcLastAxis = s2SrcExtend;
  info.maskLastAxis = s2MaskExtend;
  dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
  srcTensor.SetSize(info.firstAxis * info.maskLastAxis);

  if (maskType == 0) {
    SelectWithBytesMask<float, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
  } else {
    SelectWithBytesMask<float, uint8_t, false>(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
  }
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                     uint32_t softMaxOffset) {
  DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0},
              {false, 0, 0, 0});
  DataCopyPad(dstTensor[64 * 8], softmaxMaxGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0},
              {false, 0, 0, 0});
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::CalcSoftMax(LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor,
                                                   LocalTensor<float> &src1Tensor, uint32_t s1Extend, uint32_t s2Extend,
                                                   uint32_t s2ExtendAlign, const SoftMaxTiling &tiling) {
  bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);
  if (isBasicBlock) {
    LocalTensor<uint8_t> sharedTmp = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
    uint32_t shapeArray1[2];
    shapeArray1[0] = s1Extend;
    shapeArray1[1] = s2Extend;
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
    src0Tensor.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
    SimpleSoftMax<float, false, true>(dstTensor, src1Tensor, src1Tensor[64 * 8], src0Tensor, sharedTmp, tiling);
  } else {
    LocalTensor<float> vecOutBuffer = unifiedBuffer.GetWithOffset<float>(TMP_UB_SIZE / sizeof(float), TMP_UB_OFFSET);
    uint32_t sub_block_count = (s2Extend + CAL_REPEAT_NUM - 1) / CAL_REPEAT_NUM;
    for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
      uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * CAL_REPEAT_NUM) : CAL_REPEAT_NUM;
      Sub(dstTensor[subIdx * CAL_REPEAT_NUM], src0Tensor[subIdx * CAL_REPEAT_NUM], src1Tensor[64 * 8], subMaskCount,
          s1Extend,
          {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
           static_cast<uint8_t>(s2ExtendAlign / 8), 1});
      pipe_barrier(PIPE_V);
      Exp(vecOutBuffer[subIdx * CAL_REPEAT_NUM], dstTensor[subIdx * CAL_REPEAT_NUM], subMaskCount, s1Extend,
          {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
           static_cast<uint8_t>(s2ExtendAlign / 8)});
      pipe_barrier(PIPE_V);
      Div(dstTensor[subIdx * CAL_REPEAT_NUM], vecOutBuffer[subIdx * CAL_REPEAT_NUM], src1Tensor, subMaskCount, s1Extend,
          {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
           static_cast<uint8_t>(s2ExtendAlign / 8), 1});
      pipe_barrier(PIPE_V);
    }
  }
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::CalFullMask(int64_t curIdx, event_t mte2WaitMte3A, event_t mte2WaitMte3B) {
  if (curIdx > 0) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3A);
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3B);
  }
  InitOutput<INPUT_TYPE>(dropWorkSpaceGm[copyOutOffset], BASE_BLOCK_SIZE, 0.0);
  InitOutput<INPUT_TYPE>(dsWorkSpaceGm[copyOutOffset], BASE_BLOCK_SIZE, 0.0);
  if (curIdx < blockLen - 1) {
    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3A);
    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3B);
  }
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::DropOutCopy(LocalTensor<uint8_t> &dropMaskTensor, int64_t s2VBegin) {
  // for compute dropout mask offset
  dropMaskInfo.s2StartIdx = s2VBegin;
  // for copy in dropout mask
  dropMaskInfo.s2CopySize = s2Extend;
  mojo::kernels::CopyInDropMask<true>(dropMaskTensor, maskWorkSpaceGm, dropMaskGm, this->dropMaskInfo);
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::SubGrapA(int64_t curIdx, const VecBlockInfoDet &blockInfo, event_t mte2WaitMte3A,
                                                const bool skipSftMaxSumCopy) {
  if (curIdx > 0) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3A);
  }
  DataCopyPad(mm2OutTensor, mm2WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  Muls(mm2OutTensor, mm2OutTensor, scaleValue, s1Extend * s2ExtendAlign);
  pipe_barrier(PIPE_V);

  if (blockInfo.calNextToken) {
    LocalTensor<uint8_t> tmpAttenMaskTensor = attenMaskTensor[subIdx * s1VecSize * 128];
    if (!enableCausalOpt) {
      tmpAttenMaskTensor = attenMaskTensor;
      int32_t offset = blockInfo.nextTokenOffset + subIdx * s1VecSize * attenMaskDimS2;
      CopyInAttenMaskBool(tmpAttenMaskTensor, offset, 64, 128);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    CalcAttenMaskBool(mm2OutTensor, tmpAttenMaskTensor, s1Extend, s2ExtendAlign, 128, 0);
    pipe_barrier(PIPE_V);
  }

  if (blockInfo.calPreToken) {
    auto tmpAttenMaskTensor = attenMaskTensor[64 * 128];
    int32_t offset = blockInfo.preTokenOffset + subIdx * s1VecSize * attenMaskDimS2;
    CopyInAttenMaskBool(tmpAttenMaskTensor, offset, 64, 128);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    CalcAttenMaskBool(mm2OutTensor, tmpAttenMaskTensor, s1Extend, s2ExtendAlign, 128, 1);
    pipe_barrier(PIPE_V);
  }

  if (!skipSftMaxSumCopy) {
    CopyInSoftMax(sftMaxSumTensor, s1Extend, sfmgOffset);
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  CalcSoftMax(softmaxResTensor, mm2OutTensor, sftMaxSumTensor, s1Extend, s2Extend, s2ExtendAlign,
              tilingData->softmaxTilingData);
  pipe_barrier(PIPE_V);

  if constexpr (DROP_ENABLE == true) {
    DropOutCopy(dropMaskTensor, blockInfo.s2Idx);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    DataCopy(dropMaskCopyTensor, dropMaskTensor, 8 * 1024);
    pipe_barrier(PIPE_V);
  }
  LocalTensor<float> vecDropBuffer = softmaxResTensor;

  if constexpr (DROP_ENABLE == true) {
    vecDropBuffer = mm2OutTensor;
    dropMaskInfo.lstAxis = s2ExtendAlign;
    dropMaskInfo.maskLstAxis = s2ExtendAlign;
    mojo::kernels::ComputeDropMask<float, true>(vecDropBuffer, softmaxResTensor, dropMaskTensor, tmpTensor,
                                                this->dropMaskInfo);
    pipe_barrier(PIPE_V);
  }

  Cast(softmaxCastTensor, vecDropBuffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
  pipe_barrier(PIPE_V);

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  DataCopyPad(dropWorkSpaceGm[copyOutOffset], softmaxCastTensor, copyOutParam);
  if (curIdx < blockLen - 1) {
    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3A);
  }
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::SubGrapB(int64_t curIdx, const VecBlockInfoDet &blockInfo, event_t mte2WaitMte3B,
                                                const bool skipSftMaxSumCopy) {
  if (curIdx > 0) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3B);
  }
  DataCopyPad(mm1OutTensor, mm1WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

  if constexpr (DROP_ENABLE == true) {
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    mojo::kernels::ComputeDropMask<float, true>(mm1OutTensor, mm1OutTensor, dropMaskCopyTensor, tmpTensor,
                                                this->dropMaskInfo);
    pipe_barrier(PIPE_V);
  }

  if (!skipSftMaxSumCopy) {
    DataCopy(sfmgTensor, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  uint32_t sub_block_cout = (s2ExtendAlign + CAL_REPEAT_NUM - 1) / CAL_REPEAT_NUM;
  for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
    uint32_t subMaskCout = (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * CAL_REPEAT_NUM) : CAL_REPEAT_NUM;
    Sub(mm1OutTensor[subIdx * CAL_REPEAT_NUM], mm1OutTensor[subIdx * CAL_REPEAT_NUM], sfmgTensor, subMaskCout, s1Extend,
        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
         static_cast<uint8_t>(s2ExtendAlign / 8), 1});
  }
  pipe_barrier(PIPE_V);

  Mul(mm1OutTensor, mm1OutTensor, softmaxResTensor, s1Extend * s2ExtendAlign);
  pipe_barrier(PIPE_V);

  Cast(softmaxGradCastTensor, mm1OutTensor, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
  pipe_barrier(PIPE_V);

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // dyv = dp -> ds
  DataCopyPad(dsWorkSpaceGm[copyOutOffset], softmaxGradCastTensor, copyOutParam);
  if (curIdx < blockLen - 1) {
    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3B);
  }
}

template <typename FAGT> __aicore__ inline void VecOpDet<FAGT>::DetVector1(const VecAddrInfoDet &addrs) {
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_V);

  int32_t taskId = addrs.taskId;
  int32_t pingpongIdx = taskId % 2;
  int32_t batchIdx = addrs.batchIdx;
  SEQLEN_TYPE actualS1Len;
  SEQLEN_TYPE actualS2Len;
  blockLen = addrs.blockLength;
  if (taskId == 0) {
    attenMaskDimS2 = addrs.attenMaskDimS2;
    enableCausalOpt = addrs.enableCausalOpt;
    if (enableCausalOpt) {
      CopyInAttenMaskBool(attenMaskTensor, 0, 128, 128);
    }
  }
  GetSeqQlenKvlenByBidx(batchIdx, actualS1Len, actualS2Len);
  if constexpr (DROP_ENABLE == true) {
    dropMaskInfo.s2TotalSize = actualS2Len;
    dropMaskInfo.s1Size = actualS1Len;
    dropMaskInfo.s2Size = actualS2Len;
    int64_t bSSOffset = 0;
    SEQLEN_TYPE tmpS1Len;
    SEQLEN_TYPE tmpS2Len;
    for (int64_t bidx = 0; bidx < batchIdx; bidx++) {
      GetSeqQlenKvlenByBidx(bidx, tmpS1Len, tmpS2Len);
      bSSOffset += tmpS1Len * AlignUp(tmpS2Len, (int64_t)8);
    }
    dropMaskInfo.bSSOffset = bSSOffset;
  }

  for (uint32_t i = 0; i < blockLen; ++i) {
    auto &blockInfo = addrs.VecBlkInfo[i];
    bool skipSftMaxSumCopy = false;
    s1VecSize = (blockInfo.s1Len + 1) / 2;
    s1Extend = subIdx ? blockInfo.s1Len - s1VecSize : s1VecSize;
    s2Extend = blockInfo.s2Len;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    if constexpr (DROP_ENABLE == true) {
      dropMaskInfo.gOutIdx = blockInfo.gIdx;
      dropMaskInfo.n2OutIdx = blockInfo.n2Idx;
      dropMaskInfo.s1OutIdx = blockInfo.s1Idx;
      dropMaskInfo.firstAxis = (blockInfo.s1Len + 1) / 2;
      dropMaskInfo.s1CopySize = (blockInfo.s1Len + 1) / 2;
      dropMaskInfo.splitS1BaseSize = (blockInfo.s1Len + 1) / 2;
    }
    sfmgOffset = 0;
    if (batchIdx > 0) {
      sfmgOffset = ((__gm__ SEQLEN_TYPE *)actual_seq_qlen_addr)[batchIdx - 1] * n2 * g * 8;
    }
    sfmgOffset += ((blockInfo.n2Idx * g + blockInfo.gIdx) * actualS1Len + blockInfo.s1Idx + subIdx * s1VecSize) * 8;

    copyInOffset = cubeCoreIdx * CUBE_BLOCK_SIZE * 2 + pingpongIdx * CUBE_BLOCK_SIZE + blockInfo.cubeBlockOffset +
                   subIdx * s1VecSize * 128;
    copyOutOffset = 2 * copyInOffset;
    copyInParam = {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                   static_cast<uint16_t>((128 - s2ExtendAlign) * sizeof(float)), 0};
    copyOutParam = {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(INPUT_TYPE)), 0,
                    static_cast<uint16_t>((128 - s2ExtendAlign) * sizeof(INPUT_TYPE))};
    ///////////////////////////////////////////////////////////////
    // do vector calculate
    ///////////////////////////////////////////////////////////////
    event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    if (unlikely(blockInfo.isFullMask)) {
      CalFullMask(i, mte2WaitMte3A, mte2WaitMte3B);
    } else {
      SubGrapA(i, blockInfo, mte2WaitMte3A, skipSftMaxSumCopy);
      SubGrapB(i, blockInfo, mte2WaitMte3B, skipSftMaxSumCopy);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3A);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3B);
  }
}

template <typename FAGT> __aicore__ inline void VecOpDet<FAGT>::DetVector2(const VecAddrInfoDet &addrs) {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_V);
  set_flag(PIPE_MTE3, PIPE_MTE2, eventIdAPing);
  set_flag(PIPE_MTE3, PIPE_MTE2, eventIdAPong);
  set_flag(PIPE_V, PIPE_MTE2, eventIdBPing);
  set_flag(PIPE_V, PIPE_MTE2, eventIdBPong);

  uint32_t pingpongFlag = 0;

  if (vecCoreIdx < 16) {
    if (dqPostAbsorb) {
      for (uint32_t groupIdx = 0; groupIdx < addrs.s1GroupNum; groupIdx++) {
        OrderAccumDq(dqGm, dqDetWorkSpaceGm, addrs.detS1GroupList[groupIdx], pingpongFlag, 0, 16);
        pingpongFlag = 1 - pingpongFlag;
      }
    } else {
      for (uint32_t groupIdx = 0; groupIdx < addrs.s1GroupNum; groupIdx++) {
        OrderAccum(dqWorkSpaceGm, dqDetWorkSpaceGm, addrs.detS1GroupList[groupIdx], pingpongFlag, n2 * g, 0, 16);
        pingpongFlag = 1 - pingpongFlag;
      }
    }
  }

  if (vecCoreIdx >= 16 && vecCoreIdx < 32) {
    // dk
    for (uint32_t groupIdx = 0; groupIdx < addrs.s2GroupNum; groupIdx++) {
      if (addrs.detS2GroupList[groupIdx].accumNum == 1) {
        continue;
      }
      OrderAccum(dkWorkSpaceGm, dkDetWorkSpaceGm, addrs.detS2GroupList[groupIdx], pingpongFlag, n2, 16, 16);
      pingpongFlag = 1 - pingpongFlag;
    }
  }

  if (vecCoreIdx >= 32) {
    // dv
    for (uint32_t groupIdx = 0; groupIdx < addrs.s2GroupNum; groupIdx++) {
      if (addrs.detS2GroupList[groupIdx].accumNum == 1) {
        continue;
      }
      OrderAccum(dvWorkSpaceGm, dvDetWorkSpaceGm, addrs.detS2GroupList[groupIdx], pingpongFlag, n2, 32, 16);
      pingpongFlag = 1 - pingpongFlag;
    }
  }

  wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdAPing);
  wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdAPong);
  wait_flag(PIPE_V, PIPE_MTE2, eventIdBPing);
  wait_flag(PIPE_V, PIPE_MTE2, eventIdBPong);
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::OrderAccumDq(GlobalTensor<INPUT_TYPE> dstTensor, GlobalTensor<float> srcTensor,
                                                    DetGroup detGrop, uint32_t pingpongFlag, uint32_t startBlkIdx,
                                                    uint32_t usedCubeCore) {
  LocalTensor<float> addTensor = pingpongFlag ? addPingTensor : addPongTensor;
  LocalTensor<INPUT_TYPE> addCastTensor = pingpongFlag ? addCastPingTensor : addCastPongTensor;
  uint32_t processS1 = detGrop.recoderS / usedCubeCore;
  uint32_t tailS1 = detGrop.recoderS % usedCubeCore;
  uint32_t loclBlockIdx = vecCoreIdx - startBlkIdx;
  uint64_t copyInGmOffset, copyOutGmOffset;
  uint32_t headNum = n2 * g;
  uint32_t eventIdA = pingpongFlag ? eventIdAPing : eventIdAPong;

  if (loclBlockIdx < tailS1) {
    processS1 += 1;
    copyInGmOffset = loclBlockIdx * processS1 * headDim;
    copyOutGmOffset = detGrop.outAddr + (loclBlockIdx * processS1) * headNum * headDim;
  } else {
    copyInGmOffset = (loclBlockIdx * processS1 + tailS1) * headDim;
    copyOutGmOffset = detGrop.outAddr + (loclBlockIdx * processS1 + tailS1) * headNum * headDim;
  }

  uint32_t dataSize = processS1 * headDim;
  uint32_t resIdx = detGrop.accumList[0];
  AscendC::DataCopyExtParams intriParams;
  intriParams.blockCount = processS1;
  intriParams.blockLen = headDim * sizeof(INPUT_TYPE);
  intriParams.srcStride = 0;
  intriParams.dstStride = (headNum - 1) * headDim * sizeof(INPUT_TYPE);

  // first process
  wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdA);
  DataCopy(addTensor, srcTensor[copyInGmOffset + resIdx * 512 * headDim], dataSize);

  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // middle process
  for (uint32_t i = 1; i < detGrop.accumNum; i++) {
    LocalTensor<float> copyTensor = i % 2 ? copyPingTensor : copyPongTensor;
    uint32_t eventId = i % 2 ? eventIdBPing : eventIdBPong;
    resIdx = detGrop.accumList[i];

    wait_flag(PIPE_V, PIPE_MTE2, eventId);
    DataCopy(copyTensor, srcTensor[copyInGmOffset + resIdx * 512 * headDim], dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Add(addTensor, addTensor, copyTensor, dataSize);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, eventId);
  }

  // end process
  if (detGrop.existLastBlock) {
    if (!detGrop.existFirstBlock) {
      // 最后一块，从临时GM上读取之前的累加数据，直接做post操作并输出
      LocalTensor<float> copyTensor = (detGrop.accumNum % 2) ? copyPingTensor : copyPongTensor;
      uint32_t eventId = (detGrop.accumNum % 2) ? eventIdBPing : eventIdBPong;

      wait_flag(PIPE_V, PIPE_MTE2, eventId);
      DataCopy(copyTensor, dqWorkSpaceGm[copyInGmOffset], dataSize);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      Add(addTensor, addTensor, copyTensor, dataSize);
      set_flag(PIPE_V, PIPE_MTE2, eventId);
      pipe_barrier(PIPE_V);
    }

    // 既是第一块也是最后一块，直接做post操作并输出
    Muls(addTensor, addTensor, (float)tilingData->scaleValue, dataSize);
    pipe_barrier(PIPE_V);

    Cast(addCastTensor, addTensor, AscendC::RoundMode::CAST_ROUND, dataSize);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopyPad(dstTensor[copyOutGmOffset], addCastTensor, intriParams);
  } else if (detGrop.existFirstBlock && !detGrop.existLastBlock) {
    // 第一块，直接覆盖输出到临时GM上
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dqWorkSpaceGm[copyInGmOffset], addTensor, dataSize);
  } else {
    // 既不是最后一块也是不是最后一块，
    // 则在临时GM上累加
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    AscendC::SetAtomicType<float>();
    DataCopy(dqWorkSpaceGm[copyInGmOffset], addTensor, dataSize);
    AscendC::SetAtomicNone();
  }

  set_flag(PIPE_MTE3, PIPE_MTE2, eventIdA);
}

template <typename FAGT>
__aicore__ inline void VecOpDet<FAGT>::OrderAccum(GlobalTensor<float> dstTensor, GlobalTensor<float> srcTensor,
                                                  DetGroup detGrop, uint32_t pingpongFlag, int32_t headNum,
                                                  int32_t startBlkIdx, int32_t usedCubeCore) {
  LocalTensor<float> addTensor = pingpongFlag ? addPingTensor : addPongTensor;
  uint32_t eventIdA = pingpongFlag ? eventIdAPing : eventIdAPong;
  uint32_t processS1 = detGrop.recoderS / usedCubeCore;
  uint32_t tailS1 = detGrop.recoderS % usedCubeCore;
  uint32_t loclBlockIdx = vecCoreIdx - startBlkIdx;
  uint64_t copyInGmOffset, copyOutGmOffset;
  if (loclBlockIdx < tailS1) {
    processS1 += 1;
    copyInGmOffset = loclBlockIdx * processS1 * headDim;
    copyOutGmOffset = detGrop.outAddr + (loclBlockIdx * processS1) * headNum * headDim;
  } else {
    copyInGmOffset = (loclBlockIdx * processS1 + tailS1) * headDim;
    copyOutGmOffset = detGrop.outAddr + (loclBlockIdx * processS1 + tailS1) * headNum * headDim;
  }

  uint32_t dataSize = processS1 * headDim;
  uint32_t resIdx = detGrop.accumList[0];
  // first process
  wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdA);
  DataCopy(addTensor, srcTensor[copyInGmOffset + resIdx * 512 * headDim], dataSize);

  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // middle process
  for (uint32_t i = 1; i < detGrop.accumNum; i++) {
    int32_t resIdx = detGrop.accumList[i];
    uint32_t eventId = i % 2 ? eventIdBPing : eventIdBPong;
    LocalTensor<float> copyTensor = i % 2 ? copyPingTensor : copyPongTensor;

    wait_flag(PIPE_V, PIPE_MTE2, eventId);
    DataCopy(copyTensor, srcTensor[copyInGmOffset + resIdx * 512 * headDim], dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Add(addTensor, addTensor, copyTensor, dataSize);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, eventId);
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  // end process
  AscendC::DataCopyExtParams intriParams;
  intriParams.blockCount = processS1;
  intriParams.blockLen = headDim * sizeof(float);
  intriParams.srcStride = 0;
  intriParams.dstStride = (headNum - 1) * headDim * sizeof(float);

  AscendC::SetAtomicType<float>();
  DataCopyPad(dstTensor[copyOutGmOffset], addTensor, intriParams);
  AscendC::SetAtomicNone();

  set_flag(PIPE_MTE3, PIPE_MTE2, eventIdA);
}
#endif
