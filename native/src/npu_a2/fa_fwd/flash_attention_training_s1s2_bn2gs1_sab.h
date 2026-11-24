#ifndef FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H
#define FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H

#include "../common/fa_dropmask.h"
#include "flash_attention_training_common.h"
#include "kernel_operator.h"
#include "fa_tiling.h"
#include "lib/matmul_intf.h"
#include "../common/fa_util.h"

using matmul::MatmulType;
constexpr int64_t GM_DOUBLE_BUFFER = 2;
constexpr int64_t INVALID_OFFSET = INT64_MIN;
constexpr int64_t SPLIT_S2_SIZE_LIMIT = 1024;
constexpr AscendC::SoftmaxConfig MOJO_FA_SOFTMAX_CFG = {false};

namespace mojo {
namespace kernels {

struct SplitSameABExtraInfo {
  int64_t s2StartIdx;
  int64_t s2EndIdx;
  int64_t s2LoopCount;
  int64_t s1oIdx;
  int64_t boIdx;
  int64_t n2oIdx;
  int64_t goIdx;
  int64_t taskId;
  int8_t taskIdMod2;
  int8_t multiCoreInnerIdxMod2;
  int8_t needNz2Nd;
  int32_t s1RealSize;
  int32_t cubeS1RealSize;
  int32_t s2RealSize;
  int32_t s2AlignedSize;

  int32_t s2RealSizeAlign64 = 0;
  uint64_t duplicateMask = 0;
  int32_t s2RealSizeFloorAlign8 = 0;
  bool s2LastLoop = false;

  int32_t vec1S1BaseSize;
  int32_t vec1S1RealSize;
  int32_t vec2S1BaseSize;
  int32_t vec2S1RealSize;
  int32_t realSplitN;
  int32_t s2LoopLimit;
  int64_t multiCoreInnerIdx;
  int64_t qCoreOffset;
  int64_t s1SizeDelta;
  int64_t s1SizeAcc;
  int64_t s2SizeAcc;
  int64_t attenB1SSOffset;
  int64_t attenMaskS2Size;
  int64_t s1Size;
  int64_t s2Size;
  int64_t softmaxMaxOffset;
  int64_t vecCoreOffset;
};

constexpr int64_t VEC_BUFFER_SIZE = 1024;

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, typename SEQLEN_TYPE = int64_t, CubeFormat bmm1Format = CubeFormat::NZ>
class FlashAttentionTrainingS1s2Bn2gs1SameAB {
public:
  __aicore__ inline FlashAttentionTrainingS1s2Bn2gs1SameAB(){};

  __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                              __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                              __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                              const FlashAttentionTrainingGeneralTilingData *__restrict tiling, TPipe *tPipe);
  __aicore__ inline void Process();

  // define matmul
  using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
  using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, true>;
  using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
  matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

  using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
  matmul::Matmul<a1Type, b1Type, c1NzType, bias1Type, CFG_EXCEED> bmm1Nz;

  // define batchmatmul
  using a2Type = MatmulType<TPosition::GM, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
  using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
  using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
  matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED> bmm2;

protected:
  __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                   __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                   __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                   const FlashAttentionTrainingGeneralTilingData *__restrict tiling, TPipe *tPipe);
  __aicore__ inline void WaitBmm1Result(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void WaitBmm2Result(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                     matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                                                    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm2);
  __aicore__ inline void NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx);
  __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
  __aicore__ inline void SetTiling(const FlashAttentionTrainingGeneralTilingData *__restrict tilingData);
  __aicore__ inline void InitBuffer();
  __aicore__ inline void ComputeConstexpr();
  __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
  template <typename T2>
  __aicore__ inline void IterateBmm1(SplitSameABExtraInfo &extraInfo,
                                     matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1);
  template <typename T2>
  __aicore__ inline void Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                       matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1);
  template <typename T2>
  __aicore__ inline void SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                       matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1);
  __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void ProcessVec1(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset,
                                         bool secondTime = false);
  __aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                                 SplitSameABExtraInfo &extraInfo);
  __aicore__ inline int64_t ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
  __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
  __aicore__ inline void GetBmm1Result(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
  __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                          LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

  __aicore__ inline void SoftMaxCompute(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
  __aicore__ inline void SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo, int64_t vec1S1realSplitN);
  __aicore__ inline void InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                   LocalTensor<T> &maxUb, int64_t loopIdx);
  __aicore__ inline void ProcessVec2(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void Bmm2ResultMul(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
  __aicore__ inline void Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx);
  __aicore__ inline void Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize);
  __aicore__ inline void SoftmaxDataCopyOut(SplitSameABExtraInfo &extraInfo);
  __aicore__ inline void AttnOutFp32DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx,
                                                GlobalTensor<float> &outGm, LocalTensor<float> &srcUb);

  // sparse 用函数
  __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
  __aicore__ inline void GetS2LoopRange();

  uint32_t cubeS1BaseSize;
  uint32_t vecS1BaseSize;
  uint32_t s2BaseSize;
  uint32_t dSize;
  int64_t dSizeAlign16;
  int64_t s1Size;
  int64_t s2Size;
  int64_t s1OuterSize;

  // sparse 用参数
  int64_t s2StartIdx;
  int64_t s2EndIdx;
  int64_t nextS2EndIdx;

  // BNG 外循环
  int64_t bngStartIdx;
  int64_t bngEndIdx;
  int64_t qCoreOffset;

  // 资源分配
  TBuf<> maskTBufPing;
  TBuf<> maskTBufPong;
  TBuf<> halfTBuf;
  TBuf<> stage1PingBuf;
  TBuf<> stage1PongBuf;
  TBuf<> stage2TBuf;
  TBuf<> softmaxSumBuf[2];
  TBuf<> softmaxExpBuf[2];
  TBuf<> softmaxMaxBuf;
  TBuf<> commonTBuf; // common的复用空间

  LocalTensor<T> softmaxExpUb;
  GlobalTensor<T> mm1Res[2];
  GlobalTensor<T> mm2Res[2];
  GlobalTensor<T> vec2Res[2];
  GlobalTensor<INPUT_T> stage1Res[2];

  constexpr static int32_t repeatMaxBytes = 256;
  constexpr static int32_t repeatMaxTimes = 255;
  // 轴的乘积
  int64_t gS1o;
  int64_t n2GS1o;
  int64_t s1D;
  int64_t gS1D;
  int64_t n2GS1D;
  int64_t s2D;
  int64_t n2S2D;
  int64_t s1S2;
  int64_t gS1S2;
  int64_t n2GS1S2;
  int64_t gS1;
  int64_t n2GS1;
  int64_t gD;
  int64_t n2D;
  int64_t bN2D;
  int64_t n2G;
  int64_t n2GD;
  int64_t bN2GD;

  // s2base*N之后的长度
  int64_t s2BaseNratioSize;
  int64_t s2BaseN2D;
  int64_t s2BaseBN2D;
  int64_t s1BaseN2GD;
  int64_t s1BaseBN2GD;
  int64_t s1BaseD;
  int64_t s2BaseNratioD;
  int64_t s2BaseNratioN2D;
  int64_t s2BaseNratioBN2D;
  int64_t bN2G;
  int64_t s2SizeSum;

  int64_t mm1Ka;
  int64_t mm1Kb;
  int64_t mm2Kb;
  // 当splitN大于16时，需要修改softMaxCheckRes数据类型
  uint16_t softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
  uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
  T negativeFloatScalar;
  T positiveFloatScalar;

  AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

  int32_t vecBlockIdx;
  int32_t cubeBlockIdx;
  int32_t cubeSubIdx;
  const FlashAttentionTrainingGeneralTilingData *__restrict tilingData;

  int64_t boIdx;
  int64_t n2oIdx;
  int64_t goIdx;
  int64_t s1oIdx;

  TPipe *pipe;

  GlobalTensor<INPUT_T> queryGm;
  GlobalTensor<INPUT_T> keyGm;
  GlobalTensor<INPUT_T> valueGm;
  GlobalTensor<float> attentionOutGm;
  GlobalTensor<float> softmaxMaxGm;
  GlobalTensor<float> softmaxSumGm;
  GlobalTensor<uint8_t> dropMaskGm;
  GlobalTensor<uint8_t> attenMaskGmInt;

  bool dropMaskUnAligned;
  int64_t attenMaskOffsetPre = 0;
  DropMaskInfo dropMaskInfo = {0};
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
         __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
         __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
         const FlashAttentionTrainingGeneralTilingData *__restrict tiling, TPipe *tPipe) {
  this->InitInput(query, key, value, dropMask, attenMask, softmaxMax, softmaxSum, attentionOut, workspace, tiling,
                  tPipe); // gm设置

  this->ComputeConstexpr();
  this->InitBuffer();
  if constexpr (hasDrop == true) {
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
              __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
              __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
              const FlashAttentionTrainingGeneralTilingData *__restrict tiling, TPipe *tPipe) {
  this->vecBlockIdx = GetBlockIdx();
  this->cubeBlockIdx = this->vecBlockIdx / 2;
  this->cubeSubIdx = this->vecBlockIdx % 2;
  this->pipe = tPipe;
  this->SetTiling(tiling);

  // init global buffer
  this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
  this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
  this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
  this->dropMaskUnAligned = this->tilingData->inputParams.needDropMaskOp == 1;
  if (this->dropMaskUnAligned) {
    this->dropMaskGm.SetGlobalBuffer(workspace);
    if constexpr (hasDrop == true) {
      workspace += CeilDiv(this->tilingData->dropmaskParams.shapeTotalSize, 512) * 512;
    }
  } else {
    this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
  }
  this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
  this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
  this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
  this->attentionOutGm.SetGlobalBuffer((__gm__ float *)attentionOut);

  // 补齐到512， 统一按T处理
  int64_t mm1ResultSize = cubeS1BaseSize * s2BaseSize;
  int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
  int64_t mm2ResultSize = cubeS1BaseSize * dSizeAlign16;
  int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
  int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
  int64_t vector1OffsetPing = 0;
  int64_t vector1OffsetPong = mmNRatioOffset;

  // bmm1、bmm2 NZ场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
  vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
  vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
  bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;

  int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;

  // bmm1Result，占用2倍mmNRatioOffset空间
  this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset));
  this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset));

  // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
  this->stage1Res[0].SetGlobalBuffer(
      (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPing));
  this->stage1Res[1].SetGlobalBuffer(
      (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPong));

  // bmm2Result，占用2倍mmOffset空间
  this->mm2Res[0].SetGlobalBuffer(
      (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
  this->mm2Res[1].SetGlobalBuffer(
      (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

  // vec2阶段，占用2倍mmOffset空间，仅在D轴大于64的情况下出现
  this->vec2Res[0].SetGlobalBuffer(
      (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2));
  this->vec2Res[1].SetGlobalBuffer(
      (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 3));
  GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    SetTiling(const FlashAttentionTrainingGeneralTilingData *__restrict tilingData) {
  // copy base params
  this->tilingData = tilingData;
  this->cubeS1BaseSize = this->tilingData->coreParams.s1BaseSize;
  this->vecS1BaseSize = this->cubeS1BaseSize / 2;
  this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
  this->dSize = this->tilingData->inputParams.dSize;
  this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T,
                                                              SEQLEN_TYPE, bmm1Format>::InitBuffer() {
  uint64_t stage1Size = 8 * 1024;
  uint64_t stage1AttenSize = 9 * 1024;
  uint64_t stage1PongSize = 34 * 1024;
  uint64_t stage2Size = 64 * 128;
  uint64_t maskTBufPongSize = 16 * 1024;

  // 可选输入的buffer空间，保持和stage1处理的size一致
  this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize);  // 可以给attenmask 9k
  this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
  this->pipe->InitBuffer(this->halfTBuf, 16384);                // 16k

  this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
  this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
  this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

  this->pipe->InitBuffer(this->softmaxSumBuf[0], vecS1BaseSize * blockBytes); // 4k
  this->pipe->InitBuffer(this->softmaxSumBuf[1], vecS1BaseSize * blockBytes); // 4k
  this->pipe->InitBuffer(this->softmaxMaxBuf, vecS1BaseSize * blockBytes);    // 4k
  this->pipe->InitBuffer(this->softmaxExpBuf[0], vecS1BaseSize * blockBytes); // 4k
  this->pipe->InitBuffer(this->softmaxExpBuf[1], vecS1BaseSize * blockBytes); // 4k
  this->pipe->InitBuffer(this->stage1PongBuf, stage1PongSize);                // i.a 34k
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T,
                                                              SEQLEN_TYPE, bmm1Format>::ComputeConstexpr() {
  // 计算轴的乘积
  this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
  this->s1D = this->tilingData->inputParams.s1Size * dSize;
  this->s2D = this->tilingData->inputParams.s2Size * dSize;
  this->gD = this->tilingData->inputParams.gSize * dSize;
  this->n2D = this->tilingData->inputParams.n2Size * dSize;
  this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
  this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
  this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
  this->gS1o = this->tilingData->inputParams.gSize * this->s1OuterSize;

  this->bN2D = this->tilingData->inputParams.bSize * n2D;
  this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
  this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
  this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
  this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
  this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
  this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
  this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

  this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
  this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

  // 计算切分轴的乘积
  this->s2BaseN2D = this->s2BaseSize * this->n2D;
  this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;

  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
    // BSH/BSNGD
    this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
    this->mm1Ka = this->n2GD;
    this->mm1Kb = this->n2D;
    this->mm2Kb = this->n2D;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
    // SBH/SBNGD
    this->bN2G = this->tilingData->inputParams.bSize * this->n2G;
    this->s1BaseBN2GD = this->cubeS1BaseSize * this->tilingData->inputParams.bSize * this->n2GD;
    this->s2BaseBN2D = this->tilingData->inputParams.bSize * this->s2BaseN2D;
    this->s2BaseNratioBN2D = this->s2BaseBN2D * this->tilingData->coreParams.nRatio;
    this->mm1Ka = this->bN2GD;
    this->mm1Kb = this->bN2D;
    this->mm2Kb = this->bN2D;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
    // BNSD
    this->s1BaseD = this->cubeS1BaseSize * this->dSize;
    this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
    this->mm1Ka = this->dSize;
    this->mm1Kb = this->dSize;
    this->mm2Kb = this->dSize;
  }

  if constexpr (hasDrop == true) {
    this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
    this->dropMaskInfo.n2G = this->n2G;
    this->dropMaskInfo.s1BaseSize = this->cubeS1BaseSize;
    this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
  }
}

// sparse functions
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::GetS1LoopRange(int64_t &multiCoreInnerOffset,
                                                                   int64_t &multiCoreInnerLimit) {
  // 计算sparse场景下s1的循环范围
  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
    // sparse场景下负载均衡后每个核获取的结果
    multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx];
    if (likely((this->tilingData->multiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
      multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
    } else {
      multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
  } else {
    if (this->tilingData->inputParams.sparseType > 0) {
      // sparse场景下负载均衡后每个核获取的结果
      multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx];
      if (likely((this->tilingData->multiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
      } else {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
      }
    }
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T,
                                                              SEQLEN_TYPE, bmm1Format>::GetS2LoopRange() {
  // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
  if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
    this->s2StartIdx = 0;
    this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, this->s2Size);
  } else if (this->tilingData->inputParams.sparseType ==
             static_cast<uint8_t>(SparseModeEnum::BAND)) { // 对角线往外扩散场景, s1和s2可能不同
    this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
    this->s2EndIdx =
        Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->coreParams.s2SparseValidSize, this->s2Size);
  } else { // 其它场景, 如无attention mask
    this->s2StartIdx = 0;
    this->s2EndIdx = this->s2Size;
    return;
  }

  if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
    // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2realSize
    if (this->s2EndIdx - this->s2StartIdx <= 0) {
      this->s2StartIdx = 0;
      this->s2EndIdx = Min(this->s2Size, 128L);
    }
  }
  return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T,
                                                              SEQLEN_TYPE, bmm1Format>::Process() {
  // 确定核内切分起点
  int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
  int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
  if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
    multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
  }
  // 计算sparse场景下s1的循环范围
  this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);

  SplitSameABExtraInfo extraInfo[3];
  int64_t taskId = 0;
  event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

  bool notSecondLast = true;
  bool notLast = true;
  multiCoreInnerLimit += 2;
  for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit; ++multiCoreInnerIdx) {
    if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
      notSecondLast = false;
    } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
      notLast = false;
    }

    int64_t s2LoopLimit;
    this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
    bool notLastTwoLoop = notSecondLast && notLast;
    if (notLastTwoLoop) {
      this->ComputeAxisIdx(multiCoreInnerIdx);

      // s2轴循环计数, 支持sparse和非sparse场景
      this->GetS2LoopRange();
      s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
    } else {
      s2LoopLimit = 0;
    }
    for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
      if (taskId >= 1 && notLast) {
        // 对应extraInfo[(i+2)%3]
        WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
      }

      if (notLastTwoLoop) {
        this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
        if constexpr (bmm1Format == CubeFormat::NZ) {
          if (extraInfo[taskId % 3].needNz2Nd == 1) {
            this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
          } else {
            this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
          }
        } else {
          this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
        }
      }

      if (taskId > 0 && notLast) {
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
      }

      if (taskId > 1) {
        // 对应extraInfo[(i+1)%3]
        WaitBmm2Result(extraInfo[(taskId + 1) % 3]);
      }

      if (taskId > 0 && notLast) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
      }

      if (taskId > 1) {
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
      }
      taskId++;
    }
  }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ComputeAxisIdx(int64_t multiCoreInnerIdx) {
  // 计算轴的idx
  this->boIdx = multiCoreInnerIdx / this->n2GS1o;
  this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
  this->goIdx = multiCoreInnerIdx % this->gS1o / this->s1OuterSize;
  this->s1oIdx = multiCoreInnerIdx % this->s1OuterSize;
  this->s1Size = this->tilingData->inputParams.s1Size;
  this->s2Size = this->tilingData->inputParams.s2Size;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::WaitBmm1Result(SplitSameABExtraInfo &extraInfo) {
  if constexpr (bmm1Format == CubeFormat::NZ) {
    if (extraInfo.needNz2Nd == 1) {
      this->bmm1Nz.WaitIterateAll();
      this->bmm1Nz.End();
      return;
    }
  }
  this->bmm1.WaitIterateAll();
  this->bmm1.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId,
                                                                 int64_t s2LoopCount, int64_t s2LoopLimit,
                                                                 int64_t multiCoreInnerIdx) {
  extraInfo.s2StartIdx = this->s2StartIdx;
  extraInfo.s2EndIdx = this->s2EndIdx;
  extraInfo.s2LoopCount = s2LoopCount;
  extraInfo.s1oIdx = this->s1oIdx;
  extraInfo.boIdx = this->boIdx;
  extraInfo.n2oIdx = this->n2oIdx;
  extraInfo.goIdx = this->goIdx;
  extraInfo.taskId = taskId;
  extraInfo.taskIdMod2 = taskId % 2;
  extraInfo.s2LoopLimit = s2LoopLimit;
  extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
  extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
  extraInfo.s1Size = this->tilingData->inputParams.s1Size;
  extraInfo.s2Size = this->tilingData->inputParams.s2Size;
  extraInfo.attenB1SSOffset = extraInfo.boIdx * this->s1S2;
  extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
  extraInfo.s2SizeAcc = extraInfo.boIdx * extraInfo.s2Size;
  extraInfo.s1SizeAcc = extraInfo.boIdx * extraInfo.s1Size;
  extraInfo.cubeS1RealSize =
      Min(this->cubeS1BaseSize, this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
  extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
  extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
  if (this->cubeSubIdx == 1) {
    extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
  }

  this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo) {
  extraInfo.s2RealSize = this->s2BaseNratioSize;
  extraInfo.s2AlignedSize = extraInfo.s2RealSize;
  extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
  extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
  extraInfo.s2LastLoop = false;
  if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
    extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
    extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
    extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
    extraInfo.s2LastLoop = true;
    extraInfo.duplicateMask = (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
                              (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
    if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
      extraInfo.duplicateMask =
          0xffffffffffffffff & (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
    }
  }

  extraInfo.vec1S1BaseSize =
      Min(VEC_BUFFER_SIZE / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // 基本块为8 * 1024大小
  extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

  if (dSizeAlign16 > 64) {
    extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;
  } else {
    extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
  }
  if constexpr (bmm1Format == CubeFormat::NZ) {
    extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
  } else {
    extraInfo.needNz2Nd = 0;
  }
  return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
template <typename T2>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    IterateBmm1(SplitSameABExtraInfo &extraInfo,
                matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                               matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1) {
  bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka, this->mm1Kb, extraInfo.s2RealSize);

  this->Bmm1SetTensorA(extraInfo, bmm1);
  this->SetBmm1TensorB(extraInfo, bmm1);
  bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], false, false, true);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
template <typename T2>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                   matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                  matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1) {
  // 计算gm上的offset
  int64_t bOffset = 0;

  // s1需要考虑inner轴的影响
  int64_t s1Offset = 0;
  int64_t n2Offset = 0;
  int64_t gOffset = 0;
  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
    // BSH/BSNGD
    bOffset = extraInfo.boIdx * this->n2GS1D;
    s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    gOffset = extraInfo.goIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
    // SBH/SBNGD
    s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
    bOffset = extraInfo.boIdx * this->n2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    gOffset = extraInfo.goIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
    // bnsd
    bOffset = extraInfo.boIdx * this->n2GS1D;
    n2Offset = extraInfo.n2oIdx * this->gS1D;
    gOffset = extraInfo.goIdx * this->s1D;
    s1Offset = extraInfo.s1oIdx * this->s1BaseD;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
    bOffset = extraInfo.s1SizeAcc * this->n2GD;
    s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    gOffset = extraInfo.goIdx * this->dSize;
  }
  this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
  extraInfo.qCoreOffset = this->qCoreOffset;
  bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
template <typename T2>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                   matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                  matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm1) {
  // 计算gm上的offset
  int64_t bOffset = 0;
  int64_t n2Offset = 0;
  int64_t s2Offset = 0;
  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
    // BSH/BSND
    bOffset = extraInfo.boIdx * this->n2S2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    n2Offset = extraInfo.n2oIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
    // SBH/SBND
    s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * this->s2BaseNratioBN2D;
    bOffset = extraInfo.boIdx * this->n2D;
    n2Offset = extraInfo.n2oIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
    // BNSD
    bOffset = extraInfo.boIdx * this->n2S2D;
    n2Offset = extraInfo.n2oIdx * this->s2D;
    s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
    bOffset = extraInfo.s2SizeAcc * this->n2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    n2Offset = extraInfo.n2oIdx * this->dSize;
  }
  int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
  bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
  bmm1.SetTail(extraInfo.cubeS1RealSize, extraInfo.s2RealSize);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::NdToNz(SplitSameABExtraInfo &extraInfo,
                                                           LocalTensor<INPUT_T> &nzResUb,
                                                           LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx) {
  auto nzResUbTmp = nzResUb.template ReinterpretCast<half>();
  auto vec1ResUbTmp = vec1ResUb.template ReinterpretCast<half>();
  int64_t s21 = CeilDiv(extraInfo.s2AlignedSize, 16L);
  int64_t s2InnerLoop = s21 / 8L;
  int64_t s2InnerRemain = s21 % 8L;
  int64_t repeatMaxSize16 = this->repeatMaxBytes / sizeof(INPUT_T);
  CopyRepeatParams repeatParams;
  repeatParams.srcStride = 1;
  repeatParams.dstStride = extraInfo.vec1S1RealSize + 1; // 防止bank冲突，所以+1
  repeatParams.srcRepeatSize = extraInfo.s2RealSizeAlign64 / 16;
  repeatParams.dstRepeatSize = 1;
  int32_t s1OuterLoop = extraInfo.vec1S1RealSize / this->repeatMaxTimes;
  int32_t s1OuterRemain = extraInfo.vec1S1RealSize % this->repeatMaxTimes;
  int32_t s1OuterBmm1Offset = this->repeatMaxTimes * extraInfo.s2RealSizeAlign64;
  int32_t s1OuterTempOffset = this->repeatMaxTimes * 16;
  int64_t vec1Offset = extraInfo.vecCoreOffset * 16 + loopIdx * extraInfo.vec1S1BaseSize * 16;
  int64_t offsetJ = 8 * extraInfo.vec1S1RealSize * 16 + 128;
  pipe_barrier(PIPE_V);
  for (int64_t outerIndex = 0; outerIndex < s1OuterLoop; ++outerIndex) {
    for (int64_t j = 0; j < s2InnerLoop; ++j) {
      Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + j * offsetJ],
           vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + j * 128], repeatMaxSize16, this->repeatMaxTimes, repeatParams);
    }
    if (s2InnerRemain) {
      Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + s2InnerLoop * offsetJ],
           vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + s2InnerLoop * 128], s2InnerRemain * 16, this->repeatMaxTimes,
           repeatParams);
    }
  }

  if (s1OuterRemain) {
    for (int64_t j = 0; j < s2InnerLoop; ++j) {
      Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + j * offsetJ],
           vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + j * 128], repeatMaxSize16, s1OuterRemain, repeatParams);
    }
    if (s2InnerRemain) {
      Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + s2InnerLoop * offsetJ],
           vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + s2InnerLoop * 128], s2InnerRemain * 16, s1OuterRemain,
           repeatParams);
    }
  }

  event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = s21;
  dataCopyParams.blockLen = extraInfo.vec1S1RealSize;
  dataCopyParams.srcStride = 1;
  dataCopyParams.dstStride = CeilDiv(extraInfo.cubeS1RealSize, 16) * 16 - extraInfo.vec1S1RealSize;
  DataCopy(this->stage1Res[extraInfo.taskIdMod2][vec1Offset], nzResUb, dataCopyParams);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ProcessVec1(SplitSameABExtraInfo &extraInfo) {
  if (extraInfo.s1RealSize == 0) {
    return;
  }
  LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
  LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
  LocalTensor<T> &actualUseTensor = (extraInfo.needNz2Nd == 0) ? stage1PongTensor : stage1PingTensor;
  LocalTensor<T> commonTBuf = this->commonTBuf.template Get<T>(); // t.b 32k

  event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
  event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
  event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
  event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
  event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
  event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  event_t eventIdDropMte3ToMte2;
  extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
  for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; ++loopIdx) {
    if (loopIdx == extraInfo.realSplitN - 1) {
      extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
    }
    if (loopIdx > 0) {
      WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
    } else {
      event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
      WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }

    // FP32场景，需要等待vec1上一轮输出搬完
    if constexpr (IsSameType<INPUT_T, float>::value) {
      event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
    this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx);

    // mul需要等bmm结果搬完
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    this->CopyInAttenMask(extraInfo, loopIdx, -1);
    pipe_barrier(PIPE_V);
    Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
         extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
    if constexpr (hasAtten) {
      SelectWithBytesMaskShapeInfo shapeInfo;
      shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
      shapeInfo.srcLastAxis = extraInfo.s2RealSizeAlign64;
      shapeInfo.maskLastAxis = extraInfo.s2RealSizeAlign64;
      stage1PingTensor.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
      if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
        uint8_t maskType = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ? 1 : 0;
        LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        this->ComputeAttenMask(shapeInfo, stage1PingTensor, attenMaskUb, maskType, eventIdMte2ToV);
      }

      if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE) {
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
        this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre, true);
        LocalTensor<uint8_t> secondTimeMaskUb;
        uint8_t maskType = 1;
        secondTimeMaskUb = this->halfTBuf.template Get<uint8_t>();
        this->ComputeAttenMask(shapeInfo, stage1PingTensor, secondTimeMaskUb, maskType, eventIdMte2ToV);
      }
    }
    if (loopIdx < extraInfo.realSplitN - 1) {
      SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
    }

    if (loopIdx > 0) {
      WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    }

    if constexpr (hasDrop == true) {
      LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
      this->dropMaskInfo.s1InnerIdx = loopIdx;
      this->dropMaskInfo.s2StartIdx = extraInfo.s2StartIdx;
      this->dropMaskInfo.s2Idx = extraInfo.s2LoopCount;
      this->dropMaskInfo.bSSOffset = extraInfo.attenB1SSOffset;
      this->dropMaskInfo.n2OutIdx = extraInfo.n2oIdx;
      this->dropMaskInfo.s1Size = extraInfo.s1Size;
      this->dropMaskInfo.s2Size = extraInfo.s2Size;
      this->dropMaskInfo.gOutIdx = extraInfo.goIdx;
      this->dropMaskInfo.s1OutIdx = extraInfo.s1oIdx;
      this->dropMaskInfo.splitS1BaseSize = extraInfo.vec1S1BaseSize;
      this->dropMaskInfo.s1CopySize = static_cast<uint32_t>(extraInfo.vec1S1RealSize);
      this->dropMaskInfo.s2CopySize = static_cast<uint32_t>(extraInfo.s2RealSize);
      this->dropMaskInfo.s2TotalSize = extraInfo.s2Size;
      this->dropMaskInfo.boolMode = this->dropMaskUnAligned;
      this->dropMaskInfo.vecCoreOffset = extraInfo.vecCoreOffset;
      CopyInDropMask<hasDrop>(dropMaskUb, this->dropMaskGm, this->dropMaskGm, this->dropMaskInfo, 64);
    }

    this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);

    if constexpr (hasDrop == true) {
      LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
      LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
      pipe_barrier(PIPE_V);

      SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      this->dropMaskInfo.firstAxis = static_cast<uint32_t>(extraInfo.vec1S1RealSize);
      this->dropMaskInfo.lstAxis = static_cast<uint32_t>(extraInfo.s2RealSizeAlign64);
      this->dropMaskInfo.maskLstAxis = this->dropMaskInfo.lstAxis;
      // this->dropMaskInfo.keepProb = this->tilingData->inputParams.keepProb;
      this->dropMaskInfo.keepProb = 1.0f;
      ComputeDropMask<T, hasDrop>(stage1PingTensor, stage1PingTensor, dropMaskUb, apiTmpBuffer, this->dropMaskInfo);
    }

    if (loopIdx < extraInfo.realSplitN - 1) {
      SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    }

    if (loopIdx > 0) {
      WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    pipe_barrier(PIPE_V);
    LocalTensor<INPUT_T> stage1CastTensor;
    stage1CastTensor = this->halfTBuf.template Get<INPUT_T>();
    Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
         extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    LocalTensor<INPUT_T> stage1NzTensor = this->stage2TBuf.template Get<INPUT_T>();
    NdToNz(extraInfo, stage1NzTensor, stage1CastTensor, loopIdx);

    if (loopIdx < extraInfo.realSplitN - 1) {
      SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
  }
  GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
  GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
  GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
  GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
  return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx,
                                                                    int64_t maskOffset, bool secondTime) {
  if constexpr (hasAtten == true) {
    LocalTensor<uint8_t> attenMaskUb;
    if (secondTime) {
      attenMaskUb = this->halfTBuf.template Get<uint8_t>();
    } else {
      attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
    }
    if (maskOffset == -1) {
      maskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
    }
    if (this->attenMaskComputeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
      return;
    }
    if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) {
      maskOffset = this->attenMaskOffsetPre;
    }

    int64_t s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
      if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
        s2StrideSize = this->tilingData->inputParams.s2Size;
      } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
        s2StrideSize = this->s2SizeSum;
      }
      // band compress mode
      if (this->tilingData->inputParams.attenMaskCompressMode !=
          static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
      }
    }
    BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
               s2StrideSize, 64);
    return;
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline int64_t
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo,
                                                                           int64_t loopIdx) {
  if constexpr (hasAtten == true) {
    if (this->tilingData->inputParams.attenMaskCompressMode ==
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
      return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
    }
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
      // compress mode
      int64_t delta = 0;
      int64_t deltaPre = 0;
      int64_t deltaN = static_cast<int64_t>((extraInfo.s1Size)) - static_cast<int64_t>((extraInfo.s2Size));
      int64_t s1Offset =
          extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize;
      int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
      if (this->tilingData->inputParams.attenMaskCompressMode ==
          static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
        delta = s1Offset - s2Offset;
      } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                 static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
        delta = s1Offset - s2Offset - deltaN;
      } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                 static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
        int64_t tmpPre = this->tilingData->inputParams.preTokens;
        int64_t tmpNext = this->tilingData->inputParams.nextTokens;
        int64_t transPreTokens = extraInfo.s1Size - Max(extraInfo.s2Size - tmpPre, 0);
        int64_t transNextTokens = extraInfo.s2Size - Max(extraInfo.s1Size - tmpNext, 0);
        deltaPre = s1Offset - s2Offset - transPreTokens - 1;
        int64_t maskOffsetPre = ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                                       this->tilingData->inputParams.attenMaskS2Size);
        this->attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
        delta = s1Offset - s2Offset + transNextTokens;
      } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                 static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE)) {
        if (extraInfo.boIdx == this->tilingData->inputParams.bandIndex) {
          delta = s1Offset - s2Offset - deltaN + this->tilingData->inputParams.nextTokens;
        } else {
          delta = s1Offset - s2Offset - deltaN;
        }
      } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                 static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE)) {
        if (extraInfo.boIdx == this->tilingData->inputParams.bandIndex) {
          delta = s1Offset - s2Offset + extraInfo.s2Size -
                  Max(extraInfo.s1Size - this->tilingData->inputParams.nextTokens, 0);
        } else {
          delta = s1Offset - s2Offset;
        }
      } else {
        return 0;
      }
      this->GetAttenMaskComputeMode(delta, deltaPre, s1Offset, extraInfo);
      return ComputeOffsetForCausal(delta, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                    this->tilingData->inputParams.attenMaskS2Size);
    }
    // compress mode
    int64_t deltaCausalOrNext = 0;
    int64_t deltaPre = 0;
    int64_t deltaN = extraInfo.s1Size - extraInfo.s2Size;
    int64_t s1Offset =
        extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
    if (this->tilingData->inputParams.attenMaskCompressMode ==
        static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
      deltaCausalOrNext = s1Offset - s2Offset;
    } else if (this->tilingData->inputParams.attenMaskCompressMode ==
               static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
      deltaCausalOrNext = s1Offset - s2Offset - deltaN;
    } else if (this->tilingData->inputParams.attenMaskCompressMode ==
               static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
      deltaPre = s1Offset - s2Offset - this->tilingData->inputParams.preTokens - 1;
      this->attenMaskOffsetPre = ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                                        this->tilingData->inputParams.attenMaskS2Size);
      deltaCausalOrNext = s1Offset - s2Offset + this->tilingData->inputParams.nextTokens;
    } else {
      return 0;
    }
    this->GetAttenMaskComputeMode(deltaCausalOrNext, deltaPre, s1Offset, extraInfo);
    return ComputeOffsetForCausal(deltaCausalOrNext, extraInfo.vec1S1BaseSize, s2BaseNratioSize,
                                  this->tilingData->inputParams.attenMaskS2Size);
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
                                                                            int64_t s1Offset,
                                                                            SplitSameABExtraInfo &extraInfo) {
  if constexpr (hasAtten == true) {
    int64_t causalOrNextFactor = deltaCausalOrNext - extraInfo.s2AlignedSize;
    if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
        this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
      if (causalOrNextFactor >= 0) {
        this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
      } else {
        this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
      }
      return;
    }
    if (this->tilingData->inputParams.attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
      int64_t preFactor = deltaPre + 1 + extraInfo.vec1S1BaseSize;
      if (causalOrNextFactor >= 0 && preFactor <= 0) {
        this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
      } else if (causalOrNextFactor < 0 && preFactor <= 0) {
        this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
      } else if (causalOrNextFactor >= 0 && preFactor > 0) {
        this->attenMaskComputeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
      } else {
        this->attenMaskComputeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
      }
    }
    return;
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline int64_t
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo,
                                                                               int64_t loopIdx) {
  if constexpr (hasAtten == true) {
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset =
        (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
        extraInfo.s2Size;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;
    if (this->tilingData->inputParams.attenMaskShapeType == attenMaskBN2GS1S2) {
      bOffset = extraInfo.attenB1SSOffset * this->n2G;
      n2Offset = extraInfo.n2oIdx * this->tilingData->inputParams.gSize * extraInfo.s1Size * extraInfo.s2Size;
      gOffset = extraInfo.goIdx * extraInfo.s1Size * extraInfo.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskBS1S2) {
      bOffset = extraInfo.attenB1SSOffset;
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
      s1Offset =
          (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
          this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
      s1Offset = extraInfo.s1SizeAcc + extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                 loopIdx * extraInfo.vec1S1BaseSize;
      s1Offset = s1Offset * this->s2SizeSum;
      s2Offset = s2Offset + extraInfo.s2SizeAcc;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::GetBmm1Result(SplitSameABExtraInfo &extraInfo,
                                                                  LocalTensor<T> &bmm1ResUb, int64_t loopIdx) {
  if constexpr (bmm1Format == CubeFormat::NZ) {
    if (extraInfo.needNz2Nd == 1) {
      Nz2NdInfo nz2NdInfo;
      nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
      nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec1S1BaseSize;
      nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec1S1RealSize;
      nz2NdInfo.ndLastAxis = extraInfo.s2RealSizeAlign64;
      nz2NdInfo.loopIdx = loopIdx;
      nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
      LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
      NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
      return;
    }
  }
  if (likely(extraInfo.s2RealSizeAlign64 == extraInfo.s2RealSize)) {
    DataCopy2D(bmm1ResUb,
               this->mm1Res[extraInfo.taskIdMod2]
                           [(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2RealSize],
               extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
  } else {
    DataCopyParams dataCopyParams;
    DataCopyPadParams dataCopyPadParams;
    dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
    dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    if (extraInfo.s2LastLoop) {
      dataCopyParams.dstStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSize) / 8;
      dataCopyPadParams.isPad = false;
    } else {
      dataCopyParams.dstStride = 0;
      dataCopyPadParams.isPad = true;
      dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
      if (dataCopyPadParams.rightPadding > blockSize) {
        dataCopyPadParams.rightPadding -= blockSize;
        dataCopyParams.dstStride = 1;
        int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
        Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                     extraInfo.s2AlignedSize * sizeof(T) / blockBytes);
      }
      dataCopyPadParams.paddingValue = 0;
    }
    DataCopyPad(bmm1ResUb,
                this->mm1Res[extraInfo.taskIdMod2]
                            [(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2RealSize],
                dataCopyParams, dataCopyPadParams);
  }
  uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                               static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
  bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo,
                                                                     LocalTensor<T> &bmm1ResUb,
                                                                     LocalTensor<uint8_t> &attenMaskUb,
                                                                     const uint8_t maskType, event_t vWaitMte2) {
  if constexpr (hasAtten == true) {
    LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
    attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
    SetFlag<HardEvent::MTE2_V>(vWaitMte2);
    WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    if (maskType == 0) {
      SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
    } else {
      SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
    }
    return;
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo,
                                                                            int64_t vec1S1realSplitN) {
  if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
    if (unlikely(extraInfo.realSplitN == 1)) {
      bool res = IsIncludeInvalidLine(this->softMaxCheckRes, vec1S1realSplitN);
      UpdateSoftMaxCheckRes(this->softMaxCheckRes, 0, res);
    } else {
      int64_t avg = CeilDiv(vec1S1realSplitN, extraInfo.realSplitN);
      for (int64_t i = 0; i < extraInfo.realSplitN; ++i) {
        int64_t endIdx = Min((i + 1) * avg, vec1S1realSplitN) - 1;
        if (IsIncludeInvalidLine(this->softMaxCheckRes, endIdx, i * avg)) {
          UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, true);
        } else {
          UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, false);
        }
      }
    }
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo,
                                                                              LocalTensor<T> &srcTensor,
                                                                              LocalTensor<T> &maxUb, int64_t loopIdx) {
  if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
    if (loopIdx == 0 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
      int64_t vec1S1BaseSize = Min(8, extraInfo.s1RealSize);
      int64_t vec1S1realSplitN = CeilDiv(extraInfo.s1RealSize, vec1S1BaseSize);
      if (extraInfo.realSplitN != vec1S1realSplitN) {
        this->SoftMaxCheckResCompress(extraInfo, vec1S1realSplitN);
      }
    }

    if (hasInvalidLine(this->softMaxCheckRes, loopIdx)) {
      SoftMaxShapeInfo softmaxShapeInfo{
          static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
          static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
      bool res = AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
      UpdateSoftMaxCheckRes(this->softMaxCheckRes, loopIdx, res);
    }

    if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit &&
        IsIncludeInvalidLine(this->softMaxCheckRes, extraInfo.realSplitN)) {
      LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
      LocalTensor<T> sumTensor;
      sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

      SoftMaxShapeInfo softmaxShapeInfo{
          static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
          static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
      AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                             softmaxShapeInfo);
    }
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::SoftMaxCompute(SplitSameABExtraInfo &extraInfo,
                                                                   LocalTensor<T> &srcTensor, int64_t loopIdx) {
  int64_t vec1S1RealSize = CeilDiv(extraInfo.vec1S1RealSize, 8) * 8;
  uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                               static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
  uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                                  static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
  srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));
  srcTensor.SetSize(vec1S1RealSize * extraInfo.s2RealSizeAlign64);
  uint32_t maxSumShape[] = {static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(fp32BaseSize)};

  LocalTensor<T> sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                             .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];
  LocalTensor<T> maxUb = this->softmaxMaxBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];

  sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
  maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
  if (extraInfo.s2LastLoop) {
    if constexpr (hasAtten == false) {
      uint64_t mask[2] = {extraInfo.duplicateMask, 0};
      Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vec1S1RealSize, 1,
                   extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
    } else if (this->tilingData->inputParams.attenMaskCompressMode !=
                   static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) ||
               extraInfo.s2RealSize < 64) {
      uint64_t mask[2] = {extraInfo.duplicateMask, 0};
      Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vec1S1RealSize, 1,
                   extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
    }
  }

  LocalTensor<T> expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                             .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * blockBytes / sizeof(T)];

  expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
  LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
  pipe_barrier(PIPE_V);
  if (unlikely(extraInfo.s2LoopCount == 0)) {
    SoftMaxTiling newTiling =
        AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize, extraInfo.s2RealSizeAlign64, sizeof(T), sizeof(T),
                                              apiTmpBuffer.GetSize() / sizeof(T), false, true);
    SoftmaxFlashV2<T, false, true, true, false, MOJO_FA_SOFTMAX_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb,
                                                                     maxUb, apiTmpBuffer, newTiling);
  } else {
    SoftMaxTiling newTiling =
        AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize, extraInfo.s2RealSizeAlign64, sizeof(T), sizeof(T),
                                              apiTmpBuffer.GetSize() / sizeof(T), true, true);
    SoftmaxFlashV2<T, true, true, true, false, MOJO_FA_SOFTMAX_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb,
                                                                    maxUb, apiTmpBuffer, newTiling);
  }

  if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
    if (extraInfo.s2Size > SPLIT_S2_SIZE_LIMIT) {
      this->InvalidLineSplitS2Process(extraInfo, srcTensor, maxUb, loopIdx);
    } else {
      SoftMaxShapeInfo softmaxShapeInfo{
          static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
          static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
      AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
      if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
        LocalTensor<T> sumTensor;
        sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

        SoftMaxShapeInfo softmaxFullShapeInfo{
            static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
            static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
        AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                               softmaxFullShapeInfo);
      }
    }
  }

  if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
    extraInfo.softmaxMaxOffset =
        (extraInfo.s1SizeAcc * this->n2G + extraInfo.n2oIdx * this->tilingData->inputParams.gSize * extraInfo.s1Size +
         extraInfo.goIdx * extraInfo.s1Size + extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset) *
        static_cast<int64_t>(fp32BaseSize);
    int64_t calculateSize = extraInfo.s1RealSize * fp32BaseSize;
    LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], maxTensor, calculateSize);
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::WaitBmm2Result(SplitSameABExtraInfo &extraInfo) {
  this->bmm2.WaitIterateAll();
  this->bmm2.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE, bmm1Format>::
    IterateBmm2(SplitSameABExtraInfo &extraInfo,
                matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                               matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> &bmm2) {
  int64_t bOffset = 0;
  int64_t n2Offset = 0;
  int64_t s2Offset = 0;

  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
    // BSH/BSND
    bOffset = extraInfo.boIdx * this->n2S2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * s2BaseNratioSize * this->n2D;
    n2Offset = extraInfo.n2oIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
    // SBH/SBND
    s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * s2BaseNratioSize * this->bN2D;
    bOffset = extraInfo.boIdx * this->n2D;
    n2Offset = extraInfo.n2oIdx * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
    // BNSD
    bOffset = extraInfo.boIdx * this->n2S2D;
    n2Offset = extraInfo.n2oIdx * this->s2D;
    s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * s2BaseNratioSize * dSize;
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
    // TND
    bOffset = extraInfo.s2SizeAcc * this->n2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D;
    n2Offset = extraInfo.n2oIdx * this->dSize;
  }

  int64_t vCoreOffset = bOffset + n2Offset + s2Offset;

  bmm2.SetOrgShape(extraInfo.cubeS1RealSize, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->dSize);
  bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);

  bmm2.SetTensorB(this->valueGm[vCoreOffset]);
  bmm2.SetTail(extraInfo.cubeS1RealSize, this->dSize, extraInfo.s2RealSize);

  bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::ProcessVec2(SplitSameABExtraInfo &extraInfo) {
  if (extraInfo.s1RealSize == 0) {
    return;
  }
  // 获取缓存bmm2的计算结果
  LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
  LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
  int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

  event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
  event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
  event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;
  for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; ++s1oIdx) {
    if (s1oIdx == vec2LoopLimit - 1) {
      extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
    }
    int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * this->dSizeAlign16;
    int64_t mm2ResOffset = (extraInfo.vecCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize) * this->dSizeAlign16;
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    Nz2NdInfo nz2NdInfo;
    nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
    nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec2S1BaseSize;
    nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec2S1RealSize;
    nz2NdInfo.ndLastAxis = this->dSizeAlign16;
    nz2NdInfo.loopIdx = s1oIdx;
    nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
    NzToNd(nz2NdInfo, this->mm2Res[extraInfo.taskIdMod2], tempUb, stage2BufTensor);
    pipe_barrier(PIPE_V);

    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    DataCopy(bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (unlikely(extraInfo.s2LoopCount == 0)) {
      DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
    } else {
      this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
      pipe_barrier(PIPE_V);
      Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
    }

    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
      Bmm2ResultDiv(extraInfo, s1oIdx);
      Muls(bmm2ResUb, bmm2ResUb, static_cast<T>(1.0f / this->tilingData->inputParams.keepProb), mm2ResCalcSize);
      Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize);
      if (s1oIdx == vec2LoopLimit - 1) {
        SoftmaxDataCopyOut(extraInfo);
      }
      event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
      SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
      WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    } else {
      SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
  }
  return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::Bmm2ResultMul(SplitSameABExtraInfo &extraInfo,
                                                                  LocalTensor<T> &bmm2ResUb, int64_t s1oIdx) {
  pipe_barrier(PIPE_V);
  LocalTensor<T> expUb = softmaxExpBuf[extraInfo.taskIdMod2].template Get<T>();

  BinaryRepeatParams repeatParams;
  repeatParams.src0BlkStride = 0;
  repeatParams.src0RepStride = 1;
  repeatParams.src1RepStride = dSizeAlign16 / blockSize;
  repeatParams.dstRepStride = dSizeAlign16 / blockSize;

  // s1长度可能会超过255限制，修改成双重循环
  // 根据一次最多计算的byte数量，对bmm2Res分组mul
  int32_t loop = dSizeAlign16 / repeatMaxSize;
  int32_t remain = dSizeAlign16 % repeatMaxSize;
  for (int i = 0; i < loop; ++i) {
    Mul(bmm2ResUb[i * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], bmm2ResUb[i * repeatMaxSize],
        repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
  }
  if (likely(remain)) {
    Mul(bmm2ResUb[loop * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], bmm2ResUb[loop * repeatMaxSize],
        remain, extraInfo.vec2S1RealSize, repeatParams);
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx) {
  LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

  BinaryRepeatParams repeatParams;
  repeatParams.src0BlkStride = 1;
  repeatParams.src0RepStride = dSizeAlign16 / blockSize;
  repeatParams.src1BlkStride = 0;
  repeatParams.src1RepStride = 1;
  repeatParams.dstRepStride = dSizeAlign16 / blockSize;
  int32_t loop = dSizeAlign16 / repeatMaxSize;
  int32_t remain = dSizeAlign16 % repeatMaxSize;

  LocalTensor<float> sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
  int32_t calcSize = sumUb.GetSize();
  // 用optionalInputQueue的queue
  pipe_barrier(PIPE_V);
  if constexpr (IsSameType<T, half>::value) {
    LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
    Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
    Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
    LocalTensor<T> sumCastTensor =
        softmaxExpBuf[0].template Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);

    Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
    for (int i = 0; i < loop; i++) {
      Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
          extraInfo.vec2S1RealSize, repeatParams);
    }
    if (likely(remain)) {
      Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
          extraInfo.vec2S1RealSize, repeatParams);
    }
  } else {
    for (int i = 0; i < loop; i++) {
      Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
          repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
    }
    if (likely(remain)) {
      Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
          sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
    }
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::AttnOutFp32DataCopyOut(SplitSameABExtraInfo &extraInfo,
                                                                           int64_t s1oIdx, GlobalTensor<float> &outGm,
                                                                           LocalTensor<float> &srcUb) {
  DataCopyParams dataCopyParams;
  dataCopyParams.blockLen = this->dSize * sizeof(float);
  dataCopyParams.srcStride = 0;
  int64_t dstStride = 0;
  int64_t attenOutOffset = this->dSize;
  int64_t datacopyOffset = this->dSize;

  if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
    datacopyOffset = this->n2GD;
    attenOutOffset = this->n2GD;
    dstStride =
        (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize * sizeof(float);
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
    datacopyOffset = this->bN2G * this->dSize;
    attenOutOffset = this->bN2GD;
    dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
                     this->tilingData->inputParams.gSize -
                 1) *
                this->dSize * sizeof(float);
  } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
    datacopyOffset = this->n2GD;
    attenOutOffset = this->n2GD;
    dstStride =
        (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize * sizeof(float);
  }

  // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
  if (likely(dstStride <= 65535)) {
    dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
    dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
    DataCopyPad(
        outGm[extraInfo.qCoreOffset + (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset],
        srcUb, dataCopyParams);
  } else {
    dataCopyParams.blockCount = 1;
    dataCopyParams.dstStride = 0;

    for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
      DataCopyPad(
          outGm[extraInfo.qCoreOffset + (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset +
                i * datacopyOffset],
          srcUb[i * this->dSizeAlign16], dataCopyParams);
    }
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx,
                                                                    int64_t mm2ResCalcSize) {
  LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
  bmm2ResUb.SetSize(mm2ResCalcSize);
  {
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
  }
  AttnOutFp32DataCopyOut(extraInfo, s1oIdx, this->attentionOutGm, bmm2ResUb);
  {
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
  }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          typename SEQLEN_TYPE, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionTrainingS1s2Bn2gs1SameAB<implMode, layOutType, hasAtten, hasDrop, INPUT_T, T, SEQLEN_TYPE,
                                       bmm1Format>::SoftmaxDataCopyOut(SplitSameABExtraInfo &extraInfo) {
  LocalTensor<float> sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
  DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset], sumTensor, extraInfo.s1RealSize * fp32BaseSize);
}

} // namespace kernels
} // namespace mojo

#endif // FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H
