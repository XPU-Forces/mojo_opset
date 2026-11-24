#ifndef FLASH_ATTENTION_SCORE_COMMON_H
#define FLASH_ATTENTION_SCORE_COMMON_H

#include "kernel_operator.h"
#include "fa_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
using AscendC::AdjustSoftMaxRes;
#endif
using AscendC::AIC;
using AscendC::BinaryRepeatParams;
using AscendC::Cast;
using AscendC::CopyRepeatParams;
using AscendC::Div;
using AscendC::DropOut;
using AscendC::DROPOUT_MODE_BIT_MISALIGN;
using AscendC::DROPOUT_MODE_BYTE_MISALIGN;
using AscendC::DropOutShapeInfo;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
using AscendC::GetUserWorkspace;
#endif
using AscendC::Nd2NzParams;
using AscendC::RoundMode;
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
using AscendC::SelectWithBytesMask;
#endif
using AscendC::SelectWithBytesMaskShapeInfo;
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
using AscendC::SoftmaxFlashV2;
#endif
using AscendC::SoftMaxShapeInfo;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TSCM;

namespace mojo {
namespace kernels {

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig CFG_IBSHARE_EXCEED = GetIBShareNormConfig(true);
constexpr static uint64_t BLOCK_BYTE = 32;
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr static uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr static uint16_t SOFTMAX_CHECK_RES_DEFAULT_VALUE = 0xFFFF;
constexpr static int64_t attenMaskBN2GS1S2 = 0;
constexpr static int64_t attenMaskBS1S2 = 1;
constexpr static int64_t attenMaskS1S2 = 2;
constexpr static int64_t attenMaskTT = 99;

// 0级接口的block间隔范围需要满足32B对齐
constexpr static int32_t fp32BaseSize = 8;

enum class SparseModeEnum {
  ALL = 0,
  NONE = 1,
  ANY = 2,
  CAUSAL = 3,
  BAND = 4,
  PREFIX = 5,
  BAND_COMPRESS = 6,
  RIGHT_DOWN_CAUSAL = 7,
  RIGHT_DOWN_CAUSAL_BAND = 8,
  BAND_LEFT_UP_CAUSAL = 9
};

enum class ImplModeEnum { AA_HIGH_PRECISION = 0, AA_HIGH_PERFORMANCE = 1, AA_INVALID_LINE_HIGH_PRECISION = 2 };

enum class AttenMaskCompressMode {
  NO_COMPRESS_MODE = 0,
  LEFT_UP_CAUSAL_MODE = 1,
  RIGHT_DOWN_CAUSAL_MODE = 2,
  BAND_MODE = 3,
  PREFIX_MODE = 4,
  RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
  BAND_LEFT_UP_CAUSAL_MODE = 6
};

enum class AttenMaskComputeMode {
  NORMAL_MODE = 0,
  CAUSAL_OR_NEXT_ONLY_MODE,
  PRE_ONLY_MODE,
  PRE_AND_NEXT_MODE,
  NO_NEED_COMPUTE_MODE,
  PREFIX_COMPUTE_MODE,
  PREFIX_N_COMPUTE_MODE
};

struct Nz2NdInfo {
  int64_t ndFirstAxisRealSize = 0;
  int64_t ndFirstAxisBaseSize = 0;
  int64_t ndFirstAxisLoopSize = 0;
  int64_t ndLastAxis = 0;
  int64_t loopIdx = 0;
  int64_t vecCoreOffset = 0;
};

// scalar 常量化枚举
enum class STemplateType {
  S_TEMPLATE_UNKNOW = 0,
  ALIGNED_16 = 16,
  ALIGNED_32 = 32,
  ALIGNED_48 = 48,
  ALIGNED_64 = 64,
  ALIGNED_80 = 80,
  ALIGNED_96 = 96,
  ALIGNED_112 = 112,
  ALIGNED_128 = 128,
};

enum class DTemplateType {
  D_TEMPLATE_UNKNOW = 0,
  ALIGNED_80 = 80,
  ALIGNED_96 = 96,
  ALIGNED_128 = 128,
};

__aicore__ inline bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK) {
  return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
}

__aicore__ inline bool hasInvalidLine(uint16_t softMaxCheckRes, uint32_t bitIdx) {
  return ((softMaxCheckRes >> bitIdx) & 0x01);
}

__aicore__ inline void UpdateSoftMaxCheckRes(uint16_t &softMaxCheckRes, uint32_t bitIdx, bool bitValue) {
  if (bitValue) {
    softMaxCheckRes |= 1 << bitIdx;
  } else {
    softMaxCheckRes &= ~(1 << bitIdx);
  }
}

__aicore__ inline bool IsIncludeInvalidLine(uint16_t softMaxCheckRes, uint32_t bitIdxB, uint32_t bitIdxA = 0) {
  if (bitIdxA == 0) {
    return (softMaxCheckRes & ((1 << bitIdxB) - 1));
  } else {
    uint16_t mask = (1 << (bitIdxB - bitIdxA + 1)) - 1;
    mask = mask << bitIdxA;
    return (softMaxCheckRes & mask);
  }
}

template <typename T> __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar) {
  if constexpr (IsSameType<T, float>::value) {
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
    positiveScalar = *((float *)&tmp2);
  } else {
    uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
    uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
    negativeScalar = *((half *)&tmp1);
    positiveScalar = *((half *)&tmp2);
  }
}

template <typename T>
__aicore__ inline void DataCopy2D(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal, const uint32_t d0,
                                  const uint32_t d1, const uint32_t orgD1, uint64_t paddingValue = 0) {
  if (d1 % (BLOCK_BYTE / sizeof(T)) == 0 && orgD1 % (BLOCK_BYTE / sizeof(T)) == 0) {
    auto d1Blocks = math::Ceil(d1 * sizeof(T), BLOCK_BYTE);
    auto orgD1Blocks = math::Ceil(orgD1 * sizeof(T), BLOCK_BYTE);
    DataCopyParams copyParams(d0, d1Blocks, orgD1Blocks - d1Blocks, 0);
    DataCopy(dstLocal, srcGlobal, copyParams);
  } else {
    auto d1Bytes = d1 * sizeof(T);
    auto d1Aligned = math::Align(static_cast<int64_t>(d1), static_cast<int64_t>(BLOCK_BYTE / sizeof(T)));
    DataCopyParams copyParams(static_cast<uint16_t>(d0), static_cast<uint16_t>(d1Bytes), orgD1 * sizeof(T) - d1Bytes,
                              0);
    DataCopyPadParams padParams(true, 0, static_cast<uint8_t>(d1Aligned - d1), paddingValue);
    DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
  }
}

__aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                 const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size) {
  if (delta <= 0) {
    return Min(-1 * delta, s1BaseSize);
  } else {
    return Min(delta, s2BaseSize) * attenMaskS2Size;
  }
}

__aicore__ inline int64_t ComputeOffsetForPrefixRectangle(const int64_t &delta, const uint32_t &s2BaseSize,
                                                          const uint32_t &attenMaskS2Size) {
  // attenMask S1 is same to S2
  if (delta <= 0) {
    return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2; // 2048 * 2048 + 1024
  } else if (delta >= s2BaseSize) {
    return attenMaskS2Size * attenMaskS2Size; // 2048 * 2048 + 0
  } else {
    return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta; // 2048 * 2048 + (1024 - delta)
  }
}

template <typename T>
__aicore__ inline void NzToNd(Nz2NdInfo &nz2NdInfo, const GlobalTensor<T> &bmmResGm, LocalTensor<T> &tempUb,
                              LocalTensor<T> &bmmResUb) {
  // 1.将bmm1结果由GM搬至UB，每块数据在UB上间隔1个block，防止BANK冲突
  DataCopyParams dataCopyParams;
  int64_t nzFirstAxis = CeilDiv(nz2NdInfo.ndLastAxis, 16L);
  dataCopyParams.blockCount = nzFirstAxis;
  dataCopyParams.blockLen = nz2NdInfo.ndFirstAxisLoopSize * 2;
  dataCopyParams.srcStride = (nz2NdInfo.ndFirstAxisRealSize - nz2NdInfo.ndFirstAxisLoopSize) * 2;
  dataCopyParams.dstStride = 1;
  int64_t bmmResOffset = nz2NdInfo.vecCoreOffset * 16 + nz2NdInfo.loopIdx * nz2NdInfo.ndFirstAxisBaseSize * 16;
  int64_t innerLoop = nzFirstAxis / 8L;
  int64_t innerRemain = nzFirstAxis % 8L;

  CopyRepeatParams repeatParams;
  repeatParams.srcStride = nz2NdInfo.ndFirstAxisLoopSize * 2 + 1;
  repeatParams.dstStride = 2;
  repeatParams.srcRepeatSize = 2;
  repeatParams.dstRepeatSize = nz2NdInfo.ndLastAxis / 8;
  int32_t outerLoop = nz2NdInfo.ndFirstAxisLoopSize / repeatMaxTimes;
  int32_t outerRemain = nz2NdInfo.ndFirstAxisLoopSize % repeatMaxTimes;
  int32_t outerBmmOffset = repeatMaxTimes * nz2NdInfo.ndLastAxis;
  int32_t outerTempOffset = repeatMaxTimes * 16;
  int64_t offsetJ = 128 * nz2NdInfo.ndFirstAxisLoopSize + 64;
  DataCopy(tempUb, bmmResGm[bmmResOffset], dataCopyParams);

  // 2.使用vcopy进行transpose，[S2/16, vec1S1Base * 16 + 8] -> [vec1S1Base, S2/16 , 16]
  event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
  for (int64_t outerIndex = 0; outerIndex < outerLoop; ++outerIndex) {
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < innerLoop; ++j) {
        Copy(bmmResUb[outerIndex * outerBmmOffset + j * 128 + i * 8],
             tempUb[outerIndex * outerTempOffset + j * offsetJ + i * 8], repeatMaxSize, repeatMaxTimes, repeatParams);
      }
      if (likely(innerRemain)) {
        Copy(bmmResUb[outerIndex * outerBmmOffset + innerLoop * 128 + i * 8],
             tempUb[outerIndex * outerTempOffset + innerLoop * offsetJ + i * 8], innerRemain * 8, repeatMaxTimes,
             repeatParams);
      }
    }
  }
  if (likely(outerRemain)) {
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < innerLoop; ++j) {
        Copy(bmmResUb[outerLoop * outerBmmOffset + j * 128 + i * 8],
             tempUb[outerLoop * outerTempOffset + j * offsetJ + i * 8], repeatMaxSize, outerRemain, repeatParams);
      }
      if (likely(innerRemain)) {
        Copy(bmmResUb[outerLoop * outerBmmOffset + innerLoop * 128 + i * 8],
             tempUb[outerLoop * outerTempOffset + innerLoop * offsetJ + i * 8], innerRemain * 8, outerRemain,
             repeatParams);
      }
    }
  }
  uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(nz2NdInfo.ndFirstAxisLoopSize),
                               static_cast<uint32_t>(nz2NdInfo.ndLastAxis)};
  bmmResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}
} // namespace kernels
} // namespace mojo
#endif // FLASH_ATTENTION_SCORE_COMMON_H
