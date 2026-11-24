/*!
 * \file common_header.h
 * \brief
 */

#ifndef _COMMON_HEADER_H_
#define _COMMON_HEADER_H_

#include <limits>
#include <type_traits>

#include "kernel_event.h"
#include "kernel_macros.h"
#include "kernel_operator.h"
#include "kernel_tensor.h"

constexpr static uint32_t DYV = 0;
constexpr static uint32_t QK = 1;
constexpr static uint32_t DQ = 2;
constexpr static uint32_t DK = 3;
constexpr static uint32_t DV = 4;
constexpr static int32_t SIZE_16 = 16;
constexpr static int32_t SIZE_32 = 32;
constexpr static int32_t SIZE_64 = 64;
constexpr static int32_t SIZE_256 = 256;
constexpr static int32_t SIZE_384 = 384;
constexpr static int32_t SIZE_ONE_K = 1024;
constexpr static int32_t SIZE_128 = 128;
constexpr static int32_t BASE_BLOCK_SIZE = 128 * 128;
constexpr static int32_t BLOCK_WORKSPACE = SIZE_16 * SIZE_128 * SIZE_128;
constexpr static int32_t ATTEN_MASK_DIM_S2 = 2048;
constexpr static int32_t ATTEN_MASK_DEFAULT_OFFSET = SIZE_128 * ATTEN_MASK_DIM_S2 + SIZE_128;
constexpr static uint32_t CAL_REPEAT_NUM = 256 / sizeof(float);

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()

/////////////////////////////////////////////////////
// common struct
/////////////////////////////////////////////////////

template <class TILING_CLASS, typename INPUT_TYPE, typename SEQLEN_TYPE, bool DROP_ENABLE, bool DETERMINISTIC_ENABLE>
struct FAG_TYPE {
  using tiling_class = TILING_CLASS;
  using input_type = INPUT_TYPE;
  using seqlen_type = SEQLEN_TYPE;
  static constexpr bool drop_enable = DROP_ENABLE;
  static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

struct DynamicParams {
  int32_t l1_m_size;
  int32_t l1_n_size;
  int32_t l1_k_size;
  int32_t baseM;
  int32_t baseN;
};

struct cubeAddrInfo {
  uint64_t left;
  uint64_t right;
  uint64_t out;
  int32_t kx = 0;
  int32_t ky = 0;
  int32_t lineStride = 0;
  bool lowerLeft;
  bool upperRight;
  int32_t S1Idx;
  int32_t S2Idx;
  int32_t blockStart;
  // int32_t DiagOffset;
};

struct VecBlockInfo {
  int32_t S1Idx;
  int32_t S2Idx;
  int32_t batchIdx;
  int32_t headNumIdx;
  int32_t n1Idx;
  int32_t n2Idx;
  int32_t gIdx = 0;
  int32_t offset;
  int32_t lengthx = 0;
  int32_t lengthy = 0;
  int8_t mask = 0;
  int32_t preAttnOffset = 0;
  int32_t nextAttnOffset = 0;
  int32_t attenMaskDimS2 = ATTEN_MASK_DIM_S2;
  bool needPreToken = false;
  bool needNextToken = false;
};

struct AddrInfo {
  int32_t taskId;
  int32_t blockLength = 0;
  cubeAddrInfo addrInfo[16];
  VecBlockInfo VecBlkInfo[16];
};

struct DetGroup {
  int32_t accumList[24];
  int32_t accumNum{0};
  int32_t recoderS{0};
  int32_t existFirstBlock{0}; // 用于判断是否存在第一块累加的数据
  int32_t existLastBlock{0};  // 用于判断是否存在最后一块累加的数据
  uint64_t outAddr{0};
};

struct VecBlockInfoDet {
  int32_t n2Idx{0};
  int32_t gIdx{0};
  int32_t s1Idx{0};
  int32_t s2Idx{0};
  int32_t s1Len{0};
  int32_t s2Len{0};
  int32_t cubeBlockOffset{0};
  int32_t isFullMask{0};
  int32_t calPreToken{0};
  int32_t calNextToken{0};
  int32_t preTokenOffset{0};
  int32_t nextTokenOffset{0};
};

struct CubeAddrInfoDet {
  int32_t taskId{0};
  int32_t blockLength{0};
  int32_t enableCausalOpt{0};  // 针对sparseMode=3的优化使能开关
  int32_t atmoicAdd{0};        // 如果只存在一个块，则跳过detVec2，直接累加到GM上
  int32_t s1Idx{0};            // 需要计算的左上角顶点s1Idx
  int32_t s2Idx{0};            // 需要计算的左上角顶点s2Idx
  int32_t s1Len{0};            // 需要计算S1方向的长度
  int32_t s2Len{0};            // 需要计算S2方向的长度
  int64_t s1GmAddr{0};         // s1方向对应的GM偏移地址
  int64_t s2GmAddr{0};         // s2方向对应的GM偏移地址
  int64_t sparseLeftBound{0};  // preToken对应的计算参数
  int64_t sparseRightBound{0}; // nextToken对应的计算参数
  DetGroup detS2GroupList[24]; // 确定性计算所需结构体
};

struct VecAddrInfoDet {
  int32_t taskId{0};
  int32_t blockLength{0};
  int32_t enableCausalOpt{0};
  int32_t curCubeIdx{0};
  int32_t attenMaskDimS2{0};
  int32_t batchIdx{0};
  int32_t s1GroupNum{0};
  int32_t s2GroupNum{0};
  VecBlockInfoDet VecBlkInfo[16];
  DetGroup detS1GroupList[24];
  DetGroup detS2GroupList[24];
};

constexpr uint32_t VEC_TO_CUBE_DET = 5;
constexpr uint32_t CUBE2VECDET = 6;
constexpr uint32_t CUBE2VEC = 7;
constexpr uint32_t VEC2CUBE = 8;
constexpr uint32_t CUBE2POST = 9;

/////////////////////////////////////////////////////
// hardware
/////////////////////////////////////////////////////
enum class ArchType { ASCEND_V220, ASCEND_V200, ASCEND_M200 };

template <ArchType ArchTag> struct HardwareInfo {
  static uint32_t const l2BW = 5;
  static uint32_t const hbmBW = 1;
  static uint32_t const supportMix = 0;
  static uint32_t const l1Size = 512 * 1024;
  static uint32_t const l0ASize = 64 * 1024;
  static uint32_t const l0BSize = 64 * 1024;
  static uint32_t const l0CSize = 128 * 1024;
  static uint32_t const l2Size = 192 * 1024 * 1024;
  static uint32_t const biasSize = 1024;
  static uint32_t const fixBufSize = 7 * 1024;
  static uint32_t const ubSize = 192 * 1024;
  static uint32_t const fractalSize = 512;
  static uint32_t const l1l0BlockSize = 32;
  static uint32_t const btBlockSize = 64;
  static uint32_t const fbBlockSize = 128;
};

/////////////////////////////////////////////////////
// mem
/////////////////////////////////////////////////////
enum class BufferType { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

template <ArchType ArchTag> struct AsdopsBuffer {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  __aicore__ AsdopsBuffer() {
    constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
        HardwareInfo<ArchTag>::ubSize, HardwareInfo<ArchTag>::l1Size, HardwareInfo<ArchTag>::l0ASize,
        HardwareInfo<ArchTag>::l0BSize, HardwareInfo<ArchTag>::l0CSize};
#ifdef __DAV_C220_VEC__
    tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
    tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
#elif __DAV_C220_CUBE__
    tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
    tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
    tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
    tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
    tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
    tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
    tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
    tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#else
    tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
    tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
    tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
    tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
    tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
    tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
    tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
    tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
    tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#endif
  };
#pragma GCC diagnostic pop

  template <BufferType BufferType_, typename DstDataType>
  __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const {
    return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
  }

public:
  AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};

/////////////////////////////////////////////////////
// common function
/////////////////////////////////////////////////////

template <typename T> inline __aicore__ T RoundUp(const T val, const T align) {
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  if (align == 0 || val + align - 1 < val) {
    return val;
  }
  return (val + align - 1) / align * align;
}

template <typename T> constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T> inline __aicore__ T CeilDiv(const T dividend, const T divisor) {
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  if (divisor == 0 || dividend + divisor - 1 < dividend) {
    return T_MAX<T>;
  }
  return (dividend + divisor - 1) / divisor;
}

template <typename T> __aicore__ inline T Min(const T lhs, const T rhs) {
  return lhs < rhs ? lhs : rhs;
}

__aicore__ inline bool isLeftDownBeyondLeft(const int32_t s1Idx, const int32_t s2Idx, const int32_t leftBound) {
  // 判断矩阵左下角是否在左边界之内
  return s1Idx > (s2Idx + leftBound) ? true : false;
}

__aicore__ inline bool isLeftDownOverTouchRight(const int32_t s1Idx, const int32_t s2Idx, const int32_t rightBound) {
  // 判断矩阵左下角是否超过或触碰了右边界
  return s1Idx <= (s2Idx + rightBound) ? true : false;
}

__aicore__ inline bool isRightUpBeyondTouchLeft(const int32_t s1Idx, const int32_t s2Idx, const int32_t leftBound) {
  // 判断矩阵右上角是否触碰左边界或在左边界之内
  return s1Idx >= (s2Idx + leftBound) ? true : false;
}

__aicore__ inline bool isRightUpOverRight(const int32_t s1Idx, const int32_t s2Idx, const int32_t rightBound) {
  // 判断矩阵右上角是否超过了右边界
  return s1Idx < (s2Idx + rightBound) ? true : false;
}

__aicore__ inline bool isRightDownOverTouchRight(const int32_t s1Idx, const int32_t s2Idx, const int32_t rightBound) {
  // 判断矩阵右下角是否超过或触碰了右边界
  return s1Idx <= (s2Idx + rightBound) ? true : false;
}

__aicore__ inline bool IsFullMaskBlock(const int32_t s1Idx, const int32_t s2Idx, const int32_t s1Len,
                                       const int32_t s2Len, const int32_t leftBound, const int32_t rightBound,
                                       const uint32_t sparseMode) {
  if (sparseMode == 0) {
    // no mask 需要全部计算
    return false;
  } else if (sparseMode == 1) {
    // all mask 需要全部计算
    return false;
  } else if (sparseMode == 3) {
    return isLeftDownOverTouchRight(s1Idx + s1Len, s2Idx, rightBound);
  } else if (sparseMode == 4) {
    if (isRightUpBeyondTouchLeft(s1Idx, s2Idx + s2Len, leftBound)) {
      return true;
    }

    return isLeftDownOverTouchRight(s1Idx + s1Len, s2Idx, rightBound);
  }

  return false;
}

#endif // COMMON_HEADER_H
