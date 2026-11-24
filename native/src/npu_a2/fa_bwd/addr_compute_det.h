#include "common_header.h"
using namespace AscendC;

namespace FAG_DET {
template <typename FAGT> class AddrComputeDet {
  using TILING_CLASS = typename FAGT::tiling_class;
  using SEQLEN_TYPE = typename FAGT::seqlen_type;
  using INPUT_TYPE = typename FAGT::input_type;
  static constexpr bool DROP_ENABLE = FAGT::drop_enable;
  static constexpr bool DETERMINISTIC_ENABLE = FAGT::deterministic_enable;

public:
  __aicore__ inline void GetCubeAddrInfo(CubeAddrInfoDet *cubeAddrInfo) {
    globalCubeAddr = cubeAddrInfo;
    globalCubeAddr->blockLength = 0;
    globalCubeAddr->atmoicAdd = 0;
    s2GroupNum = 0;
    for (int32_t i = 0; i < cubeCoreNum; i++) {
      globalCubeAddr->detS2GroupList[i].accumNum = 0;
    }
    ComputeAddrInfo<false>();
    for (int32_t i = 0; i < s2GroupNum; i++) {
      if (globalCubeAddr->detS2GroupList[i].accumNum == 1 &&
          globalCubeAddr->detS2GroupList[i].accumList[0] == cubeCoreIdx) {
        // k/v 如果累加数量只有1，则直接通过atmoicAdd指令累加，不再经过vec
        globalCubeAddr->atmoicAdd = 1;
      }
    }
  }

  __aicore__ inline void GetVecAddrInfo(VecAddrInfoDet *vecAddrInfo) {
    globalVecAddr = vecAddrInfo;
    if (globalVecAddr->taskId == 0) {
      globalVecAddr->attenMaskDimS2 = sparseMode == 1 ? maxSeqK : ATTEN_MASK_DIM_S2;
      globalVecAddr->enableCausalOpt = enableCausalOpt;
    }
    globalVecAddr->blockLength = 0;
    s1GroupNum = 0;
    s2GroupNum = 0;
    for (int32_t i = 0; i < cubeCoreNum; i++) {
      globalVecAddr->detS1GroupList[i].accumNum = 0;
      globalVecAddr->detS1GroupList[i].existFirstBlock = 0;
      globalVecAddr->detS1GroupList[i].existLastBlock = 0;
      globalVecAddr->detS2GroupList[i].accumNum = 0;
    }
    ComputeAddrInfo<true>();
    globalVecAddr->curCubeIdx = curCubeIdx % cubeCoreNum;
    globalVecAddr->s1GroupNum = s1GroupNum;
    globalVecAddr->s2GroupNum = s2GroupNum;
  }

  __aicore__ void Init(__gm__ uint8_t *seqLenQ, __gm__ uint8_t *seqLenK, uint32_t cubeCoreNum, uint32_t cubeCoreIdx,
                       const TILING_CLASS *tilingData) {
    this->seqLenQ = seqLenQ;
    this->seqLenK = seqLenK;
    this->cubeCoreIdx = cubeCoreIdx;
    this->cubeCoreNum = cubeCoreNum;
    this->dimB = tilingData->b;
    this->dimN2 = tilingData->n2;
    this->dimG = tilingData->g;
    this->dimN1 = dimN2 * dimG;
    this->dimD = tilingData->d;
    this->sparseMode = tilingData->sparseMode;
    this->preToken = tilingData->preTockens;
    this->nextToken = tilingData->nextTockens;
    this->dqPostAbsorb = tilingData->dqPostAbsorb;
    UpdateSeqLen();

    if (sparseMode == 1) {
      for (int32_t i = 0; i < dimB; i++) {
        maxSeqK = max(maxSeqK, getSeqLen(i, seqLenK));
      }
    } else if (sparseMode == 3) {
      for (int32_t i = 0; i < dimB; i++) {
        if (getSeqLen(i, seqLenQ) != getSeqLen(i, seqLenK)) {
          // sparsemode=3，qk不等长情况不使能优化
          return;
        }
      }
      enableCausalOpt = 1;
    }
  }

private:
  __gm__ uint8_t *seqLenQ;
  __gm__ uint8_t *seqLenK;
  CubeAddrInfoDet *globalCubeAddr; // 用于存储Cube计算相关的地址信息
  VecAddrInfoDet *globalVecAddr;   // 用于存储Vector计算相关的地址信息
  uint32_t cubeCoreIdx{0};         // 当前核所对应的Cube核的下标
  uint32_t cubeCoreNum{0};         // Cube核的数量

  SEQLEN_TYPE maxSeqK{0};
  int32_t dimB{0};              // batch
  SEQLEN_TYPE dimS1{0};         // 当前处理的batch的s1
  SEQLEN_TYPE dimS2{0};         // 当前处理的batch的s2
  int32_t dimN1{0};             // query的HeadNum
  int32_t dimN2{0};             // key/vaule的HeadNum
  int32_t dimG{0};              // group，dimN1=dimN2*dimG
  int32_t dimD{0};              // headDim
  int32_t bIdx{0};              // 当前batch计算到的位置
  int32_t s1Idx{0};             // 当前s1方向计算到的位置
  int32_t s2Idx{0};             // 当前s2方向计算到的位置
  int32_t n1Idx{0};             // 当前n1方向计算到的位置
  SEQLEN_TYPE lastBatchQSum{0}; // 已处理的s1的总和
  SEQLEN_TYPE lastBatchKSum{0}; // 已处理的s2的总和
  int32_t curCubeIdx{0};        // 用于分配核时的中间变量，若curCubeIdx==cubeCoreIdx,则表示该核需要计算数据
  int32_t loopFinish{0};        // 用于控制Loop结束的中间变量，当所有Cube核在当前Loop都分配到任务时，一次Loop结束
  int32_t enableCausalOpt{0};
  int32_t dqPostAbsorb{0};
  // 稀疏计算参数
  uint32_t sparseMode{0}; // 稀疏模式
  int64_t preToken{0};
  int64_t nextToken{0};
  int64_t sparseLeftBound{0};  // preToken直线计算公式: s1Idx = s2Idx + sparseLeftBound
  int64_t sparseRightBound{0}; // nextToken直线计算公式: s1Idx = s2Idx + sparseRightBound
  // 确定性计算参数
  int32_t s1GroupNum{0}; // 当前s1方向计算到需要累加的分组个数
  int32_t s2GroupNum{0}; // 当前s2方向计算到需要累加的分组个数

  __aicore__ inline void UpdateSeqLen() {
    dimS1 = getSeqLen(bIdx, seqLenQ);
    dimS2 = getSeqLen(bIdx, seqLenK);
    while ((dimS1 == 0 || dimS2 == 0) && bIdx < dimB - 1) {
      bIdx++;
      dimS1 = getSeqLen(bIdx, seqLenQ);
      dimS2 = getSeqLen(bIdx, seqLenK);
    }
    sparseLeftBound = dimS1 - dimS2 + preToken + 1;
    sparseRightBound = dimS1 - dimS2 - nextToken;
    if (bIdx > 0) {
      lastBatchQSum = getTotalLen(bIdx - 1, seqLenQ);
      lastBatchKSum = getTotalLen(bIdx - 1, seqLenK);
    } else {
      lastBatchQSum = 0;
      lastBatchKSum = 0;
    }
  }

  __aicore__ inline SEQLEN_TYPE getSeqLen(int32_t i, __gm__ uint8_t *seq_Len) {
    SEQLEN_TYPE actualSeqlen;
    if (i == 0) {
      actualSeqlen = ((__gm__ SEQLEN_TYPE *)seq_Len)[0];
    } else {
      actualSeqlen = ((__gm__ SEQLEN_TYPE *)seq_Len)[i] - ((__gm__ SEQLEN_TYPE *)seq_Len)[i - 1];
    }
    return actualSeqlen;
  }

  __aicore__ inline SEQLEN_TYPE getTotalLen(int32_t i, __gm__ uint8_t *seq_Len) {
    SEQLEN_TYPE actualTotalSeqlen = ((__gm__ SEQLEN_TYPE *)seq_Len)[i];
    return actualTotalSeqlen;
  }

  __aicore__ inline uint64_t getLeftAddr(int32_t lastBatchSum, int32_t s1Idx, int32_t n1Idx) {
    return lastBatchSum * dimN1 * dimD + (s1Idx * dimN1 * dimD) + (n1Idx * dimD);
  }

  __aicore__ inline uint64_t getRightAddr(int32_t lastBatchSum, int32_t s2Idx, int32_t n1Idx) {
    return lastBatchSum * dimN2 * dimD + (s2Idx * dimN2 * dimD) + ((n1Idx / dimG) * dimD);
  }

  __aicore__ inline int32_t GetRecoderS(int32_t sIdx, int32_t sLen) { return sIdx + 512 < sLen ? 512 : (sLen - sIdx); }

  __aicore__ inline void UpdateLoopIdx() {
    curCubeIdx++;
    if (curCubeIdx % cubeCoreNum == 0) {
      loopFinish = 1;
    }
  }

  template <bool IS_DQ>
  __aicore__ inline void DetInfoProcess(DetGroup (&detGroupList)[24], int32_t recoderS1, int32_t recoderS2,
                                        int32_t &groupNum, uint64_t outAddr) {
    uint32_t groupIdx;
    int32_t find{0};
    int32_t recoderS;
    if constexpr (IS_DQ) {
      recoderS = recoderS1;
    } else {
      recoderS = recoderS2;
    }

    for (uint32_t i = 0; i < groupNum; i++) {
      if (detGroupList[i].outAddr == outAddr) {
        find = 1;
        groupIdx = i;
        break;
      }
    }

    if (!find) {
      groupNum++;
      groupIdx = groupNum - 1;
      detGroupList[groupIdx].outAddr = outAddr;
      detGroupList[groupIdx].recoderS = recoderS;
    }

    uint32_t accIdx = detGroupList[groupIdx].accumNum;
    detGroupList[groupIdx].accumList[accIdx] = curCubeIdx % cubeCoreNum;
    detGroupList[groupIdx].accumNum++;

    if constexpr (IS_DQ) {
      if (!dqPostAbsorb) {
        return;
      }
      if (s2Idx == 0) {
        detGroupList[groupIdx].existFirstBlock = 1;
      }
      if ((s2Idx + recoderS2 == dimS2)) {
        detGroupList[groupIdx].existLastBlock = 1;
      } else if (sparseMode == 3 && isRightDownOverTouchRight(s1Idx + recoderS1, s2Idx + recoderS2, sparseRightBound)) {
        detGroupList[groupIdx].existLastBlock = 1;
      }
    }
  }

  template <bool IS_AIV> __aicore__ inline void DetRecord(int32_t recoderS1, int32_t recoderS2) {
    // 保存所有核的确定性信息
    if constexpr (IS_AIV) {
      DetInfoProcess<true>(globalVecAddr->detS1GroupList, recoderS1, recoderS2, s1GroupNum,
                           getLeftAddr(lastBatchQSum, s1Idx, n1Idx));
      DetInfoProcess<false>(globalVecAddr->detS2GroupList, recoderS1, recoderS2, s2GroupNum,
                            getRightAddr(lastBatchKSum, s2Idx, n1Idx));
    } else {
      DetInfoProcess<false>(globalCubeAddr->detS2GroupList, recoderS1, recoderS2, s2GroupNum,
                            getRightAddr(lastBatchKSum, s2Idx, n1Idx));
    }
  }

  __aicore__ inline void CubeRecord(int32_t recoderS1, int32_t recoderS2) {
    if (curCubeIdx % cubeCoreNum != cubeCoreIdx) {
      // 非确定性信息只保存当前核
      return;
    }

    globalCubeAddr->s1GmAddr = getLeftAddr(lastBatchQSum, s1Idx, n1Idx);
    globalCubeAddr->s2GmAddr = getRightAddr(lastBatchKSum, s2Idx, n1Idx);
    globalCubeAddr->s1Len = recoderS1;
    globalCubeAddr->s2Len = recoderS2;
    globalCubeAddr->s1Idx = s1Idx;
    globalCubeAddr->s2Idx = s2Idx;
    globalCubeAddr->sparseLeftBound = sparseLeftBound;
    globalCubeAddr->sparseRightBound = sparseRightBound;
    globalCubeAddr->enableCausalOpt = enableCausalOpt;
    globalCubeAddr->blockLength++;
  }

  __aicore__ inline void VecRecord(int32_t recoderS1, int32_t recoderS2) {
    if (curCubeIdx % cubeCoreNum != cubeCoreIdx) {
      // 非确定性信息只保存当前核
      return;
    }

    int32_t blockId = 0;
    int32_t s1Tail = (recoderS1 % 128 == 0) ? 128 : recoderS1 % 128;
    int32_t s2Tail = (recoderS2 % 128 == 0) ? 128 : recoderS2 % 128;
    int32_t s1Loop = CeilDiv(recoderS1, 128);
    int32_t s2Loop = CeilDiv(recoderS2, 128);
    globalVecAddr->batchIdx = bIdx;

    for (int32_t x = 0; x < s2Loop; x++) {
      for (int32_t y = 0; y < s1Loop; y++) {
        int32_t curS1Idx = s1Idx + y * 128;
        int32_t curS2Idx = s2Idx + x * 128;
        int32_t curS1Len = (y == s1Loop - 1) ? s1Tail : 128;
        int32_t curS2Len = (x == s2Loop - 1) ? s2Tail : 128;
        auto vecPhyAddr = &globalVecAddr->VecBlkInfo[blockId];
        if (IsFullMaskBlock(curS1Idx, curS2Idx, curS1Len, curS2Len, sparseLeftBound, sparseRightBound, sparseMode)) {
          if (enableCausalOpt) {
            continue;
          }
          vecPhyAddr->isFullMask = true;
        } else {
          vecPhyAddr->isFullMask = false;
        }

        vecPhyAddr->n2Idx = n1Idx / dimG;
        vecPhyAddr->gIdx = n1Idx % dimG;
        vecPhyAddr->s1Idx = curS1Idx;
        vecPhyAddr->s2Idx = curS2Idx;
        vecPhyAddr->s1Len = curS1Len;
        vecPhyAddr->s2Len = curS2Len;
        vecPhyAddr->cubeBlockOffset = blockId * BASE_BLOCK_SIZE;
        blockId++;

        // cal Attenmask Info
        if (sparseMode == 0 || vecPhyAddr->isFullMask) {
          continue;
        } else if (sparseMode == 1) {
          vecPhyAddr->calNextToken = 1;
          vecPhyAddr->nextTokenOffset = curS1Idx * maxSeqK + curS2Idx;
        } else {
          if (isRightUpOverRight(curS1Idx, curS2Idx + curS2Len, sparseRightBound)) {
            int32_t sparseS2Idx = (curS1Idx - sparseRightBound - curS2Idx) % 128;
            vecPhyAddr->calNextToken = 1;
            vecPhyAddr->nextTokenOffset = sparseS2Idx >= 0
                                              ? ATTEN_MASK_DEFAULT_OFFSET - sparseS2Idx
                                              : ATTEN_MASK_DEFAULT_OFFSET + sparseS2Idx * ATTEN_MASK_DIM_S2;
          } else {
            vecPhyAddr->calNextToken = 0;
          }

          if (sparseMode == 4 && isLeftDownBeyondLeft(curS1Idx + curS1Len, curS2Idx, sparseLeftBound)) {
            int32_t sparseS2Idx = (curS1Idx - sparseLeftBound - curS2Idx) % 128;
            vecPhyAddr->calPreToken = 1;
            vecPhyAddr->preTokenOffset = sparseS2Idx >= 0 ? ATTEN_MASK_DEFAULT_OFFSET - sparseS2Idx
                                                          : ATTEN_MASK_DEFAULT_OFFSET + sparseS2Idx * ATTEN_MASK_DIM_S2;
          } else {
            vecPhyAddr->calPreToken = 0;
          }
        }
      }
    }
    globalVecAddr->blockLength = blockId;
  }

  __aicore__ inline int32_t InitBNIndex() {
    int32_t recoderS1 = GetRecoderS(s1Idx, dimS1);
    int32_t recoderS2 = GetRecoderS(s1Idx, dimS2);
    bool updateS1Idx;
    if (sparseMode == 0 || sparseMode == 1) {
      updateS1Idx = (s2Idx == dimS2) ? true : false;
    } else {
      // 当矩阵左下角超过右边界，表示当前S2Idx已遍历到最后
      updateS1Idx = isLeftDownOverTouchRight(s1Idx + recoderS1, s2Idx, sparseRightBound);
    }

    if (updateS1Idx) {
      // 处理一轮计算结束时，S2正好全部遍历完的情况。
      s1Idx += recoderS1;
      s2Idx = 0;
    }

    if (s1Idx < dimS1) {
      return false;
    }

    if (n1Idx < dimN1 - 1) {
      s1Idx = 0;
      s2Idx = 0;
      n1Idx++;
      return false;
    }

    if (bIdx < dimB - 1) {
      s1Idx = 0;
      s2Idx = 0;
      n1Idx = 0;
      bIdx++;
      UpdateSeqLen();
      return false;
    }

    return true;
  }

  template <bool IS_AIV> __aicore__ inline void ComputeAddrInfo() {
    while (true) {
      if (InitBNIndex()) {
        break;
      }

      while (s1Idx < dimS1) {
        int32_t recoderS1 = GetRecoderS(s1Idx, dimS1);
        int32_t recoderS2 = GetRecoderS(s2Idx, dimS2);

        if (recoderS2 == 0) {
          s1Idx += recoderS1;
          s2Idx = 0;
          continue;
        }

        if (!IsFullMaskBlock(s1Idx, s2Idx, recoderS1, recoderS2, sparseLeftBound, sparseRightBound, sparseMode)) {
          DetRecord<IS_AIV>(recoderS1, recoderS2);
          if constexpr (IS_AIV) {
            VecRecord(recoderS1, recoderS2);
          } else {
            CubeRecord(recoderS1, recoderS2);
          }
          UpdateLoopIdx();
        }

        s2Idx += recoderS2;

        if (loopFinish) {
          break;
        }
      }

      if (loopFinish) {
        loopFinish = 0;
        break;
      }
    }
  }
};
} // namespace FAG_DET
