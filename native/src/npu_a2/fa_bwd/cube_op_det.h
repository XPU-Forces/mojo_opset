
#ifndef __CUBE_DET_OP_H__
#define __CUBE_DET_OP_H__

#include "common_header.h"
#include "kernel_operator.h"

using namespace AscendC;
namespace FAG_DET {
template <typename FAGT> class CubeOp {
  using TILING_CLASS = typename FAGT::tiling_class;
  using SEQLEN_TYPE = typename FAGT::seqlen_type;
  using INPUT_TYPE = typename FAGT::input_type;
  static constexpr bool DROP_ENABLE = FAGT::drop_enable;
  static constexpr bool DETERMINISTIC_ENABLE = FAGT::deterministic_enable;

private:
  AscendC::Nd2NzParams commonNd2NzParams{1, SIZE_128, SIZE_128, 0, SIZE_128, SIZE_128, 1, 0};
  AscendC::LoadData2dParams commonLoadData2dParamsTranspose{0, SIZE_128, SIZE_128, 0, 0, true, 0};
  AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose{0, SIZE_128, SIZE_128, 0, 0, false, 0};

  // l1
  LocalTensor<INPUT_TYPE> l1_query_ping_tensor;
  LocalTensor<INPUT_TYPE> l1_query_pong_tensor;
  LocalTensor<INPUT_TYPE> l1_a_ping_tensor;
  LocalTensor<INPUT_TYPE> l1_a_pong_tensor;
  LocalTensor<INPUT_TYPE> l1_b_ping_tensor;
  // L0A L0B
  LocalTensor<INPUT_TYPE> l0_a_ping_tensor;
  LocalTensor<INPUT_TYPE> l0_a_pong_tensor;
  LocalTensor<INPUT_TYPE> l0_b_ping_tensor;
  LocalTensor<INPUT_TYPE> l0_b_pong_tensor;
  // L0C
  LocalTensor<float> l0_c_ping_tensor;
  LocalTensor<float> l0_c_pong_tensor;
  // GM
  GlobalTensor<INPUT_TYPE> tmp_type_gm_tensor;
  GlobalTensor<float> tmp_fp32_gm_tensor;

  uint32_t ping_pong_flag_a = 0;
  uint32_t ping_pong_flag_b = 0;
  uint32_t aEventIDPing = 4;
  uint32_t aEventIDPong = 5;
  uint32_t bEventIDPing = 6;
  uint32_t bEventIDPong = 7;

  uint64_t dimN1;
  uint64_t dimN2;
  uint64_t dimD;
  int32_t cubeBlockIdxArry[4][4];
  int32_t enableCausalOpt{0};

  uint32_t sparseMode;
  int64_t sparseLeftBound{0};  // preToken直线计算公式: s1Idx = s2Idx + sparseLeftBound
  int64_t sparseRightBound{0}; // nextToken直线计算公式: s1Idx = s2Idx + sparseRightBound
  __gm__ INPUT_TYPE *queryAddr;
  __gm__ INPUT_TYPE *keyAddr;
  __gm__ INPUT_TYPE *dyAddr;
  __gm__ INPUT_TYPE *valueAddr;
  __gm__ INPUT_TYPE *pWorkSpaceAddr;
  __gm__ INPUT_TYPE *dsWorkSpaceAddr;
  __gm__ float *mm1WorkSpaceAddr;
  __gm__ float *mm2WorkSpaceAddr;
  __gm__ float *dqDetWorkSpaceAddr;
  __gm__ float *dkDetWorkSpaceAddr;
  __gm__ float *dvDetWorkSpaceAddr;
  __gm__ float *dqWorkSpaceAddr;
  __gm__ float *dkWorkSpaceAddr;
  __gm__ float *dvWorkSpaceAddr;

public:
  __aicore__ inline CubeOp(){};

  __aicore__ inline void Init(TPipe *pipe, const TILING_CLASS *tilingData, __gm__ INPUT_TYPE *queryAddr,
                              __gm__ INPUT_TYPE *keyAddr, __gm__ INPUT_TYPE *dyAddr, __gm__ INPUT_TYPE *valueAddr,
                              __gm__ float *mm1WorkSpaceAddr, __gm__ float *mm2WorkSpaceAddr,
                              __gm__ float *dqDetWorkSpaceAddr, __gm__ float *dkDetWorkSpaceAddr,
                              __gm__ float *dvDetWorkSpaceAddr, __gm__ float *dqWorkSpaceAddr,
                              __gm__ float *dkWorkSpaceAddr, __gm__ float *dvWorkSpaceAddr) {
    dimN2 = tilingData->n2;
    dimN1 = dimN2 * tilingData->g;
    dimD = tilingData->d;
    sparseMode = tilingData->sparseMode;
    this->queryAddr = queryAddr;
    this->keyAddr = keyAddr;
    this->dyAddr = dyAddr;
    this->valueAddr = valueAddr;
    this->mm1WorkSpaceAddr = mm1WorkSpaceAddr;
    this->mm2WorkSpaceAddr = mm2WorkSpaceAddr;
    this->dsWorkSpaceAddr = (__gm__ INPUT_TYPE *)mm1WorkSpaceAddr;
    this->pWorkSpaceAddr = (__gm__ INPUT_TYPE *)mm2WorkSpaceAddr;
    this->dqDetWorkSpaceAddr = dqDetWorkSpaceAddr;
    this->dkDetWorkSpaceAddr = dkDetWorkSpaceAddr;
    this->dvDetWorkSpaceAddr = dvDetWorkSpaceAddr;
    this->dqWorkSpaceAddr = dqWorkSpaceAddr;
    this->dkWorkSpaceAddr = dkWorkSpaceAddr;
    this->dvWorkSpaceAddr = dvWorkSpaceAddr;
    uint64_t config = 0x1;
    TBuf<AscendC::TPosition::A1> L1Buffer;
    TBuf<AscendC::TPosition::CO1> L0CBuffer;
    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    pipe->InitBuffer(L1Buffer, HardwareInfo<ArchType::ASCEND_V220>::l1Size);
    pipe->InitBuffer(L0CBuffer, HardwareInfo<ArchType::ASCEND_V220>::l0CSize);
    tmp_type_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(0));
    tmp_fp32_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(0));
    AscendC::SetNdParaImpl(config);
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);

    // init L1 tensor
    // query数据常驻L1
    l1_query_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(0);
    l1_query_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(SIZE_128 * SIZE_ONE_K);

    // 计算MM345左矩阵时，由于空间不足，需要pingpong
    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_a_pong_tensor =
        asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(SIZE_256 * SIZE_ONE_K + SIZE_64 * SIZE_ONE_K);
    // 计算右矩阵空间充足，直接按偏移存储即可
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(SIZE_384 * SIZE_ONE_K);

    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, INPUT_TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, INPUT_TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, INPUT_TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, INPUT_TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);
  }

  __aicore__ inline __attribute__((always_inline)) void SyncFunc() {
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
  }

  __aicore__ inline __attribute__((always_inline)) void Cube12Process(const CubeAddrInfoDet &shapeInfo) {
    SET_FLAG(M, MTE1, aEventIDPing);
    SET_FLAG(M, MTE1, aEventIDPong);
    SET_FLAG(M, MTE1, bEventIDPing);
    SET_FLAG(M, MTE1, bEventIDPong);
    SET_FLAG(FIX, M, aEventIDPing);
    SET_FLAG(FIX, M, aEventIDPong);
    if (shapeInfo.taskId == 0) {
      enableCausalOpt = shapeInfo.enableCausalOpt;
    }
    sparseLeftBound = shapeInfo.sparseLeftBound;
    sparseRightBound = shapeInfo.sparseRightBound;
    SyncFunc();
    ComputeMM12<QK>(shapeInfo, queryAddr, keyAddr, mm2WorkSpaceAddr);
    SyncFunc();
    ComputeMM12<DYV>(shapeInfo, dyAddr, valueAddr, mm1WorkSpaceAddr);
    SyncFunc();

    WAIT_FLAG(M, MTE1, aEventIDPing);
    WAIT_FLAG(M, MTE1, aEventIDPong);
    WAIT_FLAG(M, MTE1, bEventIDPing);
    WAIT_FLAG(M, MTE1, bEventIDPong);
    WAIT_FLAG(FIX, M, aEventIDPing);
    WAIT_FLAG(FIX, M, aEventIDPong);
  };

  __aicore__ inline __attribute__((always_inline)) void Cube345Process(const CubeAddrInfoDet &shapeInfo) {
    SET_FLAG(MTE1, MTE2, aEventIDPing);
    SET_FLAG(MTE1, MTE2, aEventIDPong);
    SET_FLAG(M, MTE1, aEventIDPing);
    SET_FLAG(M, MTE1, aEventIDPong);
    SET_FLAG(FIX, M, bEventIDPing);
    SET_FLAG(FIX, M, bEventIDPong);
    __gm__ float *dq_out;
    __gm__ float *dk_out;
    __gm__ float *dv_out;
    sparseLeftBound = shapeInfo.sparseLeftBound;
    sparseRightBound = shapeInfo.sparseRightBound;

    if constexpr (DETERMINISTIC_ENABLE) {
      dq_out = dqDetWorkSpaceAddr;
      dk_out = shapeInfo.atmoicAdd ? dkWorkSpaceAddr + shapeInfo.s2GmAddr : dkDetWorkSpaceAddr;
      dv_out = shapeInfo.atmoicAdd ? dvWorkSpaceAddr + shapeInfo.s2GmAddr : dvDetWorkSpaceAddr;
    } else {
      dq_out = dqWorkSpaceAddr + shapeInfo.s1GmAddr;
      dk_out = dkWorkSpaceAddr + shapeInfo.s2GmAddr;
      dv_out = dvWorkSpaceAddr + shapeInfo.s2GmAddr;
    }

    SyncFunc();
    ComputeMMDKV<DK>(shapeInfo, dsWorkSpaceAddr, queryAddr, dk_out);
    SyncFunc();
    ComputeMMDQ<DQ>(shapeInfo, dsWorkSpaceAddr, keyAddr, dq_out);
    SyncFunc();
    ComputeMMDKV<DV>(shapeInfo, pWorkSpaceAddr, dyAddr, dv_out);
    SyncFunc();

    WAIT_FLAG(MTE1, MTE2, aEventIDPing);
    WAIT_FLAG(MTE1, MTE2, aEventIDPong);
    WAIT_FLAG(M, MTE1, aEventIDPing);
    WAIT_FLAG(M, MTE1, aEventIDPong);
    WAIT_FLAG(FIX, M, bEventIDPing);
    WAIT_FLAG(FIX, M, bEventIDPong);
  };

private:
  template <uint32_t CUBE_TYPE>
  __aicore__ inline __attribute__((always_inline)) void
  ComputeMM12(const CubeAddrInfoDet &shapeInfo, __gm__ INPUT_TYPE *left, __gm__ INPUT_TYPE *right, __gm__ float *out) {
    /*
      对于Dq，ReDuce轴为headDim方向（对应K轴），具体遍历顺序示意图如下：
                        dimD(K)                          dimD(K)
                +-----------------------+         +-----------------------+
                |       0/4/8/12        |         |        0/1/2/3        |
                |-----------------------|         |-----------------------|
                |       1/5/9/13        |         |        4/5/6/7        |
           S1(M)|-----------------------|    S2(N)|-----------------------|
                |       2/6/10/14       |         |        8/9/10/11      |
                |---------- ------------|         |-----------------------|
                |       3/7/11/15       |         |       12/13/14/15     |
                +-----------------------+         +-----------------------+
    */
    uint32_t taskIdPingPong = shapeInfo.taskId % 2;
    uint64_t globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + taskIdPingPong * BLOCK_WORKSPACE;
    int32_t cubeBlockIdx = 0; // 用于计算MM12输出结果的下标，一个cubeBlock的大小是128*128
    int32_t mSize = shapeInfo.s1Len;
    int32_t nSize = shapeInfo.s2Len;
    int32_t mLoop = CeilDiv(mSize, static_cast<decltype(mSize)>(SIZE_128));
    int32_t nLoop = CeilDiv(nSize, static_cast<decltype(nSize)>(SIZE_128));
    int32_t mTail = (mSize % SIZE_128) == 0 ? SIZE_128 : (mSize % SIZE_128);
    int32_t nTail = (nSize % SIZE_128) == 0 ? SIZE_128 : (nSize % SIZE_128);
    auto gm_a = left + shapeInfo.s1GmAddr;
    auto gm_b = right + shapeInfo.s2GmAddr;
    auto gm_c = out + globalBlockOffset;
    const uint64_t left_offset = gm_a - (__gm__ INPUT_TYPE *)0;
    const uint64_t right_offset = gm_b - (__gm__ INPUT_TYPE *)0;
    const uint64_t out_offset = gm_c - (__gm__ float *)0;

    for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
      int32_t s2Idx = shapeInfo.s2Idx + nIdx * SIZE_128;
      int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
      int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
      bool needCopyInAMatrix = nIdx == 0 ? true : false;
      bool needCopyInBMatrix = true;
      LocalTensor<INPUT_TYPE> l1_b_tensor = l1_b_ping_tensor[nIdx * SIZE_128 * dimD];
      LocalTensor<INPUT_TYPE> l0_b_tensor = ping_pong_flag_b ? l0_b_ping_tensor : l0_b_pong_tensor;
      uint32_t bEventID = ping_pong_flag_b ? bEventIDPing : bEventIDPong;

      for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
        int32_t s1Idx = shapeInfo.s1Idx + mIdx * SIZE_128;
        int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
        if (enableCausalOpt &&
            IsFullMaskBlock(s1Idx, s2Idx, mProcess, nProcess, sparseLeftBound, sparseRightBound, sparseMode)) {
          continue;
        }
        LocalTensor<INPUT_TYPE> l1_a_tensor;
        LocalTensor<INPUT_TYPE> l0_a_tensor = ping_pong_flag_a ? l0_a_ping_tensor : l0_a_pong_tensor;
        LocalTensor<float> l0_c_tensor = ping_pong_flag_a ? l0_c_ping_tensor : l0_c_pong_tensor;
        uint32_t aEventID = ping_pong_flag_a ? aEventIDPing : aEventIDPong;
        int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);

        if constexpr (CUBE_TYPE == QK) {
          // reuse query
          l1_a_tensor = taskIdPingPong ? l1_query_ping_tensor[mIdx * SIZE_128 * dimD]
                                       : l1_query_pong_tensor[mIdx * SIZE_128 * dimD];
        } else {
          l1_a_tensor = l1_a_ping_tensor[mIdx * SIZE_128 * dimD];
        }

        if (needCopyInAMatrix) {
          commonNd2NzParams.nValue = mProcess;
          commonNd2NzParams.dValue = dimD;
          commonNd2NzParams.srcDValue = dimD * dimN1;
          commonNd2NzParams.dstNzC0Stride = mProcessAlign;
          AscendC::DataCopy(l1_a_tensor, tmp_type_gm_tensor[left_offset + mIdx * SIZE_128 * dimN1 * dimD],
                            commonNd2NzParams);

          SET_FLAG(MTE2, MTE1, EVENT_ID0);
          WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        WAIT_FLAG(M, MTE1, aEventID);
        commonLoadData2dParamsNoTranspose.repeatTimes = dimD / SIZE_16;
        commonLoadData2dParamsNoTranspose.srcStride = mProcessAlign / SIZE_16;
        for (int32_t i = 0; i < mProcessAlign / SIZE_16; i++) {
          AscendC::LoadData(l0_a_tensor[i * dimD * SIZE_16], l1_a_tensor[i * SIZE_256],
                            commonLoadData2dParamsNoTranspose);
        }

        if (needCopyInBMatrix) {
          commonNd2NzParams.nValue = nProcess;
          commonNd2NzParams.dValue = dimD;
          commonNd2NzParams.srcDValue = dimD * dimN2;
          commonNd2NzParams.dstNzC0Stride = nProcessAlign;
          AscendC::DataCopy(l1_b_tensor, tmp_type_gm_tensor[right_offset + nIdx * SIZE_128 * dimN2 * dimD],
                            commonNd2NzParams);

          SET_FLAG(MTE2, MTE1, EVENT_ID0);
          WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
          WAIT_FLAG(M, MTE1, bEventID);

          commonLoadData2dParamsNoTranspose.repeatTimes = dimD * nProcessAlign / SIZE_256;
          commonLoadData2dParamsNoTranspose.srcStride = 1;
          AscendC::LoadData(l0_b_tensor, l1_b_tensor, commonLoadData2dParamsNoTranspose);
        }

        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, aEventID);

        MmadParams madParams;
        madParams.m = mProcess == 1 ? 2 : mProcess;
        madParams.n = nProcess;
        madParams.k = dimD;
        madParams.cmatrixInitVal = true;
        madParams.unitFlag = 3;
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        PipeBarrier<PIPE_M>();

        SET_FLAG(M, MTE1, aEventID);
        if (needCopyInBMatrix) {
          SET_FLAG(M, MTE1, bEventID);
          needCopyInBMatrix = false;
        }

        copy_matrix_cc_to_gm(
            (__gm__ float *)tmp_fp32_gm_tensor.GetPhyAddr() + out_offset + cubeBlockIdx * BASE_BLOCK_SIZE,
            (__cc__ float *)l0_c_tensor.GetPhyAddr(),
            (uint8_t)0,              // uint8_t sid,
            (uint16_t)nProcess,      //  uint16_t NSize,
            (uint16_t)mProcess,      // uint16_t MSize,
            (uint32_t)128,           // uint32_t dstStride_dst_D,
            (uint16_t)mProcessAlign, // uint16_t srcStride,
            3,                       // uint8_t UnitFlagMode,
            0,                       //  uint64_t QuantPRE,
            0,                       //  uint8_t ReLUPRE,
            0,                       // bool channelSplit,
            1                        // bool NZ2ND_EN
        );
        SET_FLAG(FIX, M, aEventID);
        cubeBlockIdx++;
        ping_pong_flag_a = 1 - ping_pong_flag_a;
      }
      ping_pong_flag_b = 1 - ping_pong_flag_b;
    }
  };

  template <uint32_t CUBE_TYPE>
  __aicore__ inline __attribute__((always_inline)) void
  ComputeMMDQ(const CubeAddrInfoDet &shapeInfo, __gm__ INPUT_TYPE *left, __gm__ INPUT_TYPE *right, __gm__ float *out) {
    /*
      对于Dq，ReDuce轴为S2方向（对应N轴），具体遍历顺序示意图如下：
                        S2(N)                           dimD(K)
                +-----------------------+         +-----------------------+
                |  0  |  1  |  2  |  3  |         |       0/4/8/12        |
                |-----|-----|-----|-----|         |-----|-----|-----|-----|
                |  4  |  5  |  6  |  7  |         |       1/5/9/13        |
           S1(M)|-----|-----|-----|-----|    S2(N)|-----|-----|-----|-----|
                |  8  |  9  |  10 |  11 |         |       2/6/10/14       |
                |-----|-----|-----+-----|         |-----|-----|-----+-----|
                |  12 |  13 |  14 |  15 |         |       3/7/11/15       |
                +-----|-----|-----+-----+         +-----|-----|-----+-----+
    */
    uint32_t taskIdPingPong = shapeInfo.taskId % 2;
    uint64_t globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + taskIdPingPong * BLOCK_WORKSPACE;
    uint32_t mSize = shapeInfo.s1Len;
    uint32_t nSize = shapeInfo.s2Len;
    int32_t mTail = (mSize % SIZE_128) == 0 ? SIZE_128 : (mSize % SIZE_128);
    int32_t nTail = (nSize % SIZE_128) == 0 ? SIZE_128 : (nSize % SIZE_128);
    int32_t mLoop = CeilDiv(mSize, static_cast<decltype(mSize)>(SIZE_128));
    int32_t nLoop = CeilDiv(nSize, static_cast<decltype(nSize)>(SIZE_128));
    uint64_t left_offset = (__gm__ INPUT_TYPE *)(left + globalBlockOffset * 2) - (__gm__ INPUT_TYPE *)0;
    uint64_t right_offset = (__gm__ INPUT_TYPE *)(right + shapeInfo.s2GmAddr) - (__gm__ INPUT_TYPE *)0;
    uint64_t out_offset = (__gm__ float *)(out) - (__gm__ float *)0;
    bool isCausalBlock = shapeInfo.s1Idx == shapeInfo.s2Idx;
    bool visitBMatrix[4] = {false, false, false, false};

    for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
      int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
      int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);
      int32_t s1Idx = shapeInfo.s1Idx + mIdx * SIZE_128;
      bool initL0C = true;
      uint32_t bEventID = ping_pong_flag_b ? bEventIDPing : bEventIDPong;
      LocalTensor<float> l0_c_buf_tensor = ping_pong_flag_b ? l0_c_ping_tensor : l0_c_pong_tensor;

      for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
        int32_t s2Idx = shapeInfo.s2Idx + nIdx * SIZE_128;
        int32_t cubeBlockIdx = cubeBlockIdxArry[nIdx][mIdx];
        int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
        int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
        bool needCopyInBMatrix = false;
        bool needCopyOut = (nIdx == nLoop - 1) ? true : false;

        if (enableCausalOpt &&
            IsFullMaskBlock(s1Idx, s2Idx, mProcess, nProcess, sparseLeftBound, sparseRightBound, sparseMode)) {
          continue;
        }
        if (!visitBMatrix[nIdx]) {
          visitBMatrix[nIdx] = true;
          needCopyInBMatrix = true;
        }

        if (enableCausalOpt && isCausalBlock && s1Idx == s2Idx) {
          needCopyOut = true;
        }
        uint64_t real_left_offset = left_offset + cubeBlockIdx * BASE_BLOCK_SIZE * 2;
        uint64_t real_right_offset = right_offset + nIdx * SIZE_128 * (dimN2 * dimD);
        uint32_t aEventID = ping_pong_flag_a ? aEventIDPing : aEventIDPong;
        LocalTensor<INPUT_TYPE> l1_a_buf_tensor = ping_pong_flag_a ? l1_a_ping_tensor : l1_a_pong_tensor;
        LocalTensor<INPUT_TYPE> l0_a_buf_tensor = ping_pong_flag_a ? l0_a_ping_tensor : l0_a_pong_tensor;
        LocalTensor<INPUT_TYPE> l1_b_buf_tensor = l1_b_ping_tensor[nIdx * SIZE_128 * dimD];
        LocalTensor<INPUT_TYPE> l0_b_buf_tensor = ping_pong_flag_a ? l0_b_ping_tensor : l0_b_pong_tensor;

        WAIT_FLAG(MTE1, MTE2, aEventID);
        WAIT_FLAG(M, MTE1, aEventID);
        this->template MM345LoadA<CUBE_TYPE>(l1_a_buf_tensor, l0_a_buf_tensor, tmp_type_gm_tensor[real_left_offset],
                                             mProcess, nProcess, mProcessAlign, nProcessAlign, true);
        SET_FLAG(MTE1, MTE2, aEventID);

        MM345LoadB(l1_b_buf_tensor, l0_b_buf_tensor, tmp_type_gm_tensor[real_right_offset], nProcess, nProcessAlign,
                   dimN2, needCopyInBMatrix);

        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, bEventID);

        MmadParams madParams;
        madParams.m = mProcess == 1 ? 2 : mProcess;
        madParams.n = dimD;
        madParams.k = nProcess;
        madParams.cmatrixInitVal = initL0C;
        madParams.unitFlag = needCopyOut ? 3 : 2;
        AscendC::Mmad(l0_c_buf_tensor, l0_a_buf_tensor, l0_b_buf_tensor, madParams);
        SET_FLAG(M, MTE1, aEventID);
        PipeBarrier<PIPE_M>();

        if (needCopyOut) {
          if constexpr (DETERMINISTIC_ENABLE) {
            MM345CopyOut<false>(l0_c_buf_tensor, out_offset + mIdx * SIZE_128 * dimD, mProcess, dimD, mProcessAlign);
          } else {
            MM345CopyOut<true>(l0_c_buf_tensor, out_offset + mIdx * SIZE_128 * dimN1 * dimD, mProcess, dimN1 * dimD,
                               mProcessAlign);
          }
        }
        SET_FLAG(FIX, M, bEventID);

        initL0C = false;
        ping_pong_flag_a = 1 - ping_pong_flag_a;
      }
      ping_pong_flag_b = 1 - ping_pong_flag_b;
    }
  };

  template <uint32_t CUBE_TYPE>
  __aicore__ inline __attribute__((always_inline)) void
  ComputeMMDKV(const CubeAddrInfoDet &shapeInfo, __gm__ INPUT_TYPE *left, __gm__ INPUT_TYPE *right, __gm__ float *out) {
    /*
      对于Dk，Dv，ReDuce轴为S1方向（对应M轴），具体遍历顺序示意图如下：
                        S2(N)                           dimD(K)
                +-----------------------+         +-----------------------+
                |  0  |  4  |  8  |  12 |         |       0/4/8/12        |
                |-----|-----|-----|-----|         |-----------------------|
                |  1  |  5  |  9  |  13 |         |       1/5/9/13        |
            S1(M)|-----|-----|-----|-----|    S1(M)|-----------------------|
                |  2  |  6  |  10 |  14 |         |       2/6/10/14       |
                |-----|-----|-----+-----|         |-----------------------|
                |  3  |  7  |  11 |  15 |         |       3/7/11/15       |
                +-----|-----|-----+-----+         +-----------------------+
    */
    uint32_t taskIdPingPong = shapeInfo.taskId % 2;
    uint64_t globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + taskIdPingPong * BLOCK_WORKSPACE;
    uint32_t cubeBlockIdx = 0;
    uint32_t mSize = shapeInfo.s1Len;
    uint32_t nSize = shapeInfo.s2Len;
    int32_t mTail = (mSize % SIZE_128) == 0 ? SIZE_128 : (mSize % SIZE_128);
    int32_t nTail = (nSize % SIZE_128) == 0 ? SIZE_128 : (nSize % SIZE_128);
    int32_t mLoop = CeilDiv(mSize, static_cast<decltype(mSize)>(SIZE_128));
    int32_t nLoop = CeilDiv(nSize, static_cast<decltype(nSize)>(SIZE_128));
    uint64_t left_offset = (__gm__ INPUT_TYPE *)(left + globalBlockOffset * 2) - (__gm__ INPUT_TYPE *)0;
    uint64_t right_offset = (__gm__ INPUT_TYPE *)(right + shapeInfo.s1GmAddr) - (__gm__ INPUT_TYPE *)0;
    uint64_t out_offset = (__gm__ float *)(out) - (__gm__ float *)0;

    for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
      int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
      int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
      int32_t s2Idx = shapeInfo.s2Idx + nIdx * SIZE_128;
      bool initL0C = true;
      uint32_t bEventID = ping_pong_flag_b ? bEventIDPing : bEventIDPong;
      LocalTensor<float> l0_c_buf_tensor = ping_pong_flag_b ? l0_c_ping_tensor : l0_c_pong_tensor;
      for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
        int32_t s1Idx = shapeInfo.s1Idx + mIdx * SIZE_128;
        int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
        if (enableCausalOpt &&
            IsFullMaskBlock(s1Idx, s2Idx, mProcess, nProcess, sparseLeftBound, sparseRightBound, sparseMode)) {
          continue;
        }

        int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);
        uint64_t real_left_offset = left_offset + cubeBlockIdx * BASE_BLOCK_SIZE * 2;
        uint64_t real_right_offset = right_offset + mIdx * SIZE_128 * (dimN1 * dimD);
        uint32_t aEventID = ping_pong_flag_a ? aEventIDPing : aEventIDPong;
        bool needCopyInBMatrix;
        LocalTensor<INPUT_TYPE> l1_a_buf_tensor = ping_pong_flag_a ? l1_a_ping_tensor : l1_a_pong_tensor;
        LocalTensor<INPUT_TYPE> l1_b_buf_tensor;
        LocalTensor<INPUT_TYPE> l0_a_buf_tensor = ping_pong_flag_a ? l0_a_ping_tensor : l0_a_pong_tensor;
        LocalTensor<INPUT_TYPE> l0_b_buf_tensor = ping_pong_flag_a ? l0_b_ping_tensor : l0_b_pong_tensor;

        if constexpr (CUBE_TYPE == DK) {
          // reuse query
          l1_b_buf_tensor = taskIdPingPong ? l1_query_ping_tensor[mIdx * SIZE_128 * dimD]
                                           : l1_query_pong_tensor[mIdx * SIZE_128 * dimD];
          needCopyInBMatrix = false;
        } else {
          l1_b_buf_tensor = l1_b_ping_tensor[mIdx * SIZE_128 * dimD];
          needCopyInBMatrix = nIdx == 0;
        }

        WAIT_FLAG(MTE1, MTE2, aEventID);
        WAIT_FLAG(M, MTE1, aEventID);
        this->template MM345LoadA<CUBE_TYPE>(l1_a_buf_tensor, l0_a_buf_tensor, tmp_type_gm_tensor[real_left_offset],
                                             mProcess, nProcess, mProcessAlign, nProcessAlign, true);
        SET_FLAG(MTE1, MTE2, aEventID);

        MM345LoadB(l1_b_buf_tensor, l0_b_buf_tensor, tmp_type_gm_tensor[real_right_offset], mProcess, mProcessAlign,
                   dimN1, needCopyInBMatrix);

        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, bEventID);

        MmadParams madParams;
        madParams.m = nProcess == 1 ? 2 : nProcess;
        madParams.n = dimD;
        madParams.k = mProcess;
        madParams.cmatrixInitVal = initL0C;
        madParams.unitFlag = mIdx == mLoop - 1 ? 3 : 2;
        AscendC::Mmad(l0_c_buf_tensor, l0_a_buf_tensor, l0_b_buf_tensor, madParams);
        SET_FLAG(M, MTE1, aEventID);
        PipeBarrier<PIPE_M>();

        if (mIdx == mLoop - 1) {
          if constexpr (DETERMINISTIC_ENABLE) {
            if (shapeInfo.atmoicAdd) {
              MM345CopyOut<true>(l0_c_buf_tensor, out_offset + nIdx * SIZE_128 * dimN2 * dimD, nProcess, dimN2 * dimD,
                                 nProcessAlign);
            } else {
              MM345CopyOut<false>(l0_c_buf_tensor, out_offset + nIdx * SIZE_128 * dimD, nProcess, dimD, nProcessAlign);
            }
          } else {
            MM345CopyOut<true>(l0_c_buf_tensor, out_offset + nIdx * SIZE_128 * dimN2 * dimD, nProcess, dimN2 * dimD,
                               nProcessAlign);
          }
        }

        SET_FLAG(FIX, M, bEventID);
        if constexpr (CUBE_TYPE == DK) {
          // 记录cubeBlockIdx, 用于dQ的计算
          cubeBlockIdxArry[nIdx][mIdx] = cubeBlockIdx;
        }
        initL0C = false;
        cubeBlockIdx++;
        ping_pong_flag_a = 1 - ping_pong_flag_a;
      }

      ping_pong_flag_b = 1 - ping_pong_flag_b;
    }
  };

  template <uint32_t CUBE_TYPE>
  __aicore__ inline __attribute__((always_inline)) void
  MM345LoadA(LocalTensor<INPUT_TYPE> dstL1Tensor, LocalTensor<INPUT_TYPE> dstL0Tensor,
             GlobalTensor<INPUT_TYPE> srcTensor, const int32_t baseM, const int32_t baseN, const int32_t baseMAlign,
             const int32_t baseNAlign, const bool needCopyIn) {
    if (needCopyIn) {
      uint32_t mUp = (baseM + 1) / 2;
      uint32_t mDown = baseM - mUp;
      commonNd2NzParams.nValue = mUp;
      commonNd2NzParams.dValue = baseN;
      commonNd2NzParams.dstNzC0Stride = baseMAlign;
      commonNd2NzParams.srcDValue = SIZE_128;
      AscendC::DataCopy(dstL1Tensor, srcTensor, commonNd2NzParams);
      if (mDown > 0) {
        commonNd2NzParams.nValue = mDown;
        AscendC::DataCopy(dstL1Tensor[mUp * SIZE_16], srcTensor[mUp * SIZE_128 * 2], commonNd2NzParams);
      }

      SET_FLAG(MTE2, MTE1, EVENT_ID0);
      WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
    }

    if constexpr (CUBE_TYPE == DQ) {
      commonLoadData2dParamsNoTranspose.repeatTimes = baseNAlign / SIZE_16;
      commonLoadData2dParamsNoTranspose.srcStride = baseMAlign / SIZE_16;
      for (int32_t i = 0; i < baseMAlign / SIZE_16; i++) {
        AscendC::LoadData(dstL0Tensor[i * baseNAlign * SIZE_16], dstL1Tensor[i * SIZE_16 * SIZE_16],
                          commonLoadData2dParamsNoTranspose);
      }
    } else {
      commonLoadData2dParamsTranspose.repeatTimes = (baseMAlign / SIZE_16) * (baseNAlign / SIZE_16);
      commonLoadData2dParamsTranspose.srcStride = 1;
      AscendC::LoadData(dstL0Tensor, dstL1Tensor, commonLoadData2dParamsTranspose);
    }
  };

  __aicore__ inline __attribute__((always_inline)) void
  MM345LoadB(LocalTensor<INPUT_TYPE> dstL1Tensor, LocalTensor<INPUT_TYPE> dstL0Tensor,
             GlobalTensor<INPUT_TYPE> srcTensor, const int32_t processLen, const int32_t processLenAlign,
             const int32_t headNum, const bool needCopyIn) {
    if (needCopyIn) {
      commonNd2NzParams.nValue = processLen;
      commonNd2NzParams.dValue = dimD;
      commonNd2NzParams.srcDValue = headNum * dimD;
      commonNd2NzParams.dstNzC0Stride = processLenAlign;
      AscendC::DataCopy(dstL1Tensor, srcTensor, commonNd2NzParams);

      SET_FLAG(MTE2, MTE1, EVENT_ID0);
      WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
    }

    commonLoadData2dParamsTranspose.repeatTimes = dimD / SIZE_16;
    commonLoadData2dParamsTranspose.srcStride = processLenAlign / SIZE_16;
    for (int32_t j = 0; j < processLenAlign / SIZE_16; j++) {
      AscendC::LoadData(dstL0Tensor[j * dimD * SIZE_16], dstL1Tensor[j * SIZE_16 * SIZE_16],
                        commonLoadData2dParamsTranspose);
    }
  };

  template <bool ATMOIC_ADD>
  __aicore__ inline __attribute__((always_inline)) void MM345CopyOut(LocalTensor<float> srcTensor, uint64_t outOffset,
                                                                     uint16_t mSize, uint32_t dstStride,
                                                                     uint16_t srcStride) {
    if constexpr (ATMOIC_ADD) {
      AscendC::SetAtomicType<float>();
    }

    copy_matrix_cc_to_gm((__gm__ float *)tmp_fp32_gm_tensor.GetPhyAddr() + outOffset,
                         (__cc__ float *)srcTensor.GetPhyAddr(),
                         (uint8_t)0,          // uint8_t sid,
                         (uint16_t)dimD,      //  uint16_t NSize,
                         (uint16_t)mSize,     // uint16_t MSize,
                         (uint32_t)dstStride, // uint32_t dstStride_dst_D,
                         (uint16_t)srcStride, // uint16_t srcStride,
                         3,                   // uint8_t UnitFlagMode,
                         0,                   //  uint64_t QuantPRE,
                         0,                   //  uint8_t ReLUPRE,
                         0,                   // bool channelSplit,
                         1                    // bool NZ2ND_EN
    );

    if constexpr (ATMOIC_ADD) {
      AscendC::SetAtomicNone();
    }
  };
};
} // namespace FAG_DET
#endif // __CUBEFORWARD_H__
