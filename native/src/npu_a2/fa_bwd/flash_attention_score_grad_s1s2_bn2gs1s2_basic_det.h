/*!
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_basic_det.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BASIC_DET_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BASIC_DET_H_

#include "addr_compute_det.h"
#include "common_header.h"
#include "cube_op_det.h"
#include "kernel_operator.h"
#include "vec_op_det.h"
#include "vec_post.h"
#include "vec_pre.h"
#include "vec_sfmg.h"
using namespace AscendC;

template <typename FAGT> class FlashAttentionScoreGradBasicDet {
  using TILING_CLASS = typename FAGT::tiling_class;
  using SEQLEN_TYPE = typename FAGT::seqlen_type;
  using INPUT_TYPE = typename FAGT::input_type;
  static constexpr bool DROP_ENABLE = FAGT::drop_enable;
  static constexpr bool DETERMINISTIC_ENABLE = FAGT::deterministic_enable;

public:
  __aicore__ inline FlashAttentionScoreGradBasicDet(){};
  __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask,
                                 GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_in,
                                 GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                 GM_ADDR user, const TILING_CLASS *tilingData);

private:
  __aicore__ inline void CubeProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask,
                                     GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_in,
                                     GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk,
                                     GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData);

  __aicore__ inline void VectorProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask,
                                       GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                       GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                       GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR user,
                                       const TILING_CLASS *tilingData);

  __aicore__ inline void VecMainProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask,
                                        GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                        GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR user,
                                        const TILING_CLASS *tilingData);

  __aicore__ inline void VecDetMainProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask,
                                           GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                           GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                           GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR user,
                                           const TILING_CLASS *tilingData);
  int64_t dimB;
  int64_t dimD;
  uint32_t cubeCoreNum; // Cube核数量
  uint32_t cubeCoreIdx; // Cube核所对应的Index
  GM_ADDR mm1WorkSpaceAddr;
  GM_ADDR mm2WorkSpaceAddr;
  GM_ADDR dqWorkSpaceAddr;
  GM_ADDR dkWorkSpaceAddr;
  GM_ADDR dvWorkSpaceAddr;
  // 确定性
  GM_ADDR dqDetWorkSpaceAddr;
  GM_ADDR dkDetWorkSpaceAddr;
  GM_ADDR dvDetWorkSpaceAddr;
};

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradBasicDet<FAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData) {
  dimB = tilingData->b;
  dimD = tilingData->d;
  cubeCoreNum = tilingData->coreNum / 2;
  mm1WorkSpaceAddr = user + tilingData->mm1WorkspaceOffset;
  mm2WorkSpaceAddr = user + tilingData->mm2WorkspaceOffset;
  dqWorkSpaceAddr = user + tilingData->dqWorkSpaceOffset;
  dkWorkSpaceAddr = user + tilingData->dkWorkSpaceOffset;
  dvWorkSpaceAddr = user + tilingData->dvWorkSpaceOffset;
  // printf("==============idx=%d==============\n",GetBlockIdx());
  if ASCEND_IS_AIC {
    cubeCoreIdx = GetBlockIdx();
    if constexpr (DETERMINISTIC_ENABLE) {
      dqDetWorkSpaceAddr = user + tilingData->dqDetWorkspaceOffset + cubeCoreIdx * 512 * dimD * sizeof(float);
      dkDetWorkSpaceAddr = user + tilingData->dkDetWorkspaceOffset + cubeCoreIdx * 512 * dimD * sizeof(float);
      dvDetWorkSpaceAddr = user + tilingData->dvDetWorkspaceOffset + cubeCoreIdx * 512 * dimD * sizeof(float);
    }
    CubeProcess(query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_in, actual_seq_qlen,
                actual_seq_kvlen, dq, dk, dv, user, tilingData);
  }

  if ASCEND_IS_AIV {
    cubeCoreIdx = GetBlockIdx() / 2;
    VectorProcess(query, key, value, dy, drop_mask, atten_mask, softmax_max, softmax_sum, attention_in, actual_seq_qlen,
                  actual_seq_kvlen, dq, dk, dv, user, tilingData);
  }
  // printf("=================================\n",GetBlockIdx());
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradBasicDet<FAGT>::CubeProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
    GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR user, const TILING_CLASS *tilingData) {
  AscendC::TPipe pipeCube;
  FAG_DET::CubeOp<FAGT> cubeOp;
  FAG_DET::AddrComputeDet<FAGT> addrComputeDet;
  CubeAddrInfoDet cubeAddrInfo[2];
  uint32_t taskId = 0;

  cubeOp.Init(&pipeCube, tilingData, (__gm__ INPUT_TYPE *)query, (__gm__ INPUT_TYPE *)key, (__gm__ INPUT_TYPE *)dy,
              (__gm__ INPUT_TYPE *)value, (__gm__ float *)mm1WorkSpaceAddr, (__gm__ float *)mm2WorkSpaceAddr,
              (__gm__ float *)dqDetWorkSpaceAddr, (__gm__ float *)dkDetWorkSpaceAddr,
              (__gm__ float *)dvDetWorkSpaceAddr, (__gm__ float *)dqWorkSpaceAddr, (__gm__ float *)dkWorkSpaceAddr,
              (__gm__ float *)dvWorkSpaceAddr);
  addrComputeDet.Init(actual_seq_qlen, actual_seq_kvlen, cubeCoreNum, cubeCoreIdx, tilingData);

  while (true) {
    // init addr info
    cubeAddrInfo[taskId % 2].taskId = taskId;
    addrComputeDet.GetCubeAddrInfo(&cubeAddrInfo[taskId % 2]);

    // process mm12
    if (cubeAddrInfo[taskId % 2].blockLength > 0) {
      cubeOp.Cube12Process(cubeAddrInfo[taskId % 2]);
      AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
    }

    // process mm345
    if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
      AscendC::WaitEvent(VEC2CUBE);
      if constexpr (DETERMINISTIC_ENABLE) {
        if (taskId > 1) {
          AscendC::WaitEvent(VEC_TO_CUBE_DET);
        }
        cubeOp.Cube345Process(cubeAddrInfo[(taskId - 1) % 2]);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VECDET);
      } else {
        cubeOp.Cube345Process(cubeAddrInfo[(taskId - 1) % 2]);
      }
    }

    if (cubeAddrInfo[taskId % 2].blockLength == 0) {
      break;
    }
    taskId++;
  }
  AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradBasicDet<FAGT>::VectorProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData) {
  AscendC::TPipe pipePre;
  VectorInitOuput<TILING_CLASS, DETERMINISTIC_ENABLE> opPre;
  opPre.Init(&pipePre, dq, dk, dv, user, tilingData);
  opPre.Process();
  pipePre.Destroy();

  AscendC::TPipe pipeSoftmaxGrad;
  VectorSoftmaxGrad<INPUT_TYPE, SEQLEN_TYPE> opSoftmaxGrad;
  opSoftmaxGrad.Init(&pipeSoftmaxGrad, dy, attention_in, nullptr, actual_seq_qlen, user, dimB, tilingData);
  opSoftmaxGrad.Process();
  pipeSoftmaxGrad.Destroy();
  AscendC::SyncAll();

  if constexpr (DETERMINISTIC_ENABLE) {
    VecDetMainProcess(query, key, value, dy, drop_mask, atten_mask, softmax_max, softmax_sum, attention_in,
                      actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user, tilingData);
  } else {
    VecMainProcess(query, key, value, dy, drop_mask, atten_mask, softmax_max, softmax_sum, attention_in,
                   actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user, tilingData);
  }

  AscendC::TPipe pipePost;
  VectorPost<INPUT_TYPE, TILING_CLASS, DETERMINISTIC_ENABLE> vecPost;
  vecPost.Init(&pipePost, dq, dk, dv, user, tilingData);
  vecPost.Process();
  pipePost.Destroy();
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradBasicDet<FAGT>::VecMainProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData) {
  AscendC::TPipe pipeVec;
  VecOpDet<FAGT> op;
  FAG_DET::AddrComputeDet<FAGT> addrComputeDet;
  VecAddrInfoDet vecAddrInfo;
  uint32_t taskId = 0;

  op.Init(drop_mask, softmax_max, softmax_sum, atten_mask, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user,
          tilingData, &pipeVec);
  addrComputeDet.Init(actual_seq_qlen, actual_seq_kvlen, cubeCoreNum, cubeCoreIdx, tilingData);

  while (true) {
    addrComputeDet.GetVecAddrInfo(&vecAddrInfo);
    if (vecAddrInfo.blockLength > 0) {
      AscendC::WaitEvent(CUBE2VEC);
      vecAddrInfo.taskId = taskId;
      op.DetVector1(vecAddrInfo);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
    }
    if (vecAddrInfo.blockLength == 0) {
      break;
    }
    taskId++;
  }
  pipeVec.Destroy();
  AscendC::WaitEvent(CUBE2POST);
  AscendC::SyncAll();
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradBasicDet<FAGT>::VecDetMainProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR drop_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData) {
  AscendC::TPipe pipeVec;
  VecOpDet<FAGT> op;
  FAG_DET::AddrComputeDet<FAGT> addrComputeDet;
  VecAddrInfoDet vecAddrInfo[2];
  uint32_t taskId = 0;

  op.Init(drop_mask, softmax_max, softmax_sum, atten_mask, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user,
          tilingData, &pipeVec);
  addrComputeDet.Init(actual_seq_qlen, actual_seq_kvlen, cubeCoreNum, cubeCoreIdx, tilingData);

  while (true) {
    addrComputeDet.GetVecAddrInfo(&vecAddrInfo[taskId % 2]);

    if (taskId > 1) {
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      AscendC::SyncAll();
      AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(VEC_TO_CUBE_DET);
    }

    if (vecAddrInfo[taskId % 2].blockLength > 0) {
      AscendC::WaitEvent(CUBE2VEC);
      vecAddrInfo[taskId % 2].taskId = taskId;
      op.DetVector1(vecAddrInfo[taskId % 2]);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
    }

    if (taskId > 0) {
      AscendC::WaitEvent(CUBE2VECDET);
      AscendC::SyncAll();
      op.DetVector2(vecAddrInfo[(taskId - 1) % 2]);
    }

    if (vecAddrInfo[taskId % 2].blockLength == 0) {
      break;
    }

    taskId++;
  }

  if (cubeCoreIdx >= vecAddrInfo[taskId % 2].curCubeIdx) {
    if (taskId > 0) {
      AscendC::SyncAll();
    }
    AscendC::SyncAll();
    op.DetVector2(vecAddrInfo[taskId % 2]);
  }

  pipeVec.Destroy();
  AscendC::WaitEvent(CUBE2POST);
  AscendC::SyncAll();
}

#endif
