#pragma once
#include "kernel_operator.h"
using namespace AscendC;
struct FlashAttentionTrainingGradTilingData {
  uint32_t coreNum;
  uint32_t sparseMode;
  float scaleValue;
  float keepProb;
  uint32_t dqPostAbsorb;
  uint32_t rsv;
  int64_t b;
  int64_t t1;
  int64_t t2;
  int64_t n2;
  int64_t g;
  int64_t d;
  int64_t qSize;
  int64_t kvSize;
  int64_t preTockens;
  int64_t nextTockens;
  int64_t dqWorkSpaceOffset;
  int64_t dkWorkSpaceOffset;
  int64_t dvWorkSpaceOffset;
  int64_t sfmgWorkspaceOffset;
  int64_t mm1WorkspaceOffset;
  int64_t mm2WorkspaceOffset;
  int64_t dqDetWorkspaceOffset;
  int64_t dkDetWorkspaceOffset;
  int64_t dvDetWorkspaceOffset;
  SoftMaxTiling softmaxTilingData;
  SoftMaxTiling softmaxGradTilingData;
};
