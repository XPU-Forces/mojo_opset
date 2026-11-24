#pragma once
#include <tiling/tiling_api.h>
namespace optiling {
BEGIN_TILING_DATA_DEF(FlashAttentionTrainingGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, dqPostAbsorb); // 确定性场景下优化dq Post
TILING_DATA_FIELD_DEF(uint32_t, rsv);          // 保留字段
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, t1);
TILING_DATA_FIELD_DEF(int64_t, t2);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, qSize);
TILING_DATA_FIELD_DEF(int64_t, kvSize);
TILING_DATA_FIELD_DEF(int64_t, preTockens);
TILING_DATA_FIELD_DEF(int64_t, nextTockens);

TILING_DATA_FIELD_DEF(int64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dkWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dvWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, sfmgWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, mm1WorkspaceOffset);   // dyV
TILING_DATA_FIELD_DEF(int64_t, mm2WorkspaceOffset);   // QK
TILING_DATA_FIELD_DEF(int64_t, dqDetWorkspaceOffset); // 确定性
TILING_DATA_FIELD_DEF(int64_t, dkDetWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dvDetWorkspaceOffset);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
} // namespace optiling
