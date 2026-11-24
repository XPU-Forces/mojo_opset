#include "kernel_operator.h"
#include "fa_grad_tiling.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_basic_det.h"
extern "C" __global__ __aicore__ void MOJO_FA_KERNEL(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dy, GM_ADDR mask,
                                                     GM_ADDR mx, GM_ADDR sm, GM_ADDR out, GM_ADDR cuq, GM_ADDR cuk,
                                                     GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  SetSysWorkspaceForce(workspace);
#endif
  TPipe pipeIn;
  set_mask_norm();
  FlashAttentionTrainingGradTilingData data;
  auto *dst = reinterpret_cast<uint32_t *>(&data);
  auto *src = reinterpret_cast<__gm__ uint32_t *>(tiling);
  for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i)
    dst[i] = src[i];
  using Args = FAG_TYPE<FlashAttentionTrainingGradTilingData, DTYPE_INPUT, int32_t, false, true>;
  FlashAttentionScoreGradBasicDet<Args> op;
  pipeIn.Destroy();
  op.Process(q, k, v, dy, nullptr, mask, mx, sm, out, cuq, cuk, dq, dk, dv, workspace + 16 * 1024 * 1024, &data);
}
