#define KFC_L1_RESERVER_SIZE 0
#include "kernel_operator.h"
#include "flash_attention_var_len_score.h"
using namespace AscendC;
using namespace mojo::kernels;
extern "C" __global__ __aicore__ void MOJO_FA_KERNEL(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR cuq,
                                                     GM_ADDR cuk, GM_ADDR out, GM_ADDR mx, GM_ADDR sm,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  SetSysWorkspaceForce(workspace);
  if ASCEND_IS_AIC {
    matmul::clearWorkspace(workspace);
  }
#endif
  TPipe pipe;
  set_mask_norm();
  FlashAttentionTrainingGeneralTilingData data;
  auto *dst = reinterpret_cast<uint32_t *>(&data);
  auto *src = reinterpret_cast<__gm__ uint32_t *>(tiling);
  for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i)
    dst[i] = src[i];
  using Op = FlashAttentionVarLenScore<ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_TND, MOJO_FA_MASK, false,
                                       DTYPE_INPUT, float, int32_t>;
  Op op;
  REGIST_MATMUL_OBJ(&pipe, workspace, op.bmm1, &data.bmm1TilingData, op.bmm1Nz, &data.bmm1TilingData, op.bmm2,
                    &data.bmm2TilingData);
#ifndef __DAV_C220_CUBE__
  op.UnpackInit(q, k, v, nullptr, mask, cuq, cuk, mx, sm, out, workspace + 16 * 1024 * 1024, &data, &pipe);
  op.Process();
#endif
}
