#include "../common/group_tiling.h"
#include "../common/group_gemm_training.h"
#include "../common/group_gemm_training_epilogue.h"
#include "../common/group_gemm_training_main_loop.h"
extern "C" __global__ __aicore__ void MOJO_GROUP_KERNEL(GM_ADDR input, GM_ADDR weight, GM_ADDR counts,
                                                        GM_ADDR output, GroupTiling tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
  if ASCEND_IS_AIV {
    return;
  }
  static_assert(sizeof(TCubeTiling) <= sizeof(tiling.matmul), "increase native TCubeTiling payload");
  TCubeTiling matrix = *reinterpret_cast<TCubeTiling *>(tiling.matmul);
  AscendC::TPipe pipe;
  using Config = mojo::native::a2::kernels::GroupGemmKernelConfig<false, MOJO_GROUP_TRANS_A,
                                                                  MOJO_GROUP_TRANS_B, false, false>;
  using Gemm = mojo::native::a2::kernels::GroupGemm<DTYPE_INPUT, DTYPE_INPUT, Config>;
  using Epi = mojo::native::a2::kernels::GroupGemmEpilogueNone;
  mojo::native::a2::kernels::GroupGemmMainLoop<Gemm, Config, Epi> op;
  op.init(input, weight, nullptr, counts, output, nullptr, &tiling.base, matrix, nullptr, pipe);
  op.process();
}
