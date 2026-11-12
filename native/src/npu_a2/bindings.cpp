#include <torch/extension.h>
#include <torch/library.h>

#ifdef MOJO_NATIVE_WITH_MAGI_ATTENTION
#include "magi_attention/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_VARLEN_FA
#include "varlen_fa/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_ROPE_FWD
#include "rope_fwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_ROPE_BWD
#include "rope_bwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_SWIGLU_FWD
#include "swiglu_fwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_SWIGLU_BWD
#include "swiglu_bwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_MOE_SCATTER_FWD
#include "moe_scatter_fwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_MOE_SCATTER_BWD
#include "moe_scatter_bwd/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_GROUP_GEMM_FWD
#include "group_gemm_fwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GROUP_GEMM_BWD
#include "group_gemm_bwd/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_FA_FWD
#include "fa_fwd/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_FA_BWD
#include "fa_bwd/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_GDN_CUMSUM
#include "gdn_cumsum/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_UW
#include "gdn_compute_uw/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_S
#include "gdn_compute_s/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_O
#include "gdn_compute_o/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DV_LOCAL
#include "gdn_chunk_bwd_dv_local/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DHU
#include "gdn_chunk_bwd_dhu/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DQKWG
#include "gdn_chunk_bwd_dqkwg/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_PREPARE_WY_A_GRAD
#include "gdn_prepare_wy_a_grad/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_PREPARE_WY_DKBG
#include "gdn_prepare_wy_dkbg/host.h"
#endif
#ifdef MOJO_NATIVE_WITH_GDN_REVERSE_CUMSUM
#include "gdn_reverse_cumsum/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_MAGI_ATTENTION
TORCH_LIBRARY_IMPL(mojo_npu_native_a2, PrivateUse1, m) {
  m.impl("magi_attention_forward_native", TORCH_FN(mojo::native::a2::MagiForward));
  m.impl("magi_attention_backward_native", TORCH_FN(mojo::native::a2::MagiBackwardFull));
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_MAGI_ATTENTION
  m.def("magi_forward", &mojo::native::a2::MagiForward);
  m.def("magi_build_plan", &mojo::native::a2::MagiBuildPlan);
  m.def("magi_backward_full", &mojo::native::a2::MagiBackwardFull);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CUMSUM
  m.def("gdn_cumsum", &mojo::native::a2::GdnCumsum);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_UW
  m.def("gdn_compute_uw", &mojo::native::a2::GdnComputeUW);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_S
  m.def("gdn_compute_s", &mojo::native::a2::GdnComputeS);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_COMPUTE_O
  m.def("gdn_compute_o", &mojo::native::a2::GdnComputeO);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DV_LOCAL
  m.def("gdn_chunk_bwd_dv_local", &mojo::native::a2::GdnChunkBwdDvLocal);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DHU
  m.def("gdn_chunk_bwd_dhu", &mojo::native::a2::GdnChunkBwdDhu);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_CHUNK_BWD_DQKWG
  m.def("gdn_chunk_bwd_dqkwg", &mojo::native::a2::GdnChunkBwdDqkwg);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_PREPARE_WY_A_GRAD
  m.def("gdn_prepare_wy_a_grad", &mojo::native::a2::GdnPrepareWyAGrad);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_PREPARE_WY_DKBG
  m.def("gdn_prepare_wy_dkbg", &mojo::native::a2::GdnPrepareWyDkbg);
#endif
#ifdef MOJO_NATIVE_WITH_GDN_REVERSE_CUMSUM
  m.def("gdn_reverse_cumsum", &mojo::native::a2::GdnReverseCumsum);
#endif

#ifdef MOJO_NATIVE_WITH_FA_FWD
  m.def("flash_attention_forward", &mojo::native::a2::FlashAttentionForward);
#endif
#ifdef MOJO_NATIVE_WITH_FA_BWD
  m.def("flash_attention_backward", &mojo::native::a2::FlashAttentionBackward);
#endif
#ifdef MOJO_NATIVE_WITH_GROUP_GEMM_FWD
  m.def("group_gemm", &mojo::native::a2::GroupGemm);
#endif
#ifdef MOJO_NATIVE_WITH_GROUP_GEMM_BWD
  m.def("group_gemm_wgrad", &mojo::native::a2::GroupGemmWgrad);
#endif
#ifdef MOJO_NATIVE_WITH_ROPE_FWD
  m.def("rope_forward", &mojo::native::a2::RopeForward);
#endif
#ifdef MOJO_NATIVE_WITH_ROPE_BWD
  m.def("rope_backward", &mojo::native::a2::RopeBackward);
#endif
#ifdef MOJO_NATIVE_WITH_SWIGLU_FWD
  m.def("swiglu_forward", &mojo::native::a2::SwigluForward);
#endif
#ifdef MOJO_NATIVE_WITH_SWIGLU_BWD
  m.def("swiglu_backward", &mojo::native::a2::SwigluBackward);
#endif
#ifdef MOJO_NATIVE_WITH_MOE_SCATTER_FWD
  m.def("scatter_forward", &mojo::native::a2::ScatterForward);
#endif
#ifdef MOJO_NATIVE_WITH_MOE_SCATTER_BWD
  m.def("scatter_backward", &mojo::native::a2::ScatterBackward);
#endif
#ifdef MOJO_NATIVE_WITH_VARLEN_FA
  m.def("launch_varlen_fa", &mojo::native::LaunchVarlenFaCurrentStream,
        "Launch AscendC varlen FA on the current torch-npu stream");
#endif
}
