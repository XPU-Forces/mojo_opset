#include <torch/extension.h>

#ifdef MOJO_NATIVE_WITH_TOPK
#include "topk/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_LIGHTNING_INDEXER
#include "lightning_indexer/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_SWA
#include "swa/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_SWA_BACKWARD
#include "swa_backward/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_VWN
#include "vwn/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_DENSE_GEMM
#include "dense_gemm/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_MOE_DISPATCH_COMBINE
#include "moe_dispatch_combine/host.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_TOPK
  m.def("topk_forward", &mojo::native::a5::TopKForward);
#endif
#ifdef MOJO_NATIVE_WITH_MOE_DISPATCH_COMBINE
    mojo::native::ep::Bind(m);
#endif
#ifdef MOJO_NATIVE_WITH_LIGHTNING_INDEXER
  m.def("lightning_indexer_forward", &mojo::native::a5::LightningIndexerForward);
#endif
#ifdef MOJO_NATIVE_WITH_DENSE_GEMM
    mojo::native::RegisterDenseGemm(m);
#endif
#ifdef MOJO_NATIVE_WITH_SWA
  m.def("launch_swa", &mojo::native::LaunchSwaCurrentStream,
        "Launch AscendC SWA on the current torch-npu stream");
  m.def("launch_swa_training", &mojo::native::LaunchSwaTrainingCurrentStream,
        "Launch AscendC SWA and return output plus softmax LSE");
#endif
#ifdef MOJO_NATIVE_WITH_SWA_BACKWARD
  m.def("launch_swa_backward", &mojo::native::LaunchSwaBackwardCurrentStream,
        "Launch AscendC SWA backward on the current torch-npu stream");
#endif
#ifdef MOJO_NATIVE_WITH_VWN
    using namespace mojo::native::vwn;
    m.def("vwn_width_infer", &vwn_width_connection_infer_npu);
    m.def("vwn_width_fwd", &vwn_width_connection_train_npu);
    m.def("vwn_width_bwd_partials", &vwn_width_connection_backward_partials_npu);
    m.def("vwn_depth_fwd", &vwn_depth_connection_train_npu);
    m.def("vwn_depth_infer", &vwn_depth_connection_infer_npu);
    m.def("vwn_depth_bwd_partials", &vwn_depth_connection_backward_partials_npu);
#endif
}
