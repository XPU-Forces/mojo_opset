#include <torch/extension.h>
#include <torch/library.h>

#ifdef MOJO_NATIVE_WITH_TOPK
#include "topk/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_LIGHTNING_INDEXER
#include "lightning_indexer/host.h"
#endif

#include "swa/registration.h"

#ifdef MOJO_NATIVE_WITH_VWN
#include "vwn/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_DENSE_GEMM
#include "dense_gemm/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_MOE_DISPATCH_COMBINE
#include "moe_dispatch_combine/host.h"
#endif

// The module loads registrations; C++ dispatcher operators enter through torch.ops.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_MOE_DISPATCH_COMBINE
    mojo::native::ep::Bind(m);
#endif
#ifdef MOJO_NATIVE_WITH_DENSE_GEMM
    mojo::native::RegisterDenseGemm(m);
#endif
}

TORCH_LIBRARY(mojo_native_a5, m) {
#ifdef MOJO_NATIVE_WITH_TOPK
    m.def("topk_forward(Tensor input, int k, Tensor? end, bool sorted_index, Tensor? output_idx, "
          "Tensor? output_idx_offset, int idx_fill, float value_fill, bool return_value, "
          "bool abort_when_nan_found) -> (Tensor?, Tensor)",
          &mojo::native::a5::TopKForward);
#endif
#ifdef MOJO_NATIVE_WITH_LIGHTNING_INDEXER
    m.def("lightning_indexer_forward(Tensor query, Tensor query_scale, Tensor key, Tensor key_scale) -> Tensor",
          &mojo::native::a5::LightningIndexerForward);
#endif
    mojo::native::RegisterSwa(m);
#ifdef MOJO_NATIVE_WITH_VWN
    using namespace mojo::native::vwn;
    m.def("vwn_width_infer(Tensor h, Tensor weight, Tensor fn, Tensor dynamic, Tensor statics, "
          "float eps, float? factor, float min_value, float max_value) -> (Tensor, Tensor)",
          &vwn_width_connection_infer_npu);
    m.def("vwn_width_fwd(Tensor h, Tensor weight, Tensor fn, Tensor dynamic, Tensor statics, "
          "float eps, float? factor, float min_value, float max_value) -> (Tensor, Tensor, Tensor, Tensor)",
          &vwn_width_connection_train_npu);
    m.def("vwn_width_bwd_partials(Tensor h, Tensor weight, Tensor fn, Tensor dynamic, Tensor statics, "
          "Tensor acc, Tensor invvar, Tensor u, Tensor v, Tensor? residual, float? factor, "
          "float min_value, float max_value, int workspace_blocks, ScalarType? grad_h_dtype=None, "
          "Tensor? sr_seed=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
          &vwn_width_connection_backward_partials_npu);
    m.def("vwn_depth_fwd(Tensor h, Tensor ho, Tensor gates, Tensor? weight, float eps, "
          "bool output_fp32=False, Tensor? sr_seed=None) -> (Tensor, Tensor)",
          &vwn_depth_connection_train_npu);
    m.def("vwn_depth_infer(Tensor h, Tensor ho, Tensor gates, Tensor? weight, float eps) -> Tensor",
          &vwn_depth_connection_infer_npu);
    m.def("vwn_depth_bwd_partials(Tensor h, Tensor ho, Tensor gates, Tensor dy, Tensor? weight, "
          "Tensor? invvar, Tensor? clip_mask, float clip, ScalarType? grad_h_dtype, int workspace_blocks) "
          "-> (Tensor, Tensor, Tensor, Tensor)",
          &vwn_depth_connection_backward_partials_npu);
#endif
}
