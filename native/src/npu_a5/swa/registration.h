#pragma once

#include <torch/library.h>

#ifdef MOJO_NATIVE_WITH_SWA
#include "host.h"
#endif
#ifdef MOJO_NATIVE_WITH_SWA_BACKWARD
#include "../swa_backward/host.h"
#endif

namespace mojo::native {

static inline void RegisterSwa(torch::Library& m) {
#ifdef MOJO_NATIVE_WITH_SWA
    m.def("launch_swa(Tensor query, Tensor key, Tensor value, Tensor cu_q_lens, Tensor cu_kv_lens, "
          "bool is_causal, int local_window_size, int global_window_size, float scale, "
          "int gqa_layout, int block_dim) -> Tensor", &LaunchSwaCurrentStream);
    m.def("launch_swa_training(Tensor query, Tensor key, Tensor value, Tensor cu_q_lens, Tensor cu_kv_lens, "
          "bool is_causal, int local_window_size, int global_window_size, float scale, "
          "int gqa_layout, bool output_f32, int block_dim) -> (Tensor, Tensor, Tensor)",
          &LaunchSwaTrainingCurrentStream);
#endif
#ifdef MOJO_NATIVE_WITH_SWA_BACKWARD
    m.def("launch_swa_backward(Tensor q, Tensor k, Tensor v, Tensor out, Tensor dout, Tensor softmax_lse, "
          "Tensor cu_q_lens, Tensor cu_kv_lens, int window_left, int gqa_layout, int global_window, "
          "float scale) -> (Tensor, Tensor, Tensor, Tensor)", &LaunchSwaBackwardCurrentStream);
#endif
}

}  // namespace mojo::native
