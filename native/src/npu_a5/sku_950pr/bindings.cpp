#include <torch/extension.h>

#ifdef MOJO_NATIVE_WITH_SWA
#include "../swa/host.h"
#endif

#ifdef MOJO_NATIVE_WITH_SWA_BACKWARD
#include "../swa_backward/host.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
