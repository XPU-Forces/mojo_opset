#include <torch/extension.h>

#ifdef MOJO_NATIVE_WITH_SWA
#include "swa/host.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_SWA
  m.def("launch_swa", &mojo::native::LaunchSwaCurrentStream,
        "Launch AscendC SWA on the current torch-npu stream");
#endif
}
