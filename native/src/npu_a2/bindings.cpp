#include <torch/extension.h>

#ifdef MOJO_NATIVE_WITH_VARLEN_FA
#include "varlen_fa/host.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_VARLEN_FA
  m.def("launch_varlen_fa", &mojo::native::LaunchVarlenFaCurrentStream,
        "Launch AscendC varlen FA on the current torch-npu stream");
#endif
}
