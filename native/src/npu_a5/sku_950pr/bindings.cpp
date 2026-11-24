#include <torch/extension.h>
#include <torch/library.h>

#include "../swa/registration.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Keep the public provider independent so it can coexist with the shared lib.
TORCH_LIBRARY(mojo_native_a5_950pr, m) {
    mojo::native::RegisterSwa(m);
}
