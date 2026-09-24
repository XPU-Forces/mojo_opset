# Native operators

The main package contains Python/Triton code. Separate provider lib packages
supply native binaries, built under `native/mojo_opset_lib/<provider>/`. C/C++ sources and build recipes live in `native/`.
Here `<provider>` is an architecture or architecture/SKU path, for example
`npu_a2` or `npu_a5/sku_950pr`. Public functions dispatch to Python kernel
wrappers, which internally load the matching pybind extension; users do not
import these extensions directly.

```text
native/
  CMakeLists.txt          # platform-independent build entry
  build.sh
  cmake/npu.cmake         # NPU toolchain and helpers
  src/                   # C/C++ sources only
    common/npu.h
    npu_a2/              # bindings.cpp + varlen_fa/
    npu_a5/sku_950pr/     # bindings.cpp + swa/
  third_party/catlass/    # pinned public submodule
  mojo_opset_lib/         # package layout and generated binaries
    npu_a2/
    npu_a5/sku_950pr/
```

## Adding an operator

### 1. Sources and compilation

For a new operator on an existing architecture, add
`native/src/<provider>/<operator>/`, following
[A2 Varlen FA](src/npu_a2/varlen_fa) or [A5 SWA](src/npu_a5/sku_950pr/swa).

- `kernel/`: device implementation and private headers.
- `host.h`: declaration of the C++ function exposed through pybind.
- `host.cpp`: validate dtype/device/layout/options, allocate outputs/workspace,
  obtain the input device's current stream, and launch the selected kernel.
- `CMakeLists.txt`: kernel targets, specializations and extension linkage.

Do not introduce a separate Python module entry for each operator. C++ names
and signatures may differ across architectures; Python wrappers must align
the public contract. Reuse `common/npu.h` only for genuinely shared NPU helpers.

An operator's `CMakeLists.txt` typically contains:

```cmake
mojo_native_kernel(example_kernel kernel/entry.cpp
    DEFINITIONS EXAMPLE_OPTION=1
)
mojo_native_operator(example example_kernel)
```

`mojo_native_kernel` creates an AscendC library; add `CATLASS` if needed.
`mojo_native_operator` links `host.cpp` and the listed kernel libraries into
`_native`, enabling `MOJO_NATIVE_WITH_EXAMPLE`. List multiple kernel targets
for specializations. The top level discovers operator recipes automatically.

### 2. Architecture binding

Update `native/src/<provider>/bindings.cpp`. For a hypothetical
`mojo::native::LaunchExample` declared in `example/host.h`:

```cpp
#ifdef MOJO_NATIVE_WITH_EXAMPLE
#include "example/host.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOJO_NATIVE_WITH_EXAMPLE
  m.def("launch_example", &mojo::native::LaunchExample);
#endif
  // Other operators in this architecture.
}
```

Add to the existing `PYBIND11_MODULE`; do not create a second module entry.
The macro is enabled automatically by the recipe. It selects operators within
one build selection, while CMake selects the architecture/SKU's binding/source files.

### 3. Python interface and tests

Add `mojo_opset/kernels/<arch>_native/[sku_<name>/]<operator>.py`; use
[the SWA wrapper](../mojo_opset/kernels/npu_a5_native/sku_950pr/swa.py) as an example:

1. Lazily import the matching artifact module (replace `/` with `.`), for example
   `mojo_opset_lib.npu_a5.sku_950pr._native`, and its launch function.
2. Register a leaf `torch.library.custom_op` with the correct mutation schema
   and a fake implementation describing output metadata.
3. Export `{op_id}_fwd` and, when needed, `{op_id}_bwd` with the functional
   layer's expected signatures. Leaf wrappers do not register backward.
4. Add `op_id: module_name` to the provider's `__init__.py` `OPS` mapping.
   Use an exact-target `OVERRIDES` entry for SKU-only code, as
   [A5 native](../mojo_opset/kernels/npu_a5_native/__init__.py) does for 950PR.
5. Update the appropriate `mojo_opset/config/*.yaml` entry if this should be
   the default implementation.

For an existing public operator, preserve its semantics without changing the
public API. For a new semantic operation, also add a `torch_reference`
implementation, a public function and exports, and a module only if useful.
Do not force incompatible native semantics into an existing interface:
`native_swa_infer` and `varlen_fa_infer` are temporary standalone migration APIs.

Inference operators currently run accuracy only; training operators also have
separate bitwise checks using the same case/input builders. Cover supported dtypes,
layouts/options, gradients where applicable, mutation, current-stream use and
fake/fullgraph behavior.

## Build and validate

Use the matching Python/PyTorch/vendor environment. For NPU this includes
torch-npu and CANN. `build.sh` loads the conventional CANN environment only
for NPU providers when `ASCEND_HOME_PATH` is unset.

```sh
python -m pip install -e .
bash build.sh npu_a5/sku_950pr

# For a selected operator instead of the complete provider:
bash native/build.sh npu_a5/sku_950pr swa        # selected operator

pytest -q tests/functions/test_attention.py --check accuracy -k native_swa_infer \
  --mojo-target npu.a5.950pr --mojo-implementation native
```

Root `build.sh <provider>` compiles one provider and installs its lib package
in editable mode. This makes its artifacts importable alongside an editable
main package, including from outside the checkout. It leaves `dist/` untouched.
C/C++ edits require a rebuild and a new Python process; Python/Triton edits do not.

For a selected-operator build using `native/build.sh` directly, install the lib
editable once:

```sh
MOJO_NATIVE_PROVIDER=npu_a2 python -m pip install --no-build-isolation --no-deps -e ./native
```

The lib can be installed editable before compilation; its binaries are required
only when the native operator runs.

A selected-operator build **replaces** that architecture/SKU's extension with the
selected subset; it does not append to an existing binary. Use `all` or omit
the operator before packaging the complete architecture. Avoid concurrent
builds writing the same output directory.

Intermediates live in `build/native/<provider>/<operator>/build.XXXXXX` and are
removed on success, retained on failure. Use `MOJO_NATIVE_KEEP_BUILD=1` to keep
them or `BUILD_JOBS` to control parallelism.

For A2 Varlen FA, initialize the public CATLASS submodule (pinned to the tested
v1.1.0 commit `3a28eb7bfc5fe4621f13d7aa5d39ccabd163e191`):

```sh
git submodule update --init native/third_party/catlass
CATLASS_ROOT="$PWD/native/third_party/catlass" SOC_VERSION=Ascend910B2 \
  bash native/build.sh npu_a2 varlen_fa
```

CATLASS uses the public [upstream](https://gitee.com/ascend/catlass), with no
automatic build-time fetch. An external `CATLASS_ROOT` is supported. Wheels
include its license, not its source tree.
Default SoCs are A2 `Ascend910B1` and A5 `Ascend950PR_9579`; override
`SOC_VERSION` for the intended supported device.

## Artifacts and packaging

```text
native/mojo_opset_lib/<provider>/
  _native.<Python ABI>.so     # binding + selected operators' host functions
  lib<kernel-target>.so      # linked device-kernel libraries
  _build_info.json           # successful build's provider, ABI and artifact list
```

The main `mojo-opset` wheel contains Python/Triton code and configuration. Its
`npu` extra depends on both `mojo-opset-lib-npu-a2` and
`mojo-opset-lib-npu-a5-950pr`, plus the NPU compiler. Users install
`mojo-opset[npu]`; dispatch lazily imports only the current device's extension.

Lib names follow the provider hierarchy: `npu_a2` becomes `npu-a2`, and
`npu_a5/sku_950pr` becomes `npu-a5-950pr`. The A2 lib currently targets 910B;
the architecture name alone does not establish compatibility with every SKU.
CPU architectures and Python ABIs use different wheel tags under the same lib name.

Each lib distribution owns one provider directory under the `mojo_opset_lib`
namespace. They have distinct names and disjoint files, so they can be installed
and upgraded independently. There is one Python extension per provider, not per
operator. Its runtime search path uses `$ORIGIN` for adjacent kernel libraries;
Torch/vendor runtime paths are also configured.

Build each distribution in a separate checkout with its matching environment:

```sh
# Main Python wheel: no native compilation or accelerator toolchain required.
bash build.sh --wheel

# A2 lib wheel: use the A2 build environment.
bash build.sh npu_a2 --wheel

# 950PR lib wheel: use the 950PR build environment.
bash build.sh npu_a5/sku_950pr --wheel
```

Each release command creates only its distribution's wheel in root `dist/`.
An existing `.whl` or `.tar.gz` causes an error rather than mixing releases.
There is no default `all` provider or cross-toolchain compilation. Without
`--wheel`, a provider is required and the command builds/installs an editable lib.
Operator CI follows this development path, without creating a release wheel.

To package an already compiled provider without rebuilding:

```sh
MOJO_NATIVE_PROVIDER=npu_a2 python -m pip wheel ./native \
  --no-build-isolation --no-deps -w dist
```

`native/setup.py` never runs CMake. It validates the selected provider's receipt,
complete operator build, Python ABI, CPU platform, libraries and license notices.
The main wheel is platform-independent; each lib wheel is specific to its
Python ABI and CPU platform. Each lib must publish a compatible wheel for every
Python/CPU platform supported by the `npu` extra; otherwise installation fails
even if that lib's device is not present. Pip does not select dependencies by
NPU model. Installed inactive providers are not imported or initialized.
Torch, torch-npu/CANN and device compatibility must still be verified separately;
matching wheel tags alone do not establish binary compatibility.

Root `packaging.json` owns the main version and Python dependency constraints.
`native/packaging.json` maps provider paths to independent lib versions; package
name suffixes are derived from those paths. Adjacent `packaging_*.json` overlays
add or override provider versions in filename order. Main and lib builds use the
same merged mapping.
Main packaging automatically adds exact lib dependencies to the corresponding
platform extra, using the same names and versions as lib packaging. The initial
`npu` extra pins both libs to `1.0.0rc1`. A main-only release changes only the root
version; updating a lib's version automatically updates its main extra dependency.
Libs do not depend back on the main distribution. Main builds need the lib metadata
file, but do not need native sources, binaries or an accelerator toolchain.
Root `packaging_*.json` overlays replace supplied metadata fields and individual
extra groups in filename order; they cannot override the main version. Lib names
combine the resolved main distribution name with the derived provider suffix.

Publish the required lib versions before publishing a main release that depends
on them. Prepare Python >=3.10, pip, setuptools >=64 and wheel for all builds;
native builds additionally need CMake and the matching Torch/torch-npu/CANN
toolchain. Initialize CATLASS before A2 builds. Scripts do not install vendor
dependencies or fetch submodules. Source distributions are unsupported; use an
editable repository checkout for source development.

For another platform, add `cmake/<platform>.cmake` and provider sources. Keep
vendor/toolchain setup there, not unconditionally in the top-level CMake.

## Current native implementations

| API | Target | Main restrictions |
| --- | --- | --- |
| `varlen_fa_infer` / `VarlenFAInfer` | `npu.a2` | BF16/FP16, contiguous TND, D=128, int32 offsets on the input device, nonempty sequences; causal requires Q≤KV. Supports AABB/ABAB and uneven GQA groups. |
| `native_swa_infer` | `npu.a5.950pr` | BF16, contiguous TND, D=128, causal, scale=1/√128, global window=4; int32 offsets on the input device, 0<Q≤KV, Q≤1024, KV≤9216. |

SWA's four compiled combinations are:

| Q heads | KV heads | `gqa_interleave` | Local window |
| --- | --- | --- | --- |
| 16 | 4 | False | 1023 |
| 16 | 4 | True | 255 |
| 8 | 1 | False | 1023 |
| 8 | 1 | True | 255 |

These are launch constraints, not exhaustive correctness guarantees. Original
SWA code also reproduces NaNs on some tiny-tail inputs and repeatability
failures on single-sequence Q=129/KV=257. Those cases were removed from the
routine test suite; the kernel issue remains unresolved.

Neither native inference API registers backward. Host launchers read cumulative
lengths on CPU; leaf custom ops permit compiler graph inclusion, but that does
not establish NPU execution-graph capture/replay support. A5 binaries
are isolated in `native/mojo_opset_lib/npu_a5/sku_950pr/`; future DT implementations use
`sku_950dt/` instead of replacing PR binaries.
