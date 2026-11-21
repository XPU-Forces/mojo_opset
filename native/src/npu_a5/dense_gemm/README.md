# Dense GEMM on Ascend A5

This port embeds the dense BF16, MXFP8, MXFP8 x MXFP4, and MXFP4 device kernels from
[DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) at
`3a35326271b069b8b5bf1a754435acd6cc91575b`. It does not import the Python
`deep_gemm` package. Host bindings live in the architecture's existing `_native`
extension. Device kernels are compiled ahead of time into `libmojo_dense_gemm.so`.
There is no runtime compiler, source generation, disk cache or DeepJIT dependency.

Source comments record the DeepGEMM revision, original file hashes, provenance
and complete MIT license. DeepGEMM and fmt license notices
ship with the wheel.
Only dense entry points are exposed; shared upstream grouped definitions are not
part of this API.

## Source layout

- `host.cpp` validates inputs, prepares dense descriptors and registers the binding.
- `host/` contains dense configuration types, the fallback cost model, three
  epilogues, scale preparation and integer alignment helpers.
- `aot_dispatch.cpp` selects and launches an existing compiled instance.
- `kernel/` retains the device kernels and shared device types.
  Their grouped definitions remain upstream internals, not exposed host APIs.

The unused JIT candidate search, source/type-string helpers, grouped/batched host
APIs, generic shape-spec templates and legacy launch-option wrappers are removed.
The general fallback cost equations remain upstream-derived.

## AOT catalogue and dispatch

`generate.py` runs at CMake configure time and emits both ASC launchers and the
host registry from the same records. The catalogue contains 1,232 general GEMM instances
and 46 measured specializations
and 32 scale converters, covering four arithmetic paths, four operand layouts,
two output types, identity/alpha/bias epilogues and valid accumulation modes.
The common identity path additionally includes wider K tiles, non-power-of-two
M/N tiles, direct stores and dual-AIV FP4 decoding. `specializations.json` adds
bounded fixed-N/K/core-count variants, single-M scheduling, BF16 resident-A and
scale-pipeline variants. Each specialization references a stable general base tag;
this also permits comparisons with identical tile sizes and arithmetic.

General entries take M/N/K and the actual core count at runtime. Specialized
entries embed selected constants at build time. No compilation occurs at runtime. `aot_dispatch.cpp` selects:

1. An offline-measured configuration for an exact SoC name, Cube core count,
   arithmetic/layout/output/epilogue key and M/N/K, with tightly packed operands.
2. Otherwise, the original two or three general streaming configurations ranked
   by upstream `get_gemm_k_cost`. The fallback candidate set is unchanged, so
   expanding the profiling catalogue does not change unmeasured-shape selection.

Fixed N/K/core count, single-M/residency bounds, and direct-store alignment are
checked on every dispatch, including forced profiling configurations. Build-time
tuning validation also checks specialization shape/core restrictions. The generated registry uses stable configuration
hashes; CMake rejects tuning entries referring to absent instances or duplicate
keys. Candidate ranges are indexed by metadata rather than scanning the full
catalogue. Tuning data is compiled into the host extension, with no runtime file
I/O. Strided inputs, other SoCs/core counts and unmeasured epilogues fall back.

`MOJO_GEMM_PRINT_CONFIG=1` prints the key, shape, tag, selection source and config.
`MOJO_GEMM_DISABLE_TUNING=1` selects the original cost-model fallback.
`MOJO_GEMM_FORCE_CONFIG=<tag>` is a developer-only offline profiling override;
it rejects unavailable or unsafe configurations and must not be set in serving.
Legacy `compiled_dims` values are validated but do not specialize AOT kernels;
the default is the empty string. `DG_JIT_*` variables are no longer used.

### Offline tuning

`tune.py` checks each legal candidate against the CPU reference, captures a graph
of 20 calls, checks replay correctness, and measures seven randomized replay
rounds. Its export includes only configurations at least 3% faster than the
fallback in that run. Repeat selected configurations in a fresh process before
accepting results; a small measured difference is not a performance guarantee.
The checked-in table has 67 entries measured on Ascend950PR_9579 / 28 Cube cores.
The specialization audit and paired measurements are described in
[`SPECIALIZATION_VALIDATION.md`](SPECIALIZATION_VALIDATION.md).

```sh
# Use the same source revision as the loaded native library.
python native/src/npu_a5/dense_gemm/generate.py --output build/dense_gemm_catalog
python native/src/npu_a5/dense_gemm/tune.py \
  --catalog build/dense_gemm_catalog/catalog.json \
  --output build/gemm_profiles.json --export build/gemm_tuning.json
# Also exercise all supported layouts and both output types:
python native/src/npu_a5/dense_gemm/tune.py \
  --catalog build/dense_gemm_catalog/catalog.json --layouts \
  --output build/gemm_layout_profiles.json --export build/gemm_layout_tuning.json
```

Custom workloads use `--cases cases.json`, a list such as
`[{"op":"mxfp8_gemm","shape":[128,1024,4096],"mode":"forward","output":"bf16"}]`.
`shape` always means the resulting GEMM's `[M,N,K]`, including `mode="wgrad"`.
MXFP8 uses the existing `MXFP8Tensor` API: `forward` prepares row-quantized
`[M,K]` and `[N,K]` operands; `wgrad` prepares column-quantized `[K,M]` and
`[K,N]` operands. Quantization and device transfers happen before timing.
The default workload includes MXFP8 forward; `--layouts` covers both MXFP8 modes
and all four transpose combinations for the other operators. Other operators
continue to accept `trans_input` and `trans_weight` in custom cases.
Profile rows include logical `shape` and actual `kernel_shape`; MXFP8 candidate
selection and exports use the padded kernel shape to match runtime dispatch.
Exports deduplicate shapes that map to the same dispatch key and padded geometry.
Outputs are `bf16` or `fp32`. Merge accepted entries into `tuning.json` and rebuild;
keep entries unique by SoC/core count/key/shape. A retained CMake build only needs
to regenerate/relink host dispatch when the tuning table alone changes, because
the generator preserves unchanged device source timestamps. This workflow never
adds tuning latency to a serving request.

To compare retained specializations against their general base entries while
keeping the current default dispatch as another reference:

```sh
python native/src/npu_a5/dense_gemm/tune.py \
  --catalog build/dense_gemm_catalog/catalog.json --cases cases.json \
  --specializations-only --compare-default --rounds 9 --repeats 30 \
  --output build/gemm_specializations.json
```

Changing `specializations.json` requires rebuilding both the device library and
host registry. Changing only `tuning.json` does not recompile unchanged device
sources. Keep the two artifacts from the same build.

## Build

Requirements: CANN 9.20 with Bisheng and ld.lld, torch-npu, a C++20 compiler,
the bundled fmt 9.1.0 headers (no system fmt package required). fmt is used in
header-only mode and does not add a runtime shared-library dependency.

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash native/build.sh npu_a5 dense_gemm
MOJO_NATIVE_PROVIDER=npu_a5 python -m pip install --no-build-isolation --no-deps -e native
```

A selected build replaces the architecture extension. Build `npu_a5` without
an operator argument before packaging the complete library. Binary wheels carry
the compiled GEMM library, not device source headers. Deployment requires the
matching CANN runtime and driver; GEMM does not invoke Bisheng or ld.lld.

## APIs

Use `mojo_opset.functions.gemm`, `mxfp8_gemm`, `mxfp8_dequant_gemm`, or
`mxfp4_gemm` with `implementation="native"`. A5 `mxfp8_gemm` defaults to
`torch_npu`; selecting `native` explicitly uses these AOT kernels.

`mxfp8_gemm` takes two `MXFP8Tensor` operands, `mode="forward"` or `"wgrad"`,
and `output_dtype` (FP32 by default). Forward computes `input @ weight.T`;
wgrad computes `input.T @ weight`. The adapter handles paired scales, padding,
and cropping to the logical output shape. It does not accept separate scale
arguments, transpose flags, C, alpha, or recipe.

The following raw-tensor options apply to `gemm`, `mxfp8_dequant_gemm`, and
`mxfp4_gemm`:

- Logical operands default to A[M,K] and B[K,N]. `trans_input` and `trans_weight`
  select the four transpose combinations. One matrix axis must be contiguous;
  the other may have padding. FP4 packs two E2M1 values per int8 byte along the
  contiguous axis, low nibble first; that logical dimension is twice its storage
  dimension. Ambiguous FP4 strides are rejected, and logical K must be even.
- Low-precision scales always describe logical [M,K] / [N,K], independently of
  transpose flags. `recipe=(gm,gn,32)` defines scale block sizes; gm/gn must be powers of two in [1,128]. FP32 scales
  must be finite positive powers of two representable as normal UE8M0 values.
  Alternatively, int16 packs two consecutive K exponents (low byte first).
  Prepacked MN-major scales with gm=gn=1 avoid the scale conversion kernel.
- Output is a fresh [M,N] BF16 or FP32 tensor. C may be [M,N] for accumulation
  or contiguous [1,N] for row bias. C and other inputs are never mutated.
  Full C is copied into the output before upstream atomic accumulation.
  `alpha` scales the product and cannot be combined with row bias.
- These entry points are forward-only; autograd inputs are rejected while grad
  mode is enabled. They have fake implementations and support Dynamo fullgraph
  capture after wrapper warmup. NPU execution-graph replay is covered by the offline profiling checks.
- Device and stream follow the input tensor and current torch-npu stream. The
  host queries the actual device core count for both launch and scheduling.

```python
import torch
from mojo_opset import functions as F
out = F.gemm(a, b, output_dtype=torch.float32)
# x is BF16 [M,K], w is BF16 [N,K]. Quantize outside the GEMM timing region.
qx = F.mxfp8_quantize(x, axis=1)
qw = F.mxfp8_quantize(w, axis=1)
out = F.mxfp8_gemm(qx, qw, mode="forward", implementation="native")
out = F.mxfp8_dequant_gemm(a8, packed_b4, sa, sb, trans_weight=True)
```

## Port adaptations

- Host formatting uses fmt instead of std::format so GCC 12 can build the
  extension. Cache hints are printed as enum values; magic_enum is not required.
- `common.h` includes type_traits directly instead of unused utility; this
  avoids Bisheng's incompatibility with GCC 12's C++20 std::pair declarations.
- The scale conversion kernel explicitly includes CANN's device assertion header.
- ASC AOT uses the upstream VF loop-unroll and argument-preload compiler flags.
- The shared pointer constructor supports host and device compilation; scheduler
  and scale converter receive runtime core counts for different A5 SKUs.
- A relative common-header include was adjusted to the vendored include layout.
- Host bindings validate devices, shapes and dtypes, handle functional C,
  normalize unpacked singleton strides, and guard the input NPU device.


## Validation

```sh
pytest -q tests/framework/test_dense_gemm_contract.py
pytest -q tests/functions/test_dense_gemm.py --check accuracy \
  --mojo-target npu.a5.950pr --mojo-implementation native
```

The suite compares against independent CPU decoding/FP32 multiplication and
covers NN/NT/TN/TT, BF16/FP32 outputs, tails, strided inputs, FP32 and packed
scales, scale recipes, bias/C/alpha, input immutability, current streams, and
fullgraph capture. Validate another A5 SKU on its own hardware before claiming
numerical or performance parity there.

### AOT host cleanup validation (950PR, 2026-10-27)

The dense-only host refactor preserves all 164 generated files (161 ASC sources,
registry, catalogue and tuning table) byte for byte. An old/new C++ comparison
checks all 1,232 configurations over 600 shapes, two core counts and two stride
patterns: 2,956,800 cost estimates and direct-store predicates match exactly.
The clean A5 build passes 171 NPU correctness tests; all 492 framework checks
pass, including public export and native packaging. Validation artifacts are in
`build/dense_gemm_cleanup_validation/` on the development checkout.

## Initial AOT measurements (950PR, 2026-10-22)

CANN 9.20, 28 Cube cores. Compared with the JIT wheel from commit `6c2530b1`.
BF16 output; A[M,K], B[N,K], trans_weight=True; low-precision inputs use
prepacked scales. After warmup, each NPU graph replays 20 calls; results below
are the median of three event timings divided by 20. Correctness is checked
for both ordinary execution and graph replay. These are pilot measurements,
not a full performance qualification. Lower is better.

| API | M,N,K | AOT us | warm JIT us | AOT/JIT |
|---|---|---:|---:|---:|
| gemm | 32,4096,4096 | 18.23 | 25.30 | 0.72 |
| gemm | 256,512,1024 | 6.78 | 5.13 | 1.32 |
| gemm | 1024,1024,1024 | 13.99 | 9.43 | 1.48 |
| fp8_gemm | 32,4096,4096 | 13.87 | 13.72 | 1.01 |
| fp8_gemm | 256,512,1024 | 4.39 | 3.66 | 1.20 |
| fp8_gemm | 1024,1024,1024 | 8.41 | 6.27 | 1.34 |
| mxfp8_dequant_gemm | 32,4096,4096 | 14.01 | 9.67 | 1.45 |
| mxfp8_dequant_gemm | 256,512,1024 | 5.25 | 4.93 | 1.07 |
| mxfp8_dequant_gemm | 1024,1024,1024 | 8.79 | 9.68 | 0.91 |
| mxfp4_gemm | 32,4096,4096 | 9.76 | 8.47 | 1.15 |
| mxfp4_gemm | 256,512,1024 | 4.08 | 3.10 | 1.32 |
| mxfp4_gemm | 1024,1024,1024 | 6.51 | 4.95 | 1.31 |

Reproduce with `python native/src/npu_a5/dense_gemm/benchmark.py --output results.json`.
The current benchmark targets AOT wheels. The historical JIT measurements above
used the earlier benchmark revision.

Validation: 161 native correctness cases (97 original + 64 AOT coverage),
102 framework/packaging checks, complete A5 build and wheel packaging, and a fresh installed-wheel process
with an unusable compiler PATH/toolkit discovery path and no device headers.
All four arithmetic paths pass in that process.

## Tuned AOT measurements (950PR, 2026-10-23)

Compared with the initial AOT commit `a4678a0d`, using the same 950PR, CANN 9.20,
BF16 output, NT layout and prepacked scales. These fresh-process measurements use
15 NPU-event rounds, each timing a graph of 20 calls. Lower is better.

| API | M,N,K | Initial AOT us | Tuned AOT us |
|---|---|---:|---:|
| gemm | 32,4096,4096 | 17.77 | 11.89 |
| gemm | 256,512,1024 | 6.46 | 4.90 |
| gemm | 1024,1024,1024 | 13.60 | 9.23 |
| fp8_gemm | 32,4096,4096 | 13.76 | 8.83 |
| fp8_gemm | 256,512,1024 | 4.29 | 3.44 |
| fp8_gemm | 1024,1024,1024 | 8.27 | 6.09 |
| mxfp8_dequant_gemm | 32,4096,4096 | 13.73 | 9.31 |
| mxfp8_dequant_gemm | 256,512,1024 | 4.95 | 4.67 |
| mxfp8_dequant_gemm | 1024,1024,1024 | 8.65 | 7.50 |
| mxfp4_gemm | 32,4096,4096 | 9.55 | 5.83 |
| mxfp4_gemm | 256,512,1024 | 3.88 | 2.88 |
| mxfp4_gemm | 1024,1024,1024 | 6.28 | 4.57 |

Across the 40 profiled NT/BF16-output workloads (10 shapes per arithmetic path),
geometric-mean latency decreased by 17.1%; this is a workload-specific result,
not a claim for all GEMM shapes. Sixteen held-out workloads retain the original
fallback choices; repeated latency ratios were 0.9979–1.0034. The table also
contains selected results from 32 cases covering all layouts and both outputs.
Other SKUs have no measured entries and use the original fallback.

Validation: 171 NPU tests (163 regression/dispatch checks plus eight tests that
force all direct-store/dual-decode candidates on K tails and strided operands),
107 framework/build-contract tests, a complete A5 build, and eight installed-wheel
smoke cases with compiler executables/toolkit discovery paths unavailable.
Profiling checks each candidate and its captured replay against a CPU reference.
Full raw profiles, repeat measurements, logs and the wheel are retained in
`build/dense_gemm_aot_optimization_validation/` on the development checkout.
