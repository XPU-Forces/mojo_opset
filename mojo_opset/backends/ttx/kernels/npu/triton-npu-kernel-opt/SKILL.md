---
name: triton-npu-kernel-opt
description: >-
Build, port, optimize, integrate, and validate Triton kernels for Ascend NPU
  on the mojo_opset ttx backend. Use when adding a new NPU kernel, migrating a
  GPU Triton kernel to NPU, tuning an existing NPU kernel, wiring it into
  core/operator/backend dispatch, or preparing accuracy and performance
  acceptance for Ascend 910B/910C.
---

# Triton Ascend NPU Kernel Skill

Backend: `ttx` in `mojo_opset`

This skill is for end-to-end delivery of a new or updated Triton kernel on
Ascend NPU. It is not only an optimization note. It defines the required
workflow for:

- checking that the op contract already exists in `core`
- checking whether a `torch` or `torch_npu` implementation already exists
- studying existing optimized NPU kernels before writing new code
- implementing and registering the new kernel
- running accuracy, opcheck, and performance validation
- deciding whether the kernel is ready for acceptance

## Mandatory workflow

Follow this order. Do not skip steps.

### 1. Confirm the contract exists in `core`

Inspect the relevant API first:

- `mojo_opset/core/operators/`
- `mojo_opset/core/functions/`

You need a stable semantic contract before writing a backend kernel.

If there is no corresponding `core` operator/function, or there is no usable
`torch` reference implementation for semantics and correctness comparison, stop
and tell the user exactly what is missing. Ask the user to provide or approve
the missing core/torch implementation first.

Practical rule:

- Existing `core` contract + existing `torch` reference: proceed
- Existing `core` contract but no `torch` reference: proceed only if there is a
  `torch_npu` implementation or a trusted mathematical reference you can test
  against; otherwise ask the user to provide one
- No `core` contract: do not invent API semantics silently

### 2. Read the closest existing NPU kernels before coding

Before writing any new NPU kernel, read all relevant existing optimized NPU
kernels in this repository. Start with:

- [guides/repo/kernel_inventory.md](guides/repo/kernel_inventory.md)

At minimum, inspect:

- the same operator family if it exists
- kernels with similar memory access patterns
- kernels with similar reduction or tiling structure
- kernels with similar registration and test shape coverage

Do not treat the new kernel as greenfield work if the repo already solved a
similar scheduling, layout, or validation problem elsewhere.

### 3. Read only the needed Triton-Ascend references

Use the curated guides as the primary documentation layer. Prefer English only.

Start with:

- [guides/platform/index.md](guides/platform/index.md)

Load only the sections needed for the current task:

- programming model and migration rules for GPU-to-NPU ports
- memory/tensor descriptor APIs when layout is non-trivial
- debugging/profiling docs when compile or runtime issues appear
- examples only when a close pattern exists

Use only the distilled local references that have a clear role:

- [guides/tuning/best_practice.md](guides/tuning/best_practice.md) for repo-specific NPU coding patterns
- [guides/tuning/compile_option.md](guides/tuning/compile_option.md) for curated compiler/runtime knobs
- [guides/tuning/ascend-910b-gemm.md](guides/tuning/ascend-910b-gemm.md) for GEMM-only tuning data
- [guides/tuning/optimization-log.md](guides/tuning/optimization-log.md) for experiment history

### 4. Check existing backend implementations

Inspect the full dispatch path before coding:

- `mojo_opset/backends/ttx/operators/`
- `mojo_opset/backends/ttx/kernels/npu/`
- `mojo_opset/backends/torch_npu/operators/`
- `mojo_opset/backends/ttx/kernels/npu/__init__.py`

The questions to answer are:

- Is there already a `ttx` operator class for this core op?
- Is there already a native `torch_npu` backend path to use as a baseline?
- Is the new work a new kernel only, or also an operator wiring change?
- Is the kernel used through `torch.ops.ttx.*`, a `Mojo*` operator, or both?

### 5. Implement with NPU-first constraints

Default assumptions for Ascend NPU:

- launch overhead is expensive; prefer fewer logical programs
- persistent scheduling is often better for large tiled workloads
- host-side padding is usually safer than tail peeling
- UB is the hard resource limit; tile shapes must justify their footprint
- GPU heuristics like `num_stages > 1` do not automatically transfer

Start from proven patterns in existing NPU kernels before inventing a new
schedule.

For GEMM-like kernels, read only:

- [guides/tuning/ascend-910b-gemm.md](guides/tuning/ascend-910b-gemm.md)

For non-GEMM kernels, start from:

- [guides/tuning/best_practice.md](guides/tuning/best_practice.md)

### 6. Integrate the kernel completely

A kernel is not done when the Triton body compiles.

Finish all needed integration work:

- export the kernel or impl symbol from `backends/ttx/kernels/npu`
- wire the `ttx` operator/function backend if required
- preserve existing backend dispatch behavior
- keep the `torch` implementation as the correctness reference
- keep the `torch_npu` implementation as a native-performance baseline when it
  exists

Use:

- [guides/repo/integration_and_acceptance.md](guides/repo/integration_and_acceptance.md)

## Acceptance requirements

Do not present a new NPU kernel as complete unless all applicable checks pass.

### Accuracy

You must validate against a trusted reference:

- `torch` backend first when available
- otherwise `torch_npu` native operator or mathematically equivalent reference

Minimum expectations:

- representative production shapes
- edge shapes: very small, single-row, tail shapes, uneven groups, non-power-of-two
- expected dtypes
- optional arguments and alternate layouts
- bias/no-bias or fused/non-fused branches when applicable

Use existing `accuracy` tests where possible. Extend them when the current shape
set does not cover the new behavior.

### API correctness

If the kernel surfaces through `torch.ops.ttx.*`, add or update opcheck tests in
`mojo_opset/tests/test_ttx_graph/`.

### Performance

You must run performance validation, not only correctness.

Compare against the strongest available baseline:

- existing `ttx` kernel, if replacing one
- `torch_npu` native implementation, if it exists
- `torch` reference only as a correctness baseline, not as the final
  performance target

Record or summarize:

- tested shapes
- dtype/layout
- latency or throughput
- which baseline was used
- whether the new kernel wins, matches, or regresses

### Recommended acceptance bar

Treat the kernel as accepted only when all of the following are true:

- semantics are covered by `core`
- a trusted reference path exists
- accuracy tests pass for normal and boundary cases
- opcheck passes when `torch.ops.ttx.*` is involved
- performance tests pass on target NPU hardware
- no unexplained regression remains against the best available baseline

If performance is worse but the kernel is still needed for coverage, state that
explicitly and keep the gap quantified.

## What to read next

- Kernel inventory and example selection:
  [guides/repo/kernel_inventory.md](guides/repo/kernel_inventory.md)
- Repo integration and validation workflow:
  [guides/repo/integration_and_acceptance.md](guides/repo/integration_and_acceptance.md)
- Triton-Ascend doc map:
  [guides/platform/index.md](guides/platform/index.md)

## Maintaining this skill

When new NPU-specific findings appear:

- add reusable workflow changes to this `SKILL.md`
- add detailed kernel-family notes to the smallest fitting reference file
- append experiment outcomes to [guides/tuning/optimization-log.md](guides/tuning/optimization-log.md)
- keep repo and tuning guides focused and avoid duplicating large platform-guide content
