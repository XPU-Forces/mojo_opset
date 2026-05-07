# Compile Options For Triton Ascend NPU

This file is a curated English summary of compile options that matter when
working on Triton kernels for Ascend NPU in this repo. It is intentionally much
smaller than the full option dump and focuses on options that are useful during
development, debugging, or measured tuning.

Use this file for:

- understanding which knobs are worth trying first
- avoiding repeated experiments on options that did not help
- mapping local findings to BiShengIR/HIVM terminology

Use `../platform/environment_variable_reference.md` and the profiling/debug
guides for broader environment and tooling details.

## 1. High-value options

These are the options or behaviors that most directly affect Triton NPU kernel
work in this repo.

| Option / Concept | Recommended stance | Why it matters |
|---|---|---|
| `multibuffer=True` at launch | Usually enable | Often helps overlap load and compute on NPU |
| `tl.multibuffer(x, 2)` | Try early for looped loads | Repo-proven optimization for GEMM-like kernels |
| `num_stages=1` | Keep at `1` unless proven otherwise | GPU-style stage tuning does not transfer well here |
| target selection | Match the real SoC | Performance conclusions are target-dependent |
| deterministic computing | Keep enabled unless you are explicitly exploring a non-deterministic tradeoff | Safer default for validation |

## 2. Options that were explicitly tested and did not help for GEMM

These results are local observations, not universal truths for every kernel.
Still, they are strong defaults for new work in this repo.

| Option / Hint | Observed result |
|---|---|
| `num_stages > 1` | No measurable benefit on tested GEMM kernels |
| `enable_hivm_auto_cv_balance` | No measurable benefit on tested GEMM kernels |
| `unit_flag` style synchronization tuning | No measurable benefit on tested GEMM kernels |
| `tile_mix_vector_loop` / `tile_mix_cube_loop` | No measurable GEMM improvement in tested cases |

See also:

- [optimization-log.md](optimization-log.md)
- [ascend-910b-gemm.md](ascend-910b-gemm.md)

## 3. Options and hints that need caution

| Option / Hint | Caution |
|---|---|
| `tl.compile_hint(..., "dot_pad_only_k")` | Unsafe with large `BLOCK_K` in tested GEMM cases; can trigger `aicore timeout` |
| aggressive block/workspace overrides | Can hide the real scheduling issue instead of fixing it |
| undocumented or broad compiler toggles | Do not include them in the delivered path without measured evidence |

## 4. Practical tuning order

When a kernel underperforms, try tuning in this order before expanding into a
large compiler-flag search:

1. Fix grid shape and scheduling strategy.
2. Fix data layout and host-side preprocessing.
3. Revisit tile sizes under the UB budget.
4. Add or remove multibuffering.
5. Measure only a small set of compiler hints that are known to affect the
   kernel family.

This order is deliberate. In this repo, layout and schedule usually matter more
than broad compiler flag exploration.

## 5. Common target names

Examples seen in the local option inventory:

- `Ascend910B1`
- `Ascend910B2`
- `Ascend910B2C`
- `Ascend910B3`
- `Ascend910B4`
- `Ascend910C`-family values when available in the toolchain

Always record the exact target used for a benchmark or optimization claim.

## 6. Raw option inventories belong elsewhere

Do not keep expanding this file into a full compiler manual. If you need the
complete option universe, rely on the upstream/local tool documentation and keep
this file limited to:

- options that are demonstrably relevant to Triton kernel work
- options that were measured locally
- options that are known pitfalls
