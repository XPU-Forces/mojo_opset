# `layernorm_gated` — fused gated normalization on Ascend NPU

## 1. Scope

`layernorm_gated` is a forward-only inference primitive carried over from SGLang
v0.5.22 (`python/sglang/kernels/ops/attention/fla/layernorm_gated.py`, itself a
port of `fla-org/flash-linear-attention`). One kernel launch performs the whole
post-attention normalization: optional gate application, LayerNorm/RMSNorm
statistics (optionally per group), affine transform, optional gate multiply and
an optional per-token FP8 e4m3 quantisation epilogue.

Call sites are the GDN out-norm and QK-Norm paths of linear-attention and
`Mixer2RMSNormGated`-style blocks, i.e. tensors of `[tokens*heads, head_dim]`
with a gate that is a strided `[tokens, heads, head_dim]` view.

Three public entry points are provided, all dispatching through
`functions._dispatch.load_impl`:

| API | Purpose |
| --- | --- |
| `functions.layernorm_gated(x, weight, bias, eps, z=, out=, group_size=, norm_before_gate=, is_rms_norm=, activation=, quant_heads=)` | native API, returns `(result, mean, rstd)` |
| `functions.rms_norm_gated(x=, weight=, ...)` | SGLang `rms_norm_gated` signature, returns only the output |
| `functions._layer_norm_fwd(x, weight, bias, eps, z=, ...)` | SGLang kernel-level signature for a 2D `x` |
| `modules.LayerNormGated` / `modules.RMSNormGated` | parameterised layers |

## 2. Semantics

For each row `m` and group `g` (`ngroups = N // group_size`, `N = group_size`
when the caller does not group):

```
if not norm_before_gate: xg = xg * act(zg)
mean, rstd = statistics(xg)                      # RMS: mean=0, var=mean(x^2)
y = (xg - mean) * rstd * w + b                   # RMS: y = xg * rstd * w + b
if norm_before_gate:     y = y * act(zg)
act(z) = z*sigmoid(z) for "swish"/"silu", sigmoid(z) for "sigmoid"
```

* Statistics accumulate in FP32 and `mean`/`rstd` are written as FP32 with
  shape `[ngroups * M]`, group-major (`index = group * M + row`).
* `weight`/`bias` have shape `[N]` and are sliced per group by the kernel's
  second grid dimension.
* A 3D gate is addressed as `(row // H) * stride_z_token + (row % H) * stride_z_head`
  where `T * H == M` and `D == N`; only the head dim must be contiguous.
* `quant_heads > 0` replaces the output with `(q, s)`:
  `q[M/quant_heads, quant_heads*N]` in FP8 e4m3 and `s[M/quant_heads, 1]` in FP32.
  The epilogue rounds the normalised value to the output dtype first, takes
  `amax = max(max|y|, 1e-10)`, `scale = amax / 448.0`, and stores
  `clamp(y * (1.0 / scale), -448, 448)`. That reproduces the unfused
  `per_token_group_quant_fp8` bit for bit, including the rounding boundary.

## 3. Layout of the implementation

```
functions/layernorm_gated.py            validation, reshape, dispatch (no math)
modules/layernorm_gated.py              LayerNormGated / RMSNormGated parameters
kernels/_layernorm_gated.py             shared kernel + launch policy
kernels/npu_a2_triton/layernorm_gated.py   provider -> shared kernel
kernels/npu_a5_triton/layernorm_gated.py   provider -> shared kernel + 3D gate tile
kernels/torch/layernorm_gated.py        FP32 upcast reference (also the test oracle)
```

The provider modules are two lines each on purpose: the kernel body is shared so
that an A2/A5 tuning change cannot drift the numerics, and A5 differ only by the
`use_z3d_tile` launch flag. Registration: `OPS["layernorm_gated"]` in the three
providers, `layernorm_gated: triton` in `config/npu_a2.yaml` and `npu_a5.yaml`,
and one row in `docs/operator_support.md`.

## 4. Kernel design

### 4.1 Grid and work distribution

`grid = (min(vector_cores * (2 if quant else 1), ceil(M / rows_per_block)), ngroups)`.
Program `(pid, group)` walks `row_task_id in range(pid, num_row_tasks, num_programs)`,
so a fixed number of programs covers any `M` without a per-row launch. The upper
bound is the vector-core count (`get_num_cores("vector")`), because every stage of
this kernel is a vector operation; the quant path queues two blocks per core to
overlap the FP8 conversion and stores of one tile with the loads of the next.

### 4.2 Tile shape

* `BLOCK_N = min(65536 // element_size, next_power_of_2(group_size))`. The 64 KiB
  cap is the SGLang limit (a group must fit the on-chip buffer); `group_size >
  BLOCK_N` raises `RuntimeError` instead of silently chunking.
* `ROWS_PER_BLOCK` from `calc_rows_per_block`:

| condition | rows |
| --- | --- |
| batch-invariant mode | 4 (constant) |
| `N <= 128` and `M*N >= 32M` elements | 32 |
| `N <= 128` otherwise | 64 |
| `N <= 256` | 32 |
| `N <= 512` | 8 |
| `N <= 1024` | 4 |
| `N > 1024` | 2 |

  A 64-row tile wins on short decoding tensors; on streaming tensors the extra
  rows cost occupancy (measured on A2 at `32768x128`: AIV vector ratio 68.4% ->
  91.9%, MTE2 72.6% -> 97.5% when moving from 64 to 32 rows). The result is
  capped by `next_power_of_2(M)` so a 1-row decode does not build a 32-row block.
* `num_warps = min(max(BLOCK_N // 256, 1), 8)`; the quant path widens it to
  `max(min(num_warps * quant_heads // rows_per_block, 16), 1)` so that one
  `quant_heads`-row block (one token) spreads across more vector lanes. The
  `max(..., 1)` guard is required: for `N = 128`, `quant_heads = 8` the SGLang
  formula yields 0. Core occupancy is decided by the grid (vector-core count),
  as in the other normalization kernels in this repository, so `num_warps` is a
  scheduling hint inherited from the reference and revisited by the sweep
  (`LN_GATED_NUM_WARPS`).
* `EVEN_ROWS = M % ROWS_PER_BLOCK == 0` removes the row mask and the
  `tl.where` guards entirely; with `N == BLOCK_N` it also removes the column
  mask. The tail case (`M = 32771`, `N = 192`) keeps a masked last block, which
  is covered by a NaN-sentinel test for out-of-bounds stores.

### 4.3 Gate addressing

The default is a 2D gather: `z_row = (rows // H) * stride_z_token + (rows % H) * stride_z_head`,
then `z_row + col_offsets`. This is exact for any `T`, `H`, `D` and needs no
alignment, but on a strided view the DMA engine issues one scattered segment per
row. A5 providers additionally enable `Z_3D_TILE` when the gate is 3D, the row
tile is a multiple of `H` (so `row_start // H` is exact) and `M*N >= 32M`
elements: the block becomes `[rows_per_block // H, H, BLOCK_N]`, which is one
dense contiguous copy per token, and is `tl.reshape`d to `[rows_per_block, BLOCK_N]`.
The reshape is row-major, so tile row `t*H + h` is exactly the `(t, h)` gate row
of the same linear index. Measured on A5 at `524288x128` BF16 the scattered
addressing cost 526 us versus 315 us for the dense tile.

### 4.4 FP8 epilogue

`QUANT` is compiled only when `quant_heads > 0`; the epilogue runs after the gate
so `q` sees the same value the unfused path would quantise. Two details decide
bit-exactness and are therefore deliberate:

* the normalised value is rounded to the output dtype **before** `amax`
  (`y.to(Y.dtype.element_ty).to(tl.float32)`), matching the unfused order of
  write-then-quantise;
* `scale` uses `/ 448.0` and the reciprocal `1.0 / scale`, not a reciprocal
  multiply, so the device computes the same IEEE quotient as the torch reference.

`rows_per_block` is forced to `quant_heads` for this path so that one block is
one token; `s` is written at `S + row_task_id`, which is the token index. The
quantised output never touches `Y`, and `x` is passed as the dtype carrier, so no
`[M, N]` buffer is allocated for the fused case (memory report: one `[M, N]`
fp16/bf16 output saved, i.e. 2 bytes per element).

### 4.5 Batch invariance

`LN_GATED_BATCH_INVARIANT=1` pins `rows_per_block` to `MAX_ROWS_PER_BLOCK = 4`,
independent of `M` and of the core count, and the switch is read on every
dispatch (like SGLang's mode flag) rather than cached. A row reduction then
always runs with the same tile shape, so the same row has the same result in any
batch. `tests/functions/test_layernorm_gated.py::test_batch_invariant_rows`
checks it byte for byte across `M = 64` and `M = 128`.

## 5. Numerical policy

Inputs are FP16/BF16; every intermediate (`x`, gate, mean, variance, affine) is
FP32 on the vector unit and only the final store rounds to the output dtype.
Tolerances follow `tests/_checks.py`: FP16 `1e-3/1e-3`, BF16 `1e-2/1e-2`
relative/absolute, `mean`/`rstd` compared at `1e-3/1e-3` against the FP32
reference. Three boundaries are checked for exact equality (byte comparison):

1. 3D strided gate vs `.contiguous()` gate, both gate orders;
2. fused `(q, s)` vs the unfused quantiser applied to the same normalised output;
3. batch-invariant rows across batch sizes.

## 6. Performance method

Per the task definition: 5 warm-up launches, 30 profiled launches, and the metric
is the sum of every device kernel recorded in the window divided by 30
(`triton.backends.ascend.testing.do_bench_npu_profiler` with no
`target_kernel_name`). MBU = bytes / time / calibrated bandwidth, where bytes are
`(3*M*N + 2*N) * element_size` with a gate, `(2*M*N + 2*N) * element_size`
without, and `2*M*N*element_size + M*N + 4*M/quant_heads + 2*N*element_size` for
the fused quantiser. The denominator is calibrated in-process at the start of the
run with a 128M-element BF16 `copy_` probe (1 read + 1 write), 5 warm-ups and 30
profiled iterations. `tests/performance/functions/test_layernorm_gated.py`
implements this matrix and writes a JSON report plus the profiler CSV paths and
`npu-smi info`; raw profiler output stays under `tests/performance/results/layernorm_gated_raw/`.

Tuning knobs (`LN_GATED_ROWS_PER_BLOCK`, `LN_GATED_NUM_WARPS`) exist so the sweep
can be done on hardware without editing the kernel; the chosen values are baked
into `calc_rows_per_block` / the launch policy and documented with the measured
comparison table in the PR.

## 7. Known limits

* `group_size` must divide `N` and must fit 64 KiB, as in the reference kernel.
* The quantised path requires a single group (`ngroups == 1`) and
  `M % quant_heads == 0`; anything else raises `ValueError`.
* A 3D gate requires `T*H == M`, `D == N` and a contiguous last dim; `z` is never
  silently copied, so a bad view is an error rather than a performance cliff.
* Forward only: this is an inference primitive, there is no backward wrapper.
* `mean` and `rstd` keep the reference layout `[ngroups*M]` (group-major) rather
  than `[M, ngroups]`.

## 8. Reproduce

```bash
# accuracy + bitwise
pytest tests/functions/test_layernorm_gated.py tests/modules/test_layernorm_gated.py -q
# benchmark (writes JSON + raw profiler csv)
pytest tests/performance/functions/test_layernorm_gated.py -q --perf-output tests/performance/results/npu.json
```
