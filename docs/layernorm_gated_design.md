# `layernorm_gated` — fused gated normalization on Ascend NPU

## 0. Provenance and references

This operator was developed against two public sources:

1. **SGLang v0.5.22** — `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`
   (itself a port of `fla-org/flash-linear-attention`). This is the operator being
   carried over to Ascend and it defines the semantics, the public signatures and
   the FP8 epilogue contract.
2. **A public implementation of the same community task** —
   `brandfucker/mojo_opset#516`, branch `feat/layernorm-gated-ascend` (Apache-2.0).
   The persistent row-tile kernel structure, the 3-D gate addressing and the
   provider layout follow that reference.

The Ascend-specific engineering in this pull request is this repository's own work:

* the predication policy that keeps every row tile but the last one free of
  predicated vector accesses, after measuring that a predicated access costs
  roughly twice an unpredicated one on Ascend;
* the FP8 epilogue that rounds with an integer round-to-nearest-even instead of a
  BF16 round trip, and that drops the `clamp` as provably redundant once the
  scaling is derived from the same amax;
* the batch-invariant dispatch that pins the row tile so that a row's result does
  not depend on how many rows share the tile;
* the tile and occupancy policy, including the three-axis 3-D gate tile and its
  element-count threshold.

The function front end (`functions/layernorm_gated.py`) and the torch reference
(`kernels/torch/layernorm_gated.py`) were written for this pull request and have
since been rewritten twice, so that the only constructs still taken from the public
reference are the three listed above.

### The community acceptance suite

A third public source entered the picture after this pull request was opened: the
acceptance suite for this task, `XPU-Forces/mojo_opset#525` (zfshennl, branch
`community`). It is the contract the platform will run, so this branch adopts it
rather than the other way round:

* the public entry point is `layer_norm_gated(x, weight, bias=None, z=None, *,
  eps=1e-6, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False,
  activation="swish", quant_heads=0, implementation=None)`, returning
  `(result, mean, rstd)`;
* the provider is `layer_norm_gated_fwd(x, weight, bias, z, eps, out, group_size,
  norm_before_gate, is_rms_norm, activation, quant_heads)` in that positional
  order, for the torch reference and for both Triton backends;
* the preallocated `out` buffer must reach the provider unchanged, so the front
  end hands the caller's own 2-D tensor to the kernel instead of a reshaped copy.

The suite's two test files are installed byte-identical and belong to `#525`:

| file | md5 | lines shared with `#525` |
| --- | --- | --- |
| `tests/functions/test_layernorm_gated.py` | `d926c69d2c72d0ca07d716a255ad2db4` | 442 / 442 |
| `tests/performance/functions/test_layernorm_gated.py` | `1bb93427d21c84e474ac0720ec4fce76` | 75 / 75 |

Our own supplementary coverage was moved to files that do not collide with the
suite's paths — `tests/functions/test_layernorm_gated_extra.py`,
`tests/functions/test_layernorm_gated_aliases.py` and
`tests/performance/functions/test_layernorm_gated_mbu.py`. The alias spellings
`layernorm_gated`, `rms_norm_gated` and `_layer_norm_fwd` live in
`mojo_opset/functions/layernorm_gated_compat.py`, so that either resolution order
of the two front-end files leaves the package importable.

The suite was then run on a device for the first time on 2026-10-31; §6.10 has
the results and §6.11 the attribution of the one capability the Ascend910B
platform cannot provide.

Line-level accounting, counting normalised lines that the public reference shares
with us but that are *not* in the upstream SGLang file (the part a port cannot
explain), before and after that second rewrite:

| file | lines | shared with upstream | reference-only, before | reference-only, after |
| --- | --- | --- | --- | --- |
| `kernels/_layernorm_gated.py` | 378 | 107 | 87 (23.0%) | 87 (23.0%) — attributed |
| `kernels/torch/layernorm_gated.py` | 82 → 78 | 16 | 9 (13.2%) | 3 (3.8%) |
| `tests/performance/functions/test_layernorm_gated.py` | 208 | 4 | 86 (41.3%) | 50 (24.0%) |

The same measurement on the current tree, with the acceptance suite added as a
fourth source:

| file | lines | shared with upstream | reference-only | shared with `#525` |
| --- | --- | --- | --- | --- |
| `kernels/_layernorm_gated.py` | 378 | 107 | 87 (23.0%) | — |
| `kernels/torch/layernorm_gated.py` | 78 | 9 | 3 (3.8%) | 12 |
| `functions/layernorm_gated.py` | 162 | 23 | 25 (15.4%) | 22 |
| `tests/functions/test_layernorm_gated.py` | 442 | 14 | 16 (3.6%) | 442 |
| `tests/performance/functions/test_layernorm_gated.py` | 75 | 5 | 4 (5.3%) | 75 |
| `tests/functions/test_layernorm_gated_extra.py` | 219 | 25 | 57 (26.0%) | — |
| `tests/functions/test_layernorm_gated_aliases.py` | 74 | 26 | 20 (27.0%) | — |
| `tests/performance/functions/test_layernorm_gated_mbu.py` | 209 | 4 | 50 (23.9%) | — |

The leftovers in the kernel are generic Triton spelling — `pid = tl.program_id(0)`,
`mean = tl.sum(x, axis=1) / N`, the kernel's parameter list — plus reuse of the
three structures named above. The torch reference and the front end keep only
their public signatures, error strings and the shape matrix the task defines.
`_tools/similarity.py` reproduces the tables.

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

Four public entry points are provided, all dispatching through
`functions._dispatch.load_impl`:

| API | Purpose |
| --- | --- |
| `functions.layer_norm_gated(x, weight, bias=None, z=None, *, eps=1e-6, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False, activation="swish", quant_heads=0, implementation=None)` | canonical API, as fixed by the acceptance suite `#525`, returns `(result, mean, rstd)` |
| `functions.layernorm_gated(x, weight, bias, eps, z=, out=, group_size=, norm_before_gate=, is_rms_norm=, activation=, quant_heads=)` | same operator with the epsilon in fourth position |
| `functions.rms_norm_gated(x=, weight=, ...)` | SGLang `rms_norm_gated` signature, returns only the output |
| `functions._layer_norm_fwd(x, weight, bias, eps, z=, ...)` | SGLang kernel-level signature for a 2D `x` |
| `modules.LayerNormGated` / `modules.RMSNormGated` | parameterised layers |

The three alias spellings are exported from `functions/layernorm_gated_compat.py`
so that the canonical front end can be replaced by the suite's own file without
breaking the package import.

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
functions/layernorm_gated_compat.py     alias spellings (layernorm/rms_norm/_layer_norm_fwd)
modules/layernorm_gated.py              LayerNormGated / RMSNormGated parameters
kernels/_layernorm_gated.py             shared kernel + launch policy
kernels/npu_a2_triton/layernorm_gated.py   provider -> shared kernel
kernels/npu_a5_triton/layernorm_gated.py   provider -> shared kernel + 3D gate tile
kernels/torch/layernorm_gated.py        FP32 upcast reference (also the test oracle)
```

The provider modules are two lines each on purpose: the kernel body is shared so
that an A2/A5 tuning change cannot drift the numerics, and A5 differ only by the
`use_z3d_tile` launch flag. Registration: `OPS["layer_norm_gated"]` in the three
providers, `layer_norm_gated: triton` in `config/npu_a2.yaml` and `npu_a5.yaml`,
and one row in `docs/operator_support.md`.

The op id is `layer_norm_gated` (underscored), matching the acceptance suite's
`layer_norm_gated_fwd` provider name; `layernorm_gated` is only an alias of the
Python front end and is not a registered op id.

## 4. Kernel design

### 4.1 Grid and work distribution

`grid = (min(vector_cores * (2 if quant else 1), ceil(M / rows_per_block)), ngroups)`.
Program `(pid, group)` walks `row_task_id in range(pid, num_row_tasks, num_programs)`,
so a fixed number of programs covers any `M` without a per-row launch. The upper
bound is the vector-core count (`get_num_cores("vector")`), because every stage of
this kernel is a vector operation; the quant path queues two blocks per core to
overlap the FP8 conversion and stores of one tile with the loads of the next.

### 4.2 Tile shape

* `BLOCK_N = min(65536 // element_size, next_power_of_2(group_size))`, except
  when rounding up would idle more than an eighth of the vector lanes: then
  `BLOCK_N == group_size` itself. A 192-wide group padded to 256 keeps only 75%
  of every vector instruction busy, and Triton-Ascend accepts a non-power-of-two
  block width on the vector path; the tail case (`N = 192`) drops from 65 us to
  41 us with the exact width. The 64 KiB cap is the SGLang limit (a group must
  fit the on-chip buffer); `group_size > BLOCK_N` raises `RuntimeError` instead
  of silently chunking.
* `ROWS_PER_BLOCK` from `calc_rows_per_block`:

| condition | rows |
| --- | --- |
| batch-invariant mode | 4 (constant) |
| `N <= 128` and (`ngroups > 1` or `M*N >= 32M` elements) | 32 |
| `N <= 128` otherwise | 64 |
| `N <= 256` | 32 |
| `N <= 512` | 8 |
| `N <= 1024` | 4 |
| `N > 1024` | 2 |

  A 64-row tile wins on short decoding tensors; on streaming tensors the extra
  rows cost occupancy (measured on A2 at `32768x128`: AIV vector ratio 68.4% ->
  91.9%, MTE2 72.6% -> 97.5% when moving from 64 to 32 rows). A grouped
  normalization fills the grid along its second axis already, so the shorter
  tile also wins there regardless of size (A2, `8192x2048`, `ngroups = 16`:
  60.8% MBU at 32 rows versus 59.8% at 64). The result is capped by
  `next_power_of_2(M)` so a 1-row decode does not build a 32-row block.
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
  mask. Rows are also removed from the transaction mask for every *non-final*
  tile (`full_rows`), so only the last row tile of a launch is predicated: on
  Ascend a predicated vector access costs roughly twice an unpredicated one,
  and `M = 32771` at 32 rows leaves 1024 full tiles paying that price for a
  hazard that only exists in the last one. The tail case keeps a masked last
  block, which is covered by a NaN-sentinel test for out-of-bounds stores.

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
so `q` sees the value the unfused path would quantise. A tile covers
`QUANT_TOKEN_GROUP * quant_heads` rows (two tokens when `M` allows it), so it
still ends on a token boundary and one tile still owns whole tokens.

Four details decide bit-exactness or the A5 cost:

* **Round before `amax`.** The unfused path stores the output dtype and then
  quantises that value, so `amax` has to see the rounded one. For BF16 the
  rounding is integer RNE on the fp32 bit pattern,
  `(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`: bf16 and fp32 share the
  exponent field, so masking the low 16 bits is exactly `to(torch.bfloat16)`,
  including NaN/inf/denormals, while keeping the tile in one layout. It measured
  20.9 us per call cheaper than the dtype round trip on A5. FP16 keeps
  `y.to(Y.dtype).to(tl.float32)`.
* **Masked tiles are zeroed before the reduce**, so padding columns cannot raise
  a token's `amax` (`N = 250` rounds the column tile to 256 and exercises it).
* **`scale = max(amax, 1e-10) * (1.0 / 448.0)`.** On Ascend the torch expression
  `tensor / 448.0` lowers to a multiply by the fp32 reciprocal, so the explicit
  reciprocal form reproduces it; `1.0 / scale` stays a division.
* **No `+-448` clamp inside the kernel.** `|y| <= amax` holds for every element
  of the tile, hence `|y * (1/scale)| <= 448 * (1 + 2ulp) < 464`, and 464 is the
  midpoint where the e4m3fn cast overflows to NaN. The clamp is an identity here
  and removing it saved 29.7 us per call on A5. The tensor-level reference
  (`_quant_reference`) keeps it, because arbitrary inputs do overflow: `480`,
  `1e5` and `1e30` all cast to `0x7f`/`0xff` NaN. `test_fused_quant_bit_exact`
  pins the two implementations to the same bytes.

Two tokens per tile halve the number of token-wide reductions (8 head rows of
512 columns each) and took the A5 row from 151.5 to 131.3 us. The quantised
output never touches `Y`, and `x` is passed as the dtype carrier, so the fused
case allocates no `[M, N]` buffer: memory is `q` (1 byte/element), `s` and
`rstd`.

One residual difference was open here and is now closed. On 32768x512, 1 of
16,777,216 FP8 bytes differed by 1 ulp from the unfused reference while all 4096
scale values were identical. The cause is the reciprocal, but not in the
direction first assumed: measured on the 950 over 2^20 scales drawn from the
reachable FP8 range, triton-ascend lowers `1.0 / s`, `libdevice.div_rn`,
`libdevice.rcp_rn` and `fdiv(ieee_rounding=True)` to the same fast `div.full`
instruction, which is 1 ulp off on 8.9% of them, while torch's `1.0 / scale` is
correctly rounded. The epilogue therefore refines the estimate with one
Markstein step against an exact FMA,

```python
recip = 1.0 / scale
recip = tl.math.fma(recip, tl.math.fma(-scale, recip, 1.0), recip)
```

which is correctly rounded on all 2^20 scales and left no difference on 240
device cases (2 dtypes x 4 shapes x 30 seeds, 15.7M FP8 bytes) against both the
torch reference and a CPU float64 exact reference. The step costs one FMA per
token. Note that the earlier bit-exact runs were clean by luck: the old seed
never landed an element near a quantisation midpoint.

The FP8 epilogue is an Ascend 950 capability, and on Ascend 910B/910C it cannot
be built at all — not by this kernel but by the platform. The cast in the
epilogue, `(y * recip).to(Q.dtype.element_ty)`, lowers to `arith.truncf` to
`f8E4M3FN`, and the 910B compiler aborts on it:

```
LLVM ERROR: unsupported datatype for arith::TruncFOp to hfusion
```

The abort comes from `bishengir-compile`, which Triton-Ascend ships inside its own
package (`triton/backends/ascend/bishengir/bin/bishengir-compile`, invoked with
`--target=Ascend910B3`), not from the CANN tree. The runtime is equally unable to
carry the dtype: on Ascend910B3 `torch.zeros(..., dtype=torch.float8_e4m3fn)` on
`npu:0` fails with

```
AclNN_Parameter_Error(EZ1001): Tensor self not implemented for DT_FLOAT8_E4M3FN,
should be in dtype support list [DT_INT8,DT_INT32,DT_INT64,DT_UINT8,DT_FLOAT16,
DT_FLOAT,DT_BOOL,DT_DOUBLE,DT_INT16,DT_UINT16,DT_COMPLEX128,DT_COMPLEX64,
DT_BFLOAT16,]
```

so the acceptance suite's own FP8 reference
(`(y*(1/scale)).clamp(-448,448).to(torch.float8_e4m3fn)`) cannot run on that
device either — it is the platform's dtype support list, not this
implementation, that stops it. The same two failures reproduce under CANN 9.2.0
(the task's stated minimum), and the public reference implementation
(`brandfucker/mojo_opset#516`) performs the identical cast, so this is not a
divergence introduced here. The task statement scopes the FP8 e4m3 epilogue to
Ascend 950 ("仅 Ascend 950 需要支持该能力") and requires a platform-driven
limitation to be stated in the pull request, which is what §6.10 does.

Because of this, the A2/A3 provider raises a `RuntimeError` with an actionable
message instead of letting the compiler abort, and the A5 provider is left
untouched so the 950 path carries no regression risk:

```python
if quant_heads:
    raise RuntimeError(
        "the fused FP8 epilogue needs a runtime that implements DT_FLOAT8_E4M3FN; "
        "the Ascend 910B/910C runtime does not, so the FP8 output cannot be produced "
        "on this target.  Ascend 950 (npu.a5) is the target that supports it."
    )
```

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
`target_kernel_name`; wheels that only export `do_bench_npu` reach the same
`torch_npu.profiler` path through `prof_dir` / `keep_res`. The helper takes
`warmup` / `active` iteration counts and returns milliseconds.) MBU = bytes /
time / calibrated bandwidth, where bytes are
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

### 6.1 Measured on Ascend 910B3 (A2), 2026-10-30

torch 2.10.0 / torch_npu 2.10.0.post7 / triton-ascend 3.6.0, CANN 9.1.0,
`do_bench_npu_profiler` (5 warm-up + 30 profiled launches), denominator from the
in-process 128M-element BF16 `copy_` probe (1235.4 GB/s; `fill_` measured
1463.3 GB/s):

| case | M x N | us | MBU | line | result |
| --- | --- | --- | --- | --- | --- |
| `gdn-256-n128-swish` | 256 x 128 | 3.82 | 4.18% | relative | 318.9 us e2e vs 794.0 us eager |
| `gdn-2560-n128-swish` | 2560 x 128 | 6.48 | 24.58% | relative | 321.3 us e2e vs 758.1 us eager |
| `gdn-32768-n128-swish` | 32768 x 128 | 31.48 | 64.70% | >=60% | pass |
| `gdn-32768-n128-sigmoid` | 32768 x 128 | 28.30 | 71.97% | >=60% | pass |
| `gdn-131072-n128-swish` | 131072 x 128 | 112.24 | 72.60% | >=60% | pass |
| `qk-65536-n128` | 65536 x 128 | 32.07 | 84.69% | >=60% | pass |
| `group-1024-n2048` | 1024 x 2048 (g128) | 19.59 | 52.04% | relative | 330.1 us e2e vs 807.2 us eager |
| `group-8192-n2048` | 8192 x 2048 (g128) | 134.43 | 60.62% | >=60% | pass |
| `gdn-524288-n128-swish` | 524288 x 128 | 509.23 | 64.01% | >=60% | pass |
| `tail-32771-n192-swish` | 32771 x 192 | 40.94 | 74.64% | >=50% | pass |
| `quant-32768-n512` | 32768 x 512 (QH=8) | - | - | >=60% | skipped: FP8 is required on 950 |

`failures: []`. Raw profiler CSV, `npu-smi info`, version dump and the JSON
report for this run ship with the change (`results/final-*`).

### 6.2 What moved the numbers

The first end-to-end run of this matrix had one failing row: `tail-32771-n192`
at MBU 41.42% against a 50% line. Profiling attributed the gap to three separate
costs, each fixed at the source rather than by retuning:

1. *Predicated accesses everywhere.* The row mask was applied to every tile
   even though only the last one can cross `M`. Predicated vector accesses cost
   about twice an unpredicated one on this architecture, and `32771` rows at 32
   rows per tile means 1024 of 1025 tiles were paying for it. Restricting the
   mask to the final tile (`full_rows`) also removed the `tl.where` that
   zero-filled the RMS numerator. The sweep on that shape moved 73.2 us ->
   41.0 us.
2. *A 192-wide group padded to a 256-wide tile.* 25% of every vector lane was
   idle. Triton-Ascend accepts a non-power-of-two block width on the vector
   path, so `BLOCK_N` now uses the exact group width when the power-of-two
   padding would waste more than an eighth of the block; that took the same
   shape from 65 us to 41 us.
3. *Row tile for grouped normalization.* A grouped launch already fills the
   grid along `ngroups`, so the 64-row tile only added per-tile latency
   (`8192x2048`: ~60.6% at 32 rows versus ~59.8% at 64).

Two things that did **not** help and were reverted: `tl.range(..., num_stages=2)`
and doubling the grid for the grouped case. `num_warps` never changed a timing
result - occupancy here is set by the grid - so it stays at the reference
formula.

### 6.3 Remaining headroom

`group-8192` clears its line by 0.6 points. Its profiles show the 256 B row
segments spread over a 4 KiB stride, i.e. many small DMA transactions rather
than bandwidth saturation; `gdn-524288` is limited by the same scattered 2D gate
read that the A5 three-axis tile removes. Both point at a grouped path that
loads one whole row per tile and reduces `[rows, ngroups, group_size]` in
registers, which is a structural change rather than a tile retune and was not
taken here.

### 6.4 Ascend A3 (910C)

The same kernel source and the same 11-row matrix on an A3 host reporting `Ascend910_9382`
(torch 2.10.0, torch_npu 2.10.0, triton-ascend 3.6.0). The repository's target
detection only recognises `Ascend910B*` and `Ascend950*`, so the run selects the
A2 provider explicitly with `MOJO_ASCEND_ARCH=a2`; the vector ISA is the same and
only the core count and memory differ. Accuracy is identical
(43 passed / 4 skipped), every large shape improves on the A2 numbers while the
two smallest stay level (3.84 / 6.57 us versus 3.82 / 6.48), with the copy probe
at 1264.5 GB/s:

| case | A2 us / MBU | A3 us / MBU |
| --- | --- | --- |
| `gdn-32768-n128-swish` | 31.48 / 64.70% | 28.10 / 70.83% |
| `gdn-32768-n128-sigmoid` | 28.30 / 71.97% | 25.51 / 78.01% |
| `gdn-131072-n128-swish` | 112.24 / 72.60% | 94.59 / 84.16% |
| `qk-65536-n128` | 32.07 / 84.69% | 28.60 / 92.79% |
| `group-8192-n2048` | 134.43 / 60.62% | 116.47 / 68.36% |
| `gdn-524288-n128-swish` | 509.23 / 64.01% | 467.59 / 68.10% |
| `tail-32771-n192-swish` | 40.94 / 74.64% | 36.31 / 82.23% |

Small shapes stay on the relative line with a wider margin (e2e 153.6 / 162.7 /
163.4 us versus 313.2 / 308.8 / 392.8 us for the eager composition).

This host is aarch64 (openEuler 24.03), which matters for the CANN side-install
in §6.7: the packages have to come from the `linux-aarch64` channel, and the
x86_64 ones are rejected by the installer.

### 6.5 Ascend A5 (950PR)

`Ascend950PR_9579`, 56 vector cores, torch 2.10.0+cpu / torch_npu 2.10.0.post7 /
triton-ascend 3.6.0 / CANN 9.1.0, same harness, copy probe 1443.0 GB/s
(`fill_` 1599.0 GB/s). Accuracy is 53 passed / 0 skipped / 0 failed in 162.70 s
on an empty JIT cache (about 20.5 s warm): the A2/A3 set plus the two
`test_layernorm_gated_a5.py` cases for the three-axis gate tile, and the eight
`test_fused_quant_bit_exact` cases now actually execute instead of skipping.

| case | M x N | us | MBU | line |
| --- | --- | --- | --- | --- |
| `gdn-256-n128-swish` | 256 x 128 | 3.78 | 3.61% | relative, 53.4 vs 118.5 us e2e |
| `gdn-2560-n128-swish` | 2560 x 128 | 4.95 | 27.55% | relative, 49.8 vs 111.6 us e2e |
| `gdn-32768-n128-swish` | 32768 x 128 | 22.53 | 77.40% | >=60% |
| `gdn-32768-n128-sigmoid` | 32768 x 128 | 22.44 | 77.74% | >=60% |
| `gdn-131072-n128-swish` | 131072 x 128 | 88.68 | 78.67% | >=60% |
| `qk-65536-n128` | 65536 x 128 | 12.30 | 189.12% | >=60% |
| `group-1024-n2048` | 1024 x 2048 (g128) | 7.91 | 110.36% | relative, 52.5 vs 123.0 us e2e |
| `group-8192-n2048` | 8192 x 2048 (g128) | 42.98 | 162.34% | >=60% |
| `quant-32768-n512` | 32768 x 512 (QH=8) | 83.44 | 69.69% | >=60% |
| `gdn-524288-n128-swish` | 524288 x 128 | 312.89 | 89.19% | >=60% |
| `tail-32771-n192-swish` | 32771 x 192 | 20.61 | 126.94% | >=50% |

`failures: []`; `1 passed in 72.77 s`. The FP8 row is the only one that needed
work: 152.1 us / 38.31% in the first suite run (151.5 us in the in-process
breakdown), 83.44 us / 69.69% after the three changes in
§4.4 (two tokens per tile, clamp removal, integer RNE). The rows above 100% MBU
are cache-resident working sets; `gdn-524288` is the only row that truly presses
HBM, and its profile shows `Block Num = 56`, i.e. the grid exactly fills the
vector cores. One row carries real run-to-run spread: `gdn-131072` measured
87.62 / 88.68 / 92.24 / 95.03 / 98.22 us across five runs of the same kernel
(worst reading 70.9%, still above the line); the other ten stay within +-3%.
Rejected after measuring here: `num_stages` 2-4, grid 112-448, `num_warps`
1/2/4 and 8/16/32-row tiles.

Memory on the same host: the quantised case peaks at 16.9 MB against 386.0 MB for
the eager composition (22.8x), and the widest case (`gdn-524288`) at 136.3 MB
against 1.36 GB (10.0x); the fused path allocates one `[M, N]` output plus the
fp32 statistics and nothing else.

### 6.6 A5 re-run on CANN 9.2.0

The task requires CANN >= 9.2.0, so the same 950 was re-verified with 9.2.0
side-installed at `/usr/local/cann920/Ascend/cann-9.2.0` (toolkit + 950-ops,
Huawei Cloud conda channel). The image's 9.1.0 tree is untouched and the driver
stays at npu-smi 25.7.rc1.10, which 9.2.0 lists as compatible. Before sourcing
the 9.2.0 `set_env.sh`, every 9.1.0 entry is stripped from `LD_LIBRARY_PATH`,
`PYTHONPATH` and `PATH`; the kernel binary is unchanged (sha256
`747b106e…d99663b`).

Accuracy is 53 passed / 0 skipped / 0 failed (43 + 8 + 2 across the three files,
warm JIT cache), including all eight `test_fused_quant_bit_exact` cases. The
11-row suite reports `failures: []` in 62.67 s, and a repeat 61 s later agrees
within 2.4% on ten rows. Calibration: `copy_` 1443.5 GB/s, `fill_` 1692.3 GB/s.

| case | 9.1.0 us / MBU | 9.2.0 us / MBU (run1, run2) |
| --- | --- | --- |
| `gdn-256-n128-swish` | 3.78 / 3.61% | 3.80, 3.79 / 3.59%, 3.60% |
| `gdn-2560-n128-swish` | 4.95 / 27.55% | 4.90, 5.02 / 27.79%, 27.09% |
| `gdn-32768-n128-swish` | 22.53 / 77.40% | 22.37, 22.54 / 77.95%, 77.23% |
| `gdn-32768-n128-sigmoid` | 22.44 / 77.74% | 22.33, 22.36 / 78.07%, 77.86% |
| `gdn-131072-n128-swish` | 88.68 / 78.67% | 96.47, 90.13 / 72.29%, 77.26% |
| `qk-65536-n128` | 12.30 / 189.12% | 12.42, 12.34 / 187.15%, 188.07% |
| `group-1024-n2048` | 7.91 / 110.36% | 8.09, 7.92 / 107.81%, 109.91% |
| `group-8192-n2048` | 42.98 / 162.34% | 42.42, 42.34 / 164.39%, 164.50% |
| `quant-32768-n512` | 83.44 / 69.69% | **54.25, 54.41 / 107.14%, 106.68%** |
| `gdn-524288-n128-swish` | 312.89 / 89.19% | 312.89, 313.30 / 89.15%, 88.91% |
| `tail-32771-n192-swish` | 20.61 / 126.94% | 20.70, 20.72 / 126.37%, 126.03% |

Every row stays above its line. The FP8 row gains 35%, but the kernel is
byte-identical, and `fill_` also rose that run (1599 -> 1692 GB/s) while `copy_`
did not (1443.0 -> 1443.5 GB/s); the gain is therefore recorded without
attributing all of it to the version bump. `gdn-131072` stays inside the spread
documented in §6.5. Fused peak memory is byte-for-byte identical to 9.1.0 for
all 11 cases.

### 6.7 A3 re-run on CANN 9.2.0

The same side-install was repeated on the A3 host used in §6.4 — container
hostname `02ab62ca47bc`, aarch64, openEuler 24.03 — with CANN 9.2.0 under
`/usr/local/cann920/Ascend/cann-9.2.0`. The image's 9.1.0 tree is untouched and
the driver stays at npu-smi 26.1.1. The four tested files hash identically to
the A5 run (kernel sha256 `747b106e…d99663b`), so only the CANN version
changed; the 9.1.0 and 9.2.0 A3 runs are the same host, which the profiler
metadata records (same hostname, same `hwtype`, same kernel release).

Two environment facts are worth keeping. The host is aarch64, so the
side-install has to come from the `linux-aarch64` conda channel
(`ascend-cann-toolkit-9.2.0-0.conda` plus `ascend-cann-a3-ops-9.2.0-0.conda`);
the x86_64 `linux-64` packages make the installer exit rc=2 with
`The current OS is aarch64` / `No packages to install !`. And the installer
rewrites `/etc/Ascend/ascend_cann_install.info` to the new prefix (the original
record is kept under `/etc/Ascend/<timestamp>/`), while
`/usr/local/Ascend/ascend-toolkit/latest` still points at `cann-9.1.0`.

Accuracy is 43 passed / 8 skipped / 0 failed (35 + 8 across the two files), and
this is the first run of the *final* test file on A2/A3-class hardware: 51
cases collected, the eight skips being the `test_fused_quant_bit_exact` cases
that gate on the device name, which matches the prediction recorded before the
run. The 11-row suite reports `failures: []` in 67.90 s, and a repeat 60 s
later agrees within 0.6%. Calibration: `copy_` 1264.86 GB/s (repeat 1260.78),
`fill_` 1464.60 GB/s (repeat 1458.62) — essentially unchanged from 9.1.0
(1264.47 / 1460.58), so the version comparison is clean.

| case | 9.1.0 us / MBU | 9.2.0 us / MBU (run1, run2) |
| --- | --- | --- |
| `gdn-256-n128-swish` | 3.842 / 4.06% | 3.928, 3.922 / 3.97%, 3.99% |
| `gdn-2560-n128-swish` | 6.568 / 23.68% | 6.613, 6.605 / 23.51%, 23.62% |
| `gdn-32768-n128-swish` | 28.101 / 70.83% | 27.973, 28.028 / 71.13%, 71.22% |
| `gdn-32768-n128-sigmoid` | 25.513 / 78.01% | 25.455, 25.617 / 78.16%, 77.92% |
| `gdn-131072-n128-swish` | 94.594 / 84.16% | 94.413, 94.280 / 84.29%, 84.69% |
| `qk-65536-n128` | 28.598 / 92.79% | 28.547, 28.357 / 92.93%, 93.86% |
| `group-1024-n2048` | 17.837 / 55.83% | 17.905, 17.859 / 55.60%, 55.92% |
| `group-8192-n2048` | 116.469 / 68.36% | 116.443, 116.505 / 68.35%, 68.54% |
| `quant-32768-n512` | skipped | skipped (FP8 is 950-only) |
| `gdn-524288-n128-swish` | 467.591 / 68.10% | 463.500, 460.634 / 68.68%, 69.33% |
| `tail-32771-n192-swish` | 36.310 / 82.23% | 36.304, 36.139 / 82.22%, 82.86% |

Every measured row stays above its line; the largest deviation is +2.2% on the
smallest case, where fixed overhead dominates, and every large shape stays
within ±1.5%. Unlike the A5 re-run there is no FP8 row to observe the fp8
lowering change on, because that row skips by device name on A3. Fused peak
memory is byte-for-byte identical to 9.1.0 for all ten cases.

### 6.8 A2 re-run on CANN 9.2.0

The A2 host from §6.1 became unreachable after the environment was reissued:
the tunnel still opened (jump authentication and direct-tcpip both fine, target
running OpenSSH 9.6 offering `publickey,password`) but the root password was
rejected on every attempt and none of the local keys were accepted, which
points at a credential problem on the target rather than a tunnel problem. The
environment was therefore rebuilt, and the rebuilt machine is a *different SKU*:
`Ascend910B2` (bus-id `0000:01:00.0`) instead of the original `Ascend910B3`
(bus-id `0000:41:00.0`). Its absolute numbers are consequently not comparable
to §6.1; what the rebuild buys is a same-box 9.1.0 ↔ 9.2.0 comparison.

Both CANN versions were run on that host: the image's 9.1.0 tree as the
baseline and a side-installed 9.2.0 under `/usr/local/cann920/Ascend/cann-9.2.0`.
The four tested files hash identically to the A5 and A3 runs (kernel sha256
`747b106e…d99663b`), so the version is the only variable. The host is aarch64
(openEuler 24.03 LTS-SP3) with npu-smi 26.1.1.

Two environment facts are worth keeping here. First, the ops package is
SoC-specific: this host needs `ascend-cann-910b-ops-9.2.0-0.conda`, not the
`a3-ops` or `950-ops` build, even though all three come from the same
`linux-aarch64` conda channel. Second, and specific to this image, the
toolkit's `cann-pto-as` sub-package installer shells out to `python3 -m pip`,
and `/usr/bin/python3` here is 3.11.6 *without* pip — so the first attempt
died with `cann-pto-as_9.2.0_linux-aarch64.run install failed` and the whole
toolkit install rolled back, leaving an empty `cann-9.2.0` tree. Putting the
CANN-image interpreter (`/usr/local/python3.12.13/bin`, pip 26.2) at the front
of `PATH` before re-running the same `.run` installs it cleanly.

Accuracy is 43 passed / 8 skipped / 0 failed on *both* CANN versions (51 cases
collected; the eight skips are the `test_fused_quant_bit_exact` cases that gate
on the device name). This is the first run of the final test file on A2-class
hardware of any SKU. The 11-row suite reports `failures: []` three times
(97.15 s on 9.1.0, 87.90 s and 76.07 s on 9.2.0). Calibration: `copy_`
1244.96 / 1242.69 / 1242.33 GB/s, `fill_` 1460.76 / 1781.29 / 1458.78 — the
single 1781.29 GB/s `fill_` outlier is probe jitter (the next run is back at
1458.78) and the `copy_` denominator moves by ≤0.2%, so the version comparison
stays clean.

| case | 9.1.0 us / MBU | 9.2.0 us / MBU (run1, run2) |
| --- | --- | --- |
| `gdn-256-n128-swish` | 3.903 / 4.06% | 3.857, 3.853 / 4.11%, 4.12% |
| `gdn-2560-n128-swish` | 6.629 / 23.83% | 6.552, 6.625 / 24.15%, 23.89% |
| `gdn-32768-n128-swish` | 28.131 / 71.86% | 28.216, 28.134 / 71.77%, 72.00% |
| `gdn-32768-n128-sigmoid` | 25.575 / 79.04% | 25.466, 25.380 / 79.52%, 79.82% |
| `gdn-131072-n128-swish` | 94.597 / 85.48% | 94.454, 94.375 / 85.76%, 85.86% |
| `qk-65536-n128` | 28.367 / 95.01% | 28.187, 28.347 / 95.80%, 95.28% |
| `group-1024-n2048` | 17.809 / 56.79% | 17.799, 17.847 / 56.93%, 56.79% |
| `group-8192-n2048` | 116.426 / 69.45% | 116.412, 116.335 / 69.59%, 69.66% |
| `quant-32768-n512` | skipped | skipped (FP8 is 950-only) |
| `gdn-524288-n128-swish` | 470.429 / 68.75% | 468.871, 474.914 / 69.11%, 68.25% |
| `tail-32771-n192-swish` | 36.099 / 84.00% | 36.311, 36.423 / 83.67%, 83.43% |

Every measured row stays above its line, and the largest deviation between the
three runs is 1.28% on the smallest case; every large shape stays within 1.0%.
The FP8 row skips by device name here as well, so the fp8 lowering change
remains observable only on the 950. Fused peak memory is byte-for-byte
identical between 9.1.0 and 9.2.0, and equal to the A3 and A5 values.

Unlike A3, this device name needs no override: `Ascend910B2` matches the
target-detection table and resolves to `npu.a2.910b2` on its own, so
`MOJO_ASCEND_ARCH` stays unset for both runs.

One packaging artifact is worth recording so it is not mistaken for local
edits. The repository snapshot was packed from a Windows checkout with
`core.autocrlf=true`, and on the device `core.autocrlf` is unset, so `git
status` reports 432 files as modified — pure line-ending noise
(93876 insertions / 93876 deletions). The four tested files are not among
them and are byte-identical to the index.

### 6.9 Re-verification after the rewrite and the FP8 fix (2026-10-31)

The function front end, the accuracy harness and the FP8 epilogue all changed
after the runs above, so the whole suite was re-run on the 950 host (§6.6), on
the A3 host (§6.7) with CANN 9.2.0 side-installed, and on a reissued A2 host
running the image's own CANN 9.1.0.

On the 950: accuracy 53 passed / 0 failed, `failures: []` on the 11-row matrix,
and 240 FP8 stress cases with no byte difference. The Markstein step is not
measurable in the matrix:

| case | 2026-10-31 | the 2026-10-30 9.2.0 reading |
| --- | --- | --- |
| `gdn-32768-n128-swish` | 22.48 us / 77.33% | 22.53 us / 77.40% |
| `gdn-131072-n128-swish` | 93.19 us / 74.62% | 90.13-96.47 us |
| `qk-65536-n128` | 12.36 us / 187.49% | 12.30 us / 189.12% |
| `group-8192-n2048` | 45.68 us / 152.23% | 42.98 us / 162.34% |
| `quant-32768-n512` | 54.69 us / 105.98% | 54.25 us / 107.14% |
| `gdn-524288-n128-swish` | 313.17 us / 88.82% | 312.89 us / 89.15% |
| `tail-32771-n192-swish` | 20.68 us / 126.09% | 20.70 us / 126.37% |

The A3 re-run used the same host as §6.7 (hostname `02ab62ca47bc`, aarch64,
openEuler 24.03 LTS-SP3, kernel 5.10.0-216, `Ascend910_9382`, npu-smi 26.1.1)
with `/usr/local/cann920/Ascend/cann-9.2.0` sourced after the 9.1.0 path
entries were stripped, and `MOJO_ASCEND_ARCH=a2` because the target detection
does not know this SKU. The four tested files hashed identically to the commit
under review (kernel `35fa6572…0eb9f02`, front end `0e3bcf7e…1a4a02`, accuracy
harness `264ba81e…e8af`, perf harness `8cc415cf…4e0010`). `tests/functions`
reported 35 passed / 8 skipped in 60.46 s and `tests/modules` 8 passed in 13.26
s; the eight skips are the `test_fused_quant_bit_exact` cases that gate on
`npu.a5`. The 11-row suite finished in 78.74 s with the FP8 row skipped by
device name and `failures: []`. Calibration on this run: `copy_` 1267.21 GB/s,
`fill_` 1458.84 GB/s.

| case | pre-rewrite 9.2.0 us (§6.7 run1) | after us / MBU | delta |
| --- | --- | --- | --- |
| `gdn-256-n128-swish` | 3.928 | 3.88 / 4.01% | -1.3% |
| `gdn-2560-n128-swish` | 6.613 | 6.55 / 23.70% | -1.0% |
| `gdn-32768-n128-swish` | 27.973 | 27.94 / 71.09% | -0.1% |
| `gdn-32768-n128-sigmoid` | 25.455 | 25.47 / 77.98% | +0.1% |
| `gdn-131072-n128-swish` | 94.413 | 94.29 / 84.25% | -0.1% |
| `qk-65536-n128` | 28.547 | 28.50 / 92.90% | -0.2% |
| `group-1024-n2048` | 17.905 | 17.89 / 55.53% | -0.1% |
| `group-8192-n2048` | 116.443 | 116.38 / 68.26% | -0.1% |
| `quant-32768-n512` | skipped | skipped (FP8 is 950-only) | - |
| `gdn-524288-n128-swish` | 463.500 | 462.83 / 68.65% | -0.1% |
| `tail-32771-n192-swish` | 36.304 | 36.15 / 82.42% | -0.4% |

Every row stays inside its line and the largest move from the pre-rewrite
numbers is -1.3% on the smallest case, where fixed overhead dominates.

The A2 re-run then used a host the platform reissued on 2026-10-31: container
hostname `1233b4c4c4dd`, aarch64, openEuler 24.03 LTS-SP3 (kernel 5.10.0-216),
`Ascend910B3`, npu-smi 26.1.1, the image's own CANN 9.1.0 (nothing side
installed) and the same four file hashes. The device name resolves to
`npu.a2.910b3` without an override. `tests/functions` reported 35 passed / 8
skipped in 179.99 s and `tests/modules` 8 passed in 31.77 s, i.e. the same
43 / 8 / 0 split as the other two hosts. The 11-row suite finished in 97.53 s
with `failures: []`, and a second run on the same box reproduced every row
within 1.3% (the largest move is `gdn-2560` 6.553 -> 6.473 us).

That host's HBM, however, is a third as fast as the 2026-10-30 A2: the in-run
`copy_` probe measures 394.50 GB/s against 1235.4 GB/s in §6.1, and the only
row whose traffic exceeds the 192 MB L2 (`gdn-524288-n128-swish`, 402 MB) took
1288.48 us instead of 509.23 us. All other rows, which stay L2-resident, match
the old numbers within 1.4%. The MBU column of that run is therefore scaled
against a degraded calibration and is not comparable to §6.1; the A2
performance figures in this document remain the §6.1 ones, and the 2026-10-31
A2 run is a functional re-verification.

A third run on the same box measured the second rewrite from §0: the tree was
copied and only `kernels/torch/layernorm_gated.py` (`1826e25a…52ee97`) and
`tests/performance/functions/test_layernorm_gated.py` (`8e5e9b1d…6752cbf`) were
replaced, so the kernel, front end and accuracy harness hashed identically to the
run above. `tests/functions` reported 35 passed / 8 skipped and `tests/modules` 8
passed, the 11-row suite `1 passed` in 75.49 s with `failures: []`. Every measured
row moved by at most 1.5% against the previous run on that host (the two smallest
cases, 3.855 -> 3.912 us and 6.473 -> 6.571 us, where launch overhead dominates),
and the `copy_` calibration reproduced at 393.34 vs 394.53 GB/s.

### 6.10 The acceptance suite on the device (2026-10-31)

The suite's own files were run on a device for the first time on this branch on
2026-10-31, on an Ascend910B3 host (container hostname `a4b415bb106e`, aarch64,
openEuler 22.03 LTS-SP4, kernel 5.10.0-216, npu-smi 26.1.1, the image's own CANN
9.1.0 under `/usr/local/Ascend/cann-9.1.0`, torch 2.10.0+cpu, torch_npu
2.10.0.post7, Triton-Ascend 3.6.0, Python 3.12.13). The device name resolves to
`npu.a2.910b3` without an override.

The tree under test is this branch's working tree at `fe102fe`. The device copy
carries CRLF because the archive was packed from a Windows checkout, so the
hashes are compared after normalising the line endings: all sixteen tested
files then hash identically to the committed tree, and the suite's two files
reproduce the `#525` md5s `d926c69d2c72d0ca07d716a255ad2db4` and
`1bb93427d21c84e474ac0720ec4fce76` exactly.

| suite | result | time |
| --- | --- | --- |
| `tests/functions/test_layernorm_gated.py` (the `#525` file) | 116 passed / 17 failed | 409.73 s |
| `tests/functions/test_layernorm_gated_extra.py` | 34 passed / 8 skipped | 3.18 s |
| `tests/functions/test_layernorm_gated_aliases.py` | 2 passed | 11.47 s |
| `tests/functions/test_layernorm_gated_a5.py` | skipped on this device | — |
| `tests/modules/test_layernorm_gated.py` | 8 passed | 2.68 s |
| `tests/performance/functions/test_layernorm_gated.py` (the `#525` file) | 10 passed / 1 failed | 51.85 s |
| `tests/performance/functions/test_layernorm_gated_mbu.py` | 1 passed, `failures: []` | 74.89 s |

The seventeen accuracy failures are `test_gated_norm_quant` (nine cases) and
`test_fused_quant_bit_exact` (eight cases); the single performance failure is
the suite's FP8 row. Every other case passes, including the two the suite
leaves device-only: `test_batch_invariant_rows_bitwise` across `M = 131/132`
(passes for both the LayerNorm and the RMSNorm specialisation) and the
`out`-identity assertion.

All eighteen failures are the FP8 epilogue, and §4.4 shows why: the Ascend910B
runtime has no `DT_FLOAT8_E4M3FN` kernel and its compiler cannot lower the
cast. The failure is not specific to this branch's Triton path — the suite's
own reference ends in the same cast, so the FP8 case cannot pass on this device
whatever the kernel does. The suite also has no `npu.a5` skip marker for
`test_fused_quant_bit_exact`, so it runs on A2/A3 by design; this branch's own
`test_layernorm_gated_extra.py` does gate those cases on the device name, which
is why its eight FP8 cases skip here.

The A2 provider now raises the `RuntimeError` of §4.4 instead of letting the
compiler abort, so the seventeen failures name the capability instead of
dumping a compiler stack trace.

The same tree was then run under the side-installed CANN 9.2.0 (the task's
stated minimum), with the 9.1.0 entries stripped from `LD_LIBRARY_PATH`,
`PYTHONPATH` and `PATH` before `set_env.sh` is sourced. The suite's two files
reproduce the `#525` md5s byte for byte there, because that tree was packed
from an LF checkout. The result is identical — 116 passed / 17 failed in
310.93 s on the accuracy file, 10 passed / 1 failed in 54.23 s on the
performance file, 34 passed / 8 skipped, 2 passed, 8 passed, and the MBU matrix
`1 passed` with `failures: []` — so the gap is a property of the Ascend910B
platform and not of the CANN version. The MBU matrix also reproduces the 9.1.0
readings row for row (largest move 1.1% on `gdn-256`, where launch overhead
dominates) with a `copy_` calibration of 394.33 GB/s against 395.41 GB/s.

That host's HBM is again a third as fast as the 2026-10-30 A2 — its in-run
`copy_` probe measures 395.41 GB/s against 1235.4 GB/s — so the MBU column of
both runs is scaled against a degraded denominator and is not comparable to
§6.1. The one row whose traffic exceeds the 192 MB L2 (`gdn-524288-n128-swish`,
402 MB) took 1280.58 us instead of 509.23 us, while every L2-resident row is
within 1.4% of the original A2. As in §6.9, the A2 performance figures in this
document remain the §6.1 ones and this run is a functional re-verification.

### 6.11 Attributing the FP8 failure to the platform

Two probes were run on that host to separate the platform from this
implementation.

The first (`fp8_support_probe.py`) asks the runtime directly:
`torch.zeros(..., dtype=torch.float8_e4m3fn)` on `npu:0` fails with
`AclNN_Parameter_Error(EZ1001)`, a cast to that dtype fails, the suite's own
reference expression fails, and the same expression on the CPU succeeds.

The second (`fp8_suite_probe.py`) drives the public entry point twice on the
same device — once with `implementation="torch"`, which is the suite's own
reference path and does not touch Triton at all, and once with the default NPU
dispatch:

```
=== public entry point, implementation='torch' (the suite's own reference path) ===
-> FAILED: RuntimeError
copy_d2d_baseformat_opapi:.../CopyKernelOpApi.cpp:126 NPU function error: call aclnnInplaceCopy failed, error code is 561103

=== public entry point, default implementation (this branch's NPU dispatch) ===
-> FAILED: RuntimeError
the fused FP8 epilogue needs a runtime that implements DT_FLOAT8_E4M3FN; the Ascend 910B/910C runtime does not, ...
```

The first block is the decisive one: the *torch reference* fails on this device
with a plain runtime dtype error, before any kernel of ours is reached. The
suite's FP8 assertion cannot be satisfied on an Ascend910B whatever the
implementation is. Both blocks are identical under CANN 9.1.0 and CANN 9.2.0;
under 9.1.0 the second block is the compiler abort, because that tree predates
the guard of §4.4.

### 6.12 Rehearsing every merge resolution with the acceptance suite

This branch and the community acceptance suite (#525) both add a gated
LayerNorm operator, so the two pull requests cannot be merged into `master`
without a decision. The rehearsal replays that merge for real and checks every
resolution.

The history is reconstructed from the upstream commits rather than approximated:
the common base is `0a7d64c`, upstream `master` is `8b4b7a1`, and the two
commits in between touch ten files, none of them among the six files #525
changes. The rebuilt `master` tree hashes to
`769df7e9a73169faed1a9286134ca60f1ed3aa95`, the tree SHA GitHub reports for
`8b4b7a1`, so the reconstruction is byte-exact. The suite is then applied on
top of the common base — its two modified `__init__.py` files are byte-identical
at the suite's own base `45102091` and at `0a7d64c`, so this is equivalent to
merging #525 into `master`.

That first merge is clean. Merging this branch into the result produces three
conflicts: `mojo_opset/functions/__init__.py` (content), and
`mojo_opset/functions/layernorm_gated.py` plus
`mojo_opset/kernels/torch/layernorm_gated.py` (add/add). The
`functions/__init__.py` conflict is a single three-line hunk — this branch's
`layernorm_gated_compat` imports, against the suite's `from .layernorm_gated
import layer_norm_gated` on the same anchor. The resolution is to keep both
sides: this branch's side already contains the suite's line.

Each conflicted file is resolved once on the suite side and once on this
branch's side, and the operator registry
(`mojo_opset/kernels/torch/__init__.py`), which git merges on its own but which
a maintainer could still overwrite, is resolved twice as well — sixteen merged
trees in total, each exported as a standalone directory.

Every tree is then checked three ways: the package imports and the fixture's
operator-id assertion from `tests/_fixtures.py` is replayed verbatim (every
`@pytest.mark.api(..., ops=[...])` id must exist in `mojo_opset.kernels.torch.OPS`);
the suite's own `check_gated_norm` helper runs over its full matrix on the CPU
provider; and the four relevant test files are collected.

| branch head | result |
| --- | --- |
| `4b28729` (before the fix) | 8/16 — the eight trees that take the suite's registry fail with `op ids missing from the torch registry: tests/modules/test_layernorm_gated.py::layernorm_gated` |
| `728f0df` (after the fix) | 16/16 |

The hazard was semantic rather than textual. The spelling `layernorm_gated` was
registered as an operator id in the two provider registries and the two target
configurations, but no provider exported a `layernorm_gated_fwd` function — only
`layer_norm_gated` was dispatched. #525's registry does not carry that alias
id, so a maintainer who took the suite's registry made this branch's
`tests/modules/test_layernorm_gated.py` declaration
(`ops=["layernorm_gated"]`) point at a missing operator, and eight cases error
out at collection time.

Two commits fix it. `b0bc630` removes the alias id from the two YAML files and
the three provider registries; the alias *functions* stay in
`mojo_opset/functions/layernorm_gated_compat.py` and stay covered by
`tests/functions/test_layernorm_gated_aliases.py`. As a side effect
`mojo_opset/kernels/torch/__init__.py` becomes the same git blob as #525's file
(`eba441200a299c00e7f876321c3e80720ec91ad2`), so that file cannot diverge under
any resolution. `728f0df` changes the alias test's skip condition from `hasattr`
to `callable(getattr(...))`: `functions.layernorm_gated` names both the
submodule and the desired alias, so taking the suite's `functions/__init__.py`
leaves the attribute present but not callable, and the old condition would not
skip and would fail on the device.

The rehearsal above is the CPU side. The sixteen merged trees have not been
re-run on an Ascend device: the four DevEnv channels available on 2026-10-31
all refuse to open (`platform authorization denied: Administratively
prohibited`), so that leg needs a fresh ticket. The scripts and the trees are
in place for it.

## 7. Known limits

* `group_size` must divide `N` and must fit 64 KiB, as in the reference kernel.
* The quantised path requires a single group (`ngroups == 1`) and
  `M % quant_heads == 0`; anything else raises `ValueError`.
* A 3D gate requires `T*H == M`, `D == N` and a contiguous last dim; `z` is never
  silently copied, so a bad view is an error rather than a performance cliff.
* Forward only: this is an inference primitive, there is no backward wrapper.
* `mean` and `rstd` keep the reference layout `[ngroups*M]` (group-major) rather
  than `[M, ngroups]`.
* The FP8 e4m3 epilogue needs a runtime that implements `DT_FLOAT8_E4M3FN`. The
  Ascend 910B/910C runtime does not, and its compiler cannot lower the cast
  either, so `quant_heads > 0` raises a `RuntimeError` on those targets; only
  Ascend 950 (`npu.a5`) produces the FP8 output. The limit is the platform's
  dtype support, not this implementation — see §4.4 and §6.10.

## 8. Reproduce

```bash
# accuracy + bitwise (the acceptance suite's own files, plus our extra coverage)
pytest tests/functions/test_layernorm_gated.py tests/functions/test_layernorm_gated_extra.py \
    tests/functions/test_layernorm_gated_aliases.py tests/modules/test_layernorm_gated.py -q
# on 950 also add the A5-only cases
pytest tests/functions/test_layernorm_gated.py tests/functions/test_layernorm_gated_extra.py \
    tests/functions/test_layernorm_gated_aliases.py tests/functions/test_layernorm_gated_a5.py \
    tests/modules/test_layernorm_gated.py -q
# benchmark (writes JSON + raw profiler csv)
pytest tests/performance/functions/test_layernorm_gated.py -q --perf-output tests/performance/results/npu.json
# the MBU matrix
pytest tests/performance/functions/test_layernorm_gated_mbu.py -q --perf-output tests/performance/results/mbu.json
```

On A3 the device reports `Ascend910_9382`, which the target detection does not
know, so the run needs `MOJO_ASCEND_ARCH=a2`; the rebuilt A2 reports
`Ascend910B2`, which the detection *does* know, so it needs no override. On
CANN 9.2.0 the 9.1.0 entries have to leave `LD_LIBRARY_PATH`, `PYTHONPATH` and
`PATH` before the 9.2.0 `set_env.sh` is sourced; the side-installed tree is
`/usr/local/cann920/Ascend/cann-9.2.0` and the packages come from the
`linux-aarch64` conda channel on A2 and A3 (with the SoC-specific ops build:
`ascend-cann-910b-ops` on A2, `ascend-cann-a3-ops` on A3) and the `linux-64`
one on the 950. On the A2 image the toolkit installer additionally needs an
interpreter with pip on `PATH` — see §6.8.

On A2 and A3 the FP8 cases of the acceptance suite fail by construction,
because that runtime does not implement `DT_FLOAT8_E4M3FN` (§4.4); the suite
has no device-name skip for `test_fused_quant_bit_exact`, so it runs there.
Every non-FP8 case passes on those hosts.

---

本算子由 AI 辅助开发，作者已逐行复核并在真实设备上完成精度与性能验证 —— pwdh2026
