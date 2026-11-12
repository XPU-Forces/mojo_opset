# AOT specialization validation

Measured on 2026-10-29, Ascend950PR_9579 (28 Cube cores), CANN 9.2.0,
PyTorch 2.7.1 / torch-npu 2.7.1.post11. Baseline: `2f28f2a0`.

## What changed

The original general catalogue made M/N/K and core count dynamic. It retained
compile-time tile sizes, memory layouts, arithmetic, buffer geometry and output
operators, but did not instantiate BF16 residency or single-M scheduling.

`specializations.json` now defines 46 bounded additions to the 1,232 general
instances. Each references a general `base_tag` and overrides selected template
parameters. Retained variants include fixed N/K/core count, BF16 resident-A,
single-M scheduling combined with fixed dimensions, and one scale-chunk variant.
All M values remain dynamic. Device kernels and compiler optimization flags are
unchanged. The device library remains approximately 54 MiB (previously 53 MiB).

Host dispatch checks fixed dimensions, core count, residency/single-M bounds and
existing direct-store restrictions before launching any entry. The same checks
apply to forced profiling entries. The 67-entry measured dispatch table is scoped
to exact SoC/core count, metadata, shape and tightly packed strides. Unmeasured
shapes retain the original general fallback selection; no runtime compilation or
autotuning is introduced.

## Measurement method

- 77 cases: the previous 57 tuned cases (including all four layouts and both
  output types), plus 20 additional small-M/small-N cases across four operations.
- 596 initial specialization variants, followed by 12 combinations of promising
  transformations. Four fresh-process sweeps; combination entries appear in the
  last two sweeps.
- Every candidate is checked against the CPU reference both eagerly and after
  graph replay. Comparisons keep the same general base tile/MAD configuration.
- 30 GEMM calls per NPU graph, nine randomized paired timing rounds per case,
  medians of NPU event times. Inputs and scale layout preparation are excluded.
- Selection initially required >3% improvement over the previous default in every
  applicable sweep. Prefer fewer overridden parameters when within 1% of the
  fastest eligible variant. Rebuild the reduced catalogue and measure actual
  default dispatch twice more. Drop eight marginal entries whose default-path
  improvement was only 2.6-2.9%, leaving 46 retained entries.
- Repeat correctness and default-path timings after that final reduction.

The same-base comparisons isolate specialization effects from tile selection.
The before/after default comparison also includes dispatch and graph placement
variation; it is not an instruction-level attribution. These are warm graph
measurements on this SKU, not a guarantee for arbitrary shapes or other devices.

## Findings

Fixed N/K and core count together provided the broadest improvement, especially
for FP8/FP4. BF16 resident-A was useful for a small-M, many-N-tile case. Standalone
single-M specialization did not consistently improve performance, although some
combined variants were worthwhile. Changing L2 hints or epilogue buffer count
alone did not produce repeatable >3% gains in the tested set and no such variants
are retained. Resident-B improved its paired base in one case, but did not beat
the existing default and is not retained.

Final default dispatch was checked on all 77 cases in two fresh processes.
44/77 cases improved by more than 3%; no case regressed by more than 3%.
Two retained marginal entries measured just below 3% in this final repeat,
illustrating the timing variation near the selection threshold.
Representative BF16-output results:

| Operation | M,N,K | Before (us) | After (us) | Speedup |
|---|---|---:|---:|---:|
| BF16 | 32,8192,1024 | 12.795 | 10.225 | 1.251x |
| MXFP8 | 16,4096,512 | 3.875 | 3.390 | 1.143x |
| MXFP8 x MXFP4 | 64,2048,2048 | 5.586 | 4.856 | 1.150x |
| MXFP4 | 4096,32,128 | 3.307 | 2.787 | 1.187x |


## Validation and reproduction

- Full A5 native build passed, including the existing non-GEMM operators.
- CPU framework and CI unit suites: 567 passed.
- Dense GEMM NPU regression: 179 passed, one skipped (no retained resident-B entry).
- New checks cover incompatible shape/core guards, invalid tuning, small-M tails
  and padded strides. Catalogue/contract checks and Ruff passed.

Generate the catalogue from the same revision as the loaded library. `tune.py`
accepts `--specializations-only --compare-default --rounds 9 --repeats 30` to
compare retained specializations, their general bases and current default dispatch.
Case JSON uses the existing `op`, `shape`, `trans_input`, `trans_weight`, `output`
fields documented in README.md. Compare with a saved baseline tuning table when
reproducing a before/after dispatch result.

Raw sweeps, per-round timings, the full experimental catalogue, selection inputs
and build/test logs are retained under
`build/dense_gemm_specialization_validation/` in the development checkout.
