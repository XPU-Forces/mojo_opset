# triton_convert_req_index_to_global_index validation

## Scope

This report covers the public mojo_opset API on Ascend A2 and A5, including the original 22 workloads, interface boundaries, FakeTensor metadata, fullgraph capture/execution, and public-call performance. It does not claim production compiler backend, NPU Graph replay, dynamic-shape, whole-model, cold-L2, MFU or MBU acceptance.

The implementation under test is commit `d300d0fd923f2b662f47a12995a974c4d0097295` (including upstream `6115199`). The comparison baseline is PR head `899baf449fb3e14d7fe58edd8e892c509d98239f`, with the identical performance harness and deterministic input builder added for measurement. Input preparation is outside timing; validation, dispatch, layout copies, output allocation and launch remain inside the public call.

## Fixes found by the new tests

- **FakeTensor executed a real kernel.** The original unregistered wrapper entered Triton launch with fake pointers. The subsequent synchronization reported an invalid device-memory access. Register an opaque operator with a metadata-only fake implementation; tests also patch the real implementation to fail if FakeTensor reaches it. Follow-on failures in that poisoned process are not independent kernel correctness failures.
- **A5 one-entry page table failed to compile.** The backend could not lower a singleton gather source (`arith.constant` legalization failure). Pad the gather source to at least two lanes; the existing page-table mask protects the extra lane. This covers K=1 as well as larger K without changing legal inputs or outputs.
- **The first graph fix added host overhead.** High-level `custom_op` registration preserved kernel timing but slowed A5 public-call E2E (initial 22-case geometric speedup about 0.76x). The final implementation uses direct `torch.library.Library` dispatcher registration, with schema/no-mutation/FakeTensor checks and fullgraph execution tests. Device-specific models cache hardware metadata only; neither tensors nor outputs are cached.
- **Test-tool limitation:** `SchemaCheckMode`/`clone_preserve_strides` in this environment cannot clone certain empty strided NPU views (negative allocation size). Schema opcheck uses contiguous empty inputs; separate public-API boundary and FakeTensor tests still cover empty strided views.
- Resolved the upstream Torch-provider registration conflict by retaining both `topk_infer` and `triton_convert_req_index_to_global_index`.

- **A2 profiler export failure:** some captures on the original card contained host launch flows but no device kernels (`TASK` table missing). Such results were rejected, and their raw trace/logs retained locally. The complete parent/current/Torch comparison was rerun on another idle healthy card; no parser validation was relaxed.

## Correctness and graph coverage

Each platform runs 100 cases with exact integer equality, with no tolerance relaxation:

| Coverage | Cases |
| --- | ---: |
| Existing matrix/reference/basic API tests | 25 |
| Empty/singleton/tail shapes, all/mixed/no active prefill, contiguous and offset strided inputs; input immutability | 30 |
| Explicit page-edge, invalid sentinel and prefill-workspace expected values | 1 |
| Invalid dtype/rank/shape/device/optional tensors/block-size/TopK | 23 |
| FakeTensor shape/dtype/device/stride and no real implementation execution | 6 |
| `fullgraph=True`, `eager` and `aot_eager`, decode/prefill/empty, contiguous/strided, two different inputs per compiled call | 12 |
| Registered operator schema/no-mutation and fake metadata (`opcheck`) | 3 |

These graph backends validate graph integration; they do not establish accelerated compiler lowering or NPU Graph replay. Preload the selected provider before compilation, as required by the repository dispatcher.

## Measurement contract

- Same physical card and software stack for each platform's parent/current/Torch comparison; runs are serial on each card. No frequency locking or explicit cache eviction.
- 22 workloads from `tests/_convert_req_index.py`; int32 inputs built on CPU with deterministic seeds 42 and 43, copied to the target device before timing. Two independent input instances alternate.
- 10 warmups, 20 samples per timer and case; report medians. Ratios are baseline/current latency, aggregated as a geometric mean across the 22 cases.
- `profiler_span_us`: first device kernel start to last kernel end within a public call, including intervening device gaps. `e2e_us`: host wall time for the public call followed by device synchronization. They must not be mixed.
- Rotating two inputs is not a cold-L2 guarantee. No historical L20 timings, bandwidth calibration, simulator cycles or PMU ratios enter these results.
- Repository `scripts/compare_perf.py` validates compatible environments, measurement settings, workload hashes and complete samples. Its 5% threshold/noise labels are reported, not CI gates.

## Supplemental paired timing

Because the 20-sample E2E results were noisy, both platforms were also measured in parent/current/current/parent order, with 50 warmups and 200 samples per timer per run. Each side contributes 400 samples per case. Pooled medians are descriptive; the 3xMAD rule is not a confidence interval. Event time is current-stream elapsed time including submission gaps, not the profiler kernel duration.

Remote runs used Git archives, so the raw harness `revision.commit` is null. The evidence bundle includes a source manifest binding the archive and tracked source hashes (CRLF normalized to LF for Git-blob comparison) to the reported Git commit. Reports do not substitute a fabricated Git checkout revision.

## Reproduction

```sh
# Source the platform's CANN environment and activate its torch_npu/Triton environment first.
# Select an idle physical card; --perf-device 0 is the logical card within this visibility mask.
export ASCEND_RT_VISIBLE_DEVICES=<physical_id>
python -m pytest tests/functions/test_triton_convert_req_index_to_global_index.py \
  --mojo-target npu.a2 --mojo-implementation triton --junitxml=accuracy.xml -q
python -m pytest tests/performance/functions/test_triton_convert_req_index_to_global_index.py \
  --mojo-target npu.a2 --mojo-implementation triton --perf-device 0 \
  --perf-timers profiler,e2e --perf-instances 2 --perf-warmup 10 --perf-repeats 20 \
  --perf-output triton.json -q
# Use npu.a5 on A5; use --mojo-implementation torch for the reference measurement.
# Run the parent checkout with the same benchmark file and shared input helper.
python scripts/compare_perf.py parent.json triton.json --output comparison.md
```

## Results

A2: Ascend910B3, physical card 0 -> logical 0, CANN 9.3.0. A5: Ascend950PR_957c, physical card 2 -> logical 0, CANN 9.2.0-beta.2. Python versions are recorded in the raw environment JSON. Both use torch 2.7.1+cpu, torch_npu 2.7.1.post11.dev20261022 and Triton 3.6.0. The selected cards were checked before each run; unrelated work on other cards was retained.

| Platform | Correctness / boundary / graph | Performance per implementation | Profiler vs parent | E2E vs parent (20 samples) | Paired E2E vs parent (400 samples) | Profiler vs Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 | 100/100 | 22/22 | 0.9903x | 1.0889x | 1.0547x | 273.4339x |
| A5 | 100/100 | 22/22 | 0.9904x | 0.9553x | 1.1785x | 263.1357x |

Profiler latency is approximately unchanged. E2E results depend on host load and show substantial sample dispersion; the paired run reduces ordering bias but does not prove a speedup for every case. See per-case statuses and raw samples rather than treating the geometric mean as a per-case guarantee.

- A2 parent comparison status counts: profiler {'noisy': 16, 'stable': 6}; 20-sample E2E {'noisy': 11, 'stable': 7, 'improvement': 4}; paired E2E {'noisy': 22}.
- A5 parent comparison status counts: profiler {'noisy': 20, 'stable': 2}; 20-sample E2E {'noisy': 21, 'stable': 1}; paired E2E {'noisy': 22}.

### A2 per-case medians (microseconds)

P = parent `899baf4`; C = current `d300d0f`; Torch = same-device reference.

| Case | T / R / M / block size / K / prefill | Profiler P | Profiler C | Profiler Torch | E2E P | E2E C | E2E Torch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 / 4 / 10 / 16 / 128 / False | 1.780 | 1.800 | 623.315 | 379.410 | 262.870 | 665.940 |
| 2 | 8 / 4 / 10 / 16 / 256 / False | 1.900 | 1.970 | 620.325 | 302.290 | 293.665 | 703.100 |
| 3 | 8 / 4 / 10 / 16 / 512 / False | 2.220 | 2.200 | 621.485 | 339.565 | 284.910 | 713.345 |
| 4 | 8 / 4 / 10 / 64 / 128 / False | 1.760 | 1.830 | 646.318 | 291.545 | 293.600 | 639.245 |
| 5 | 8 / 4 / 10 / 64 / 256 / False | 1.900 | 1.950 | 640.557 | 291.015 | 330.520 | 619.050 |
| 6 | 8 / 4 / 10 / 64 / 512 / False | 2.250 | 2.191 | 619.254 | 312.910 | 288.415 | 634.470 |
| 7 | 8 / 4 / 10 / 128 / 128 / False | 1.770 | 1.830 | 614.444 | 292.185 | 280.305 | 632.425 |
| 8 | 8 / 4 / 10 / 128 / 256 / False | 1.890 | 1.960 | 637.866 | 303.440 | 290.055 | 622.900 |
| 9 | 8 / 4 / 10 / 128 / 512 / False | 2.200 | 2.210 | 609.023 | 525.200 | 298.350 | 640.770 |
| 10 | 8 / 4 / 10 / 8 / 128 / False | 1.770 | 1.820 | 687.032 | 286.700 | 293.640 | 620.515 |
| 11 | 12 / 4 / 8 / 16 / 128 / True | 2.310 | 2.360 | 868.294 | 319.690 | 315.110 | 863.985 |
| 12 | 1 / 1 / 10 / 16 / 128 / False | 1.500 | 1.530 | 759.872 | 273.675 | 252.095 | 667.840 |
| 13 | 64 / 8 / 10 / 16 / 128 / False | 5.111 | 5.061 | 588.281 | 296.890 | 286.480 | 661.225 |
| 14 | 512 / 32 / 64 / 64 / 128 / False | 6.221 | 6.191 | 904.038 | 333.345 | 280.345 | 995.695 |
| 15 | 2048 / 64 / 128 / 128 / 128 / False | 11.862 | 11.931 | 1670.081 | 313.600 | 306.930 | 1741.885 |
| 16 | 8192 / 256 / 1024 / 64 / 512 / False | 61.538 | 61.857 | 13509.001 | 347.250 | 362.165 | 46662.080 |
| 17 | 512 / 16 / 64 / 16 / 512 / False | 7.821 | 7.811 | 1656.298 | 333.940 | 310.665 | 1977.220 |
| 18 | 2048 / 128 / 1024 / 64 / 512 / False | 18.112 | 18.072 | 4050.996 | 292.590 | 293.675 | 4250.440 |
| 19 | 2048 / 128 / 1024 / 128 / 2048 / False | 45.165 | 45.425 | 13936.963 | 360.835 | 287.690 | 14198.185 |
| 20 | 512 / 32 / 64 / 128 / 2048 / False | 15.282 | 15.312 | 4722.717 | 319.650 | 299.725 | 4632.335 |
| 21 | 8192 / 512 / 1024 / 128 / 2048 / False | 176.421 | 176.351 | 75652.210 | 423.105 | 415.685 | 75587.640 |
| 22 | 2048 / 64 / 1024 / 64 / 512 / True | 23.683 | 23.533 | 4576.920 | 332.810 | 311.665 | 4558.860 |

### A2 paired medians (400 samples per side)

| Case | E2E P (us) | E2E C (us) | E2E speedup | E2E status | Event speedup | Event status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | 255.775 | 255.840 | 1.000x | noisy | 0.965x | noisy |
| 2 | 280.815 | 292.275 | 0.961x | noisy | 1.046x | noisy |
| 3 | 287.725 | 252.905 | 1.138x | noisy | 1.025x | noisy |
| 4 | 279.335 | 260.805 | 1.071x | noisy | 1.137x | noisy |
| 5 | 288.035 | 258.470 | 1.114x | noisy | 1.065x | noisy |
| 6 | 283.125 | 270.755 | 1.046x | noisy | 1.037x | noisy |
| 7 | 276.705 | 264.615 | 1.046x | noisy | 0.956x | noisy |
| 8 | 279.405 | 278.970 | 1.002x | noisy | 0.951x | noisy |
| 9 | 289.305 | 281.885 | 1.026x | noisy | 0.888x | noisy |
| 10 | 269.830 | 281.855 | 0.957x | noisy | 0.958x | noisy |
| 11 | 279.400 | 277.325 | 1.007x | noisy | 1.015x | noisy |
| 12 | 271.625 | 272.865 | 0.995x | noisy | 0.990x | noisy |
| 13 | 286.775 | 272.230 | 1.053x | noisy | 1.021x | noisy |
| 14 | 287.440 | 268.290 | 1.071x | noisy | 1.057x | noisy |
| 15 | 273.630 | 284.740 | 0.961x | noisy | 0.972x | noisy |
| 16 | 311.415 | 270.145 | 1.153x | noisy | 1.034x | noisy |
| 17 | 284.865 | 255.460 | 1.115x | noisy | 1.074x | noisy |
| 18 | 286.865 | 259.955 | 1.104x | noisy | 1.039x | noisy |
| 19 | 303.105 | 254.980 | 1.189x | noisy | 1.094x | noisy |
| 20 | 290.355 | 263.285 | 1.103x | noisy | 1.060x | noisy |
| 21 | 433.875 | 393.760 | 1.102x | noisy | 1.034x | noisy |
| 22 | 317.770 | 308.160 | 1.031x | noisy | 0.902x | noisy |

### A5 per-case medians (microseconds)

P = parent `899baf4`; C = current `d300d0f`; Torch = same-device reference.

| Case | T / R / M / block size / K / prefill | Profiler P | Profiler C | Profiler Torch | E2E P | E2E C | E2E Torch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 / 4 / 10 / 16 / 128 / False | 1.120 | 1.137 | 196.994 | 38.243 | 36.234 | 206.500 |
| 2 | 8 / 4 / 10 / 16 / 256 / False | 1.131 | 1.137 | 215.053 | 47.286 | 53.841 | 216.795 |
| 3 | 8 / 4 / 10 / 16 / 512 / False | 1.133 | 1.170 | 279.032 | 52.985 | 38.788 | 265.934 |
| 4 | 8 / 4 / 10 / 64 / 128 / False | 1.113 | 1.120 | 201.166 | 35.929 | 50.000 | 191.537 |
| 5 | 8 / 4 / 10 / 64 / 256 / False | 1.154 | 1.164 | 202.275 | 89.169 | 49.885 | 195.003 |
| 6 | 8 / 4 / 10 / 64 / 512 / False | 1.166 | 1.146 | 261.513 | 43.871 | 49.930 | 258.749 |
| 7 | 8 / 4 / 10 / 128 / 128 / False | 1.107 | 1.136 | 193.789 | 47.351 | 48.242 | 220.877 |
| 8 | 8 / 4 / 10 / 128 / 256 / False | 1.143 | 1.156 | 198.278 | 75.193 | 49.034 | 196.975 |
| 9 | 8 / 4 / 10 / 128 / 512 / False | 1.135 | 1.137 | 249.568 | 39.073 | 48.552 | 317.005 |
| 10 | 8 / 4 / 10 / 8 / 128 / False | 1.112 | 1.139 | 172.293 | 46.455 | 49.163 | 165.228 |
| 11 | 12 / 4 / 8 / 16 / 128 / True | 1.284 | 1.274 | 194.593 | 42.448 | 49.730 | 210.486 |
| 12 | 1 / 1 / 10 / 16 / 128 / False | 0.877 | 0.880 | 161.039 | 36.830 | 51.707 | 153.715 |
| 13 | 64 / 8 / 10 / 16 / 128 / False | 1.784 | 1.784 | 282.780 | 31.532 | 50.972 | 274.481 |
| 14 | 512 / 32 / 64 / 64 / 128 / False | 2.492 | 2.521 | 512.331 | 38.603 | 57.346 | 502.499 |
| 15 | 2048 / 64 / 128 / 128 / 128 / False | 4.831 | 4.933 | 1138.196 | 47.676 | 51.988 | 1206.074 |
| 16 | 8192 / 256 / 1024 / 64 / 512 / False | 18.267 | 18.287 | 13595.946 | 59.860 | 58.663 | 13605.619 |
| 17 | 512 / 16 / 64 / 16 / 512 / False | 3.478 | 3.539 | 1116.328 | 53.880 | 55.097 | 1184.487 |
| 18 | 2048 / 128 / 1024 / 64 / 512 / False | 6.971 | 7.247 | 3693.114 | 46.660 | 54.621 | 3706.606 |
| 19 | 2048 / 128 / 1024 / 128 / 2048 / False | 13.356 | 13.415 | 11869.163 | 50.752 | 61.277 | 11898.282 |
| 20 | 512 / 32 / 64 / 128 / 2048 / False | 4.601 | 4.712 | 3218.129 | 41.557 | 50.626 | 3201.979 |
| 21 | 8192 / 512 / 1024 / 128 / 2048 / False | 97.872 | 98.431 | 72114.903 | 139.419 | 142.143 | 72454.977 |
| 22 | 2048 / 64 / 1024 / 64 / 512 / True | 12.466 | 12.214 | 3602.631 | 82.153 | 48.513 | 3523.166 |

### A5 paired medians (400 samples per side)

| Case | E2E P (us) | E2E C (us) | E2E speedup | E2E status | Event speedup | Event status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | 33.671 | 38.638 | 0.871x | noisy | 0.815x | noisy |
| 2 | 33.060 | 35.579 | 0.929x | noisy | 1.038x | noisy |
| 3 | 32.543 | 35.344 | 0.921x | noisy | 0.997x | noisy |
| 4 | 34.337 | 36.054 | 0.952x | noisy | 1.023x | noisy |
| 5 | 35.468 | 31.547 | 1.124x | noisy | 1.204x | noisy |
| 6 | 38.743 | 26.585 | 1.457x | noisy | 1.199x | noisy |
| 7 | 38.248 | 26.064 | 1.467x | noisy | 1.207x | noisy |
| 8 | 39.014 | 23.370 | 1.669x | noisy | 1.227x | noisy |
| 9 | 38.047 | 23.345 | 1.630x | noisy | 1.208x | noisy |
| 10 | 38.543 | 23.124 | 1.667x | noisy | 1.263x | noisy |
| 11 | 39.349 | 32.133 | 1.225x | noisy | 1.142x | noisy |
| 12 | 38.012 | 33.896 | 1.121x | noisy | 1.174x | noisy |
| 13 | 40.225 | 36.831 | 1.092x | noisy | 1.229x | noisy |
| 14 | 37.572 | 36.700 | 1.024x | noisy | 1.115x | noisy |
| 15 | 39.194 | 32.759 | 1.196x | noisy | 1.096x | noisy |
| 16 | 50.947 | 46.965 | 1.085x | noisy | 1.016x | noisy |
| 17 | 39.835 | 31.817 | 1.252x | noisy | 1.196x | noisy |
| 18 | 40.305 | 34.387 | 1.172x | noisy | 1.208x | noisy |
| 19 | 45.929 | 40.145 | 1.144x | noisy | 1.084x | noisy |
| 20 | 37.947 | 30.796 | 1.232x | noisy | 1.160x | noisy |
| 21 | 135.813 | 128.543 | 1.057x | noisy | 1.008x | noisy |
| 22 | 45.729 | 41.547 | 1.101x | noisy | 1.182x | noisy |
