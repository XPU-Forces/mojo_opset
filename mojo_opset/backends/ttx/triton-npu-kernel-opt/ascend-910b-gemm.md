# Ascend 910B — INT8 GEMM Optimization Reference

## Kernel Architecture

### Current: Autotuned Persistent Kernel with Fused Dequantization

Single unified kernel `_int8_gemm_dequant_kernel` with:
- `@triton.autotune` selecting from 9 configs (tile sizes × compiler flags)
- Persistent scheduling: `grid=(NUM_CUBE_CORES,)` = `(24,)`
- Fused epilogue: `int32 → fp32 → scale × scale [+ bias] → output_dtype`
- `tile_mix_vector_loop=2` on medium/small tile configs for cube/vector interleaving

### Previous: Heuristic-selected dual kernels (deprecated)
- `_int8_gemm_persistent` + `_int8_gemm_simple` with manual `select_config()`
- Replaced by autotune for better maintainability and equal/better performance

## Data Layout

```
A:   (M, K) int8, row-major, stride=(K, 1)
B_T: (N, K) int8, row-major, stride=(K, 1)  — transposed from original B(K,N)
C:   (M, N) fp16, row-major
```

Both A and B_T have K as stride-1 dimension → maximum DMA bandwidth.

## Tile Config Tuning Table

Based on 99-configuration sweep across M=1~8192, N=2048~8192, K=2048~8192.

| FLOPS Range | Tile (BM×BN×BK) | Mode | Measured TFLOPS | %QMM_ND | %QMM_NZ |
|-------------|-----------------|------|-----------------|---------|---------|
| > 2^34 (17B) | 128×256×512 | Persistent | 560~620 | 106~121% | 91~96% |
| 2^30 ~ 2^34, M≥256 | 128×256×256 | Persistent | 120~560 | 50~92% | 43~75% |
| 2^26 ~ 2^30, M≥128 | 128×128×256 | Persistent/Simple | 60~420 | 20~51% | 17~43% |
| < 2^26, M≥64 | 64×128×256 | Simple | 12~100 | 12~27% | 10~22% |
| M < 64 | 64×64×256 | Simple | 0.2~12 | 10~20% | 9~17% |

## Heuristic Selection Code

```python
def select_config(M: int, N: int, K: int):
    flops = 2 * M * N * K
    if flops > (1 << 34):
        return 128, 256, 512, True       # persistent, BK=512
    if flops > (1 << 30) and M >= 256:
        return 128, 256, 256, True       # persistent, BK=256
    if flops > (1 << 26) and M >= 128:
        BM, BN, BK = 128, 128, 256
        use_p = M >= 256 and N >= 256
        return BM, BN, BK, use_p
    if M >= 64:
        return 64, 128, 256, False       # simple
    return 64, 64, 256, False             # simple, smallest tile
```

## Optimization Impact Chain (Cumulative)

Baseline: naive INT8 GEMM, FP16 accumulator, no mask elimination, B not transposed.

| Step | Optimization | 4096³ TFLOPS | Cumulative Gain |
|------|-------------|-------------|-----------------|
| 0 | Baseline (naive) | ~80 | — |
| 1 | INT8 dot + INT32 acc | ~120 | +50% |
| 2 | B transposed layout | ~155 | +94% |
| 3 | tl.multibuffer | ~190 | +138% |
| 4 | Large BLOCK_K (256→512) | ~210 | +163% |
| 5 | Host-side padding | ~220 | +175% |
| 6 | Persistent kernel | ~560 | +600% |
| 7 | Heuristic tuning | ~560 | (same peak, better across sizes) |
| 8 | Fused dequant epilogue | ~542 | (same, no Python overhead) |
| 9 | Autotune + vmix2 configs | ~546 | (better coverage, +17% on medium tiles) |

## Latest Performance Snapshot (2026-03-27)

Fused dequant kernel vs `npu_quant_matmul` (ND and NZ formats):

| M | K | N | Triton (T) | QMM_ND (T) | QMM_NZ (T) | vs_ND | vs_NZ |
|---|---|---|-----------|-----------|-----------|-------|-------|
| 1024 | 4096 | 4096 | 183 | 429 | 489 | 42.7% | 37.5% |
| 2048 | 4096 | 4096 | 377 | 501 | 574 | 75.4% | 65.7% |
| 4096 | 4096 | 4096 | 546 | 522 | 602 | **104.7%** | 90.7% |
| 8192 | 4096 | 4096 | 577 | 543 | 632 | **106.4%** | 91.4% |
| 4096 | 8192 | 4096 | 591 | 538 | 622 | **109.9%** | 95.0% |
| 8192 | 8192 | 8192 | 601 | 556 | 632 | **108.2%** | **95.1%** |

Key: Triton beats QMM_ND for M≥4096. Reaches 91-95% of QMM_NZ (hardware NZ format).

## NZ Format Investigation Results

NZ (FRACTAL_NZ, format=29) for INT8:
- Memory layout: `(M, N)` → `(N//32, ceil(M/16)*16, 32)` row-major
- Address formula: `nz_offset(r, c) = (c // 32) * M_pad * 32 + r * 32 + (c % 32)`

| Approach | Result |
|----------|--------|
| NZ scattered pointers in Triton | UB overflow (770KB > 192KB limit) |
| 32-wide K sub-tile iteration | Compiles correctly, but 50x slower (Cube utilization <2%) |
| nd2nz compiler flag | Not exposed via Triton API |
| NZ for npu_quant_matmul | +16% speedup (517T → 600T), recommended for native path |

## Benchmark Template

```python
def manual_bench(fn, warmup=10, reps=100):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / reps * 1000

# Always compare against: npu_quant_matmul (ND), npu_quant_matmul (NZ)
scale = torch.tensor([1.0], device='npu', dtype=torch.float32)
ms_qmm = manual_bench(lambda: torch_npu.npu_quant_matmul(a, b, scale))
b_nz = torch_npu.npu_format_cast(b, 29)
ms_qmm_nz = manual_bench(lambda: torch_npu.npu_quant_matmul(a, b_nz, scale))
```

## Techniques Attempted but Rejected

| Technique | Why Rejected |
|-----------|-------------|
| Loop peeling (if-else K tail) | NPU compiler crash |
| FP16 accumulator for int8 dot | NaN output |
| num_stages > 1 | NPU ignores; no effect |
| dot_pad_only_k compile hint | ~90% slowdown on GEMM (all BK sizes, not just 512) |
| NZ format in Triton | UB overflow / 50x slowdown |
| grouped_launch_diagonal | No improvement over linear tile ordering for GEMM |
| tl.multibuffer(x, 3) triple buffer | UB overflow on all tile configs |
| make_block_ptr / tl.advance | Inconsistent; worse at M=1024 (-22%), M=4096 (-14%) |
| Tile swizzle (GROUP_M) | Degrades at large M (-13% at M=8192 with GROUP_M=8) |
| Split-K (parallel K reduction) | -32% at M=1024 with split4; marginal for tiny M |
| Scale prefetch / multibuffer | No improvement; load latency hidden by pipeline |
| enable_ubuf_saving | Hurts large M (-12% at M=4096) |
| unit_flag | Fails at M=128; marginal elsewhere |
| enable_hivm_auto_cv_balance | Inconsistent results |
| set_workspace_multibuffer=4 | No measurable improvement |
| limit_auto_multi_buffer_only_for_local_buffer=False | **aicore timeout** (for attention patterns only) |
| limit_auto_multi_buffer_of_local_buffer="no-limit" | UB overflow on large tiles; neutral on medium |
| tile_mix_cube_loop=2 | Requires hanging flag above; unusable for GEMM |
