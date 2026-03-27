# Ascend 910B — INT8 GEMM Optimization Reference

## Kernel Architecture

Two kernel variants, selected by heuristic:

### Persistent Kernel (`_int8_gemm_persistent`)
- Grid: `(NUM_CUBE_CORES,)` = `(24,)`
- Each core loops over assigned output tiles
- Best for: large matrices (FLOPS > 2^30)

### Simple Kernel (`_int8_gemm_simple`)
- Grid: `(tiles_m, tiles_n)` — standard 2D launch
- Best for: small matrices or few tiles

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
| dot_pad_only_k with BK=512 | aicore timeout (hardware hang) |
| NZ format in Triton | UB overflow / 50x slowdown |
| grouped_launch_diagonal | No improvement over linear tile ordering for GEMM |
