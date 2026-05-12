# Optimization Log

Chronological record of optimization experiments on the ttx (Triton NPU) backend.
**Append new entries at the bottom** when discovering new techniques or testing new kernel types.

## Format

```
### YYYY-MM-DD | Kernel: <type> | SoC: <model>
**Technique:** <description>
**Result:** <measured impact or failure reason>
**Action:** Added as OPT-XX / Rejected / Updated existing entry
```

---

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Native INT8 tl.dot with INT32 accumulator
**Result:** Correct output. FP16 accumulator produces NaN.
**Action:** Added as OPT-03

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** B transposed layout (N,K) row-major
**Result:** +29% (275T → 357T @ 4096³)
**Action:** Added as OPT-02

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** tl.multibuffer(x, 2) double buffering
**Result:** +15-25% across sizes, overlaps HBM load with Cube compute
**Action:** Added as OPT-04

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** BLOCK_K=512 (from 128)
**Result:** +10-20% for large matrices (fewer K-loop iterations)
**Action:** Added as OPT-05

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Loop peeling for K tail
**Result:** REJECTED — NPU compiler crash (Triton error)
**Action:** Replaced with host-side padding (OPT-06)

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Persistent kernel (grid=24 cores, internal tile loop)
**Result:** +57% (357T → 560T @ 4096³). Dominant optimization for large matrices.
**Action:** Added as OPT-01

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Heuristic tile selection (5-tier based on FLOPS + M)
**Result:** Eliminates autotune overhead. Same peak perf, better cross-size coverage.
**Action:** Added as OPT-10

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** tl.compile_hint("dot_pad_only_k")
**Result:** REJECTED — aicore timeout with BLOCK_K=512. Only safe with BK≤256.
**Action:** Added to Known Pitfalls

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** NZ (FRACTAL_NZ) format for B matrix in Triton
**Result:** REJECTED for Triton (UB overflow / 50x slowdown). +16% for npu_quant_matmul.
**Action:** Added as OPT-12 (for native operators only). Detailed in [ascend-910b-gemm.md](ascend-910b-gemm.md).

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** BiShengIR flags (enable_ubuf_saving, workspace_multibuffer=4, cv_balance)
**Result:** No measurable improvement on GEMM kernel
**Action:** Added to "Flags That Do NOT Help" in SKILL.md

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Full M/N/K sweep (99 configs: M=1~8192, N/K=2048~8192)
**Result:** Data used to build select_config() heuristic. Peak: 620T (121% QMM_ND).
**Action:** Tuning table in [ascend-910b-gemm.md](ascend-910b-gemm.md)

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->

### 2026-05-08 | Kernel: Conformer Attention Varlen | SoC: Ascend 910B2C, 24 AICore

**Technique:** Replaced dense BSHD + padding-mask path with a THD varlen encoder API. The kernel uses cumulative query/KV lengths and direct window masking by absolute query position.
**Result:** Accuracy passed for bf16/float32, cache/no-cache encoder cases, and uneven lengths. Perf profiler device latency: 64.9472 us / 173.7712 us / 108.5136 us / 23.5040 us for the four perf cases. Effective throughput by visible-window FLOPs peaked at 9.641 TFLOP/s; normalized to a 354 TFLOP/s bf16 peak, current best observed MFU is about 2.72%.
**Action:** Added `MojoConformerSlidingWindowAttention`, unified TTX varlen backend, and updated accuracy/perf tests.

---

### 2026-05-09 | Kernel: Conformer Attention Varlen | SoC: Ascend 910B2C, 24 AICore

**Technique:** OPT-MB — Enable multibuffering (`multibuffer=True` + `tl.multibuffer(k_t, 2)` + `tl.multibuffer(v, 2)`) on the persistent conformer attention kernel. Overlaps HBM→UB DMA transfers with Cube dot-product compute in the inner KV-block loop.
**Result:** Expected +15-25% throughput from DMA/compute overlap.

**Technique:** OPT-TILE — Rebalanced tile sizes for double-buffered UB budget (192 KB). D=64: BM=128, BN=128 (AI=85, UB=181KB). D=128: BM=128, BN=48 (AI=55, UB=173KB). D=96: BM=192, BN=48 (AI=64, UB=186KB). fp32: BM=64, BN=64 (no multibuffer).
**Result:** Larger BN (from 64→128 for D=64, from 32→48 for D=128) reduces KV-loop iterations and increases AI.

**Technique:** OPT-MASK — Precompute fp32 query positions (`q_abs_f32`, `left_bound_f32`, `right_bound_f32`) once outside the KV-block loop. Eliminates repeated i32→fp32 casts and arithmetic per KV block.
**Result:** Reduces per-KV-block vector operations.

**Technique:** OPT-MASK-LOOKUP (optional, env-controlled) — Pre-computed auxiliary mask lookup using triu/tril slices (SWA pattern) as an alternative to inline fp32 comparisons. Controlled by `MOJO_CONFORMER_MASK_LOOKUP=1`.
**Result:** Experimental path; fp32 comparison path (default) already avoids int32 scalarization.

**MFU Formula (from core definition):**
```
core_FLOPs  = 4 * head_dim * num_heads * sum_batch(Q_b * KV_b)   // dense Q×KV matmul
MFU         = core_FLOPs / (peak_bf16_TFLOPS * device_time_us * 1e-6 * 1e12)
```
where `peak_bf16_TFLOPS ≈ 320` for Ascend 910B2C (24 AICores, Cube units).
This counts the full dense matmul FLOPs that the core reference operator performs (the kernel's window-skipping is an optimization that reduces actual FLOPs).

**Action:** Updated `_conformer_sliding_window_attention_kernel` with multibuffering, optimized tiles, hoisted mask precomputation; added optional mask-lookup path.
