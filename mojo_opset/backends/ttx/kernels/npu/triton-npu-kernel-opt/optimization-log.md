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
**Action:** Added as OPT-12 (for native operators only). Detailed in ascend-910b-gemm.md.

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** BiShengIR flags (enable_ubuf_saving, workspace_multibuffer=4, cv_balance)
**Result:** No measurable improvement on GEMM kernel
**Action:** Added to "Flags That Do NOT Help" in SKILL.md

### 2026-03-26 | Kernel: INT8 GEMM | SoC: Ascend 910B2C

**Technique:** Full M/N/K sweep (99 configs: M=1~8192, N/K=2048~8192)
**Result:** Data used to build select_config() heuristic. Peak: 620T (121% QMM_ND).
**Action:** Tuning table in ascend-910b-gemm.md

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
