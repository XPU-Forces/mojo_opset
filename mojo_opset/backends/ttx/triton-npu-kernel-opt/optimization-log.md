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

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** tile_mix_vector_loop=2 (cube/vector interleaving)
**Result:** +3-17% for medium tile configs (BM128/BN128/BK256). +17% at M=4096 (308T→360T). Neutral for BM128/BN256/BK512 (large tiles). vmix=8 degrades at large M.
**Action:** Added as OPT-13. Added to autotune configs.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** limit_auto_multi_buffer_of_local_buffer="no-l0c"
**Result:** +1-2% consistent improvement (M=1024: +2.4%, M=2048: +2.2%). "no-ub" is invalid compiler option.
**Action:** Added as OPT-14.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** dot_pad_only_k compile hint on GEMM
**Result:** REJECTED — catastrophic ~90% slowdown (e.g., M=4096: 454T→100T with BK=256). Previously only known to fail with BK=512; now confirmed harmful at all block sizes for GEMM.
**Action:** Updated SKILL.md pitfall. Never use for GEMM.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** tl.multibuffer(x, 3) (triple buffering)
**Result:** REJECTED — UB overflow for all tested tile configs. NPU cannot fit 3× operand buffers.
**Action:** Added to pitfalls.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** tl.make_block_ptr / tl.advance (structured block pointers)
**Result:** REJECTED — inconsistent results. Worse at M=1024 (162T vs 207T) and M=4096 (466T vs 540T). Regular pointer arithmetic is already well-optimized by compiler.
**Action:** Added to "Flags That Do NOT Help" table.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** Tile swizzle (GROUP_M-based tile ordering for L2 cache reuse)
**Result:** REJECTED — marginal positive at medium M (GROUP_M=4: +1% at M=2048), but degrades at large M (GROUP_M=8: -13% at M=8192, 496T vs 568T).
**Action:** Not adopted. NPU L2 cache is large enough (192MB) that linear tile ordering already provides good reuse.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** Split-K (partition K across cores, atomic reduce)
**Result:** REJECTED — marginal for tiny M (M=128: 26.4T vs 25.8T = +2%). Hurts at M=1024 split4 (146T vs 215T = -32%). Python overhead of extra output tensor + atomic reduction.
**Action:** Not adopted. Persistent kernel handles small-M better.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** Scale prefetch + tl.multibuffer(scale, 2)
**Result:** REJECTED — no improvement. Scale vectors are small (M or N elements of float32), load latency fully hidden by K-loop pipeline drain.
**Action:** Not adopted.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** enable_ubuf_saving=True (UB memory optimization)
**Result:** Mixed — +7% at M≤1024 (209T→217T), but **-12% at M=4096** (543T→478T when combined with vmix2). Size-dependent behavior makes it unsuitable for unconditional use.
**Action:** Updated SKILL.md flag table. Added to pitfalls.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** Comprehensive compiler flag sweep (unit_flag, enable_hivm_auto_cv_balance, set_workspace_multibuffer=4)
**Result:** unit_flag: FAIL at M=128 + vmix2, marginal elsewhere. cv_balance: inconsistent. ws_mb=4: marginal. No flag provides consistent improvement beyond vmix2 and no-l0c.
**Action:** Updated "Flags That Do NOT Help" table.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** limit_auto_multi_buffer_only_for_local_buffer=False (full GM workspace CV pipelining)
**Result:** REJECTED — **aicore timeout** (hardware hang). This flag is for cube→vector→cube patterns (flash attention), not simple cube-loop→vector-epilogue (GEMM). Documentation in best_practice.md confirms this is only for alternating cube/vector kernels.
**Action:** Added as pitfall #11. Added CV pipelining section to SKILL.md.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** limit_auto_multi_buffer_of_local_buffer="no-limit" (allow L0C multi-buffering)
**Result:** REJECTED — Compilation failure on large tiles (BM128/BN256/BK512). Works on medium tiles (BM128/BN128/BK256) but no improvement. Default "no-l0c" is optimal for GEMM.
**Action:** Added as pitfall #12.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** tile_mix_cube_loop=2 (cube sub-tiling)
**Result:** REJECTED — Requires limit_auto_multi_buffer_only_for_local_buffer=False to take effect, which causes aicore timeout on GEMM. Only applicable to attention-style kernels.
**Action:** Added as pitfall #13.

### 2026-03-27 | Kernel: INT8 GEMM Dequant | SoC: Ascend 910B2C

**Technique:** Study of compile_option.md and best_practice.md for new optimization avenues
**Result:** No new applicable optimizations found for GEMM beyond tile_mix_vector_loop=2. CV pipelining flags designed for attention kernels. mayDiscretememaccess, bitwise_mask, and hivm.tile_mix_cube_num are for other kernel types (where, layernorm, multi-dot patterns).
**Action:** Added both documents as references in SKILL.md. Updated compiler flag documentation.

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
