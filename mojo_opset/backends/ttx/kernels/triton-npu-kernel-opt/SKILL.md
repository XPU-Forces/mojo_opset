---
name: triton-npu-kernel-opt
description: >-
  Triton kernel optimization guide for Ascend NPU (910B/910C). Use when writing
  or optimizing Triton kernels targeting NPU backend, including GEMM, attention,
  normalization, or any compute-intensive kernel. Covers hardware constraints,
  proven optimization patterns, compiler flags, tile tuning, and known pitfalls
  specific to the ttx backend with Ascend SoCs.
---

# Triton NPU Kernel Optimization Guide

Backend: **ttx** (Triton for Ascend NPU)

## SoC Hardware Specs

### Ascend 910B2C

| Resource | Value |
|----------|-------|
| Cube Cores (AI Cores) | 24 |
| Vector Cores | 48 |
| Unified Buffer (UB) per core | 192 KB (1,572,864 bits) |
| L2 Cache | 192 MB |
| HBM | 62 GB |
| INT8 peak throughput | ~512 TFLOPS |
| FP16 peak throughput | ~320 TFLOPS |

Get core counts dynamically:

```python
from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
num_cube = get_num_cores("cube")   # 24 on 910B
num_vec  = get_num_cores("vector") # 48 on 910B
```

## Optimization Catalog

Each optimization is tagged with **applicability** (which kernel types benefit) and **measured impact**.

### OPT-01: Persistent Kernel

**Applies to:** GEMM, GroupGEMM, any kernel with many output tiles
**Impact:** +57% on GEMM 4096³

Standard Triton launches one program per tile. On NPU, dispatch overhead is ~6µs per tile. For 512 tiles that's 3ms overhead — often exceeding compute time.

Persistent kernel: launch `grid=(NUM_CUBE_CORES,)`, each core loops over assigned tiles.

```python
@triton.jit
def _persistent_kernel(..., NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1
    tile_id = start_pid - NUM_SMS
    for _ in range(0, tiles_per_sm):
        tile_id += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        # ... compute tile ...
```

**Decision rule:** Use persistent when `num_tiles > 2 * NUM_CUBE_CORES`.

### OPT-02: B Transposed Layout (for GEMM)

**Applies to:** GEMM, GroupGEMM
**Impact:** +29% on GEMM 4096³

Store weight B as `(N, K)` row-major instead of `(K, N)`. Both A and B_T then have K as the stride-1 (contiguous) dimension, doubling DMA efficiency.

```python
def prepare_b(b):
    """B(K,N) → B_T(N,K) row-major + padding. Call once for static weights."""
    return b.T.contiguous()  # + padding to block multiples
```

### OPT-03: Native INT8 dot + INT32 Accumulator

**Applies to:** INT8 GEMM
**Impact:** Enables correct INT8 computation; avoids FP16 intermediate loss

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
acc = tl.dot(a_int8, tl.trans(bt_int8), acc=acc)  # INT8×INT8 → INT32
```

**Pitfall:** `tl.dot(int8, int8)` without `acc=tl.int32` may produce NaN. Always specify accumulator dtype.

### OPT-04: tl.multibuffer (Double Buffering)

**Applies to:** All kernels with K-loop or sequential block loads
**Impact:** +15-25% (overlaps HBM load with Cube compute)

```python
a = tl.load(a_ptrs)
bt = tl.load(bt_ptrs)
tl.multibuffer(a, 2)   # NPU-specific: hint compiler to double-buffer
tl.multibuffer(bt, 2)
acc = tl.dot(a, tl.trans(bt), acc=acc)
```

Also enable via launch kwarg: `multibuffer=True`.

**Pitfall:** `num_stages > 1` (GPU-style software pipelining) is NOT supported on NPU. Use `tl.multibuffer` instead.

### OPT-05: Large BLOCK_K

**Applies to:** GEMM, attention QK dot
**Impact:** +10-20% (fewer K-loop iterations → less per-iteration overhead)

NPU K-loop has ~1µs fixed overhead per iteration. Increasing BLOCK_K from 128 to 512 reduces iterations 4x.

| BLOCK_K | K-iters (K=4096) | Overhead |
|---------|------------------|----------|
| 128     | 32               | ~32µs    |
| 256     | 16               | ~16µs    |
| 512     | 8                | ~8µs     |

**Constraint:** Total UB usage must fit 192KB. For INT8: `(BM + BN) * BK * 1 byte + BM * BN * 4 bytes` (with multibuffer ×2). Typical max for INT8: `BM=128, BN=256, BK=512` → ~192KB.

### OPT-06: Host-side Padding (Eliminate Masks)

**Applies to:** All kernels
**Impact:** +5-10% (removes branch/mask overhead in inner loop)

Pad inputs to block multiples on the host (Python side) so the kernel never needs `tl.where` or mask predicates.

```python
pM = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
if pM != M:
    a_pad = torch.zeros(pM, K, device=a.device, dtype=a.dtype)
    a_pad[:M, :K] = a
    a = a_pad
```

**Pitfall:** Loop peeling (if-else for tail K iteration) crashes the NPU compiler. Always pad K instead.

### OPT-07: Narrower Dtype Loads

**Applies to:** All kernels
**Impact:** 2x bandwidth savings (INT8 vs FP16)

Keep data in narrowest dtype as long as possible. INT8 loads use half the HBM bandwidth of FP16. Cast to wider types only in epilogue.

### OPT-08: Epilogue Fusion

**Applies to:** GEMM, normalization, activation
**Impact:** Avoids extra kernel launch + memory round-trip

Fuse post-compute operations (scale, cast, bias add, activation) into the same kernel:

```python
# All in registers, no extra HBM access
c = (acc.to(tl.float32) * alpha).to(tl.float16)
tl.store(c_ptrs, c)
```

### OPT-09: Hybrid Dispatch

**Applies to:** Kernels that serve a wide range of input sizes
**Impact:** Optimal across all M/N/K ranges

Use persistent kernel for large problems, simple 2D grid for small ones:

```python
if use_persistent:
    _persistent_kernel[(min(NUM_CORES, num_tiles),)](...)
else:
    _simple_kernel[(tiles_m, tiles_n)](...)
```

### OPT-10: Heuristic Tuning (Replace Autotune)

**Applies to:** Performance-critical kernels in inference
**Impact:** Eliminates autotune warmup cost; deterministic first-call performance

After sweeping M/N/K parameter space, encode optimal configs as a lookup table:

```python
def select_config(M, N, K):
    flops = 2 * M * N * K
    if flops > (1 << 34):
        return 128, 256, 512, True   # persistent
    if flops > (1 << 30) and M >= 256:
        return 128, 256, 256, True
    # ... more tiers ...
```

**When to use autotune instead:** Training kernels, or when shape space is unpredictable.

### OPT-11: Weight Preprocessing (prepare_b)

**Applies to:** Inference GEMM (static weights)
**Impact:** Removes per-forward transpose + pad cost

Pre-transpose and pad weight matrices once at model load time. Pass preprocessed weights to the kernel.

### OPT-12: NZ Format (FRACTAL_NZ) for Native Operators

**Applies to:** `npu_quant_matmul` and other torch_npu native operators
**Impact:** +16% on npu_quant_matmul (600T vs 517T @ 4096³)

```python
b_nz = torch_npu.npu_format_cast(b.contiguous(), 29)  # format 29 = FRACTAL_NZ
out = torch_npu.npu_quant_matmul(a, b_nz, scale)
```

INT8 NZ memory layout: `(M, N)` → `(N//32, ceil(M/16)*16, 32)` row-major.
Conversion cost: 0.026ms for 4096×4096 (one-time).

**Critical: NZ format is NOT usable in Triton kernels:**
- NZ address pattern causes UB overflow (770KB > 192KB) with BLOCK_K > 32
- Forcing 32-wide K steps → Cube Core utilization <2%, 50x slowdown
- `nd2nz-on-vector` compiler flag is not exposed via Triton API
- Triton kernels should use ND (row-major) data layout

## Compiler Flags & Hints

### Supported Launch Kwargs

| Flag | Effect | Recommendation |
|------|--------|----------------|
| `multibuffer=True` | Enable auto multi-buffering | Always use |
| `num_stages=1` | Pipeline stages | Keep at 1 (NPU ignores >1) |
| `num_warps=4` | Warp count | Default 4 works well |
| `enable_ubuf_saving=True` | UB memory saving | Marginal effect, safe to enable |
| `set_workspace_multibuffer=N` | Workspace buffer count | 2 (default) is optimal |

### tl.compile_hint

```python
tl.compile_hint(..., "dot_pad_only_k")   # Pad only K dim for dot
tl.compile_hint(..., "tile_cube_loop")   # Tile-level cube loop
```

**Pitfall:** `dot_pad_only_k` is INCOMPATIBLE with `BLOCK_K=512`. Causes `aicore timeout`. Only use with `BLOCK_K ≤ 256`.

### Flags That Do NOT Help (Tested)

| Flag | Result |
|------|--------|
| `num_stages > 1` | Ignored by NPU compiler |
| `enable_hivm_auto_cv_balance` | No measurable effect |
| `unit_flag` | No measurable effect |
| `tile_mix_vector_loop` / `tile_mix_cube_loop` | No improvement for GEMM |

## UB Budget Calculator

```
UB_total = 192 KB = 196,608 bytes

Per-tile UB (with multibuffer=2):
  A tile:   BLOCK_M × BLOCK_K × elem_size × 2
  B tile:   BLOCK_N × BLOCK_K × elem_size × 2
  Acc:      BLOCK_M × BLOCK_N × 4 bytes (int32/fp32)

Example INT8 GEMM (BM=128, BN=256, BK=512):
  A: 128 × 512 × 1 × 2 = 128 KB
  B: 256 × 512 × 1 × 2 = 256 KB  ← exceeds if both double-buffered
  Acc: 128 × 256 × 4 = 128 KB
  → Compiler auto-manages, but may fail if too large
```

When UB overflows: reduce BLOCK_K or BLOCK_N, or disable multibuffer for one operand.

## Known Pitfalls (NPU-Specific)

1. **Loop peeling crashes compiler:** No if-else for K tail iterations. Use host-side padding.
2. **Zero strides → MLIR error:** `M=1` with modulo arithmetic can produce zero strides. Use explicit clamping.
3. **arange with int8:** `tl.arange(..., dtype=int8)` not supported. Create as int32 then cast.
4. **NZ format in Triton:** Cannot use. See OPT-12 for details.
5. **dot_pad_only_k + large BLOCK_K:** Causes hardware timeout. Only safe with BK ≤ 256.
6. **Corrupted compiler cache:** After compilation errors, clear `/root/.triton/cache/` before retrying.

## Updating This Skill

**This skill is a living document.** When generating or optimizing new kernel types (attention, normalization, convolution, etc.) on this backend, follow this protocol:

### When to Update

Update this skill whenever you discover:
- A new optimization technique that applies to NPU kernels
- A new compiler flag/hint that works (or doesn't work)
- A new hardware constraint or UB budget formula
- A new pitfall or compiler bug workaround
- Performance data for a new kernel type

### How to Update

1. **Add the new technique** to the Optimization Catalog section with:
   - Tag: `OPT-XX` (next sequential number)
   - Applicability: which kernel types benefit
   - Measured impact: % improvement with test conditions
   - Code example
   - Any pitfalls discovered

2. **Add kernel-type-specific notes** to the appropriate reference file:
   - GEMM patterns → [ascend-910b-gemm.md](ascend-910b-gemm.md)
   - New kernel type → create `ascend-910b-{kernel_type}.md`

3. **Update the optimization log** in [optimization-log.md](optimization-log.md) with:
   - Date, kernel type, technique, result

4. **Update Known Pitfalls** if new compiler issues are found.

### Reference Files

- [ascend-910b-gemm.md](ascend-910b-gemm.md) — Detailed GEMM optimization results and tuning tables
- [optimization-log.md](optimization-log.md) — Chronological log of all optimization experiments
