# Best Practices For Triton Kernels On Ascend NPU

This file is a distilled repo-local tuning guide. It intentionally avoids
duplicating the broader `guides/platform/` material. Use it for practical
kernel writing patterns that show up repeatedly in `mojo_opset`.

Primary audience:

- developers porting a Triton GPU kernel to Ascend NPU
- developers adding a new `ttx` kernel in this repo

## Scope

Use this file for:

- scheduling and tiling choices that are usually better on NPU than on GPU
- common code-shape rewrites seen in existing optimized NPU kernels
- tail handling and vectorization pitfalls

Do not use this file as a replacement for:

- [../../SKILL.md](../../SKILL.md) for the full delivery workflow
- [../platform/index.md](../platform/index.md) for the broader Triton-Ascend guides
- [ascend-910b-gemm.md](ascend-910b-gemm.md) for GEMM-specific tuning data

## 1. Launch fewer logical programs

Ascend NPU is much more sensitive to launch and scheduling overhead than a
typical Triton GPU target. A direct GPU-style launch often emits far more
logical programs than the number of physical compute cores, which can dominate
runtime.

Default rule:

- prefer one-dimensional or persistent scheduling when the original GPU launch
  creates many tiny logical programs
- make each program do more work before increasing grid size

### Example pattern: fold one launch dimension into an inner loop

Instead of launching one program per batch row and one per K tile, launch fewer
programs and iterate over the K dimension inside the kernel.

```python
@triton.jit
def gather_dim1_kernel(
    x_ptr,
    idx_ptr,
    out_ptr,
    stride_xb, stride_xc,
    stride_ib, stride_ik,
    stride_ob, stride_ok,
    B, K,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_idx < B

    for k_start in range(0, K, BLOCK_K):
        ks = tl.arange(0, BLOCK_K)
        k_mask = ks < K - k_start

        idx_off = b_idx[:, None] * stride_ib + (k_start + ks)[None, :] * stride_ik
        col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)

        x_off = b_idx[:, None] * stride_xb + col_idx * stride_xc
        x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)

        out_off = b_idx[:, None] * stride_ob + (k_start + ks)[None, :] * stride_ok
        tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)
```

Typical launch change:

```python
grid = (triton.cdiv(B, BLOCK_B),)
```

Use this rewrite when:

- the original GPU kernel had a second launch dimension mainly for convenience
- each program does too little work
- runtime is sensitive to dispatch overhead

## 2. Prefer host-side padding over loop peeling

For many NPU kernels, special-case tail handling inside the hot loop is less
robust than padding on the host.

Prefer:

- padding `M`, `N`, or `K` to block multiples before launch
- simple masked stores or fully mask-free inner loops when possible

Avoid:

- K-tail loop peeling with separate fast/slow loop bodies
- complicated branch-heavy inner loops

Reason:

- compiler stability is better
- inner loops stay regular
- persistent schedules become easier to reason about

## 3. Be explicit about layout and stride assumptions

Many successful NPU kernels in this repo rely on very strong layout
assumptions. Declare them early and normalize inputs on the host if needed.

Common pattern:

- require last-dimension contiguity for the reduction axis
- transpose or reorder weights once on the host
- preprocess static tensors once instead of every call

For GEMM-like kernels, this often means:

- `A` kept row-major with contiguous `K`
- `B` pre-transposed to `(N, K)` row-major so both operands are contiguous along `K`

## 4. Reduce scalar compare paths

Some integer compare flows that are cheap on GPU may degrade to scalar behavior
on Ascend NPU. When that happens in a hot path, convert the compare inputs to a
vector-friendly dtype and keep the rest of the computation vectorized.

### Example pattern: compare through fp32 in tail handling

```python
cols = tl.arange(0, BLOCK_N)
x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

Use this only when profiling or inspection shows that integer compare lowers to
an inefficient path. Do not cast blindly.

## 5. Fuse epilogues when the math is naturally local

If the kernel already has the needed values in registers, fuse cheap epilogue
work instead of emitting another kernel:

- scale application
- dtype cast
- bias add
- simple activation

This is especially important on NPU where extra launches and HBM round-trips
are expensive.

## 6. Prefer repo-proven schedules over generic autotune-first design

For inference-oriented kernels with predictable shape ranges:

- start from the closest proven NPU kernel in this repo
- encode heuristic config selection once you know the good shape regions
- do not default to a GPU-style autotune-everything approach

Autotune is still useful for exploration, but the delivered path should usually
be deterministic and cheap on first call.

## 7. Keep testing tied to the operator contract

A kernel is not accepted because it compiles or benchmarks once.

Always validate through the repo's public contract:

- accuracy tests under `mojo_opset/tests/accuracy/`
- opcheck tests under `mojo_opset/tests/test_ttx_graph/` when `torch.ops.ttx.*` is exposed
- perf tests under `mojo_opset/tests/perf/`

If there is no stable `core` or `torch` reference path, stop and ask for one
before treating the kernel as complete.
