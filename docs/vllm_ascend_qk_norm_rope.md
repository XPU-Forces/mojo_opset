# vllm-ascend integration: `fused_qk_norm_rope` dispatch

`qk_norm_rope_infer` is the Ascend Triton port of vLLM-Omni's `fused_qk_norm_rope`
(`vllm_omni/diffusion/layers/fused_qk_norm_rope.py`), the MiniMax-H3 contract:
`head_dim=128`, `rotary_dim=96`, BF16, `[tokens, heads, head_dim]`, half-split pairing.

This note records **when a consumer should route to it and when it should not**. The short
answer is: only for launch-bound shapes. It is not a drop-in replacement for the NPU
composition upstream already ships.

## Current state upstream

`_fused_qk_norm_rope_impl` routes NPU tensors to `_npu_qk_norm_rope`, which is two fused
Ascend primitives:

```python
q_norm = torch_npu.npu_rms_norm(q, q_weight, epsilon=eps)[0]
k_norm = torch_npu.npu_rms_norm(k, k_weight, epsilon=eps)[0]
# then the rotary dimensions go through MindIE-SD's rotary kernel, or
# npu_rotary_mul when MindIE-SD is not packaged
```

The gate is `_fused_npu_supported`: NPU device, BF16, `head_dim == 128`,
`rotary_dim == 96`, `interleaved == False`. Everything else falls back to
`_eager_qk_norm_rope`.

That composition costs **16 kernels** (2 norms + 2 rotary applications, each split into
several primitives), which is why it loses on tiny inputs and wins on every large one.

## What this operator changes

One launch per stream, two launches for the Q+K pair, instead of sixteen. Measured on
A2 910B3 (CANN 9.0.0, triton-ascend 3.6.0, torch_npu 2.10.0.post7), median of five:

| tokens x heads | bytes | triton | torch_npu composition | ratio | MBU |
| --- | --- | --- | --- | --- | --- |
| 32 x 8 | 0.27 MB | 26.4 us | 74.2 us | **0.36x (2.8x faster)** | 0.008 |
| 320 x 8 | 2.7 MB | 105.9 us | 91.1 us | 1.16x | 0.020 |
| 320 x 56 | 18 MB | 654.8 us | 178.5 us | 3.67x | 0.022 |
| 1024 x 56 | 59 MB | 2055 us | 285.5 us | 7.20x | 0.023 |
| 4096 x 56 | 236 MB | 7814 us | 828.9 us | 9.43x | 0.024 |
| 4096 x 57 | 240 MB | 8600 us | 829.6 us | 10.37x | 0.022 |
| 16384 x 56 | 943 MB | 31240 us | 4307.7 us | 7.25x | 0.024 |

`MBU` is against the `fill_` bandwidth calibrated in the same process (2.93 TB/s); the
target line for this task is 0.60.

**The crossover is between 32 and 320 tokens.** Below roughly 64-128 tokens the launch count
dominates and Triton wins outright; above it the operator is 1.2x-10.4x slower.

## Why it cannot be fixed by scheduling

Each head is a 256 B row, and on triton-ascend 3.6.0 every `tl.load` pays a fixed cost of
roughly 3.5 ns regardless of transfer size. The kernel needs four loads and one store per
row (the activation, the two shifted partner halves, the packed cos/sin table, the store),
i.e. ~17 ns/row. A pure row-wise copy of the same layout reaches only 157 GB/s on the same
box, against 288 GB/s for the composition, so the gap is in the per-row structure, not in
the grid shape.

Four forms were measured and rejected on the 235 MB case in one round: 1-D rows with the
table per head (53.2 ms), a 2-D `(8, head_dim)` tile in upstream's shape (40.3 ms), a flat
2 KB tile with `tl.reshape` and a 2-D reduction (676 ms), and a 256-row tile (UB overflow:
3.7 Mbit required against 1.57 Mbit available). The shipped form is the fastest of them.

The 2-D regression has the same signature as
[triton-lang/triton-ascend#2865](https://github.com/triton-lang/triton-ascend/issues/2865)
(~11 us of fixed cost per tile). If that is fixed, the merged-tile form becomes worth
re-measuring, and the stage ablation in `kernels/npu_a2_triton/qk_norm_rope.py` says the
partner re-read (44% of the kernel) is the next thing to attack.

## Recommended dispatch

Keep the NPU composition as the default for this operator, and route to Triton only below
the launch-bound crossover:

```python
# In the consumer, not in mojo_opset: the threshold is a property of the deployment.
from mojo_opset import functions

_TRITON_MAX_TOKENS = 64  # measured crossover on A2 910B3; re-tune per architecture


def qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps):
    if q.shape[0] <= _TRITON_MAX_TOKENS:
        return functions.qk_norm_rope_infer(
            q, k, q_weight, k_weight, rope_table, eps, implementation="triton"
        )
    # keep upstream's _npu_qk_norm_rope: npu_rms_norm + the rotary kernel
    ...
```

`mojo_opset`'s own selection (`config/npu_a2.yaml`) points at `triton`, because the package
ships Triton operators; a consumer that carries the composition as well should follow the
gate above rather than call `qk_norm_rope_infer` unconditionally.

## Validation status

| Platform | Accuracy | Performance |
| --- | --- | --- |
| A2 910B3 (CANN 9.0.0, triton-ascend 3.6.0) | `tests/functions` **63 passed, 0 skipped**, `tests/modules` 6 passed; max abs diff 3.12e-2, inside upstream's `atol=2e-2`/`rtol=2e-2` band | measured, table above |
| A3 (910_9382) | not run | not run |
| 950PR | not run | not run |

The A5 provider re-exports the A2 kernel (no Cube unit, no L0/L1 buffers, no
architecture-specific intrinsic), which is what the sibling operators in this package do as
well. It has **not** been executed on 950 hardware, so `npu_a5.yaml` carries the entry with a
comment saying so.

Geometry beyond the MiniMax-H3 contract is measured on hardware for all five accepted
shapes — `128/96`, `128/128`, `128/32`, `64/64` and the non-power-of-two `100/64` — by
`test_qk_norm_rope_geometry_hardware`, so the kernel's `d_pad` masking is exercised rather
than assumed. The head-count branches (`hpg=1` below 8 heads, one-head tail at 57), grouped
GQA (56 q heads against 8 kv heads) and the empty-input early return are measured the same
way.

Activation dtype: BF16 is the contract, FP16 and FP32 are accepted as upstream does. The
kernel keeps the pre-rotation rounding on the activation dtype, so all three match the eager
chain — FP32 at ~1e-6, i.e. no hidden BF16 rounding in the middle of the FP32 path.
