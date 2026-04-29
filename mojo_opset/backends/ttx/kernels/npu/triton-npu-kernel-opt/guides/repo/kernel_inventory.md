# Existing Optimized NPU Kernels

Read this file before writing a new NPU Triton kernel. The goal is to reuse
existing scheduling, layout, integration, and testing patterns from this repo.

Kernel root:

- `mojo_opset/backends/ttx/kernels/npu/`

## Operator families

### GEMM and linear

- `int8_gemm.py`
  Use for persistent scheduling, transposed-weight preprocessing, host-side
  padding, multibuffer usage, and heuristic tile selection.
- `group_gemm.py`
  Use for grouped matmul patterns, segmented workloads, and operator-level
  wiring with grouped metadata.

Related backend/operator files:

- `mojo_opset/backends/ttx/operators/gemm.py`
- `mojo_opset/backends/torch_npu/operators/gemm.py`
- `mojo_opset/core/operators/gemm.py`

Primary references:

- [../tuning/ascend-910b-gemm.md](../tuning/ascend-910b-gemm.md)
- [../tuning/optimization-log.md](../tuning/optimization-log.md)

### Attention and sequence kernels

- `flash_attention.py`
- `sdpa.py`
- `swa.py`
- `diffution_attention.py`
- `kv_cache.py`

Use these for:

- large sequence tiling
- paged/prefill/decode split behavior
- masking and window logic
- memory-bound scheduling decisions

Related files:

- `mojo_opset/backends/ttx/operators/attention.py`
- `mojo_opset/backends/torch_npu/operators/attention.py`
- `mojo_opset/core/operators/attention.py`
- `mojo_opset/core/functions/attention.py`

### Normalization and fused norm patterns

- `layernorm.py`
- `rmsnorm.py`
- `fused_add_layernorm.py`
- `fused_add_rmsnorm.py`

Use these for:

- row-wise reductions
- tail handling
- fused residual patterns
- forward/backward/inference split structure

Related files:

- `mojo_opset/backends/ttx/operators/normalization.py`
- `mojo_opset/backends/torch_npu/operators/norm.py`
- `mojo_opset/core/operators/normalization.py`
- `mojo_opset/core/functions/normalization.py`

### Activation kernels

- `gelu.py`
- `silu.py`
- `swiglu.py`

Use these for:

- simple pointwise kernel structure
- forward/backward registration
- `torch.ops.ttx.*` opcheck coverage

Related files:

- `mojo_opset/backends/ttx/operators/activation.py`
- `mojo_opset/backends/torch_npu/operators/activation.py`
- `mojo_opset/core/operators/activation.py`
- `mojo_opset/core/functions/activation.py`

### Convolution

- `convolution.py`

Related files:

- `mojo_opset/backends/ttx/operators/convolution.py`
- `mojo_opset/core/operators/convolution.py`
- `mojo_opset/core/functions/convolution.py`

### Position embedding and indexing

- `rope.py`
- `lightning_indexer.py`

Related files:

- `mojo_opset/backends/ttx/operators/position_embedding.py`
- `mojo_opset/backends/ttx/operators/indexer.py`
- `mojo_opset/backends/torch_npu/operators/position_embedding.py`
- `mojo_opset/core/operators/position_embedding.py`
- `mojo_opset/core/operators/indexer.py`

### Quantization and sampling

- `quant.py`
- `sample.py`
- `store_lowrank.py`

Related files:

- `mojo_opset/backends/ttx/operators/quant.py`
- `mojo_opset/backends/ttx/operators/sampling.py`
- `mojo_opset/backends/ttx/operators/store_lowrank.py`
- `mojo_opset/backends/torch_npu/operators/quantize.py`
- `mojo_opset/core/operators/quantize.py`
- `mojo_opset/core/operators/sampling.py`

### Over-encoding

- `over_encoding/embedding.py`
- `over_encoding/fused_over_encoding.py`
- `over_encoding/n_gram.py`

Use when the new kernel involves specialized embedding/history layouts or
decode/prefill split logic.

Related files:

- `mojo_opset/backends/ttx/operators/over_encoding.py`
- `mojo_opset/core/operators/over_encoding.py`

## What to extract from existing kernels

For every similar kernel you inspect, answer these questions before coding:

1. How is the grid defined, and why?
2. Does it use persistent scheduling or an internal loop over tiles?
3. What layout assumptions are enforced on inputs?
4. How are tails handled: padding, masks, or both?
5. Which path is exposed to `torch.ops.ttx.*` and which is operator-only?
6. Which tests already validate similar behavior?

## Strong default examples

When unsure where to start:

- GEMM-like op: `int8_gemm.py`
- row reduction or normalization: `layernorm.py`, `rmsnorm.py`
- pointwise op: `gelu.py`, `silu.py`
- attention-like op: `flash_attention.py`, `sdpa.py`

## Anti-pattern

Do not add a new NPU kernel by only reading Triton upstream docs. In this repo,
the local NPU kernels are the first source of truth for:

- backend conventions
- dispatch wiring
- tolerated shape/layout assumptions
- test style
- proven NPU-specific workarounds
