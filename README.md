# Mojo Opset
## 1. 简介
Mojo Opset 是一个基于面向 LLM & DiT 类模型专用 Opset，支持多种硬件加速器以及不同的算子实现。用户能够基于 Mojo Opset
快速搭建 LLM 模型，轻松获取不同硬件加速器的 SOTA 性能。

## 2. 实现后端

### 2.1. ttx-kernels
ttx-kernels 提供了 Mojo Opset 的 triton 版本实现。

source code: mojo_opset/backends/ttx_kernels/src

### 2.2. xpu_ops
xpu_ops 提供了 Mojo Opset 的 C-like 版本实现。

## 3. Support matrix

| OpName | ttx_kernels | xpu_ops |
| :----- | :---------: | :-----: |
| MojoNorm | ✅ | ✅ |
| MojoResidualAddNorm | ✅ | ✅ |
| MojoRoPE | ✅ | TBD |
| MojoGelu | ✅ | TBD |
| MojoSilu | ✅ | TBD |
| MojoSiluMul | ✅ | TBD |
| MojoPagedPrefillGQA | ✅ | TBD |
| MojoPagedDecodeGQA | ✅ | TBD |
| MojoSiluFunction | ✅ | TBD |
| MojoRMSNormFunction | ✅ | TBD |
| MojoRoPEFunction | ✅ | TBD |
| MojoFusedLinearCrossEntropyFunction | ✅ | TBD |

## 4. Usage
### 4.1 apply mojo op
```python
from mojo_opset import MojoSilu

silu = MojoSilu(
    op_name="demo",
    layer_idx=0,
)

silu(torch.randn(128, 128).npu())
```

### 4.2 backend selection
您可以通过环境变量`MOJO_BACKEND`来控制您想要选用的后端，当前支持的后端包括`TTX_KERNELS`, `XPU_OPS`；当您添加多个后端后，
Mojo Opset 会按照内部的优先级顺序来选用后端实现（后续我们将添加一个 tuner 功能，自动选取当前场景下的最优实现）。
默认会开启所有后端，即`+ALL`。
```bash
export MOJO_BACKEND="+TTX_KERNELS, XPU_OPS"
```

### 4.3 modeling reference
Mojo Opset 提供了一份 qwen3 dense modeling 实现（example_models/mojo_qwen3_dense.py，modify from：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py），
并实现了相应的 monkey-patch 替换机制（mojo_opset/mojo_monkey_patch.py），仅需一行代码即可将 native modeling 中若干组件替换为 Mojo op，并进一步 dispatch 到高性能后端实现。您可以运行：
```bash
MOJO_BACKEND="+TTX_KERNELS" pytest -s tests/test_qwen3_dense_patching.py
```
跑通一个 decoder layer 的 prefill/decode 流程。



