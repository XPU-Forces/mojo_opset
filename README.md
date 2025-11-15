# Mojo Opset
## 1. 简介
Mojo Opset 是一个基于面向 LLM & DiT 类模型专用 Opset，支持多种硬件加速器以及不同的算子实现。用户能够基于 Mojo Opset
快速搭建 LLM 模型，轻松获取不同硬件加速器的 SOTA 性能。Mojo Opset 包含推理加速和训练加速两部分算子，其中推理部分以
Mojo Operator方式提供，训练部分则以 Mojo Function部分提供。

## 2. Op List

关于Mojo Opset的语义及其他信息 可以参考：https://bytedance.larkoffice.com/docx/FcTgdVpqRofDmwx2d3Pc1zRwn0A

### 2.1 Mojo Operator List

| Op Category | Op Name                     | Description       | Additional    |
| :---------- | :-------------------------- | :---------------- | :------------ |
| Embedding   | MojoEmbedding               | TBD               | TBD           |
| Embedding   | MojoParallelEmbedding       | TBD               | TBD           |
| Attention   | MojoPagedPrefillGQA         | TBD               | TBD           |
| Attention   | MojoPagedDecodeGQA          | TBD               | TBD           |
| Attention   | MojoPagedPrefillMLA         | TBD               | TBD           |
| Attention   | MojoPagedDecodeMLA          | TBD               | TBD           |
| Attention   | MojoPagedPrefillNSA         | TBD               | TBD           |
| Attention   | MojoPagedDecodeNSA          | TBD               | TBD           |
| Attention   | MojoWindownAttenton         | TBD               | TBD           |
| MoE         | MojoMoEGate                 | TBD               | TBD           |
| MoE         | MojoMoEDispatch             | TBD               | TBD           |
| MoE         | MojoMoECombine              | TBD               | TBD           |
| MoE         | MojoMoeDispatchQuant        | TBD               | TBD           |
| Sampling    | MojoTopKSampling            | TBD               | TBD           |
| Sampling    | MojoTopPSampling            | TBD               | TBD           |
| Sampling    | MojoRejectSampling          | TBD               | TBD           |
| Norm        | MojoNorm                    | TBD               | TBD           |
| Norm        | MojoResidualAddNorm         | TBD               | TBD           |
| Norm        | MojoNormQuant               | TBD               | TBD           |
| Norm        | MojoResidualAddNormQuant    | TBD               | TBD           |
| Norm        | MojoResidualAddNormCast     | TBD               | TBD           |
| PositionEmb | MojoRotaryEmb               | TBD               | TBD           |
| PositionEmb | MojoNormRotary              | TBD               | TBD           |
| PositionEmb | MojoNormRotaryStorKV        | TBD               | TBD           |
| KVCache     | MojoKVCacheCast             | TBD               | TBD           |
| KVCache     | MojoStorePagedKVCache       | TBD               | TBD           |
| KVCache     | MojoStorePagedMLAKVCache    | TBD               | TBD           |
| Linear      | MojoLinear                  | TBD               | TBD           |
| Linear      | MojoQuantLinear             | TBD               | TBD           |
| Linear      | MojoBatchLinear             | TBD               | TBD           |
| Linear      | MojoGroupLinear             | TBD               | TBD           |
| Quantize    | MojoQuant                   | TBD               | TBD           |
| Quantize    | MojoDequant                 | TBD               | TBD           |
| Activation  | MojoSilu                    | TBD               | TBD           |
| Activation  | MojoGelu                    | TBD               | TBD           |
| Activation  | MojoSwiGlu                  | TBD               | TBD           |
| Activation  | MojoSiluQuant               | TBD               | TBD           |
| Activation  | MojoGeluQuant               | TBD               | TBD           |
| Activation  | MojoSwiGluQuant             | TBD               | TBD           |
| Comm&Comp   | MojoLinearAllReduce         | TBD               | TBD           |
| Comm&Comp   | MojoAllGatherLinear         | TBD               | TBD           |
| Comm&Comp   | MojoLinearAll2All           | TBD               | TBD           |
| Comm&Comp   | MojoLinearReduceScatter     | TBD               | TBD           |


### 2.2 Mojo Function List

| Op Category | Op Name                     | Description       | Additional    |
| :---------- | :-------------------------- | :---------------- | :------------ |
| Attention   | MojoFlashAttentionFunc      | TBD               | TBD           |
| PositionEmb | MojoRotaryEmbFunc           | TBD               | TBD           |
| Activation  | MojoSwiGluFunc              | TBD               | TBD           |
| MoE         | MojoMoEGatingFunc           | TBD               | TBD           |
| Norm        | MojoRmsNormFunc             | TBD               | TBD           |
| Comm&Comp   | MojoLinearAllReduce         | TBD               | TBD           |
| Loss        | MojoLinearCrossEntropyFunc  | TBD               | TBD           |



## 3. 实现后端

### 3.1 ttx-kernels
ttx-kernels 提供了 Mojo Opset 的 triton 版本实现。

source code: mojo_opset/backends/ttx_kernels/src

### 3.2 xpu_ops
xpu_ops 提供了 Mojo Opset 的 C-like 版本实现。请注意xpu_ops是in-house的实现，不对外提供。


### 3.3 akg
【规划中】auto-kernel generation。


## 4. Support matrix

### 4.1 Mojo Operator

| Op Category | Op Name              | ttx_kernels   | xpu_ops    |
| :---------- | :------------------- | :------------ | :--------- |
| Attention   | MojoPagedPrefillGQA  | ✅            | TBD        |
| Attention   | MojoPagedDecodeGQA   | ✅            | TBD        |
| Norm        | MojoNorm             | ✅            | ✅         |
| Norm        | MojoResidualAddNorm  | ✅            | ✅         |
| PositionEmb | MojoRotaryEmb        | ✅            | TBD        |
| Activation  | MojoGelu             | ✅            | TBD        |
| Activation  | MojoSilu             | ✅            | TBD        |
| Activation  | MojoSwiGlu           | ✅            | TBD        |


### 4.2 Mojo Function

| Op Category | Op Name                     | ttx_kernels   | xpu_ops    |
| :---------- | :-------------------------- | :------------ | :--------- |
| Activation  | MojoSiluFunc                | ✅            | TBD        |
| Norm        | MojoRMSNormFunc             | ✅            | TBD        |
| PositionEmb | MojoRotaryEmbFunc           | ✅            | TBD        |
| Loss        | MojoLinearCrossEntropyFunc  | ✅            | TBD        |


## 5. Usage
### 5.1 apply mojo op
```python
from mojo_opset import MojoSilu

silu = MojoSilu(
    op_name="demo",
    layer_idx=0,
)

silu(torch.randn(128, 128).npu())
```

### 5.2 backend selection
您可以通过环境变量`MOJO_BACKEND`来控制您想要选用的后端，当前支持的后端包括`TTX`, `XPU_OPS`；当您添加多个后端后，
Mojo Opset 会按照内部的优先级顺序来选用后端实现（后续我们将添加一个 tuner 功能，自动选取当前场景下的最优实现）。
默认会开启所有后端，即`+ALL`。
```bash
export MOJO_BACKEND="+TTX, XPU_OPS"
```

### 5.3 modeling reference
以 qwen3 dense 为例（modify from：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py），您可以通过以下任意一种方式使用 Mojo Opset 构建模型：

(1) monkey patch

example_models/torch_qwen3_dense.py 中提供了原生 torch 实现的 modeling，我们实现了相应的 monkey-patch 替换机制（mojo_opset/mojo_monkey_patch.py），仅需一行代码即可将 native modeling 中若干组件替换为 Mojo op，并进一步 dispatch 到高性能后端实现。您可以运行：
```bash
MOJO_BACKEND="+TTX" pytest -s tests/test_qwen3_dense_patching.py
```
跑通一个 decoder layer 的 prefill/decode 流程。

(2) 即插即用

example_models/mojo_qwen3_dense.py 中提供了直接基于 Mojo Opset 实现的 modeling，效果等同于(1)中 monkey-patch 替换后的模型。



