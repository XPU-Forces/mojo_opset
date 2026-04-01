# 🧱 Mojo Opset

## Overview

Mojo Opset is a domain specialized opset for LLMs and multimodal models that provides operator suites for both inference acceleration and training acceleration. It supports multiple hardware accelerators and diverse operator implementations, while abstracting away the differences and complexity of implementation strategies and hardware backends for users. The goal is to help users quickly build LLM models with Mojo Opset and achieve state-of-the-art performance across different accelerators.

## Backend Implementations

### Torch native

Mojo Opset provides a baseline implementation built on PyTorch native ops. This implementation serves as the golden reference for different backends and also functions as the fallback backend while other backends are being developed.

### 🔥🔥🔥 Triton-x (TTX for short)

TTX is a triton implementation for Mojo Opset.

Supported Hardware:

- Ascend NPU 910B/C

TTX now is compatible with `torch.compile`.
You can control the run mode via the `MOJO_RUN_MODE` environment variable. The supported modes are `EAGER` and `COMPILE`; `EAGER` is enabled by default. The `COMPILE` mode requires the current Torch version to be >= 2.7.0; otherwise, an error will be raised.

```bash
# If you want the current Triton kernel to be registered in torch.library and captured by torch.dynamo
# to enable longer-term optimizations (default mode).
export MOJO_RUN_MODE="COMPILE"

# If you want the current Triton kernel to be invoked directly rather than registered in torch.library
# (this can slightly reduce PyTorch overhead in eager mode).
export MOJO_RUN_MODE="EAGER"
```

source code: mojo_opset/backends/ttx/kernels

### Backend Selection

You can control the backend you want to use via the `MOJO_BACKEND` environment variable; the currently supported backends are list as below:

- "ttx"
- "torch_npu"
- "torch"

When multiple backends are added, Mojo Opset selects the backend implementation according to its internal priority order (We plan to add a tuner feature later to automatically choose the optimal implementation for the current scenario).

## Op List

### Mojo Operator List

| Op Category | Op Name                       | torch native | torch_npu | ttx |
| :---------- | :---------------------------- | :----------- | :-------- | :-- |
| Activation  | MojoGelu                      | ✅           | ✅        | ✅  |
| Activation  | MojoSilu                      | ✅           | ✅        | ✅  |
| Activation  | MojoSwiGlu                    | ✅           | ✅        | ✅  |
| Linear      | MojoLinear                    | ✅           | TBD       | TBD |
| Gemm        | MojoGroupGemm                 | ✅           | ✅        | ✅  |
| Gemm        | MojoQuantGroupLinearReduceSum | ✅           | ✅        | TBD |
| ComputeComm | MojoGemmAllReduce             | ✅           | TBD       | TBD |
| ComputeComm | MojoAllGatherGemm             | ✅           | TBD       | TBD |
| ComputeComm | MojoGemmAll2All               | ✅           | TBD       | TBD |
| ComputeComm | MojoGemmReduceScatter         | ✅           | TBD       | TBD |
| Attention   | MojoSdpa                      | ✅           | TBD       | ✅  |
| Attention   | MojoPrefillGQA                | ✅           | ✅        | ✅  |
| Attention   | MojoPagedPrefillGQA           | ✅           | ✅        | ✅  |
| Attention   | MojoDecodeGQA                 | ✅           | TBD       | TBD |
| Attention   | MojoPagedDecodeGQA            | ✅           | ✅        | 🚧  |
| Attention   | MojoDecodeMLA                 | ✅           | TBD       | TBD |
| Attention   | MojoPagedDecodeMLA            | ✅           | TBD       | TBD |
| Attention   | MojoPrefillMLA                | ✅           | TBD       | TBD |
| Attention   | MojoPagedPrefillMLA           | ✅           | TBD       | TBD |
| Attention   | MojoDecodeNSA                 | ✅           | TBD       | TBD |
| Attention   | MojoPagedDecodeNSA            | ✅           | TBD       | TBD |
| Attention   | MojoPrefillNSA                | ✅           | TBD       | TBD |
| Attention   | MojoPagedPrefillNSA           | ✅           | TBD       | TBD |
| Attention   | MojoSlidingWindowAttention    | TBD          | TBD       | TBD |
| MoE         | MojoMoE                       | ✅           | TBD       | TBD |
| MoE         | MojoMoEGating                 | ✅           | TBD       | TBD |
| MoE         | MojoMoEDispatch               | ✅           | TBD       | TBD |
| MoE         | MojoExperts                   | ✅           | TBD       | TBD |
| MoE         | MojoMoECombine                | ✅           | TBD       | TBD |
| Sampling    | MojoTopKSampling              | TBD          | TBD       | TBD |
| Sampling    | MojoTopPSampling              | ✅           | TBD       | ✅  |
| Sampling    | MojoRejectSampling            | ✅           | TBD       | ✅  |
| Sampling    | MojoApplyPenaltiesTempurate   | ✅           | TBD       | ✅  |
| Quantize    | MojoQuant                     | ✅           | TBD       | TBD |
| Quantize    | MojoDequant                   | ✅           | TBD       | TBD |
| Quantize    | MojoDynamicQuant              | ✅           | ✅        | TBD |
| Quantize    | MojoDequantSwiGLUQuant        | ✅           | ✅        | TBD |
| Quantize    | MojoGemmDequant               | ✅           | ✅        | TBD |
| Norm        | MojoRMSNorm                   | ✅           | ✅        | ✅  |
| Norm        | MojoLayerNorm                 | ✅           | TBD       | ✅  |
| Norm        | MojoResidualAddRMSNorm        | ✅           | ✅        | ✅  |
| Norm        | MojoResidualAddLayerNorm      | ✅           | TBD       | ✅  |
| Norm        | MojoRMSNormQuant              | ✅           | ✅        | TBD |
| Norm        | MojoLayerNormQuant            | ✅           | ✅        | TBD |
| Norm        | MojoResidualAddRMSNormQuant   | ✅           | ✅        | TBD |
| Norm        | MojoResidualAddLayerNormQuant | ✅           | ✅        | TBD |
| Norm        | MojoChannelRMSNorm            | ✅           | TBD       | TBD |
| PositionEmb | MojoRoPE                      | ✅           | ✅        | ✅  |
| PositionEmb | MojoGridRoPE                  | ✅           | TBD       | TBD |
| KVCache     | MojoStorePagedKVCache         | ✅           | TBD       | ✅  |
| KVCache     | MojoStorePagedMLAKVCache      | ✅           | TBD       | TBD |
| Embedding   | MojoEmbedding                 | ✅           | TBD       | TBD |
| Embedding   | MojoParallelEmbedding         | ✅           | TBD       | TBD |
| Embedding   | MojoRelativeEmbedding         | ✅           | TBD       | TBD |


### Mojo Function List

| Op Category | Op Name                     | torch native      | ttx           |
| :---------- | :-------------------------- | :---------------- | :------------ |
| Attention   | MojoSdpaFunc                | ✅                | ✅             |
| Attention   | MojoDiffusionAttentionFunc  | ✅                | ✅             |
| PositionEmb | MojoRotaryEmbFunc           | ✅                | ✅             |
| Activation  | MojoSiluFunc                | ✅                | ✅             |
| Activation  | MojoSwiGluFunc              | TBD               | TBD           |
| Norm        | MojoRMSNormFunc             | ✅                | ✅             |
| Gemm        | MojoGemmAllReduce           | TBD               | TBD           |
| Loss        | MojoLinearCrossEntropyFunc  | ✅                | ✅             |


## Usage

### Apply mojo op

```python
from mojo_opset import MojoSilu

silu = MojoSilu()

silu(torch.randn(128, 128))
```

### Modeling with Mojo Opset

You can build the model using Mojo Opset in the following ways:

1. Build model from mojo opset

    You can also build your modeling by mojo opset directly, [Mojo qwen3 dense modeling](./mojo_opset/modeling/mojo_qwen3_dense.py) is an example.

    And you can try the example by running the following command:

    ```bash
    bash ./examples/run_model.sh

    Prompt: 你好，请介绍一下你自己。
    ----------------------------------------
    ----------------------------------------
    Generated text:  你好！我是一个大型语言模型，名叫通义千问，由通义实验室研发。我能够进行多轮对话，回答各种问题，创作文字，比如写故事、写邮件、写剧本等，还能进行逻辑推理、表达观点，甚至编写和调试程序。我的训练数据来自于互联网上的大量文本，因此我具备广泛的知识和语言理解能力。我可以用多种语言与你交流，包括中文、英文、日文、韩文等。
    ```

2. Patch for transformers models.

    For [hugging face transformers](https://github.com/huggingface/transformers) models, you can use Mojo Opset to build the model by monkey patching the original modeling code.

    ```python
    # 1. Apply mojo opset to qwen3 model
    mojo_opsetutils.patching.apply_mojo_to_qwen3()

    # 2. Instantiate patched model
    model = transformers.AutoModelForCausalLM("path/to/qwen3/model")
    ```

    And you can try the example by running the following command:

    ```python
    python ./examples/qwen3_patch.py
    ```

## Environment Variables

### MOJO_DETERMINISTIC

Controls whether deterministic computation is enabled (only TTX backend supported for now).

| Value | Description |
|-------|-------------|
| `0` (default) | Deterministic computation disabled. Best performance. |
| `1` | Deterministic computation enabled. |

**Usage:**

```bash
export MOJO_DETERMINISTIC=1
```

### MOJO_RUN_MODE

Controls the run mode for mojo kernels (only TTX backend supported for now).

| Value | Description |
|-------|-------------|
| `EAGER` (default) | Kernels are invoked directly. Reduces overhead in eager mode. |
| `COMPILE` | Kernels are registered in `torch.library`, requires Torch >= 2.7.0. |

**Usage:**

```bash
export MOJO_RUN_MODE="COMPILE"
```

### MOJO_BACKEND

Controls the backend implementation to use.

| Value | Description |
|-------|-------------|
| `ttx` | Use Triton-x implementation. |
| `torch` | Use PyTorch native implementation. |

**Usage:**

```bash
export MOJO_BACKEND="ttx"
```

## 🚧 Future Work

- Add more mojo ops.
- Support more backend implementations and support more Hardware accelerators.
  - Ascend NPU's official implementation using Ascend C language.
  - Support Cambircon MLU using triton language.
- Performance optimization.
  - A tuner for various backend implementations, ensure users can always get the best performance.
  - A compilation mechanism for replacement the original torch ops with mojo ops.
