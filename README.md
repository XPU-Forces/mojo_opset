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

### Ixformer

Ixformer is a backend powered by the [ixformer](https://github.com/AIsoftwareFactory/ixformer) library, providing high-performance fused operator implementations for Iluvatar GPUs.

Supported Hardware:

- Iluvatar GPU (MR/BI series)

source code: mojo_opset/backends/ixformer

### Backend Selection

You can control the backend you want to use via the `MOJO_BACKEND` environment variable; the currently supported backends are list as below:

- "ixformer"
- "ttx"
- "torch_npu"
- "torch"

When multiple backends are added, Mojo Opset selects the backend implementation according to its internal priority order (We plan to add a tuner feature later to automatically choose the optimal implementation for the current scenario).

## Op List

### Mojo Operator List

| Op Category | Op Name                       | torch native | torch_npu | ttx | ixformer |
| :---------- | :---------------------------- | :----------- | :-------- | :-- | :------- |
| Activation  | MojoGelu                      | ✅           | ✅        | ✅  | TBD      |
| Activation  | MojoSilu                      | ✅           | ✅        | ✅  | TBD      |
| Activation  | MojoSwiGlu                    | ✅           | ✅        | ✅  | TBD      |
| Linear      | MojoLinear                    | ✅           | TBD       | TBD | TBD      |
| Gemm        | MojoGroupGemm                 | ✅           | ✅        | ✅  | TBD      |
| Gemm        | MojoQuantGroupLinearReduceSum | ✅           | ✅        | TBD | TBD      |
| ComputeComm | MojoGemmAllReduce             | ✅           | TBD       | TBD | TBD      |
| ComputeComm | MojoAllGatherGemm             | ✅           | TBD       | TBD | TBD      |
| ComputeComm | MojoGemmAll2All               | ✅           | TBD       | TBD | TBD      |
| ComputeComm | MojoGemmReduceScatter         | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoSdpa                      | ✅           | TBD       | ✅  | TBD      |
| Attention   | MojoPrefillGQA                | ✅           | ✅        | ✅  | TBD      |
| Attention   | MojoPagedPrefillGQA           | ✅           | ✅        | ✅  | TBD      |
| Attention   | MojoDecodeGQA                 | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPagedDecodeGQA            | ✅           | ✅        | 🚧  | TBD      |
| Attention   | MojoDecodeMLA                 | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPagedDecodeMLA            | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPrefillMLA                | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPagedPrefillMLA           | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoDecodeNSA                 | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPagedDecodeNSA            | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPrefillNSA                | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoPagedPrefillNSA           | ✅           | TBD       | TBD | TBD      |
| Attention   | MojoSlidingWindowAttention    | TBD          | TBD       | TBD | TBD      |
| MoE         | MojoMoE                       | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoMoEGating                 | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoMoEDispatch               | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoExperts                   | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoQuantMoE                  | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoQuantExperts              | ✅           | TBD       | TBD | TBD      |
| MoE         | MojoMoECombine                | ✅           | TBD       | TBD | TBD      |
| Sampling    | MojoTopKSampling              | TBD          | TBD       | TBD | TBD      |
| Sampling    | MojoTopPSampling              | ✅           | TBD       | ✅  | TBD      |
| Sampling    | MojoRejectSampling            | ✅           | TBD       | ✅  | TBD      |
| Sampling    | MojoApplyPenaltiesTempurate   | ✅           | TBD       | ✅  | TBD      |
| Quantize    | MojoStaticQuant               | ✅           | TBD       | ✅  | TBD      |
| Quantize    | MojoDequant                   | ✅           | TBD       | TBD | TBD      |
| Quantize    | MojoDynamicQuant              | ✅           | ✅        | ✅  | TBD      |
| Quantize    | MojoMoEDynamicQuant           | ✅           | ✅        | ✅  | TBD      |
| Quantize    | MojoDequantSwiGLUQuant        | ✅           | ✅        | TBD | TBD      |
| Quantize    | MojoGemmDequant               | ✅           | ✅        | TBD | TBD      |
| Norm        | MojoRMSNorm                   | ✅           | ✅        | ✅  | ✅       |
| Norm        | MojoLayerNorm                 | ✅           | TBD       | ✅  | ✅       |
| Norm        | MojoResidualAddRMSNorm        | ✅           | ✅        | ✅  | ✅       |
| Norm        | MojoResidualAddLayerNorm      | ✅           | TBD       | ✅  | ✅       |
| Norm        | MojoRMSNormQuant              | ✅           | ✅        | TBD | TBD      |
| Norm        | MojoLayerNormQuant            | ✅           | ✅        | TBD | TBD      |
| Norm        | MojoResidualAddRMSNormQuant   | ✅           | ✅        | TBD | TBD      |
| Norm        | MojoResidualAddLayerNormQuant | ✅           | ✅        | TBD | TBD      |
| Norm        | MojoChannelRMSNorm            | ✅           | TBD       | TBD | TBD      |
| PositionEmb | MojoRoPE                      | ✅           | ✅        | ✅  | TBD      |
| PositionEmb | MojoGridRoPE                  | ✅           | TBD       | TBD | TBD      |
| KVCache     | MojoStorePagedKVCache         | ✅           | TBD       | ✅  | TBD      |
| KVCache     | MojoStorePagedMLAKVCache      | ✅           | TBD       | TBD | TBD      |
| Embedding   | MojoEmbedding                 | ✅           | TBD       | TBD | TBD      |
| Embedding   | MojoParallelEmbedding         | ✅           | TBD       | TBD | TBD      |
| Embedding   | MojoRelativeEmbedding         | ✅           | TBD       | TBD | TBD      |

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

    You can also build your modeling by mojo opset directly, [Mojo qwen3 dense modeling](./mojo_opset/modeling/qwen3/mojo_qwen3_dense.py) is an example.

    And you can try the LLM inference demo by running the following command:

    ```bash
    bash ./examples/run_llm.sh

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
    python -m examples.qwen3_patch
    ```

3. Run a DiT inference demo.

    For Wan2.2-based image or video generation demos, you can run:

    ```bash
    bash ./examples/run_dit.sh
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
| `ixformer` | Use ixformer implementation (Iluvatar GPU only). |
| `ttx` | Use Triton-x implementation. |
| `torch` | Use PyTorch native implementation. |

**Usage:**

```bash
export MOJO_BACKEND="ttx"
```

### MOJO_OPSET_VERBOSITY

Controls the logging verbosity level. Uses standard Python logging levels.

| Value | Description |
|-------|-------------|
| `DEBUG` | Show all messages including debug details. |
| `INFO` (default) | Show informational messages and above. |
| `WARNING` | Show warnings and errors only. |
| `ERROR` | Show errors only. |
| `CRITICAL` | Show critical errors only. |

**Usage:**

```bash
export MOJO_OPSET_VERBOSITY="DEBUG"
```

## 🚧 Future Work

- Add more mojo ops.
- Support more backend implementations and support more Hardware accelerators.
  - Ascend NPU's official implementation using Ascend C language.
  - Support Cambircon MLU using triton language.
- Performance optimization.
  - A tuner for various backend implementations, ensure users can always get the best performance.
  - A compilation mechanism for replacement the original torch ops with mojo ops.
