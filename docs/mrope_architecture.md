# MRoPE 算子类图

```mermaid
classDiagram
    direction TB

    class MojoOperator {
        <<abstract>>
        +forward(*args, **kwargs)* Any
        +_registry: Dict
        +supported_platforms_list: List[str]
    }

    class MojoRotaryEmbedding {
        +rope_theta
        +attention_scaling
        +forward(x, cu_seqlens_q, seqlens_kv, position_ids) Tuple[Tensor, Tensor]
    }

    class MojoApplyRoPE {
        +interleaved: bool
        +forward(q, k, cos, sin, head_first) Tuple[Tensor, Tensor]
        +_rotate_half(x) Tensor
        +_apply_rope(q, k, cos, sin) Tuple[Tensor, Tensor]
    }

    class MojoGridRoPE {
        +forward(x, grid_sizes, freqs_list) Tensor
    }

    class MojoMRoPE {
        +supported_platforms_list = ["npu"]
        +forward(query, key, cos_table, sin_table, mrope_section, is_interleaved, head_dim) Tuple[Tensor, Tensor]
        +_rotate_half(hidden_states) Tensor
        +_apply_interleaved_mrope(cos_table, sin_table, mrope_section) Tuple[Tensor, Tensor]
    }

    class TorchMRoPE {
        +forward(query, key, cos_table, sin_table, mrope_section, is_interleaved, head_dim) Tuple[Tensor, Tensor]
    }

    class TTXMRoPE {
        +supported_platforms_list = ["npu"]
        +forward(q, k, cos, sin, mrope_section, is_interleaved, head_dim) Tuple[Tensor, Tensor]
    }

    class mrope_fwd_impl {
        <<function>>
        +forward(q, k, cos, sin, mrope_section, is_interleaved, head_dim) Tuple[Tensor, Tensor]
    }

    class _triton_mrope_kernel {
        <<triton kernel>>
        +forward(q_ptr, k_ptr, cos_ptr, sin_ptr, num_tokens, n_qh, n_kh, hd, rope_dim, pad_n_qh, pad_n_kh, pad_hd, mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved)
    }

    MojoOperator <|-- MojoRotaryEmbedding
    MojoOperator <|-- MojoApplyRoPE
    MojoOperator <|-- MojoGridRoPE
    MojoOperator <|-- MojoMRoPE

    MojoOperator <|-- TorchMRoPE
    MojoMRoPE <|-- TTXMRoPE

    MojoMRoPE --> TorchMRoPE : registry (torch backend)
    TTXMRoPE --> mrope_fwd_impl : calls
    mrope_fwd_impl --> _triton_mrope_kernel : launches
```

## 模块架构

```
mojo_opset/
├── core/
│   └── operators/
│       └── position_embedding.py  # MojoMRoPE 基类（已合并）
│           ├── MojoRotaryEmbedding
│           ├── MojoApplyRoPE
│           ├── MojoGridRoPE
│           └── MojoMRoPE  ← MRoPE 基类
├── backends/
│   └── ttx/
│       ├── operators/
│       │   └── mrope.py           # TTXMRoPE 后端
│       └── kernels/
│           └── npu/
│               └── mrope.py       # Triton Kernel 实现
└── tests/
    └── accuracy/
        └── operators/
            └── test_mrope.py      # 测试
```

## 类关系

| 关系 | 说明 |
|------|------|
| `MojoOperator <|-- MojoMRoPE` | MojoMRoPE 继承 MojoOperator 基类 |
| `MojoMRoPE <|-- TTXMRoPE` | TTXMRoPE 继承 MojoMRoPE（NPU 后端） |
| `TTXMRoPE --> mrope_fwd_impl` | TTXMRoPE 调用 mrope_fwd_impl |
| `mrope_fwd_impl --> _triton_mrope_kernel` | mrope_fwd_impl 启动 Triton Kernel |

## 平台分发

```
                    ┌─────────────────┐
                    │   MojoMRoPE     │
                    │   (Base Class)  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │TorchMRoPE │    │ TTXMRoPE  │    │  ...       │
    │  (PyTorch) │    │   (NPU)   │    │ (Future)   │
    └────────────┘    └─────┬──────┘    └────────────┘
                           │
                           ▼
                   ┌───────────────────┐
                   │  mrope_fwd_impl  │
                   │  (Triton Wrapper) │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │_triton_mrope_kernel│
                   │   (Core Kernel)   │
                   └───────────────────┘
```
