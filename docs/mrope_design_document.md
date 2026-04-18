# MRoPE (Multimodal Rotary Position Embedding) Triton 算子需求文档

## 1. 需求概述

### 1.1 需求目的

本文档旨在描述 **MRoPE 算子在昇腾 NPU 平台上的 Triton 实现**，填补现有推理引擎在 NPU 平台上对 MRoPE（Multimodal Rotary Position Embedding，多模态旋转位置编码）支持的空白。

MRoPE 由 Qwen 团队在 Qwen2-VL 论文中提出，被广泛应用于多模态大模型（如 Qwen2-VL、Wan2.2）的注意力机制中，用于为 token 提供空间位置感知能力。

### 1.2 概念解释

#### 1.2.1 传统 RoPE

**RoPE（Rotary Position Embedding，旋转位置编码）** 是苏剑林团队于 2021 年提出的位置编码算法。

**核心思想**：将位置信息编码为旋转矩阵，作用于 query 和 key 向量，使得 attention score 能够隐式地包含相对位置信息。

**数学公式**：
```
RoPE(x, pos) = x * cos(pos * θ) + rotate_half(x) * sin(pos * θ)
```
其中 `rotate_half` 将向量后半部分翻转到前半部分。

**局限**：只能编码一维位置信息，无法利用视觉数据的空间结构。

#### 1.2.2 MRoPE

**MRoPE（Multimodal Rotary Position Embedding，多模态旋转位置编码）** 在 RoPE 基础上扩展为 3D 位置编码。

**核心改进**：

| 维度 | 含义 | 场景 |
|------|------|------|
| T（Time） | 时间帧数 | 视频理解 |
| H（Height） | 图像高度 | 图像理解 |
| W（Width） | 图像宽度 | 图像理解 |

**数学公式**：
```
cos_3d shape: [3, seq_len, rope_dim // 2]
- cos[0] 对应 T 维度
- cos[1] 对应 H 维度
- cos[2] 对应 W 维度

x' = x_half1 * cos(θ) - x_half2 * sin(θ)
x'_half2 = x_half2 * cos(θ) + x_half1 * sin(θ)
```

**灵活配置**：通过 `mrope_section = [t, h, w]` 分配 rope_dim：
```
mrope_section = [16, 24, 24] 表示：
- T 维度使用前 16 个维度
- H 维度使用接下来的 24 个维度
- W 维度使用剩余的 24 个维度
```

#### 1.2.3 演进对比

| 特性 | RoPE | MRoPE |
|------|------|-------|
| 位置维度 | 1D | 3D (T, H, W) |
| cos/sin shape | [seq_len, rope_dim] | [3, seq_len, rope_dim // 2] |
| 维度配置 | 无 | mrope_section 灵活配置 |
| 交错模式 | 无 | is_interleaved |
| 典型应用 | LLaMA, Qwen1.x | Qwen2-VL, Wan2.2 |

---

## 2. 任务概述

### 2.1 模块架构

为实现 MRoPE 算子，结合 mojo_opset 算子库，涉及以下核心模块：

```
mojo_opset/
├── core/
│   └── operators/
│       └── position_embedding.py  # MojoMRoPE 基类定义（已合并）
│           ├── MojoRotaryEmbedding
│           ├── MojoApplyRoPE
│           ├── MojoGridRoPE
│           └── MojoMRoPE  ← MRoPE 基类
├── backends/
│   └── ttx/
│       ├── operators/
│       │   └── mrope.py          # [实现] TTXMRoPE 后端
│       └── kernels/
│           └── npu/
│               └── mrope.py      # [Triton Kernel] _triton_mrope_kernel
└── tests/
    └── accuracy/
        └── operators/
            └── test_mrope.py     # [测试] 精度验证
```

### 2.2 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| MojoMRoPE | core/operators/position_embedding.py | 定义 MRoPE 算子基类，包含 PyTorch 参考实现 |
| TTXMRoPE | backends/ttx/operators/mrope.py | NPU 后端实现，调用 Triton kernel |
| mrope_fwd_impl | backends/ttx/kernels/npu/mrope.py | Triton kernel 封装，处理内存布局与核间调度 |
| _triton_mrope_kernel | backends/ttx/kernels/npu/mrope.py | 核心计算 kernel，实现旋转位置编码 |
| test_mrope | tests/accuracy/operators/test_mrope.py | 精度测试，验证 Triton 实现与 PyTorch 参考一致 |

---

## 3. 需求分析

### 3.1 MojoMRoPE 基类接口

**文件路径**：`mojo_opset/core/operators/mrope.py`

**类定义**：
```python
class MojoMRoPE(MojoOperator):
    supported_platforms_list = ["npu"]
```

**forward 接口**：
```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    mrope_section: List[int],
    is_interleaved: bool = False,
    head_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**接口说明**：

| 参数 | 类型 | 输入/输出 | 说明 | 约束 |
|------|------|-----------|------|------|
| query | torch.Tensor | 输入/输出 | 查询向量 | [num_tokens, n_qh * head_dim] |
| key | torch.Tensor | 输入/输出 | 键向量 | [num_tokens, n_kh * head_dim] |
| cos_table | torch.Tensor | 输入 | cos 预计算表 | [3, num_tokens, rotary_dim // 2] |
| sin_table | torch.Tensor | 输入 | sin 预计算表 | [3, num_tokens, rotary_dim // 2] |
| mrope_section | List[int] | 输入 | T/H/W 维度分配 | [t, h, w]，sum = rotary_dim // 2 |
| is_interleaved | bool | 输入 | 交错模式标志 | 默认 False |
| head_dim | Optional[int] | 输入 | 头维度 | 可选，用于 rope_dim < head_dim |

**内部方法**：
```python
@staticmethod
def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    ...

@staticmethod
def _apply_interleaved_mrope(
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved MRoPE pattern to cos/sin tables."""
    ...
```

### 3.2 TTXMRoPE 后端接口

**文件路径**：`mojo_opset/backends/ttx/operators/mrope.py`

**类定义**：
```python
class TTXMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]
```

**forward 接口**：
```python
@staticmethod
def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    is_interleaved: bool = False,
    head_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return mrope_fwd_impl(q, k, cos, sin, mrope_section, is_interleaved)
```

**说明**：TTXMRoPE 继承 MojoMRoPE，复写 forward 方法调用 Triton 实现。

### 3.3 mrope_fwd_impl Triton 封装接口

**文件路径**：`mojo_opset/backends/ttx/kernels/npu/mrope.py`

**接口定义**：
```python
def mrope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    is_interleaved: bool = False,
    head_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**处理流程**：
1. 确保输入张量连续（contiguous）
2. 推断或验证 head_dim
3. 计算 padding 参数（next_power_of_2）
4. 启动 Triton kernel

### 3.4 _triton_mrope_kernel 核心计算接口

**文件路径**：`mojo_opset/backends/ttx/kernels/npu/mrope.py`

**Triton Kernel 定义**：
```python
@triton.jit
def _triton_mrope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
```

**接口说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| q_ptr/k_ptr | pointer | q/k 张量指针 |
| cos_ptr/sin_ptr | pointer | cos/sin 张量指针 |
| num_tokens | int | token 数量（核间调度单位） |
| n_qh/n_kh | tl.constexpr | query/key 头数 |
| hd | tl.constexpr | 头维度 |
| rope_dim | tl.constexpr | 旋转维度 |
| pad_n_qh/pad_n_kh | tl.constexpr | 头数 padding（next_power_of_2） |
| pad_hd | tl.constexpr | 维度 padding（next_power_of_2） |
| mrope_section_t/h/w | tl.constexpr | T/H/W 各维度分配的 rope 维度 |
| is_interleaved | tl.constexpr | 交错模式标志 |

**计算逻辑**：
1. 每个 program 处理一个 token
2. 加载 T/H/W 三个维度的 cos/sin
3. 根据 mrope_section 应用 mask 合并
4. 执行旋转计算：`x_new = x_half1 * cos - x_half2 * sin`

---

## 4. 规格约束

### 4.1 输入约束

| 约束项 | 条件 |
|--------|------|
| q/k shape | [num_tokens, n_heads * head_dim] |
| cos/sin shape | [3, num_tokens, rotary_dim // 2] |
| 数据类型 | FLOAT16, BF16, FLOAT |
| num_tokens | > 0 |
| mrope_section | [t, h, w]，t + h + w = rotary_dim // 2 |

### 4.2 输出约束

| 约束项 | 条件 |
|--------|------|
| shape | 与输入 q/k 相同 |
| dtype | 与输入相同 |
| 内存 | in-place 修改 |

### 4.3 硬件约束

| 约束项 | 条件 |
|--------|------|
| UB 缓冲区 | 192KB (A2/A3) |
| 内存对齐 | 32 字节对齐 |
| 并行策略 | 按 token 维度切分 |

---

## 5. 测试方案

### 5.1 测试用例

```python
@pytest.mark.parametrize("num_tokens", [1, 16, 32])
@pytest.mark.parametrize("n_qh", [8, 16])
@pytest.mark.parametrize("n_kh", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("mrope_section", [[16, 24, 24], [0, 32, 32]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_interleaved", [False, True])
def test_mrope(...):
    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, query, key, cos_table, sin_table, mrope_section, is_interleaved)
```

### 5.2 精度要求

| 指标 | 要求 |
|------|------|
| atol | ≤ 1e-2 |
| rtol | ≤ 1e-2 |

---

## 6. 总结

本文档描述了 MRoPE 算子在昇腾 NPU 平台上的 Triton 实现，覆盖：
- 从 RoPE 到 MRoPE 的演进背景
- mojo_opset 算子库的模块架构
- 核心接口设计与规格约束
- 测试验证方案
