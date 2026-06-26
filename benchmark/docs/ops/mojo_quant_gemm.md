# MojoQuantGemm

## 算子说明

`MojoQuantGemm` 是融合量化的矩阵乘算子（W8A8），执行 int8 矩阵乘并在 int32 累加后，按 per-token 激活 scale 与 per-channel 权重 scale 反量化，最终 cast 到 `output_dtype`。

**计算公式：**

```
output = (input_i8 @ weight_i8) * input_scale * weight_scale
```

其中：
- `input_i8`：量化后的激活，shape `(M, K)`，`int8`
- `weight_i8`：量化后的权重，逻辑 shape `(K, N)`；内部存储布局由 `trans_weight` 决定
- `input_scale`：运行时 per-token 激活 scale，shape `(M,)` 或 `(M, 1)`
- `weight_scale`：per-channel 权重 scale，shape `(N,)`，`bfloat16`
- `output`：反量化结果，shape `(M, N)`，dtype 为 `output_dtype`

**实现说明：**
- Core 参考实现（`torch` backend）用 float32 模拟 int8 GEMM（`torch.matmul` 不原生支持 int8→int32 累加）
- `torch_npu` backend 调用 `torch_npu.npu_quant_matmul`，硬件融合 int8 累加与 scale
- `ttx` backend 使用 Triton kernel `int8_gemm_dequant`，融合 epilogue 反量化

源码位置：`mojo_opset/core/operators/gemm.py`

---

## 参数列表

### 构造参数 `__init__`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_features` | `int` | 必填 | 权重的逻辑 K 维 |
| `out_features` | `int` | 必填 | 权重的逻辑 N 维，也是 `weight_scale` 的长度 |
| `output_dtype` | `torch.dtype` | `torch.bfloat16` | 输出 dtype，支持 `float32` / `float16` / `bfloat16` |
| `trans_weight` | `bool` | `False` | 权重布局标志，见下方说明 |
| `quant_dtype` | `torch.dtype` | `torch.int8` | 量化 dtype，当前仅支持 `int8` |
| `weight_dtype` | `torch.dtype` / `str` | `torch.int8` | 权重 dtype，当前仅支持 `int8` |

### 注册 Buffer

| 名称 | dtype | shape | 说明 |
|------|-------|-------|------|
| `weight` | `int8` | `(K, N)` 若 `trans_weight=False`；`(N, K)` 若 `trans_weight=True` | 量化权重 |
| `weight_scale` | `bfloat16` | `(N,)` | per-channel 反量化 scale |

### 前向参数 `forward`

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `torch.Tensor` | 量化激活，2D，`int8`，shape `(M, K)` |
| `input_scale` | `torch.Tensor` | per-token 激活 scale，shape `(M,)` 或 `(M, 1)` |

**返回值：** `torch.Tensor`，shape `(M, N)`，dtype 为 `output_dtype`

### `trans_weight` 布局说明

| `trans_weight` | `weight` 存储 shape | 计算时使用的 layout | 典型场景 |
|----------------|----------------------|---------------------|----------|
| `False` | `(K, N)` | 直接使用 | 权重以 KN 布局存储 |
| `True` | `(N, K)` | forward 内部转置为 `(K, N)` | 权重以 NK 布局存储（如部分 checkpoint 格式） |

---

## 输入输出形状限制

| 约束项 | 要求 |
|--------|------|
| `input` 维度 | 必须为 2D，shape `(M, K)` |
| `input` dtype | `int8` |
| `input` 最后一维 | `input.shape[-1] == in_features`（即 K） |
| `weight` 维度 | 必须为 2D |
| `weight` 逻辑 shape | `(K, N)`，其中 `K = in_features`，`N = out_features` |
| `weight_scale` shape | 必须为 `(out_features,)`，即 `(N,)` |
| `input_scale` shape | `(M,)` 或 `(M, 1)`，M 与 `input` 的 batch/token 维一致 |
| `quant_dtype` / `weight_dtype` | 当前仅支持 `torch.int8`，其他 dtype 会在构造时 assert 失败 |
| `output` shape | `(M, N)` |
| `output` dtype | 由 `output_dtype` 指定 |

**不满足约束时的行为：**
- `input` 非 2D、K 维不匹配、`weight_scale` shape 错误 → 抛出 `ValueError`

