# MojoPagedPrefillGQA

## 算子说明

`MojoPagedPrefillGQA` 是分页 KV cache 上的 **Prefill 阶段 GQA（Grouped Query Attention）** 算子。Query 以 TND 布局（`(T, Hq, D)`）按 batch 拼接；Key/Value 存放在分页物理块中，通过 `block_tables` 做逻辑到物理块的映射。

**计算流程（每个 batch 序列）：**

1. 从 `key_cache` / `value_cache` 按 `block_tables` 收集 KV，展开为 `(kv_len, Hkv, D)`
2. 若 `Hq != Hkv`，按 `gqa_layout`（`ABAB` / `AABB`）将 KV head 扩展到 `Hq`
3. 计算 `softmax(QK^T / sqrt(D)) @ V`，默认因果 mask（`is_causal=True`）
4. 输出 shape `(T, Hq, D)`

**实现说明：**
- Core 参考实现（`torch` backend）在 Python 中逐 batch 循环，用 `einsum` 模拟 attention
- `torch_npu` backend 调用 `torch_npu.npu_fused_infer_attention_score`，TND layout + `sparse_mode=3`（Page Attention）
- 其他 backend（如 `ttx`）可提供 Triton fused kernel

源码位置：`mojo_opset/core/operators/attention.py`

---

## 参数列表

### 构造参数 `__init__`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `is_causal` | `bool` | `True` | 是否使用因果（下三角）mask |
| `gqa_layout` | `str` | `"AABB"` | GQA head 分组布局，`"ABAB"` 或 `"AABB"` |

### 前向参数 `forward`

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | `torch.Tensor` | Query tokens，shape `(T, Hq, D)` |
| `key_cache` | `torch.Tensor` | Key cache，shape `(N_blocks, Hkv, block_size, D)` |
| `value_cache` | `torch.Tensor` | Value cache，shape `(N_blocks, Hkv, block_size, D)` |
| `cu_q_lens` | `torch.Tensor` | 累积 query 长度，shape `(B+1,)`，`int32`；`cu_q_lens[-1] == T` |
| `block_tables` | `torch.Tensor` | 逻辑块到物理块 ID，shape `(B, num_blocks)`，`int32` |
| `softmax_scale` | `Optional[float]` | Attention scale，默认 `1/sqrt(D)` |
| `cu_total_seq_lens` | `Optional[torch.Tensor]` | 累积 KV 总长度，shape `(B+1,)`；`None` 时等于 `cu_q_lens` |
| `mask` | `Optional[torch.Tensor]` | 自定义 mask（`is_causal=False` 时使用） |
| `max_q_len` | `Optional[int]` | 最大 query 长度（部分 fused backend 使用） |
| `max_total_seq_len` | `Optional[int]` | 最大 KV 长度（部分 fused backend 使用） |

**返回值：** `torch.Tensor`，shape `(T, Hq, D)`

---

## 输入输出形状限制

| 约束项 | 要求 |
|--------|------|
| `query` shape | `(T, Hq, D)`，T 为所有 batch 的 query token 总数 |
| `key_cache` / `value_cache` shape | `(N_blocks, Hkv, block_size, D)` |
| `cu_q_lens` | `int32`，shape `(B+1,)` |
| `block_tables` | `int32`，shape `(B, num_blocks)`；有效物理块 ID ≥ 0 |
| GQA | `Hq % Hkv == 0` |
| `torch_npu` fused kernel | `head_dim % 128 == 0`；`block_size % 128 == 0` 且 `block_size <= 512` |
| `torch_npu` fused kernel | 通过 `npu_fused_infer_attention_score_v2`（TND + `sparse_mode=3`）支持 `cu_total_seq_lens`（`kv_len > q_len` 的 prefix cache 场景） |

---

## 序列长度语义（重要）

mojo_opset 测试与 benchmark 中 `kv_computed_len` 对应的是 **已有 prefix 的 cache 长度**（`cache_len`），**不是**总 KV 长度 `kv_len`：

| 符号 | 含义 | 关系 |
|------|------|------|
| `q_len` | 本次 prefill 新增的 query token 数 | 由 `cu_q_lens` 推出 |
| `kv_computed_len` / `cache_len` | 已在 KV cache 中计算好的 prefix 长度 | benchmark 字段名 |
| `kv_len` | 每个 batch 可见的总 KV 长度 | `kv_len = cache_len + q_len` |

原生算子通过 `cu_total_seq_lens` 表达总 KV 长度；`cu_total_seq_lens is None` 时退化为 `kv_len == q_len`（纯 prefill）。

**示例：** `q_len=4096`, `kv_computed_len=4096` → `cache_len=4096`, `kv_len=8192`。

与 `tests/perf/test_attention.py` 中 `generate_paged_prefill_data(..., max_kv_computed_len=...)` 的命名一致：`kv_lens = q_lens + kv_cache_lens`。

---

## Benchmark workload 参数

`benchmark/workloads/mojo_paged_prefill_gqa.json` 使用以下字段（与原生 API 对齐）：

| 字段 | 说明 |
|------|------|
| `batch_size` | Batch 大小 B |
| `num_q_heads` | Query head 数 Hq |
| `num_kv_heads` | KV head 数 Hkv |
| `head_dim` | Head 维度 D |
| `q_len` | 每个 batch 的 query 长度（等长 batch） |
| `kv_computed_len` | **cache_len**：已有 KV prefix 长度；`0` 表示纯 prefill |
| `block_size` | KV cache 分页块大小 |
| `is_causal` | 默认 `true` |
| `gqa_layout` | 默认 `"AABB"` |
| `dtype` | 默认 `"bfloat16"` |

`kv_computed_len > 0` 时 benchmark 会构造 `cu_total_seq_lens`；`torch_npu` 通过 FIA v2 的 `actual_seq_kvlen` 支持该场景。

**运行示例：**

```bash
python benchmark/run_perf.py --workload benchmark/workloads/mojo_paged_prefill_gqa.json
```
