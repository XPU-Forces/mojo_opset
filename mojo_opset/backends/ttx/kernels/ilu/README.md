# ILU Triton Kernels

本目录包含 Iluvatar (ILU) 平台的 Triton kernel 实现。

## 环境变量

### `TRITON_DISABLE_AUTOTUNE`

控制 Triton kernel 的 autotune 行为。

| 值 | 行为 | 适用场景 |
|---|---|---|
| 不设置（默认） | 使用 `triton.autotune` 编译并 benchmark 所有配置，选出最优 | 性能测试 |
| `1` | 绕过 `triton.autotune`，仅编译 `selected_idx` 指定的配置 | 精度测试 |

**原理**：设置 `TRITON_DISABLE_AUTOTUNE=1` 后，`smart_triton_autotune` 会改用 `triton.heuristics` 将配置参数直接注入 `triton.jit` 调用，完全避免 autotune 的编译开销。

**使用示例**：

```bash
# 精度测试（快速，跳过 autotune）
 MOJO_BACKEND=ttx TRITON_DISABLE_AUTOTUNE=1 pytest mojo_opset/tests/accuracy/operators/test_attention.py

# 性能测试（完整 autotune）
pytest mojo_opset/tests/perf/test_attention.py
```