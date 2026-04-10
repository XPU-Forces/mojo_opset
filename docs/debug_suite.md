# MojoDebugger — mojo_opset 调试套件

## 概述

`MojoDebugger` 是一个轻量级、非侵入式的 LLM 模型调试工具，专为 `mojo_opset` 构建的模型设计。它提供两项核心能力：

- **Dump** — 将指定层、指定算子的输入/输出张量保存到磁盘，供离线分析。
- **Compare** — 将加速后端（如 TTX）的算子输出与 `torch` 参考实现进行逐元素对比，实时打印绝对误差、相对误差和余弦相似度。

两项功能均支持**运行时动态切换规则**——无需重建模型，即可在不同 forward 之间更换要观测的算子。

---

## 设计方案

### 整体架构

```
MojoDebugger.enable()          ← 补丁 MojoOperator.__new__，捕获构造参数
        │
   模型构建                      ← 每个 MojoOperator 自动记录 (core_cls, init_args)
        │
dbg = MojoDebugger()
dbg.attach(model)              ← 遍历模型树，传播 layer_idx，
        │                        在每个 MojoOperator 上注册 forward hook
        │
dbg.set_dump("2:input_layernorm")
dbg.set_compare("*:mlp.act_fn")
        │
   model(input_ids)            ← 每个 MojoOperator.forward() 执行时：
        │                        1. Hook 读取当前活跃规则（API / 环境变量）
        │                        2. 匹配命中 → 执行 dump 和/或 compare
        │                        3. Compare: 惰性构建 torch 影子实例 → 运行参考 forward → 输出差异
        │
dbg.detach()                   ← 移除 hook，释放影子实例
```

### 核心设计决策

**1. Dual-Build 影子实例机制**

调试模式会补丁 `MojoOperator.__new__`，在每个算子构造时记录其构造参数。当 compare 规则被触发时，系统**惰性创建**一个 torch 影子实例——使用相同的构造参数，但强制走 `torch` 后端。权重通过 `load_state_dict` 从主实例同步。这样可以安全地对比加速后端和 torch 参考的输出，避免因后端 `__init__` 中的权重变换（如转置、量化）导致的不一致。

**内存优化**：如果后端类没有重写 `__init__`（仅重写 `forward`），则不创建影子实例，直接调用核心类的 `forward` 方法，节省显存。

**2. 语义化 `layer_idx` 传播**

调试器不通过解析 module path 字符串来识别层号，而是查找具有 `layer_idx` 属性的模块（如 `DecoderLayer.layer_idx`），并将其传播给所有子 `MojoOperator`。由此产生稳定、可读的标识符，如 `layer5.input_layernorm`。

对于不在任何 DecoderLayer 内的算子（如全局 RMSNorm、LM Head），`layer_idx` 为 `None`，在规则中用 `none` 表示。

**3. 动态规则引擎**

规则来源有两个优先级：
1. **Python API**（`set_compare`、`set_dump`）—— 优先级最高
2. **环境变量**（`MOJO_DEBUG_COMPARE`、`MOJO_DEBUG_DUMP`）—— 作为兜底

Hook 在每次 forward 时重新读取活跃规则，因此支持在迭代之间动态切换观测目标。环境变量的变更也会被即时感知（带缓存优化）。

**4. 健壮性保障**

- 所有调试逻辑的异常均被捕获并以 warning 输出——**永远不会中断推理**。
- 不匹配任何算子的规则会产生警告，并列出可用目标。
- Dump 文件按 `rank{LOCAL_RANK}/` 子目录组织，适配分布式训练。
- 步数计数器（`max_steps`）限制每个算子的 dump/compare 次数，防止磁盘和日志爆炸。

---

## 规则格式

规则格式为 `layer_idx:op_name`，多个规则用 `;` 分隔：

| 规则示例 | 含义 |
|---|---|
| `5:input_layernorm` | 第 5 层的 `input_layernorm` |
| `*:mlp.act_fn` | **所有层**的 `mlp.act_fn` |
| `none:norm` | 全局算子 `norm`（不在任何 DecoderLayer 内） |
| `*:MojoRMSNorm` | 所有 `MojoRMSNorm` 实例（按类名匹配） |
| `0:input_layernorm;0:mlp.down_proj` | 第 0 层的两个算子 |

`attach()` 后，调试器会打印所有已发现的算子及其层号归属，帮助你编写规则：

```
[MojoDebug] Attached. 58 MojoOperator instances discovered:
  layer 0-4: input_layernorm (MojoRMSNorm), mlp.act_fn (MojoSwiGLU), ...
  global: embed (MojoEmbedding), lm_head (MojoLinear), norm (MojoRMSNorm)
[MojoDebug] Rule format: "layer_idx:op_name", e.g. "5:input_layernorm" or "*:self_attn.rope"
```

---

## 使用方式

### 方式一：Python API（推荐）

```python
from mojo_opset.utils.debugger import MojoDebugger

# 第 1 步：在模型构建之前启用调试模式
MojoDebugger.enable()

# 第 2 步：正常构建和加载模型
model = build_llama_model(config)
model.load_state_dict(checkpoint)

# 第 3 步：创建调试器并挂载到模型
dbg = MojoDebugger(dump_dir="./debug_output", max_steps=5)
dbg.attach(model)

# 第 4 步：设置规则 — dump 第 0 层的 input_layernorm
dbg.set_dump("0:input_layernorm")

# 第 5 步：执行推理
output = model(input_ids)

# 第 6 步：切换到 compare 模式，观测下一次 forward
dbg.set_compare("*:mlp.act_fn")
output = model(input_ids)

# 第 7 步：清理
dbg.detach()
```

### 方式二：环境变量

通过环境变量启用调试，无需修改模型构建代码：

```bash
# 启用调试模式（import mojo_opset 时自动补丁 MojoOperator.__new__）
export MOJO_DEBUG=1

# 设置 compare / dump 规则
export MOJO_DEBUG_COMPARE="5:input_layernorm;5:mlp.act_fn"
export MOJO_DEBUG_DUMP="0:self_attn.q_proj"

# 可选：自定义 dump 目录和最大步数
export MOJO_DEBUG_DUMP_DIR="./my_debug_dumps"
export MOJO_DEBUG_MAX_STEPS=3

python run_inference.py
```

> **注意**：`MOJO_DEBUG=1` 仅自动完成 `enable()` 步骤。你仍需在脚本中创建 `MojoDebugger` 实例并调用 `attach(model)`。

```python
# run_inference.py
from mojo_opset.utils.debugger import MojoDebugger

model = build_model(config)
model.load_state_dict(checkpoint)

dbg = MojoDebugger()
dbg.attach(model)

# 规则从环境变量 MOJO_DEBUG_COMPARE / MOJO_DEBUG_DUMP 读取
output = model(input_ids)

dbg.detach()
```

### 动态规则切换

规则可以在 forward 之间随时更换——适用于逐层排查精度问题的场景：

```python
dbg.attach(model)

# 第 1 次推理：对比第 0 层
dbg.set_compare("0:input_layernorm")
model(input_ids_1)

# 第 2 次推理：切换到第 4 层
dbg.set_compare("4:mlp.down_proj")
model(input_ids_2)

# 第 3 次推理：同时 dump 和 compare
dbg.set_dump("2:post_attention_layernorm")
dbg.set_compare("2:post_attention_layernorm")
model(input_ids_3)

# 清除所有规则（后续 forward 无调试开销）
dbg.clear_rules()
model(input_ids_4)
```

环境变量同样支持运行时切换：

```python
import os
os.environ["MOJO_DEBUG_COMPARE"] = "0:self_attn.q_proj"
model(input_ids_1)

os.environ["MOJO_DEBUG_COMPARE"] = "0:self_attn.k_proj"
model(input_ids_2)
```

### Dump 输出

> **性能提示**：Dump 通过 `torch.save` 在 forward hook 中同步写入磁盘。对少量算子做偶尔 dump 是可接受的，但如果同时对大量算子启用 dump，同步 I/O 会显著阻塞执行流水线。建议配合 `max_steps` 限制写入次数，或一次只 dump 少量目标算子。

Dump 将输入和输出张量保存为 `.pt` 文件，路径格式为 `{dump_dir}/rank{LOCAL_RANK}/`：

```
./mojo_debug_dump/
  rank0/
    layer2.input_layernorm_step0_input.pt
    layer2.input_layernorm_step0_output.pt
    layer2.input_layernorm_step1_input.pt
    layer2.input_layernorm_step1_output.pt
```

加载并分析：

```python
import torch
output = torch.load("./mojo_debug_dump/rank0/layer2.input_layernorm_step0_output.pt")
print(output.shape, output.dtype, output.mean(), output.std())
```

同时，每次 dump 还会在日志中打印张量统计信息：

```
[MojoDebug] layer2.input_layernorm step=0 | output shape=(2, 16, 128) dtype=torch.float32 mean=0.0123 std=0.9876 min=-2.345 max=3.456
```

### Compare 输出

Compare 在日志中逐算子打印精度指标：

```
[MojoDebug] layer5.input_layernorm step=0 | max_abs_diff=1.526e-05 max_rel_diff=3.814e-04 cos_sim=0.999998
[MojoDebug] layer5.mlp.act_fn step=0      | max_abs_diff=7.629e-06 max_rel_diff=1.907e-04 cos_sim=1.000000
```

| 指标 | 说明 |
|---|---|
| `max_abs_diff` | 后端输出与 torch 参考之间的最大绝对误差 |
| `max_rel_diff` | 最大相对误差（分母 clamp 到 1e-12 以避免除零） |
| `cos_sim` | 展平后的输出向量之间的余弦相似度 |

> **注意**：Compare 仅报告数值差异，不做 pass/fail 判定。开发者可根据自身精度要求自行判断。

### Compare 模式：observe vs replace

Compare 支持两种模式，通过 `compare_mode` 参数或 `MOJO_DEBUG_COMPARE_MODE` 环境变量指定：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `observe`（默认） | 对比后不修改输出，加速后端结果原样传给下游 | 观察每个算子的累积误差 |
| `replace` | 对比后将加速后端输出**替换**为 torch 参考输出 | 逐层隔离误差，定位误差来源 |

**observe 模式**下，如果 Layer 0 引入了误差 `e0`，Layer 1 的 compare 看到的输入已经包含 `e0`，其报告的是累积误差。

**replace 模式**下，每个 compare 命中的算子的输出都被 torch 参考结果替换，后续算子的输入是"干净"的。每个算子的 compare 报告仅反映该算子自身引入的误差。

```python
# 逐层误差隔离
dbg = MojoDebugger(compare_mode="replace")
dbg.attach(model)
dbg.set_compare("*:input_layernorm;*:mlp.act_fn")
model(input_ids)  # 每个算子的 diff 仅反映自身误差

# 运行时切换
dbg.set_compare_mode("observe")   # 切回观察模式
model(input_ids)
```

> **提示**：建议先用 `observe` 模式粗略定位误差较大的层，再用 `replace` 模式逐层隔离，确认误差的具体来源。

---

## API 参考

### 类方法

| 方法 | 说明 |
|---|---|
| `MojoDebugger.enable()` | 补丁 `MojoOperator.__new__`，捕获构造参数。**必须在模型构建之前调用。** |
| `MojoDebugger.disable()` | 恢复原始 `MojoOperator.__new__`。 |

### 实例方法

| 方法 | 说明 |
|---|---|
| `__init__(dump_dir=None, max_steps=None, compare_mode=None)` | 创建调试器。`dump_dir` 依次取：参数 → `MOJO_DEBUG_DUMP_DIR` → `./mojo_debug_dump`。`compare_mode` 默认 `"observe"`。 |
| `attach(model)` | 遍历模型，传播 `layer_idx`，在所有 `MojoOperator` 上注册 hook。 |
| `detach()` | 移除 hook，释放影子实例，清理调试属性。 |
| `set_compare(rules)` | 设置 compare 规则，如 `"5:input_layernorm;*:mlp.act_fn"`。传入空字符串清除。 |
| `set_dump(rules)` | 设置 dump 规则。传入空字符串清除。 |
| `set_compare_mode(mode)` | 切换 compare 模式：`"observe"` 或 `"replace"`。 |
| `clear_rules()` | 清除所有通过 API 设置的规则。 |
| `set_dump_dir(path)` | 运行时更改 dump 目录。 |
| `set_max_steps(n)` | 设置每个算子的最大 dump/compare 步数。 |
| `reset_step_counters()` | 重置所有步数计数器（配合 `max_steps` 实现"再 dump 一轮"）。 |

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MOJO_DEBUG` | 设为 `1` 时，`import mojo_opset` 自动调用 `enable()` | — |
| `MOJO_DEBUG_COMPARE` | Compare 规则字符串（格式同 `set_compare`） | — |
| `MOJO_DEBUG_DUMP` | Dump 规则字符串（格式同 `set_dump`） | — |
| `MOJO_DEBUG_DUMP_DIR` | Dump 输出目录 | `./mojo_debug_dump` |
| `MOJO_DEBUG_MAX_STEPS` | 每个算子的最大 dump/compare 步数 | 无限制 |
| `MOJO_DEBUG_COMPARE_MODE` | Compare 模式：`observe` 或 `replace` | `observe` |

---

## 常见问题

**Q：忘记在模型构建前调用 `enable()` 会怎样？**

对于后端没有重写 `__init__` 的算子（大多数情况），compare 仍可正常工作。对于后端有自定义 `__init__` 的算子，会打印 warning 并退化为直接调用核心类 `forward`，如果后端在 `__init__` 中对权重做了变换，则对比结果可能不准确。

**Q：调试器会影响推理输出吗？**

在默认的 `observe` 模式下不会。Hook 仅读取输入/输出，Compare 在传入参考 forward 前会深拷贝输入，原始计算不受任何影响。在 `replace` 模式下，compare 命中的算子的输出会被 torch 参考结果替换——这是有意为之的行为，用于逐层隔离误差。

**Q：分布式训练如何使用？**

Dump 文件按 `rank{LOCAL_RANK}/` 子目录保存。日志使用 rank-0-only 打印，避免重复输出。

**Q：能否对同一个算子同时 dump 和 compare？**

可以。两种规则在每次 forward 中独立检查和执行。

**Q：规则没有匹配到任何算子怎么办？**

会打印 warning，提示检查 `attach()` 输出中列出的可用目标。推理正常继续，不受影响。

**Q：`max_steps` 到达上限后还想再 dump 怎么办？**

调用 `dbg.reset_step_counters()` 重置计数器，后续 forward 会重新从 step 0 开始。
