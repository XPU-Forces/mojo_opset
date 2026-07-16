# mojo_opset 性能测试

`mojo_opset.benchmark` 复用 xpu-perf `micro_perf` 的设备、tensor、profiling、多进程和报告能力。
开发者新增一个 `Operator` 或 `Function` 时，只维护
`mojo_opset/tests/perf_new/<target>.py`，框架会动态生成 xpu-perf 需要的 base/vendor opdef 类和
workload task。

## 运行

在仓库根目录安装项目和 xpu-perf，并准备好 Ascend/`torch_npu` 环境。

快速单进程测试：

```bash
# 全部 smoke case，默认 device 0
python -m mojo_opset.benchmark.run_perf

# 只测一个 target/provider
python -m mojo_opset.benchmark.run_perf \
  --ops mojo_quant_gemm \
  --providers torch_npu \
  --device 0

# 跑完整 case 并导出 xpu-perf 报告
python -m mojo_opset.benchmark.run_perf \
  --preset full \
  --providers torch_npu \
  --report_dir benchmark_reports
```

多进程/多卡测试：

```bash
# 单卡子进程
python -m mojo_opset.benchmark.launch \
  --backend NPU \
  --device 0 \
  --preset smoke

# 多卡
python -m mojo_opset.benchmark.launch \
  --backend NPU \
  --device 0,1 \
  --preset full
```

两个入口都支持：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--preset` | `smoke` | 按 case tag 选择；`all` 表示全部 |
| `--ops` | 全部 spec | 逗号分隔的 target 名称 |
| `--providers` | `base,torch_npu,ttx` | provider 顺序；不支持的 case 会说明原因并跳过 |
| `--timing` | spec 配置（默认 `profiler`） | 临时覆盖为 `profiler` 或 `event` |
| `--report_dir` | runner 各自默认值 | 导出 xpu-perf jsonl/csv 报告 |

`run_perf.py` 使用整数 `--device`；`launch.py` 支持 xpu-perf 的 server、NUMA 和多 device 参数。

不指定 `--providers` 时会按 `base,torch_npu,ttx` 尝试；某个 target 未声明的 provider 会自动跳过。

两个预设的约定是：

| preset | 含义 |
|---|---|
| `smoke` | 每个 target 选一个可快速验证框架和 kernel 的代表 case |
| `full` | target 的完整性能参数集 |

## 开发者只写一个 spec

### Operator

```python
import torch

from mojo_opset import MojoQuantGemm
from mojo_opset.benchmark import PerfWorkload, mojo_perf, perf_case, tensor


CASES = (
    perf_case("m16_k1024_n1024", tags=("smoke",), m=16, k=1024, n=1024),
    perf_case("m4096_k4096_n4096", tags=("full",), m=4096, k=4096, n=4096),
)


@mojo_perf(
    name="mojo_quant_gemm",
    target=MojoQuantGemm,
    cases=CASES,
    providers=("torch_npu",),
)
def quant_gemm_workload(case):
    m, k, n = case["m"], case["k"], case["n"]
    return PerfWorkload(
        op_kwargs={"in_features": k, "out_features": n},
        inputs={
            "input": tensor((m, k), torch.int8, creator=torch.zeros),
            "input_scale": tensor((m,), torch.float32, creator=torch.ones),
            "weight": tensor((k, n), torch.int8, creator=torch.zeros),
            "weight_scale": tensor((n,), torch.bfloat16, creator=torch.ones),
        },
        outputs={"y": tensor((m, n), torch.bfloat16)},
        state={"weight": "weight", "weight_scale": "weight_scale"},
        flops=2 * m * k * n,
    )
```

适配层通过 `target.get_backend_impl(...)` 找到实现。对于 `MojoOperator`，它会用
`op_kwargs` 构造实例、移动到 device、绑定 `state`，最后按 `args/kwargs` 调用。未填写 `args` 时，
框架按 `inputs` 顺序自动传递未被 `state` 或 `kwargs` 使用的 tensor；签名顺序不同或需要插入字面量时
再显式填写 `args`。

case 中可以直接使用 `torch.dtype`，例如 `perf_case("fp16", dtype=torch.float16)`。框架生成 xpu-perf
task 时会自动序列化为 `"float16"`，而 workload 构造函数仍收到原始的 `torch.float16`，不需要手动解析
或维护 dtype 映射表。

不同 provider 应保持相同的算子语义输入；如果只是支持的 shape、dtype 或 layout 范围不同，使用
`perf_provider(..., supports=...)` 声明 capability，不需要复制 case 或 workload：

```python
from mojo_opset.benchmark import perf_provider


@mojo_perf(
    ...,
    providers=(
        "ttx",
        perf_provider(
            "torch_npu",
            supports=lambda case: (
                case["head_dim"] % 128 == 0
                and case["block_size"] % 128 == 0
                and case["block_size"] <= 512
            ),
            unsupported_reason="requires aligned head_dim and block_size",
        ),
    ),
)
```

runner 会在分发前过滤 `(case, provider)` 组合并输出 skip 原因；显式选择的 provider 没有任何可运行
case 时会报错。provider 真正需要不同的物理 layout 或预打包时，仍通过 `tensor_factory` 或
`target_factory` 封装，case 中不暴露 backend 私有参数。

### Function

`MojoFunction` 使用相同装饰器，不需要额外适配类。框架不会实例化 Function，而是调用指定后端实现的
`apply(...)`。

```python
import torch

from mojo_opset import MojoSiluFunction
from mojo_opset.benchmark import PerfWorkload, mojo_perf, perf_case, tensor


@mojo_perf(
    name="mojo_silu_function",
    target=MojoSiluFunction,
    cases=(perf_case("1024x1024", tags=("smoke",), m=1024, n=1024),),
)
def silu_workload(case):
    return PerfWorkload(
        inputs={
            "input": tensor((case["m"], case["n"]), torch.float16, creator=torch.randn)
        },
        outputs={"output": tensor((case["m"], case["n"]), torch.float16)},
    )
```

`args/kwargs` 中的字符串表示 tensor 名，其他值会按原样传入；若需要字面字符串，可用
`literal("value")`。简单 Function 同样可以省略 `args`；Function 的 `apply` 只接受位置参数，签名顺序
与 `inputs` 不同或者需要 tensor 之外的标量时，应显式填写，例如
`args=("input", (hidden_size,), "weight", 1e-5)`。

## 复杂场景的扩展口

普通算子通常只需要 `inputs/outputs/op_kwargs/state`，必要时再写 `args/kwargs`。下面的扩展仍写在
同一个 spec 文件中，
不要求开发者定义 `PerfCase` 或 benchmark 子类：

| 扩展 | 使用场景 | 计时区间内执行 |
|---|---|---|
| 单 tensor `creator` | `zeros/ones/randn/randint` 等独立数据 | 否 |
| `tensor_factory(device)` | attention、KV cache 等多个 tensor 必须共享长度或索引关系 | 否 |
| `target_factory(target_cls, device)` | Operator 构造器必须接收 weight 等特殊参数 | 否 |
| `state` | 把生成的 weight/cache 绑定到 Operator 属性 | 否 |
| `run(target, tensors)` | 非标准调用顺序、通信或其他公共 args/kwargs 无法表达的调用 | 是 |
| `engine` / `profiling` | 通信引擎或特殊 kernel 统计规则 | 按 xpu-perf 语义 |

例如，关联数据应由一次 workload 级构造完成：

```python
def tensor_factory(device):
    seq_lens = torch.tensor([128, 256], dtype=torch.int32, device=device)
    block_tables = make_block_tables(seq_lens, device)
    return {"seq_lens": seq_lens, "block_tables": block_tables, "q": make_q(device)}


return PerfWorkload(
    inputs={...},
    args=("q", "seq_lens", "block_tables"),
    tensor_factory=tensor_factory,
)
```

构造签名特殊的 Operator 可提供一个很小的工厂；框架仍负责 `.to(device)`、state 绑定和调用：

```python
def target_factory(target_cls, device):
    placeholder = torch.empty(weight_shape, dtype=dtype, device=device)
    return target_cls(weight=placeholder, trans_weight=False)
```

如果调用顺序无法由公共字段表达，可以提供 `run(callable_target, tensors)`；这里的第一个参数对
Operator 是可调用实例，对 Function 是后端实现的 `apply`。

```python
def run_attention(target, tensors):
    return target(tensors["q"], tensors["k"], tensors["v"], causal=True)


PerfWorkload(..., run=run_attention)
```

需要通信引擎时可在 `@mojo_perf(..., engine="XCCLEngine")` 指定。通信生命周期或特殊 profiling
仍无法表达时，应扩展公共 `PerfWorkload`/provider hook；算子描述文件本身不需要定义 benchmark 子类。

## Profiling 与 kernel 选择

默认配置是：

```python
profile(timing="profiler", reduction="span")
```

即启用 xpu-perf profiler，选择该次调用的全部 kernel，以最早开始到最晚结束的时间跨度作为
latency。可按 provider 配置：

```python
from mojo_opset.benchmark import profile


@mojo_perf(
    ...,
    profiling={
        "base": profile(),
        "torch_npu": profile(
            kernels=("quant_matmul", "cast"),
            match="contains",
            reduction="sum",
        ),
    },
)
```

配置项：

| 字段 | 值 | 语义 |
|---|---|---|
| `timing` | `profiler` / `event` | profiler kernel 记录或 xpu-perf event 计时 |
| `kernels` | kernel selector 元组或 `None` | `None` 为全部；否则只统计匹配事件 |
| `match` | `exact` / `contains` / `regex` | selector 匹配方式 |
| `reduction` | `span` / `sum` | 取所选事件覆盖区间或逐 kernel duration 之和 |

每个 selector 必须至少匹配一个 profiler event；找不到时 benchmark 会失败，并列出可用 event，
不会静默退化到全部 kernel。`event` 计时没有 kernel 明细，因此不能配置 `kernels` 或 `sum`。

在严格单流且 kernel 首尾连续、没有 host 间隙时，全部 kernel 的 `span` 与 `sum` 接近；存在间隙时
`span` 更大，存在多流重叠时 `sum` 可能更大。二者都来自 profiler，只是聚合方法不同。
`--timing event` 则切换到 xpu-perf 的 event 路径，主要用于对照。

报告会写入 `timing_mode`；profiler 模式还包含 `latency_reduction`，选择部分 kernel 时包含
`profile_kernels/profile_match`。

## 数据由谁创建

`torch.zeros/ones/randn` 在 spec 导入时不会执行。它们作为 `OpTensorInfo.creator` 交给 xpu-perf，
由 `BasicOp` 传入 `size/dtype/device` 创建首份 tensor，再根据缓存和显存策略 clone。若指定
`tensor_factory`，框架只调用一次来得到完整 mapping，校验所有 shape/dtype 后再整体 clone，因此
`seq_lens/block_tables/cache` 等关系不会被拆散。`state` 在计时前绑定，数据构造和参数替换不计入
device latency。

## 目录

```text
mojo_opset/benchmark/
  api.py                     # perf_case/tensor/PerfWorkload/mojo_perf/profile
  xpu_adapter.py             # 动态生成 xpu-perf opdef，适配 Operator/Function
  run_perf.py                # 单进程性能 runner
  launch.py                  # xpu-perf 多进程 launcher
  runner_common.py           # case 解析、provider 与插件路径选择
  plugins/                   # descriptor 的 base/vendor 动态注册桥

mojo_opset/tests/perf_new/   # 每个领域一个 descriptor 性能测试文件

docs/benchmark/              # 框架与算子性能测试文档
```
