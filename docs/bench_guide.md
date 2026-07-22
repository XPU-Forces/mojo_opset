# mojo_opset 性能测试

`mojo_opset.benchmark` 使用 xpu-perf 执行性能测试。开发者为每个 Operator 或 Function 维护一个
`mojo_opset/tests/perf_new/operators/<target>.py` 或 `functions/<target>.py` 描述文件，定义测试参数和调用方式即可。

## 基本机制

一个性能描述包含三部分：

- `perf_case(...)`：一组 shape、dtype 等测试参数；`smoke` 用于快速验证，`full` 用于完整测试。
- `@mojo_perf(...)`：关联 Mojo target、case 和可选配置。
- `PerfWorkload`：描述输入输出、Operator 构造参数、state 和调用参数。

普通 spec 不需要声明 provider。框架会根据当前平台，从 target 已注册的 Mojo backend 中自动选择：

- NPU 环境通常选择 `base`、`ttx`、`torch_npu`。
- ILU 环境通常选择 `base`、`ixformer`、`ttx`。
- `base` 对应 Mojo core 的通用 `torch` 实现。

不传 `--providers` 时，默认运行 `base` 和当前算子的全部可用 vendor provider。

`--providers` 是严格过滤器；显式传入后只运行指定 provider。

## 运行

快速单进程测试：

```bash
# 运行全部 smoke case
python -m mojo_opset.benchmark.run_perf

# 运行一个算子，自动测试它当前可用的全部 provider
python -m mojo_opset.benchmark.run_perf \
  --ops mojo_quant_gemm \
  --device 0

# 只测试 torch_npu
python -m mojo_opset.benchmark.run_perf \
  --ops mojo_quant_gemm \
  --providers torch_npu

# 同时测试 reference 和 torch_npu
python -m mojo_opset.benchmark.run_perf \
  --ops mojo_quant_gemm \
  --providers base,torch_npu
```

多进程或多卡测试：

```bash
# 单卡
python -m mojo_opset.benchmark.launch --device 0 --preset smoke

# 多卡
python -m mojo_opset.benchmark.launch --device 0,1 --preset full
```

常用参数：

| 参数 | 默认行为 | 作用 |
|---|---|---|
| `--ops` | 全部 spec | 逗号分隔的 target 名称 |
| `--preset` | `smoke` | 选择 `smoke`、`full` 或 `all` case |
| `--providers` | `base` 和全部可用 vendor provider | 过滤要运行的 provider |
| `--device` | `run_perf` 为 0；`launch` 为全部设备 | 选择设备 |
| `--timing` | spec 配置，通常为 `profiler` | 临时切换 `profiler` 或 `event` |
| `--report_dir` | runner 默认值 | 导出 xpu-perf 报告 |

设备 backend 会根据当前环境自动选择，也可以用 `--backend` 显式覆盖。

## 编写 Operator spec

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

常用 `PerfWorkload` 字段：

| 字段 | 作用 |
|---|---|
| `inputs` / `outputs` | tensor 的 shape、dtype 和可选 creator |
| `op_kwargs` | 构造 Operator 的参数 |
| `state` | 将生成的 weight、cache 等绑定到 Operator 属性 |
| `args` / `kwargs` | 计时调用的参数；简单情况可省略 `args` 自动推导 |
| `forward_args` | Function backward 在计时外准备 ctx 的 forward 参数 |
| `flops` | 用于报告计算吞吐 |


## 编写 Function spec

Function forward 继续通过后端 Function 的 `apply(...)` 执行，名称显式带 `_forward`：

```python
from mojo_opset import MojoSiluFunction


CASES = (
    perf_case("1024x1024", tags=("smoke",), rows=1024, cols=1024),
)


@mojo_perf(
    name="mojo_silu_function_forward",
    target=MojoSiluFunction,
    cases=CASES,
)
def silu_forward_workload(case):
    shape = (case["rows"], case["cols"])
    return PerfWorkload(
        inputs={"x": tensor(shape, torch.float16, creator=torch.randn)},
        outputs={"y": tensor(shape, torch.float16)},
    )
```

需要测试 backward 时，再增加一个 `phase="backward"` descriptor：

```python
@mojo_perf(
    name="mojo_silu_function_backward",
    target=MojoSiluFunction,
    cases=CASES,
    phase="backward",
)
def silu_backward_workload(case):
    shape = (case["rows"], case["cols"])
    return PerfWorkload(
        inputs={
            "x": tensor(shape, torch.float16, creator=torch.randn),
            "dy": tensor(shape, torch.float16, creator=torch.randn),
        },
        outputs={"dx": tensor(shape, torch.float16)},
        forward_args=("x",),
    )
```

框架为每份输入在计时外调用对应 provider 的 `forward(ctx, ...)`，同步设备后只重复测量
`backward(ctx, ...)`。`args` 会自动推导为没有出现在 `forward_args` 中的输入，上例为
`("dy",)`。开发者不需要创建 ctx 或编写 prepare。

Function 参数包含标量时直接写入参数元组；字符串表示 tensor 名，字面字符串使用
`literal("value")`。Function 不支持关键字调用。

运行同一 Function 的前反向：

```bash
python -m mojo_opset.benchmark.run_perf \
  --ops mojo_silu_function_forward,mojo_silu_function_backward \
  --providers base,ttx
```

## Provider 限制

只有 workload 不适用于所有已注册 vendor backend，或不同 vendor provider 的支持范围不同，才在
spec 中写 `providers`。它会成为 vendor provider 的 allowlist，不影响 `base`：

```python
from mojo_opset.benchmark import perf_provider


@mojo_perf(
    ...,
    providers=(
        "ttx",
        perf_provider(
            "torch_npu",
            supports=lambda case: case["head_dim"] % 128 == 0,
            unsupported_reason="head_dim must be divisible by 128",
        ),
    ),
)
```

不支持的 `(case, provider)` 组合会被跳过并显示原因。

## 复杂 workload

大多数算子只需要公共字段。特殊情况仍在同一个 spec 中通过以下扩展表达：

| 扩展 | 使用场景 | 是否计入被测调用 |
|---|---|---|
| tensor `creator` | 独立创建 `zeros`、`ones`、`randn` 等数据 | 否 |
| `tensor_factory(device)` | attention、KV cache 等关联数据 | 否 |
| `target_factory(target_cls, device)` | 特殊 Operator 构造方式 | 否 |
| `run(target, tensors)` | 非标准调用或通信逻辑 | 是 |
| `engine="XCCLEngine"` | 通信算子 | 按 xpu-perf 语义 |

## Profiling 和 kernel 选择

默认使用 profiler，统计本次调用全部 kernel 的时间跨度：

```python
profile(timing="profiler", reduction="span")
```

需要只统计部分 kernel 时，可按 provider 配置：

```python
from mojo_opset.benchmark import profile


@mojo_perf(
    ...,
    profiling={
        "torch_npu": profile(
            kernels=("quant_matmul", "cast"),
            match="contains",
            reduction="sum",
        ),
    },
)
```

- `kernels=None` 表示全部 kernel。
- `match` 支持 `exact`、`contains`、`regex`。
- `reduction="span"` 统计最早开始到最晚结束；`sum` 累加 kernel duration。
- selector 找不到对应 kernel 时直接报错，并列出 profiler 中的可用名称。
- `--timing event` 可临时切换到 xpu-perf event 计时。
