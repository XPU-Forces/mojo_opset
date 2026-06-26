# mojo_opset Benchmark — 性能测试

`benchmark/` 目录提供两套性能测试入口，均复用 [xpu-perf](https://github.com/bytedance/xpu-perf) 的 `micro_perf` 框架（`BasicOp` / `ProviderRegistry` / `export_reports` 报告格式）：

| 脚本 | 定位 | 适用场景 |
|---|---|---|
| [`run_perf.py`](../run_perf.py) | 轻量级单进程 runner | 快速对比 `base` vs `torch_npu`，终端打印对比表 |
| [`launch.py`](../launch.py) | 多进程 launcher（`XpuPerfServer`） | 多卡 / NUMA 绑定，与 xpu-perf 原始 micro_perf 流程一致 |

此外还有 [`smoke_test/verify_mojo_quant_gemm.py`](../smoke_test/verify_mojo_quant_gemm.py) 用于环境可用性快速校验（非性能测试）。

---

## 环境准备

在 `mojo_opset` 仓库根目录执行：

```bash
# 安装 mojo_opset 本体
pip install -e .

# 安装 benchmark 依赖（xpu-perf）
pip install -e ../xpu-perf   # 或按 benchmark/requirements.txt 说明安装
```

需要可用的 Ascend NPU 环境（`torch_npu` 已正确安装）。

---

## 轻量级测试：`run_perf.py`

单进程、无 server / 子进程，每个 case 依次跑所有 provider，并在终端输出 micro_perf 风格的 per-case 报告和 side-by-side 对比表。

### 基本用法

```bash
# 默认：benchmark/workloads/mojo_quant_gemm.json，device 0，对比 base + torch_npu
python benchmark/run_perf.py

# 指定 workload 与 device
python benchmark/run_perf.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --device 0

# 只测 torch_npu
python benchmark/run_perf.py --providers torch_npu

# 同时导出 micro_perf 风格报告（jsonl / csv）
python benchmark/run_perf.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --device 0 \
    --report_dir benchmark/reports
```

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--workload` | `benchmark/workloads/mojo_quant_gemm.json` | workload JSON 路径 |
| `--device` | `0` | NPU device id |
| `--providers` | `base,torch_npu` | 逗号分隔的 provider 列表；`base` 为 torch 小算子参考实现 |
| `--report_dir` | 不导出 | 设置后写入 jsonl / csv 报告 |

### 输出说明

- 每个 provider 打印一张 key/value 表 + case 参数 + 性能指标（`latency(us)`、`tflops`、`mem_bw(GB/s)` 等）。
- 每个 case 末尾打印 **comparison** 表，`speedup_vs_base` 为相对 `base` 的加速比。
- 默认 demo workload 包含 `4096³` 和 `8192³` 两个 case；`base`（torch 小算子）在大 shape 下可能 OOM，对比表中显示为 `-`，`torch_npu` 仍可正常出数。

---

## 多进程测试：`launch.py`

封装 xpu-perf 的 `XpuPerfServer`：按 device 拉起子进程、支持 NUMA 绑定与队列调度，报告通过 `export_reports` 写入磁盘。适合多卡压测或与 xpu-perf 原始流程对齐。

### 基本用法

```bash
# 默认 workload，所有可见 NPU
python benchmark/launch.py --backend NPU

# 显式指定 workload
python benchmark/launch.py --backend NPU \
    --workload benchmark/workloads/mojo_quant_gemm.json

# 只跑 device 0
python benchmark/launch.py --backend NPU --device 0 \
    --workload benchmark/workloads/mojo_quant_gemm.json

# 多卡：device 0 和 1
python benchmark/launch.py --backend NPU --device 0,1 \
    --workload benchmark/workloads/mojo_quant_gemm.json
```

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend` | `NPU` | 后端类型 |
| `--workload` | `benchmark/workloads/mojo_quant_gemm.json` | 单个 workload JSON |
| `--task_dir` | `benchmark/workloads` | workload 目录（与 `--task` 配合） |
| `--task` | `all` | 跑目录下哪些 task；`--workload` 优先 |
| `--device` | 全部可见设备 | 如 `0` 或 `0,1` |
| `--report_dir` | `benchmark/reports` | 报告输出目录 |
| `--op_defs` | `benchmark/op_defs` | 算子定义目录 |
| `--vendor_ops` | `benchmark/vendor_ops/{backend}/ops` | vendor 实现目录（通常无需手动指定） |
| `--numa` | 自动 | NUMA 绑定策略 |

### 报告输出

报告写入 `--report_dir`，目录结构示例：

```
benchmark/reports/
  NPU/
    Ascend910B2C/
      info.json
      mojo_quant_gemm/
        base/
          mojo_quant_gemm-base.jsonl
          mojo_quant_gemm-base.csv
        torch_npu/
          mojo_quant_gemm-torch_npu.jsonl
          mojo_quant_gemm-torch_npu.csv
```

---

## 环境 smoke test：`smoke_test/verify_mojo_quant_gemm.py`

在跑性能或精度测试前，可用此脚本快速确认 benchmark harness 注册正常、NPU 环境可用，且 `base` / `torch_npu` 在小 shape 下数值一致。完整三层流程说明见 [`smoke_test/smoke_test.md`](../smoke_test/smoke_test.md)。

```bash
python benchmark/smoke_test/verify_mojo_quant_gemm.py
```

期望输出 `max_abs_diff(base vs torch_npu) = 0` 和 `VERIFY OK`。使用较小 shape（128×4096×4096），避免 base 大 shape OOM。完整精度对比请使用 [`run_acc.py`](../run_acc.py)。

---

## Demo Workload

默认 workload 见 [`workloads/mojo_quant_gemm.json`](../workloads/mojo_quant_gemm.json)：

```json
{
    "mojo_quant_gemm": [
        {
            "arg_type": "llm",
            "num_tokens.in_features.out_features": [
                [4096, 4096, 4096],
                [8192, 8192, 8192]
            ]
        }
    ]
}
```

参数遵从 mojo_opset 原生算子定义（见 [`op_defs/docs/ops/mojo_quant_gemm.md`](../op_defs/docs/ops/mojo_quant_gemm.md)），未指定时默认 `quant_dtype=int8`、`weight_dtype=int8`、`output_dtype=bfloat16`。

---

## 目录结构速览

```
benchmark/
  run_perf.py          # 轻量级单进程性能测试
  run_acc.py           # 轻量级单进程精度测试
  launch.py            # 多进程性能测试（XpuPerfServer）
  smoke_test/          # 环境可用性快速校验
    smoke_test.md      # smoke_test → run_acc → run_perf 流程说明
  op_defs/             # 算子 base 实现（micro_perf BasicOp）
  vendor_ops/          # vendor 实现（如 torch_npu）
  workloads/           # workload JSON
  reports/             # 报告输出（launch / run_perf --report_dir）
  docs/
    perf.md            # 本文档
    acc.md             # 精度测试文档
```
