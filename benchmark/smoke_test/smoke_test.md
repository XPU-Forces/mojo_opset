# mojo_opset Benchmark — 环境 Smoke Test

`benchmark/smoke_test/` 提供最小化的环境可用性校验，是 benchmark 三层测试流程的**第一步**：

```text
smoke_test  →  run_acc  →  run_perf
  环境 OK       数值 OK       性能对比
```

| 阶段 | 入口 | 回答的问题 |
|------|------|------------|
| 1. Smoke test | [`verify_mojo_quant_gemm.py`](verify_mojo_quant_gemm.py) | 环境能用吗？op 注册正常吗？ |
| 2. 精度 | [`run_acc.py`](../run_acc.py) | 算得对吗？ |
| 3. 性能 | [`run_perf.py`](../run_perf.py) | 跑得快吗？ |

更完整的精度 / 性能说明见 [`docs/acc.md`](../docs/acc.md) 与 [`docs/perf.md`](../docs/perf.md)。

---

## 环境准备

在 `mojo_opset` 仓库根目录执行：

```bash
pip install -e .
pip install -e ../xpu-perf   # 或按 benchmark/requirements.txt 说明安装
```

需要可用的 Ascend NPU 环境（`torch_npu` 已正确安装）。

---

## 1. Smoke test：快速验证环境

```bash
python benchmark/smoke_test/verify_mojo_quant_gemm.py
```

**用途**

- 确认 NPU 可用
- 确认 `op_defs` / `vendor_ops` 注册正常
- 确认 `base` 与 `torch_npu` 能跑通一个小 case

**特点**

- 固定小 shape（128×4096×4096），避免 `base` reference OOM
- 不走 workload JSON，几秒内完成
- 只输出 `max_abs_diff`，不做完整精度指标

**期望输出**

```text
base impl: MojoQuantGemmOp
vendor providers: ['torch_npu']
[base] out shape=(128, 4096) dtype=torch.bfloat16 ...
[torch_npu] out shape=(128, 4096) dtype=torch.bfloat16 ...
max_abs_diff(base vs torch_npu) = 0.000000e+00
VERIFY OK
```

失败时进程 exit code 为 `1`（注册缺失或 diff 超过 `1e-2`）。

---

## 2. 精度：`run_acc.py`

环境通过后，用 workload 驱动做完整数值对比：

```bash
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --providers torch_npu --device 0
```

```bash
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_paged_prefill_gqa.json \
    --providers torch_npu --device 0
```

**用途**

- 以 `base` 为 reference，对比 `torch_npu` 等 vendor
- 多 case、统一输入（`create_tensors` + clone）
- 输出 `max_abs` / `mean_abs` / `match_ratio` / `PASS` / `FAIL`

**特点**

- 不走 profiling，比性能测试轻
- 任一 vendor `FAIL` 时 exit code 为 `1`，便于 CI

---

## 3. 性能：`run_perf.py`

精度通过后，查看 latency、tflops、加速比：

```bash
python benchmark/run_perf.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --device 0
```

```bash
# 只看 torch_npu
python benchmark/run_perf.py --providers torch_npu

# 导出 micro_perf 风格报告
python benchmark/run_perf.py --report_dir benchmark/reports
```

**用途**

- 终端打印 per-case 性能指标与 `base` vs vendor 对比表
- `torch_npu` 会走 NPU profiler（含 profiling 开销）

多卡压测或写正式报告时，使用 [`launch.py`](../launch.py)。

---

## 推荐流程

```bash
# Step 1: 环境 smoke test（秒级）
python benchmark/smoke_test/verify_mojo_quant_gemm.py

# Step 2: 精度（分钟级，视 workload 而定）
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --providers torch_npu --device 0

# Step 3: 性能（含 profiling，通常更慢）
python benchmark/run_perf.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --device 0
```

---

## 目录结构

```text
benchmark/
  smoke_test/
    verify_mojo_quant_gemm.py   # 环境 smoke test 脚本
    smoke_test.md               # 本文档
  run_acc.py                    # 精度测试
  run_perf.py                   # 轻量级性能测试
  launch.py                     # 多进程性能测试
  workloads/                    # workload JSON
  docs/
    acc.md                      # 精度测试详细说明
    perf.md                     # 性能测试详细说明
```
