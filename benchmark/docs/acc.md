# mojo_opset Benchmark — 精度测试

`benchmark/run_acc.py` 提供轻量级单进程精度测试入口，复用 `xpu-perf` 的 `BasicOp` / workload 解析链路，但**不走 profiling**。脚本会：

1. 用 `base` provider 创建同一份输入；
2. 分别执行 `base` 与目标 vendor（如 `torch_npu`）；
3. 以 `base` 输出作为 reference，计算预设精度指标并给出 PASS / FAIL。

这套流程适合在改动算子实现后，快速确认 `torch_npu` 等 vendor 与 `base` 的数值一致性。

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

## 基本用法

```bash
# 默认：benchmark/workloads/mojo_quant_gemm.json，device 0，对比 base vs torch_npu
python benchmark/run_acc.py

# 指定 workload 与 device
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --providers torch_npu \
    --device 0

# 自定义容差与随机种子
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --providers torch_npu \
    --seed 42 \
    --atol 1e-1 \
    --rtol 1e-2

# 导出每个 case 的 JSON 精度报告
python benchmark/run_acc.py \
    --workload benchmark/workloads/mojo_quant_gemm.json \
    --providers torch_npu \
    --report_dir benchmark/reports_acc
```

---

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--workload` | `benchmark/workloads/mojo_quant_gemm.json` | workload JSON 路径 |
| `--device` | `0` | NPU device id |
| `--providers` | `torch_npu` | 逗号分隔的 vendor 列表；reference 固定为 `base` |
| `--seed` | `42` | `base.create_tensors()` 前设置的随机种子 |
| `--atol` | `1e-2` | 绝对误差阈值，传给 `check_tol_diff` |
| `--rtol` | `1e-2` | 相对误差阈值，传给 `check_tol_diff` |
| `--report_dir` | 不导出 | 设置后写入 per-case JSON 报告 |

---

## 运行机制

`run_acc.py` 的关键约束是：**所有 provider 使用同一份输入**。流程如下：

1. 为当前 case 构建 `base` op；
2. 在 `torch.manual_seed(seed)` 后调用 `base.create_tensors(1)[0]`；
3. 每次执行前 clone 一份 `tensor_mapping`，避免 in-place 修改互相污染；
4. 先跑 `base.core_run()` 得到 reference；
5. 再跑各个 vendor 的 `core_run()`；
6. 用 `compare_outputs()` 计算指标并输出 PASS / FAIL。

因此 vendor 侧不会重新 `create_tensors()`，可避免随机输入不一致导致的伪误差。

---

## 输出说明

每个 case 会打印一张表，核心字段如下：

| 列 | 说明 |
|---|---|
| `max_abs` | 最大绝对误差 |
| `mean_abs` | 平均绝对误差 |
| `max_rel` | 最大相对误差 |
| `match_ratio` | `torch.isclose(..., atol, rtol)` 匹配比例 |
| `status` | `PASS` / `FAIL` / `SKIP` |
| `note` | 失败或跳过原因 |

当前 `benchmark/acc_compare.py` 还会内部计算 `mean_rel_diff` 和 `rmse`；若设置 `--report_dir`，这些完整字段会保存在 JSON 报告中。

### PASS / FAIL 规则

- `PASS`：`check_tol_diff()` 通过；
- `FAIL`：数值超出容差、shape 不一致、输出结构不支持等；
- `SKIP`：例如 `base` 构建失败、reference OOM、vendor OOM。

只要任一 provider 出现 `FAIL`，脚本退出码为 `1`，便于接入 CI。`SKIP` 不会让进程失败。

---

## 示例输出

```text
####################################################################################################
# op: mojo_paged_prefill_gqa | reference: base | vendors: ['torch_npu']
####################################################################################################
+-----------+------------------------+
| key       | value                  |
+-----------+------------------------+
| op_name   | mojo_paged_prefill_gqa |
| reference | base                   |
| device_id | 0                      |
+-----------+------------------------+
{"arg_type": "llm", "block_size": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128, "batch_size": 1, "q_len": 4096, "cache_len": 0}

+-----------+--------------+--------------+--------------+--------------+--------+------+
| provider  |      max_abs |     mean_abs |      max_rel |  match_ratio | status | note |
+-----------+--------------+--------------+--------------+--------------+--------+------+
| torch_npu | 1.562500e-02 | 1.588830e-04 | 2.298355e+08 | 1.000000e+00 | PASS   |      |
+-----------+--------------+--------------+--------------+--------------+--------+------+
```

---

## Workload 注意事项

`run_acc.py` 与 `run_perf.py` 共用同一套 workload JSON，但**性能 workload 不一定适合直接做 reference 精度测试**。

例如 [`workloads/mojo_paged_prefill_gqa.json`](../workloads/mojo_paged_prefill_gqa.json) 当前默认 case 为：

```json
{
    "mojo_paged_prefill_gqa": [
        {
            "arg_type": "llm",
            "num_q_heads.num_kv_heads.head_dim": [
                [64, 8, 128]
            ],
            "batch_size.q_len.cache_len": [
                [1, 10240, 0]
            ],
            "block_size": 512
        }
    ]
}
```

该 case 在性能测试中可用，但 `base` reference 可能 OOM，结果会显示为 `SKIP`。如果希望稳定跑精度对比，建议为精度测试准备更小的 workload，例如把 `q_len` 调整为 `4096`。

---

## 报告输出

设置 `--report_dir` 后，每个 case 会生成一个 JSON 文件，文件名格式如下：

```text
<report_dir>/<op_name>-<case_hash>.json
```

报告字段包括：

- `op_name`
- `reference`
- `case`
- `seed`
- `atol` / `rtol`
- `providers.<name>.{max_abs_diff, mean_abs_diff, max_rel_diff, mean_rel_diff, rmse, match_ratio, status, note}`

适合后续做批量归档或简单 CI 收集。

---

## 目录结构速览

```text
benchmark/
  run_acc.py           # 轻量级单进程精度测试
  acc_compare.py       # 精度指标计算与 PASS / FAIL 判定
  runner_common.py     # run_acc / run_perf 共用 helper
  run_perf.py          # 轻量级单进程性能测试
  launch.py            # 多进程性能测试（XpuPerfServer）
  smoke_test/          # 环境可用性快速校验
    smoke_test.md      # smoke_test → run_acc → run_perf 流程说明
  op_defs/
  vendor_ops/
  workloads/
  docs/
    perf.md            # 性能测试文档
    acc.md             # 本文档
```
