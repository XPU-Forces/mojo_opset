# fused_qk_norm_rope 自验证报告

日期：2026-10-30

## 结论

在 Atlas 800I A2、Ascend 910B3 的 NPU 0 上，函数与模块精度用例合计 67 项全部通过，失败 0、错误 0。关键数值边界保持精确：

- BF16 norm round 生效；
- 同一 head 的两个 pair 元素共享一次 RMS 统计；
- 96 至 127 维逐位等于 BF16 归一化值。

普通精度矩阵使用 `atol=0.0625, rtol=0.02`。A2 persistent 实现每次调用只产生 1 个设备 kernel。性能大形状 MBU 为 44.56% 至 51.31%，未达到 60%/50% 目标线；完整 profile、根因和后续方向已写入性能报告。

A5 的 Triton 源码、OPS 注册和配置已经接入。本次实机证据来自 A2；A3、A5 和 950 的运行结果未标记为通过。

## 环境

| 项目 | 值 |
| --- | --- |
| 主机 | `122a0b52c5ff` |
| NPU | 8 × Ascend 910B3，测试使用 NPU 0 |
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0 |
| Triton | 3.2.0 |
| triton-ascend | 3.2.2 |
| 设备标识 | `npu-smi 26.0.rc1` |

任务书建议 triton-ascend 3.6.0 与 CANN 9.2.0。本次节点提供 triton-ascend 3.2.2 与 CANN 9.1.0，全部精度和性能结果均在该实际环境取得。

## 测试来源

任务 ZIP 包含任务书，未包含文档引用的以下目录：

```text
testsuites/task1_rms_norm_rope/tests/ops/test_fused_qk_norm_rope.py
testsuites/task1_rms_norm_rope/ops_impl/qk_norm_rope.py
testsuites/task1_rms_norm_rope/benchmarks/bench_fused_qk_norm_rope.py
```

仓库中的 `tests/functions/test_fused_qk_norm_rope.py`、`tests/modules/test_fused_qk_norm_rope.py` 和性能测试按任务书矩阵与上游提交 `717f8a3499fb7dc6b3128b58a8baeacded1a7bbb` 的语义重建。断言直接比较 eager 参考与 Triton 输出，关键边界使用独立构造的参考和 `torch.equal`。

## 执行命令

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd /root/qk_norm_rope_task/mojo_opset_repo

python -m pytest -q \
  tests/functions/test_fused_qk_norm_rope.py \
  tests/modules/test_fused_qk_norm_rope.py \
  --mojo-target npu.a2 \
  --mojo-implementation triton
```

最终结果：

```text
67 passed, 14 warnings in 373.58s (0:06:13)
```

14 条 warning 均来自 PyTorch `torch.jit.script_method` 的弃用提示。

静态检查：

```text
ruff check: All checks passed!
ruff format --check: 5 files already formatted
```

## 精度覆盖

| 类别 | 覆盖 |
| --- | --- |
| tokens | 1、257、1024、4096、16384、65536 |
| heads | 8、56 |
| 输入布局 | 连续、packed QKV strided 视图 |
| rope_table | BF16、FP32 |
| 尾块 | 257×57、4096×57，连续与 strided |
| 中间舍入 | BF16 normalized 后做 FP32 rotation，`torch.equal` |
| 共享统计 | 同一 head 的 128 维共享一个 `inv_rms` |
| 非旋转维 | 96 至 127 维对 BF16 normalized 参考 bitwise 相等 |
| 确定性 | 1、257 token 重复前向逐位一致 |
| 回退 | 6 个 interleaved 几何，连续与 strided |
| CPU | Torch provider 与空 token |
| 模块 | `FusedQKNormRope` 与别名 |

普通矩阵通过任务容差；三个关键边界通过精确相等检查。RoPE 始终使用 FP32，BF16 仅用于归一化中间值和最终输出。

## 关键语义

对每个 head，Triton 路径执行：

1. 用 FP32 输入计算平方和、均值和 \(\operatorname{inv\_rms}\)；
2. 计算 FP32 归一化值并舍入到 BF16；
3. 将 BF16 归一化值恢复为 FP32；
4. 用 packed cos/sin 在 FP32 中计算 half-split rotation；
5. 写回时舍入到 BF16；
6. 将 96 至 127 维的 BF16 归一化值直接写入输出。

`test_fused_qk_norm_rope_bf16_round_and_shared_stats_are_exact` 对上述顺序构造独立参考，Q/K 均与 Triton 结果 `torch.equal`，同时确认不做中间 BF16 舍入的版本与 Triton 结果不同。

## 性能摘要

性能按 5 次 warmup、30 次有效采样、全部设备 kernel 求和口径执行：

| tokens×heads | kernel 时间 (us) | MBU |
| --- | ---: | ---: |
| 32×8 | 8.29 | 2.58% |
| 320×8 | 8.64 | 24.68% |
| 320×56 | 32.83 | 44.56% |
| 1024×56 | 117.99 | 39.68% |
| 4096×56 | 377.26 | 49.64% |
| 16384×56 | 1484.05 | 50.48% |
| 69888×56 | 6227.21 | 51.31% |
| 4096×57 | 435.57 | 43.76% |

`4096×56` profile 的 AIV vector ratio 为 93.34%，MTE2 为 20.15%，MTE3 为 9.05%。瓶颈是 BF16/FP32 转换、RMS 归约和 FP32 RoPE 的向量计算，详见 `docs/performance_report.md`。

## 证据

- `evidence/accuracy_final.log`
- `evidence/performance_final.log`
- `evidence/performance/benchmark.json`
- `evidence/performance/profile_summary.csv`
- `evidence/performance/tokens_*/**/kernel_details.csv`
- `evidence/baseline/comparison.json`
- `evidence/npu-smi.txt`
- `evidence/environment.txt`
