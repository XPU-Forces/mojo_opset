# fused_qk_norm_rope 自验证报告

日期：2026-10-29

## 结论

在 Atlas 800I A2、Ascend 910B3 的 NPU 0 上，函数精度套件 64 项全部通过，模块套件 3 项全部通过，静态检查通过。目标几何为 BF16、`head_dim=128`、`rotary_dim=96` 的 half-split RoPE，覆盖 strided packed-QKV 输入、57-head 尾块、BF16/FP32 表、BF16 中间舍入和 32 维直通位。

性能结果按任务口径记录。两个小 shape 的主指标和端到端时长均优于原生 NPU 组合路径。大型 56/57-head 形状 MBU 为 26.7% 至 31.8%，未达到 60%/50% 线；profile 显示 AIV 向量管线利用率约 84%，瓶颈分析已写入设计与性能报告。

A5 的 Triton 源码、OPS 注册和默认分派已接入。当前机器只有 910B3，A5、A3、950 的实机验证尚未执行，报告中不把它们标记为已通过。

## 环境

| 项目 | 值 |
| --- | --- |
| 主机 | `122a0b52c5ff` |
| NPU | 8 × Ascend 910B3，测试使用 NPU 0 |
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0 |
| Triton | 3.5.0 |
| triton-ascend | 3.2.2 |
| 设备标识 | `npu-smi 26.0.rc1` |

任务书建议 triton-ascend 3.6.0 与 CANN 9.2.0。本次实测环境是节点上可用的 3.2.2 与 CANN 9.1.0；当前 kernel 在该组合上完成了全部精度与性能运行。

## 测试来源

任务 ZIP 只包含任务 Markdown，没有文档引用的以下内容：

```text
testsuites/task1_rms_norm_rope/tests/ops/test_fused_qk_norm_rope.py
testsuites/task1_rms_norm_rope/ops_impl/qk_norm_rope.py
testsuites/task1_rms_norm_rope/benchmarks/bench_fused_qk_norm_rope.py
```

仓库中的 `tests/functions/test_fused_qk_norm_rope.py`、`tests/modules/test_fused_qk_norm_rope.py` 和性能测试按任务书与上游提交 `717f8a3499fb7dc6b3128b58a8baeacded1a7bbb` 的语义重建。断言直接比较 eager 参考与 Triton 输出；关键数值边界使用独立参考和 `torch.equal`。

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

最终精度日志中的函数套件结果：

```text
64 passed, 14 warnings in 78.65s
```

模块套件结果：

```text
3 passed
```

静态检查：

```text
ruff check: All checks passed!
ruff format --check: 10 files already formatted
```

## 精度覆盖

| 类别 | 覆盖 |
| --- | --- |
| tokens | 1、257、1024、4096、16384、65536 |
| heads | 8、56 |
| 输入布局 | 连续、packed QKV strided 视图 |
| rope_table | BF16、FP32 |
| 尾块 | 257×57、4096×57，连续与 strided |
| 中间舍入 | BF16 normalized 后再做 FP32 rotation，`torch.equal` |
| 共享统计 | 同一 head 的 128 维共享一个 `inv_rms` |
| 非旋转维 | 96 至 127 维对 BF16 normalized 参考 bitwise 相等 |
| 确定性 | 1、257 token 重复前向逐位一致 |
| 回退 | 6 个 interleaved 几何，连续与 strided |
| CPU | Torch provider 与空 token |

## 关键语义

对每个 head，Triton 路径执行：

1. 由 FP32 输入计算平方和与 \(\operatorname{inv\_rms}\)；
2. 计算 FP32 归一化值并舍入到 BF16；
3. 将 BF16 归一化值恢复为 FP32；
4. 用 packed cos/sin 在 FP32 中计算 half-split rotation；
5. 写回时舍入到 BF16；
6. 将 96 至 127 维的 BF16 归一化值直接写入输出。

`test_fused_qk_norm_rope_bf16_round_and_shared_stats_are_exact` 对上述顺序构造独立参考，Q、K 均与 Triton 结果 `torch.equal`；同时确认不做中间 BF16 舍入的版本与 Triton 结果不同。

## 性能摘要

标定方法为 256 MiB BF16 `copy_`，5 次 warmup、30 次有效采样。copy 探针中位时长 0.346259 ms，对应读写带宽 1550.49 GB/s。逐形状结果如下：

| tokens×heads | kernel 时间 (us) | MBU |
| --- | ---: | ---: |
| 32×8 | 8.18 | 2.12% |
| 320×8 | 33.10 | 5.23% |
| 320×56 | 44.43 | 26.73% |
| 1024×56 | 127.28 | 29.85% |
| 4096×56 | 485.24 | 31.32% |
| 16384×56 | 2042.86 | 29.76% |
| 69888×56 | 8831.18 | 29.37% |
| 4096×57 | 486.17 | 31.82% |

4096×56 的原始 profile 显示每个 kernel 平均约 242 至 243 us，AIV vector ratio 为 79.2% 至 84.7%，AIV MTE2 ratio 约 13% 至 19%，AIV MTE3 ratio 约 5.6%。大型形状未达到 60% MBU 的直接原因是向量计算受限，数据传输管线没有饱和。

小 shape 相对基线结果：

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 8.19 | 92.72 | 0.4246 | 0.7072 |
| 320×8 | 29.32 | 111.55 | 0.4339 | 0.7065 |

## 证据

交付目录 `evidence/` 保留：

- `accuracy_final.log`
- `module_final.log`
- `performance_final.log`
- `baseline_comparison.log`
- `npu-smi.txt`
- `performance/benchmark.json`
- `performance/baseline_comparison.json`
- 逐形状 `kernel_details.csv`
- 原生与小 shape 对比 `kernel_details.csv`
