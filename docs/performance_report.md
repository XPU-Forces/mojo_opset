# fused_qk_norm_rope 性能自验证报告

## 环境与计时口径

| 项目 | 值 |
| --- | --- |
| NPU | Ascend 910B3，NPU 0 |
| CANN | 9.1.0 |
| torch_npu | 2.10.0 |
| triton-ascend | 3.2.2 |
| warmup | 5 |
| 有效采样 | 30 次调用 |
| kernel 计时 | `kernel_details.csv` 中窗口内全部设备 kernel 的 `Duration(us)` 按调用求和 |
| 每调用 kernel 数 | 1 |
| 带宽探针 | 256 MiB BF16 `copy_`，读写合计 512 MiB |
| copy 中位时长 | 0.426640 ms |
| copy 读写带宽 | 1258.37 GB/s |
| MBU 公式 | \(2(4TH\cdot128+96T+2\cdot128)/(t_{\mathrm{kernel}}\cdot BW)\) |

性能由 `tools/benchmark_fused_qk_norm_rope.py` 产出。每个 shape 使用 packed QKV 中的 Q/K strided 视图、BF16 权重和 BF16 RoPE 表。计时包含一次调用产生的全部设备 kernel。

## 主性能矩阵

| tokens×heads | kernel 时间 (us) | 读写 (MB) | MBU | 任务线 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 32×8 | 8.29 | 0.27 | 2.58% | 相对线 | 通过 |
| 320×8 | 8.64 | 2.68 | 24.68% | 相对线 | 通过 |
| 320×56 | 32.83 | 18.41 | 44.56% | 60% | 未达目标线 |
| 1024×56 | 117.99 | 58.92 | 39.68% | 60% | 未达目标线 |
| 4096×56 | 377.26 | 235.67 | 49.64% | 60% | 未达目标线 |
| 16384×56 | 1484.05 | 942.67 | 50.48% | 60% | 未达目标线 |
| 69888×56 | 6227.21 | 4021.08 | 51.31% | 60% | 未达目标线 |
| 4096×57 | 435.57 | 239.86 | 43.76% | 50% | 未达目标线 |

大形状 MBU 为 44.56% 至 51.31%。`4096×56` 的 kernel 时间从初版两 launch 约 485 us 降至 377 us。未达到 60%/50% 目标线的配置保留完整 profile、根因和后续优化方向，由验收方按任务书未达标配置条款裁定。

## 小 shape 相对基线

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 8.25 | 95.63 | 0.2757 | 0.6161 |
| 320×8 | 8.72 | 113.24 | 0.2812 | 0.6617 |

两个小 shape 的主指标和端到端时间均优于 NPU 原生组合路径。

## 4096×56 profile

30 次调用均只产生 `_persistent_qk_norm_rope_kernel`。关键管线指标如下：

| 指标 | 均值 |
| --- | ---: |
| kernel duration | 377.26 us |
| kernel rows | 30 |
| kernels per call | 1 |
| AIV vector ratio | 93.34% |
| AIV scalar ratio | 15.63% |
| AIV MTE2 ratio | 20.15% |
| AIV MTE3 ratio | 9.05% |

完整逐 shape profile 位于 `evidence/performance/profile_summary.csv`。

## 根因

大形状 kernel 是向量计算受限。AIV vector ratio 已约 93%，MTE2 约 20%，MTE3 约 9%。继续增加 tile 宽度、调节核数或增加流水不能显著提高主指标，因为主路径已经接近占满 vector pipe。

当前每个 128 维 head 需要完成：

1. FP32 平方、128 维归约和 `rsqrt`；
2. FP32 归一化、权重乘法与 BF16 舍入；
3. 两个 48 维分支的 BF16 到 FP32 恢复；
4. 四次乘法、两次加减的 FP32 RoPE；
5. 两个旋转分支的 BF16 舍入、输出组合和写回。

BF16 norm round 和 FP32 RoPE 是任务数值契约，不能通过降低旋转精度来换取速度。达到 60% 目标线要求再降低约 17% 至 20% 的设备时间，现有 Triton 表达式下主要缺少足够的向量指令缩减空间。

## 已验证方案

| 方案 | 4096×56 结果 | 结论 |
| --- | ---: | --- |
| pure persistent 8×8，当前提交 | 377.26 us / 49.64% | 最佳有效方案 |
| 旧两 launch Triton | 485 us / 31.32% | kernel 数多，已替换 |
| hybrid 多 kernel | 1031.86 us | 完整 kernel 求和后更慢 |
| flat 2D tile | 403.64 us | 布局转换收益不足 |
| packed Q/K pair | 457.93 us | 掩码和输出拆分增加开销 |
| split store | 538.74 us | MTE3 分段写回成本上升 |
| static head unroll | 483.53 us | 指令体积和调度成本上升 |
| explicit FMA | 388.05 us | Ascend 向量流水收益为负 |
| Cube dot 统计 | 约 163 ms | 同步和 tile 开销不可接受 |

也测试了常见 tile、vector core 数、`num_warps`、`num_stages`、`ops_reorder` 和编译流水选项。最终采用 40 cores、8 token、8 head 的 persistent 配置，并合并读取 96 维 RoPE 表。

## 后续优化方向

1. 使用 Ascend 低层 intrinsics 合并 RMS 平方归约、权重乘法和 BF16 舍入，减少 vector 指令数。
2. 设计 amortized Cube 统计路径，将 128 维平方和归约搬到 Cube，同时消除当前朴素 dot 方案的同步开销。
3. 在编译器层生成 Q/K 合并 lane 的无掩码连续向量操作，减少 57-head 尾块和两组输出拆分开销。
4. 升级到任务推荐版本后复查 triton-ascend 的向量融合、BF16 转换和 persistent loop 代码生成。
5. 评估专用 Ascend C kernel 或融合前后算子，以避免在同一 attention 层内重复读写 Q/K。

## 原始证据

- `evidence/performance/benchmark.json`
- `evidence/performance/profile_summary.csv`
- `evidence/performance/tokens_*/**/kernel_details.csv`
- `evidence/baseline/comparison.json`
- `evidence/performance_final.log`
- `evidence/npu-smi.txt`
- `evidence/environment.txt`
