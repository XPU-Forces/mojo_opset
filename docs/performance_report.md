# fused_qk_norm_rope 性能自验证报告

## 环境与计时口径

| 项目 | A2 | A3 | 950/A5 |
| --- | --- | --- | --- |
| NPU | Ascend 910B3 | Ascend 910_9382 | Ascend950PR |
| Provider | `npu.a2` | `npu.a2` | `npu.a5` |
| CANN | 9.1.0 | 9.1.0 | 9.1.0 |
| torch_npu | 2.10.0 | 2.10.0.post7 | 2.10.0.post7 |
| triton-ascend | 3.2.2 | 3.6.0 | 3.6.0 |
| warmup | 5 | 5 | 5 |
| 有效采样 | 30 次调用 | 30 次调用 | 30 次调用 |
| kernel 计时 | 全部设备 kernel 求和 | 全部设备 kernel 求和 | 全部设备 kernel 求和 |
| 每调用 kernel 数 | 1 | 1 | 2 |
| 带宽探针 | 256 MiB BF16 `copy_` | 同左 | 同左 |
| copy 中位时长 | 0.426640 ms | 0.426254 ms | 0.373475 ms |
| copy 读写带宽 | 1258.37 GB/s | 1259.51 GB/s | 1437.50 GB/s |
| MBU 公式 | \(2(4TH\cdot128+96T+2\cdot128)/(t_{\mathrm{kernel}}\cdot BW)\) |

性能由 `tools/benchmark_fused_qk_norm_rope.py` 产出。每个 shape 使用 packed QKV 中的 Q/K strided 视图、BF16 权重和 BF16 RoPE 表。计时包含一次调用产生的全部设备 kernel。

## A2 主性能矩阵

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

## A3 性能矩阵

A3 使用 Ascend 910_9382 实机。仓库只有 `npu.a2` 与 `npu.a5` provider，本矩阵使用 `npu.a2` provider 和 triton-ascend 3.6.0；下文准确表述为 A3 硬件上的 A2 provider 结果。

| tokens×heads | kernel 时间 (us) | MBU |
| --- | ---: | ---: |
| 32×8 | 8.22 | 2.60% |
| 320×8 | 8.53 | 24.98% |
| 320×56 | 32.76 | 44.62% |
| 1024×56 | 117.55 | 39.79% |
| 4096×56 | 379.23 | 49.34% |
| 16384×56 | 1499.48 | 49.91% |
| 69888×56 | 6294.79 | 50.72% |
| 4096×57 | 439.32 | 43.35% |

## 950/A5 性能矩阵

A5 为满足三个精确边界，使用每 program 1 token × 8 heads 的 Q/K 两次 launch 路径。每个调用包含 2 个 `_qk_norm_rope_kernel`。

| tokens×heads | kernel 时间 (us) | MBU |
| --- | ---: | ---: |
| 32×8 | 10.19 | 1.83% |
| 320×8 | 39.50 | 4.73% |
| 320×56 | 228.79 | 5.60% |
| 1024×56 | 747.82 | 5.48% |
| 4096×56 | 4323.31 | 3.79% |
| 16384×56 | 17792.29 | 3.69% |
| 69888×56 | 77376.00 | 3.62% |
| 4096×57 | 4743.85 | 3.52% |

A5 大形状 MBU 为 3.52% 至 5.60%，未达到 60%/50% 目标线。A5 精度与三个精确边界全部通过，完整 profile 和精确性约束下的性能根因见下文。

## 小 shape 相对基线

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 8.25 | 95.63 | 0.2757 | 0.6161 |
| 320×8 | 8.72 | 113.24 | 0.2812 | 0.6617 |

两个小 shape 的主指标和端到端时间均优于 NPU 原生组合路径。

950/A5 同样通过小 shape 相对线：

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 10.14 | 49.20 | 0.0533 | 0.1171 |
| 320×8 | 39.88 | 61.80 | 0.0526 | 0.1128 |

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

## 950/A5 4096×56 profile

| 指标 | 均值 |
| --- | ---: |
| 单次调用总 kernel 时间 | 4323.31 us |
| kernel rows | 60 |
| kernels per call | 2 |
| 单 kernel 平均时间 | 2161.66 us |
| AIV vector ratio | 32.06% |
| AIV scalar ratio | 32.45% |
| AIV MTE2 ratio | 55.85% |
| AIV MTE3 ratio | 4.85% |

完整逐 shape profile 位于 `evidence_a5_core/performance/profile_summary.csv`。

## A2/A3 根因

大形状 kernel 是向量计算受限。AIV vector ratio 已约 93%，MTE2 约 20%，MTE3 约 9%。继续增加 tile 宽度、调节核数或增加流水不能显著提高主指标，因为主路径已经接近占满 vector pipe。

当前每个 128 维 head 需要完成：

1. FP32 平方、128 维归约和 `rsqrt`；
2. FP32 归一化、权重乘法与 BF16 舍入；
3. 两个 48 维分支的 BF16 到 FP32 恢复；
4. 四次乘法、两次加减的 FP32 RoPE；
5. 两个旋转分支的 BF16 舍入、输出组合和写回。

BF16 norm round 和 FP32 RoPE 是任务数值契约，不能通过降低旋转精度来换取速度。达到 60% 目标线要求再降低约 17% 至 20% 的设备时间，现有 Triton 表达式下主要缺少足够的向量指令缩减空间。

## 950/A5 根因

A5 的 1 ULP 问题来自 RMS 归约树。多次实机诊断确认：

- 1D 的“先平方、前 64 维与后 64 维相加、再 64-lane 归约”可逐 head 复现 `F.rms_norm` 的 `inv_rms`；
- 将多个 head 放进同一归约轴后，编译器会选择不同的 FP32 加法树；
- 该 1 ULP 差异只在 BF16 中点附近暴露，在 460544 个直通元素中产生 2 个不同值；
- 精确 H8 路径的 AIV vector ratio 为 32.06%，scalar ratio 为 32.45%，MTE2 ratio 为 55.85%，主体被逐 head 统计与多 program 调度限制。

A5 已实测的精确配置：

| 配置 | 4096×56 结果 | 结论 |
| --- | ---: | --- |
| H8，1D RMS + 8-head 向量 RoPE，当前提交 | 4323.31 us / 3.79% | 完整精度通过，保留 |
| H4 | 4650.60 us / 3.52% | program 数增加，调度更慢 |
| 独立精确统计 kernel + Q/K 主体 kernel | 4330.32 us / 3.78% | 3 kernel，无收益 |
| 8-token 3D tile 原始归约 | 归约更快 | 直通维 2 个元素未逐位通过 |

完整精度与性能同时合格需要 triton-ascend 提供可控制的 ordered reduction，或在单条 Ascend 向量指令中保持 64-lane 加法树；当前公开 Triton 接口没有该控制能力。

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
- `evidence_a3_core/benchmark.json`
- `evidence_a3_core/profile_summary.csv`
- `evidence_a3_core/a3_performance.log`
- `evidence_a5_core/performance/benchmark.json`
- `evidence_a5_core/performance/profile_summary.csv`
- `evidence_a5_core/a5_performance.log`
