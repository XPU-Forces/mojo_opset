# fused_qk_norm_rope 性能自验证报告

## 环境与口径

| 项目 | 值 |
| --- | --- |
| NPU | Ascend 910B3，NPU 0 |
| CANN | 9.1.0 |
| torch_npu | 2.10.0 |
| triton-ascend | 3.2.2 |
| warmup | 5 |
| 有效采样 | 30 |
| kernel 计时 | `kernel_details.csv` 中目标 kernel 的 `Duration(us)` 求和后除以 30 |
| 带宽探针 | 256 MiB BF16 `copy_`，读加写共 512 MiB |
| copy 中位时长 | 0.346259 ms |
| copy 读写带宽 | 1550.49 GB/s |
| MBU 公式 | \(2(4TH\cdot128+96T+2\cdot128)/(t_{\mathrm{kernel}}\cdot BW)\) |

性能由 `tools/benchmark_fused_qk_norm_rope.py` 产出。每个 shape 使用 packed QKV 的 Q/K strided 视图、BF16 权重和 BF16 RoPE 表。

## 主性能矩阵

| tokens×heads | Q kernel (us) | K kernel (us) | 总计 (us) | MBU | 任务线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32×8 | 4.08 | 4.10 | 8.18 | 2.12% | 相对线 |
| 320×8 | 16.45 | 16.66 | 33.10 | 5.23% | 相对线 |
| 320×56 | 22.21 | 22.22 | 44.43 | 26.73% | 60% |
| 1024×56 | 63.69 | 63.59 | 127.28 | 29.85% | 60% |
| 4096×56 | 242.59 | 242.66 | 485.24 | 31.32% | 60% |
| 16384×56 | 1021.79 | 1021.07 | 2042.86 | 29.76% | 60% |
| 69888×56 | 4414.07 | 4417.11 | 8831.18 | 29.37% | 60% |
| 4096×57 | 243.17 | 243.00 | 486.17 | 31.82% | 50% |

大型 56/57-head 形状未达到 MBU 线。结果保留为实际测量，不做剔除或换口径。

## 小 shape 基线

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 8.19 | 92.72 | 0.4246 | 0.7072 |
| 320×8 | 29.32 | 111.55 | 0.4339 | 0.7065 |

两个小 shape 的主指标与端到端时间均不劣于 NPU 原生组合路径，满足相对线。

## Profile 瓶颈

4096×56 的两个 kernel 来自同一形状的 30 次采样。核心管线指标如下：

| 指标 | 两次 kernel 的范围 |
| --- | ---: |
| 单 kernel 时长 | 约 242 至 243 us |
| AIV vector ratio | 79.2% 至 84.7% |
| AIV MTE2 ratio | 约 13% 至 19% |
| AIV MTE3 ratio | 约 5.6% |
| AIV scalar ratio | 约 8.5% 至 9.4% |

AIV 向量管线接近饱和，MTE2/MTE3 仍有较大余量。当前 kernel 在每个 head 上执行 128 维平方和、归一化、BF16 舍入、两次 BF16/FP32 转换以及 96 维旋转算术，属于 compute-bound。

将相邻 token 与最多 64 head 打包到同一 program 后，4096×56 从初版约 1.26 ms 降至 0.485 ms，约提升 2.6 倍。进一步提升 MBU 需要减少向量指令而不是继续增加搬运宽度。

## 后续优化方向

1. 增加每个 program 的 token 数，复用 RoPE 表并减少循环控制，先测试 4 token tile。
2. 将 32 维直通区从 128 维旋转 tile 中拆出，减少不必要的布局和插入切片操作。
3. 分析 Triton 生成 IR 中的 BF16/FP32 转换与 broadcast 指令，用等价但更紧凑的表达式降低向量指令数。
4. 评估 Q/K 单 launch 对小型形状的收益；大型形状优先保持两个独立 launch。

## 原始证据

- `evidence/performance/benchmark.json`
- `evidence/performance/baseline_comparison.json`
- `evidence/performance_final.log`
- `evidence/baseline_comparison.log`
- `evidence/performance/kernel_details/`
- `evidence/performance/baseline_kernel_details/`
- `evidence/npu-smi.txt`
