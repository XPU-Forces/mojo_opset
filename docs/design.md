# fused_qk_norm_rope Ascend 适配设计

| 项目 | 内容 |
| --- | --- |
| 算子 | `fused_qk_norm_rope` |
| 上游实现 | `vllm-project/vllm-omni`，`vllm_omni/diffusion/layers/fused_qk_norm_rope.py` |
| 上游提交 | `717f8a3499fb7dc6b3128b58a8baeacded1a7bbb` |
| 目标平台 | Ascend A2 / A3 / 950 / A5 |
| A2 实测平台 | Atlas 800I A2，Ascend 910B3，NPU 0 |
| A2 软件环境 | CANN 9.1.0，Torch 2.10.0，torch_npu 2.10.0，triton-ascend 3.2.2 |
| A3 实测平台 | Ascend 910_9382，CANN 9.1.0，triton-ascend 3.6.0；使用 `npu.a2` provider |
| 950/A5 实测平台 | Ascend950PR，CANN 9.1.0，triton-ascend 3.6.0；使用 `npu.a5` provider |
| 目标几何 | BF16，head_dim=128，rotary_dim=96，half-split RoPE |

## 背景

MiniMax-H3 的 attention 层对每个 head 先做 RMSNorm，再应用 GPT-J 半维 RoPE。上游按 Q/K 各一次 launch 的 Triton 实现给出了功能基线和性能基线。A2 适配将 Q/K 合并为一个 persistent kernel，一次调用产生一个设备 kernel。A5 为保持三个精确数值边界，采用 Q/K 各一次 launch，一次调用产生两个设备 kernel。

输入支持调用方提供的 token/head/dim stride，覆盖从 packed QKV 中切出的 Q/K 视图。输出保持连续布局。当前 Triton 快路径处理 BF16、head_dim=128、rotary_dim=96 且 Q/K head 数相同的 half-split 输入；interleaved、其他几何、非 BF16 和不同 Q/K head 数继续进入既有 eager 路径。

## 接口

公开 API 保持上游签名，并保留 mojo_opset 的 `implementation` 选择：

```python
fused_qk_norm_rope(
    q,
    k,
    q_weight,
    k_weight,
    rope_table,
    eps,
    *,
    head_dim=None,
    rotary_dim=None,
    interleaved=False,
    cuda_rms_norm=False,
    implementation=None,
) -> tuple[Tensor, Tensor]
```

主几何参数如下：

| 参数 | 形状 | 类型 | 布局 |
| --- | --- | --- | --- |
| \(q\) | \([T,H,128]\) | BF16 | 连续或 strided |
| \(k\) | \([T,H,128]\) | BF16 | 连续或 strided |
| \(q_{weight}\)、\(k_{weight}\) | \([128]\) | BF16 | 连续 |
| \(rope_{table}\) | \([T,96]\) | BF16 或 FP32 | 连续 |
| \(q_{out}\)、\(k_{out}\) | 与 Q/K 同形 | BF16 | 连续 |

`rope_table` 按 token 打包为 \([\cos(\theta_0)\ldots\cos(\theta_{47})\mid\sin(\theta_0)\ldots\sin(\theta_{47})]\)。half-split 配对维度为 \((d,d+48)\)。

## 数值语义

每个 token、每个 head 独立计算一次 RMS 统计：

$$
\operatorname{inv\_rms}=
\operatorname{rsqrt}\left(
\frac{1}{128}\sum_{j=0}^{127}\operatorname{fp32}(x_j)^2+\epsilon
\right).
$$

RMS 统计在 FP32 中计算。归一化值在进入 RoPE 前舍入为 BF16：

$$
n_d=\operatorname{bf16}\left(
\operatorname{fp32}(x_d)\cdot\operatorname{inv\_rms}\cdot\operatorname{fp32}(w_d)
\right).
$$

旋转使用 FP32，并在写回时舍入到 BF16：

$$
\begin{aligned}
y_d&=\operatorname{bf16}\left(\operatorname{fp32}(n_d)c_d-\operatorname{fp32}(n_{d+48})s_d\right),
&&0\le d<48,\\
y_d&=\operatorname{bf16}\left(\operatorname{fp32}(n_d)c_{d-48}+\operatorname{fp32}(n_{d-48})s_{d-48}\right),
&&48\le d<96,\\
y_d&=n_d,
&&96\le d<128.
\end{aligned}
$$

同一 head 的 128 个输入共享一个 \(\operatorname{inv\_rms}\)。96 至 127 维直接使用已完成 BF16 舍入的归一化值，不经过第二次浮点运算。

## A2 persistent kernel

A2 快路径位于：

- `mojo_opset/kernels/npu_a2_triton/qk_norm_rope.py`
- `mojo_opset/kernels/npu_a2_triton/qk_norm_rope_persistent.py`

一次调用启动一个程序网格：

$$
\operatorname{grid}=
\left(\min(40,T),1,1\right).
$$

每个 program 连续处理一段 token，外层 token 循环采用 2-stage 软件流水。编译期分块参数为：

| 参数 | 值 |
| --- | ---: |
| vector cores | 40 |
| tokens per tile | 8 |
| heads per tile | 8 |
| token pipeline stages | 2 |
| launch stages | 1 |

每个 program 先加载一次 Q/K norm weight。每个 token tile 合并读取完整的 96 维 RoPE 表，再按 48 个 cos、48 个 sin 拆分。同一个 RoPE 表 tile 复用于当前 token tile 的全部 Q/K heads。

head 维按 8 个一组循环。每组的计算顺序为：

1. 按实际 stride 读取 Q 或 K 的 \([8,8,128]\) tile。
2. 转为 FP32，沿 128 维计算平方和、均值和 \(\operatorname{rsqrt}\)。
3. 应用 norm weight，并将完整归一化 tile 舍入到 BF16。
4. 从 BF16 tile 取出前 48 维、后 48 维和 32 维直通区。
5. 将前两段恢复为 FP32，计算 half-split RoPE。
6. 将两个旋转分支舍入到 BF16，与直通区组合后按输出 stride 写回。

Q 和 K 在同一个 kernel 内顺序处理，复用 token 索引与 RoPE 表。进程内地址偏移在最大 offset 小于 \(2^{31}\) 时使用 int32 索引，否则使用 int64。57 head 的尾块通过 mask 处理，越界输出不写入。

## A5 physical-vector kernel

A5 快路径位于 `mojo_opset/kernels/npu_a5_triton/qk_norm_rope.py`。每个 program 处理 1 个 token、8 个 head。Q/K 输入 stride 作为编译期参数传入，使 packed QKV 视图在每次特化中保持固定地址序列。

该路径按以下顺序生成每个 head：

1. 以 1D FP32 行计算 \(x_j^2\)，取前 64 维与后 64 维相加，再做 64-lane 归约。
2. 使用 \(1/\sqrt{\mu+\epsilon}\) 得到与 NPU `F.rms_norm` 逐位一致的 \(\operatorname{inv\_rms}\)。
3. 在该 head 内完成 BF16 归一化舍入，并写入 \(1\times8\times128\) tile。
4. 从完整 BF16 tile 取 48/48/32 三段，在 FP32 中做 RoPE，再合并写回。

逐 head 的 1D 统计顺序是 A5 精确性要求。将多个 head 放入同一个归约轴会改变 FP32 求和树，在 BF16 中点附近产生最多 1 ULP 的 `inv_rms` 差异，并导致直通维不再满足 `torch.equal`。当前 H8 配置在两个精确边界测试和完整 67 项矩阵上均通过。

## 调度与回退

A2/A5 Triton provider 在 `OPS` 和对应 YAML 中注册为默认实现。公开函数只把满足以下条件的输入分派到 Triton：

- Q/K 位于同一 NPU；
- Q/K 均为 BF16；
- head_dim=128、rotary_dim=96；
- half-split RoPE；
- Q/K head 数相同。

其余输入进入既有 eager 实现。Torch provider 提供可执行参考路径。A2、A3 和 950/A5 均已完成 67 项精度实机验证。A3 当前仓库只有 `npu.a2` 与 `npu.a5` 两个 provider，因此 A3 实机通过 `npu.a2` provider 运行，报告不声明原生 `npu.a3` provider。

## 性能测量

性能工具位于 `tools/benchmark_fused_qk_norm_rope.py`。每次测量先预热 5 次，再用 torch_npu profiler 连续记录 30 次调用。计时值从 `kernel_details.csv` 中提取窗口内全部设备 kernel 的 `Duration(us)`，按调用求和后取均值。A2 persistent 实现每次调用包含 1 个设备 kernel，A5 实现包含 2 个设备 kernel。

带宽基准使用 256 MiB BF16 `copy_`，计入 512 MiB 读写流量，在每一轮 benchmark 开始时现场标定。MBU 由任务公式计算：

$$
\operatorname{MBU}=
\frac{2(4TH\cdot128+96T+2\cdot128)}
{t_{\mathrm{kernel}}\cdot BW}.
$$

最终矩阵和瓶颈分析见 `docs/performance_report.md`。A2、A3 与 A5 原始数据分别保存在 `evidence/performance/`、`evidence_a3_core/` 与 `evidence_a5_core/performance/`。

## 验证设计

A2、A3 与 A5 精度套件覆盖 tokens 1/257/1024/4096/16384/65536、heads 8/56/57、连续与 packed-QKV strided 布局、BF16/FP32 RoPE 表和模块封装。普通矩阵使用 `atol=0.0625, rtol=0.02`；三个关键数值边界使用 `torch.equal` 或零容差检查：

- BF16 norm round 生效；
- 同一 head 的两个 pair 元素共享一次 RMS 统计；
- 96 至 127 维逐位等于 BF16 归一化值。

最终 A2、A3、950/A5 均为 67 项全部通过。详细命令和结果见 `docs/self_test_report.md`。

## 证据

- `evidence/accuracy_final.log`
- `evidence/performance_final.log`
- `evidence/performance/benchmark.json`
- `evidence/performance/profile_summary.csv`
- `evidence/performance/tokens_*/**/kernel_details.csv`
- `evidence/baseline/comparison.json`
- `evidence/npu-smi.txt`
- `evidence/environment.txt`
- `evidence_a3_core/a3_accuracy.log`
- `evidence_a3_core/a3_performance.log`
- `evidence_a3_core/benchmark.json`
- `evidence_a3_core/profile_summary.csv`
- `evidence_a5_core/a5_accuracy.log`
- `evidence_a5_core/a5_performance.log`
- `evidence_a5_core/performance/benchmark.json`
- `evidence_a5_core/performance/profile_summary.csv`
