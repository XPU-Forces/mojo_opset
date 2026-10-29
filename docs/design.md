# fused_qk_norm_rope Ascend 适配设计

| 项目 | 内容 |
| --- | --- |
| 算子 | `fused_qk_norm_rope` |
| 上游实现 | `vllm-project/vllm-omni`，`vllm_omni/diffusion/layers/fused_qk_norm_rope.py` |
| 上游提交 | `717f8a3499fb7dc6b3128b58a8baeacded1a7bbb` |
| 代码基线 | `mojo_opset` master 快照 `0a7d64c534bc96324ca58628f68b2a372b6f48e5` |
| 目标平台 | Ascend A2 / A5 |
| 验证平台 | Atlas 800I A2，Ascend 910B3 |
| 软件环境 | CANN 9.1.0，Torch 2.10.0，torch_npu 2.10.0，Triton 3.5.0，triton-ascend 3.2.2 |
| 任务几何 | BF16，`head_dim=128`，`rotary_dim=96`，half-split RoPE |

## 需求背景

MiniMax-H3 的 attention 层在每个 head 上执行 RMSNorm，随后应用 GPT-J 半维 RoPE。上游 Triton 半维路径分别对 Q 和 K 各启动一次融合 kernel。Ascend 当前生产路径由 `npu_rms_norm` 与 `npu_rotary_mul` 组合完成，只覆盖 BF16、`head_dim=128`、`rotary_dim=96`、非 interleaved 几何。

本任务把同一数值语义落到 triton-ascend：Q、K 各一次 launch，在单个 kernel 内完成 RMSNorm、BF16 中间舍入、FP32 RoPE 和连续输出写回。输入支持调用方给出的 token/head/dim stride，覆盖 MiniMax-H3 中由 packed QKV 切出的 Q/K 视图。

## 需求分析

公开接口保持上游签名，并增加 mojo_opset 的 `implementation` 选择项：

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

接受的主输入如下：

| 参数 | 形状 | 类型 | 布局 |
| --- | --- | --- | --- |
| \(q\) | \([T,Q,128]\) | BF16 | 连续或 strided |
| \(k\) | \([T,K,128]\) | BF16 | 连续或 strided |
| \(q_{weight}\)、\(k_{weight}\) | \([128]\) | BF16 | 连续 |
| \(rope_{table}\) | \([T,96]\) | BF16 或 FP32 | 连续 |
| \(q_{out}\)、\(k_{out}\) | 与 Q/K 同形 | BF16 | 连续 |

`rope_table` 按 token 打包为 \([\cos(\theta_0)\ldots\cos(\theta_{47})\mid\sin(\theta_0)\ldots\sin(\theta_{47})]\)。half-split 配对维度是 \((d,d+48)\)。

## 数值语义

对每个 token、每个 head，RMSNorm 统计量为：

$$
\operatorname{inv\_rms}=
\operatorname{rsqrt}\left(
\frac{1}{128}\sum_{j=0}^{127}\operatorname{fp32}(x_j)^2+\epsilon
\right).
$$

归一化值在进入 RoPE 前显式舍入到 BF16：

$$
n_d=\operatorname{bf16}\left(
\operatorname{fp32}(x_d)\cdot\operatorname{inv\_rms}\cdot\operatorname{fp32}(w_d)
\right).
$$

旋转阶段使用 FP32，写回时执行最终 BF16 舍入：

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

128 个输入值共享同一个 \(\operatorname{inv\_rms}\)。96 至 127 维直接复用已经舍入到 BF16 的归一化值，不再经历第二次浮点运算。

## 实现方案

### 分派与注册

公共函数位于 `mojo_opset/functions/fused_qk_norm_rope.py`，状态封装位于 `mojo_opset/modules/fused_qk_norm_rope.py`。A2 与 A5 的实现分别位于：

- `mojo_opset/kernels/npu_a2_triton/qk_norm_rope.py`
- `mojo_opset/kernels/npu_a5_triton/qk_norm_rope.py`

两个目标在各自的 `OPS` 表登记 `fused_qk_norm_rope -> qk_norm_rope`，并在 `mojo_opset/config/npu_a2.yaml`、`mojo_opset/config/npu_a5.yaml` 中把默认实现设置为 `triton`。Torch provider 同时提供可执行参考实现。

公共函数完成参数一致性检查。只有同时满足以下条件才进入 Triton：

- Q/K 位于同一 NPU；
- Q/K 为 BF16；
- `head_dim=128`；
- `rotary_dim=96`；
- 使用 half-split 而非 interleaved。

其余输入进入既有 eager 路径，interleaved 和通用 `head_dim`/`rotary_dim` 行为保持不变。

### Kernel 并行与地址

一个通用 3D kernel 同时服务 Q 和 K。program grid 为：

$$
\operatorname{grid}=
\left(
\left\lceil \frac{T}{T_{\mathrm{tile}}}\right\rceil,
\left\lceil \frac{H}{H_{\mathrm{tile}}}\right\rceil
\right).
$$

分块配置如下：

| head 数 | tokens/program | heads/program |
| --- | ---: | ---: |
| 8 | 1 | 8 |
| 56 | 2 | 64 |
| 57 尾块 | 2 | 64 |
| 其他 | 1（短输入）或 2 | \(\min(64, \operatorname{next\_power\_of\_2}(H))\) |

Q/K 的 token、head、dim stride 均作为运行参数传入。输入地址由实际 stride 计算；token 索引先转换为 `int64`，再参与地址运算。57 head 的最后一个 tile 通过 head mask 屏蔽无效 head，未覆盖位置不写入。

### Kernel 计算流程

1. 加载一个 `[TOKENS_PER_BLOCK, HEADS_PER_BLOCK, 128]` tile。
2. 转为 FP32，沿维度轴计算平方和、均值、`+ eps` 和 `rsqrt`。
3. 应用归一化权重，并把完整 128 维归一化 tile 舍入到 BF16。
4. 从 BF16 tile 中切出前 48 维、后 48 维和 32 维直通部分。
5. 按 token 加载 48 个 cos 与 48 个 sin，转为 FP32。
6. 在 FP32 中计算两个旋转分支，分别舍入到 BF16。
7. 将两个旋转分支和 BF16 直通部分组合成一个输出 tile，按输出 stride 写回。

该实现复用同一份 BF16 归一化 tile 完成两个配对分支，每次 token/head 只计算一次平方和与 `inv_rms`。

### Q/K launch

公开 Triton 实现先分配两个连续输出，然后分别调用一次 `_launch_qk_norm_rope`：

```text
Q tile: RMSNorm -> BF16 round -> FP32 RoPE -> BF16 store
K tile: RMSNorm -> BF16 round -> FP32 RoPE -> BF16 store
```

这与任务书规定的两次 launch 结构一致，便于 Q/K head 数独立配置。

## 性能设计

初版按 token 逐行处理，大小为 4096×56 的形状约需 1.26 ms。当前设计把相邻两个 token 和最多 64 个 head 合并到一个 program，扩大向量处理粒度，并把同一 token 的 RoPE 表加载复用于 64 个 head。

本机 `copy_` 带宽标定为 1550.49 GB/s。最终 kernel 数据如下：

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

大型 56/57-head 形状未达到任务书的 60%/50% MBU 线。4096×56 的 profiling 显示两次 kernel 的 AIV vector pipe 利用率分别为 79.2% 和 84.0% 至 84.7%，AIV MTE2 约 13% 至 19%，AIV MTE3 约 5.6%。数据搬运管线仍有空闲，瓶颈是归一化、BF16/FP32 转换与 RoPE 算术占满向量计算管线。

后续优化应减少向量指令数。可评估的方向包括：

- 让同一 program 处理更多 token，摊薄 RoPE 表加载与循环控制；
- 在保持 BF16 中间舍入和 FP32 旋转边界的前提下，减少重复的布局转换；
- 对 32 维直通区采用独立存储路径，避免与 128 维旋转 tile 一起形成不必要的向量操作；
- 比较 Q/K 合并 launch 对小型形状的收益，保留与任务语义一致的分派条件。

小 shape 不适用 MBU 线。32×8 与 320×8 的主指标和端到端时间均优于原生组合路径：

| tokens×heads | Triton kernel (us) | Native kernel (us) | Triton E2E (ms) | Native E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32×8 | 8.19 | 92.72 | 0.4246 | 0.7072 |
| 320×8 | 29.32 | 111.55 | 0.4339 | 0.7065 |

## 测试设计

任务 ZIP 只包含任务书，未包含文档引用的 `testsuites/task1_rms_norm_rope`。仓库因此加入等价测试套件，测试基准取自上游同一提交，并覆盖任务书列出的精度矩阵：

- tokens：1、257、1024、4096、16384、65536；
- heads：8、56、57；
- 连续 Q/K 与 packed QKV strided 视图；
- BF16 与 FP32 RoPE 表；
- BF16 中间舍入与共享 RMS 统计的 `torch.equal` 精确检查；
- 96 至 127 维直通位的 bitwise 检查；
- 确定性、interleaved eager 回退和零 token 行为。

A2 910B3 上函数套件 64 项全部通过，模块套件 3 项全部通过。A5 的源码和分派已接入，当前交付环境只有 A2 910B3，A5 与 950 实机运行仍需对应硬件复核。

## 可维可测分析

| 维度 | 设计处理 |
| --- | --- |
| 精度 | 保留 BF16 中间舍入、FP32 旋转与单次输出舍入 |
| 布局 | 显式传递 Q/K 的 token/head/dim stride |
| 大偏移 | token 地址使用 64 位索引 |
| 尾块 | 57 head 用 mask 覆盖无效 head |
| 回退 | 非目标几何进入既有 eager 实现 |
| 性能证据 | 保留逐形状 `kernel_details.csv`、带宽探针、`npu-smi` 和 JSON |
| 兼容性 | 保持上游公开参数顺序和作用；Torch provider 可执行 |

原始精度、性能、设备信息和 profiling 文件归档在任务交付目录的 `evidence/`。
