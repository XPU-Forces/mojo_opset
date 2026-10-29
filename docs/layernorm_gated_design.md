# 门控归一化融合算子设计

## 目标

在 Triton-Ascend 中实现 SGLang v0.5.22 `_layer_norm_fwd_1pass_kernel` 的等价前向，覆盖 LayerNorm/RMSNorm、2D/3D 门控、门控顺序、swish/sigmoid、仿射、GroupNorm 和 FP8 e4m3 逐头量化。公开 wrapper 保持 `_layer_norm_fwd` 的参数与返回结构，A2 和 A5 分别登记到各自 provider。

输入二维视图记为 \(X\in\mathbb{R}^{M\times N}\)。每组宽度为 \(D\)，组数为 \(G=N/D\)。3D 门控张量为 \(Z\in\mathbb{R}^{T\times H\times D}\)，并满足 \(T\cdot H=M\)。token-major 视图可带 padding，通过 `stride_z_token` 和 `stride_z_head` 寻址。

复测环境为 Python 3.10、torch 2.10.0、torch_npu 2.10.0.post7、triton-ascend 3.6.0 和 CANN 9.2.0。

平台映射方面，A2 直接检测为 `npu.a2`；A3 设备名 `Ascend910_9382` 属于 arch22 同源代码族，复测时设置 `MOJO_ASCEND_ARCH=a2` 并使用 `npu_a2_triton` provider；950 检测为 `npu.a5` 并使用 `npu_a5_triton` provider，同时承担 FP8 量化的正式验收。

## 计算

当 `HAS_Z` 且 `NORM_BEFORE_GATE=False` 时先计算：

\[
X' = X\odot f(Z)
\]

其中 `swish/silu` 使用 \(f(z)=z\,\mathrm{sigmoid}(z)\)，`sigmoid` 使用 \(f(z)=\mathrm{sigmoid}(z)\)。

LayerNorm 的每组统计量为：

\[
\mu_{m,g}=\frac{1}{D}\sum_{d=0}^{D-1}X'_{m,g,d},
\qquad
\sigma_{m,g}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}(X'_{m,g,d}-\mu_{m,g})^2+\epsilon}
\]

RMSNorm 不计算均值：

\[
\sigma_{m,g}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}(X'_{m,g,d})^2+\epsilon}.
\]

归一化和仿射为：

\[
Y_{m,g,d}=\widehat X_{m,g,d}W_{g,d}+B_{g,d},
\qquad
\widehat X=(X'-\mu)/\sigma.
\]

RMSNorm 的 \(\widehat X=X'/\sigma\)。`HAS_Z` 且 `NORM_BEFORE_GATE=True` 时在仿射后乘门控 \(f(Z)\)。输出统计量按组优先布局写入长度为 \(GM\) 的 FP32 张量。

## Kernel 映射

每轮持久循环处理 `ROWS_PER_BLOCK` 行和一个归一化组。二维 tile 为 `[ROWS_PER_BLOCK, BLOCK_N]`：

- `program_id(0)` 映射持久 row-task worker，`program_id(1)` 映射归一化组。
- 行 worker 数取 `min(AI Vector Core 数, ceil(M / ROWS_PER_BLOCK))`；每个 worker 以固定步长消费后续 row-task，权重只在进入循环前加载一次。
- `BLOCK_N=min(65536//element_size, next_power_of_2(N))`，`N` 作为 constexpr 参与特化。
- `group_size <= 256` 且不是二次幂时令 `BLOCK_N=group_size`，例如尾块用例使用精确的 192 列 tile，避免为 192 个有效列计算 256 列 mask。
- 行尾和列尾由合并 mask 处理，尾块不写越界地址。
- `M` 能被行 tile 整除时启用 `EVEN_ROWS` constexpr，编译期消除行 mask。
- 权重和偏置按组偏移广播，每组只读取对应的 \(D\) 个元素。
- LayerNorm 在 FP32 tile 上先求均值，再求中心化方差；RMSNorm 直接求平方均值。
- 2D 门控按 `stride_z_row` 寻址；3D 门控按行号分解为 token/head，并使用 token/head 步长聚合搬运。
- 门控顺序、归一化类型、bias、3D 和激活类型全部映射为 `constexpr`，消除运行时分支。

公开分派先在 `functions/layernorm_gated.py` 将输入展平为 \(M\times N\)，保留 3D 门控视图，再调用 provider 的 `layernorm_gated_fwd`。非连续门控只在调用方传入最后一维不连续张量时连续化；合法 3D strided 视图直接进入 kernel，避免额外拷贝。

## FP8 epilogue

`quant_heads=QH` 时每个 program 正好处理一个 token 的 `QH` 个 head。先按输出激活 dtype 舍入，再计算：

\[
a=\max_{d}|Y_d|,\qquad
s=\frac{\max(a,10^{-10})}{448},
\qquad
q=\mathrm{clamp}(Y/s,-448,448)
\]

\(q\) 写为 float8 e4m3，\(s\) 写为 FP32。量化块布局为 `[M/QH, QH*N]`，scale 布局为 `[M/QH, 1]`。该路径把归一化、门控、舍入、amax 和量化放在一个 kernel 内。

## 行块与批不变

普通 BF16 路径按归一化宽度选择行 tile：

\[
\begin{array}{c|ccccc}
D & \le 128 & \le 256 & \le 512 & \le 1024 & >1024 \\
\hline
ROWS & 64 & 32 & 8 & 4 & 2
\end{array}
\]

其中 \(D\le128\) 且 \(M\cdot N\ge32\text{M}\) 元素时改用 `ROWS=32`：大流式张量使用 64 行 tile 时 A2 profiler 的 AIV vector ratio 为 68.4%、MTE2 ratio 为 72.6%，降到 32 行后分别提高到 91.9%、97.5%，单次耗时由约 614 µs 降到约 498 µs。该选择把中小尺寸的 128/192 列 tile 扩成较大行块，以提高每次搬运的连续性；2048 列的 GroupNorm 则以较小的行块控制 UB 占用。

`LN_GATED_BATCH_INVARIANT=1` 时固定 `ROWS=4`，不读取 \(N\)、\(M\) 或硬件核数。共享行在不同 batch 下因此使用相同的归约树和 row-task 起点。量化路径固定 `ROWS=quant_heads`，以满足单 token 一个 program 的 epilogue。

## 精度和布局约束

- 输入、权重、偏置和输出使用 FP16/BF16；统计量始终 FP32 累加和写出。
- 3D 门控保持原视图，strided 与 contiguous 输入使用相同元素和相同计算顺序，目标为逐位一致。
- FP8 量化以舍入后的激活输出计算 amax，scale 和 clamp 顺序固定。
- 输出 `out` 与 `x` 同形状；量化输出 `q/s` 独立分配。
- 测试断言和容差不在实现中改写。

## 验证和性能

精度套件覆盖 24 个功能组合，并单独检查 strided gate、FP8 bit-exact 和 batch invariant。性能套件直接调用 Triton-Ascend 3.6.0 的 `do_bench_npu_profiler`，使用 5 次热身、30 次有效调用并保留 profiler 原始 CSV；内存带宽在同一进程内以连续 BF16 `fill_`、`copy_` 探针各热身 5 次并采样 30 次，MBU 以 `copy_` 带宽为分母。逐配置结果写入验收报告。

实际结果：

| 平台 | 精度 | 长序列 MBU | 尾块 MBU | 小 shape 相对线 |
| --- | --- | ---: | ---: | --- |
| Ascend910B3 A2 | 30 passed，4 FP8 平台跳过 | 64.31% | 70.57% | 全部优于 torch eager |
| Ascend910_9382 A3 | 30 passed，4 FP8 平台跳过 | 68.98% | 79.96% | 全部优于 torch eager |

A2/A3 的完整性能矩阵均通过。FP8 位一致与 950 性能需要 Ascend 950 实机，当前证据未覆盖该平台。
