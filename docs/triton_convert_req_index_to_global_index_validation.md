# triton_convert_req_index_to_global_index validation

## 概要

将 `triton_convert_req_index_to_global_index` 接入 mojo_opset，提供 Ascend A2/A5 Triton 实现、统一公开 API 和 PyTorch 参考。根据请求号与分页 block table 将 token 局部索引映射到全局缓存位置；workspace prefill 行使用 workspace 起点。

| 输入约定 | 语义 |
| --- | --- |
| `req_id [T]`、`block_table [R,M]`、`token_indices [T,K]` | int32；block_size 与 K 为正的 2 的幂 |
| `-1` / decode 越过页表容量 | 输出 -1 |
| workspace prefill | 不受 decode 页表容量限制；非负 workspace ID 必须有效 |
| 空 T / 非连续输入 | 支持；结果及有效索引必须在 int32 范围内 |

新性能集合为 **gb01–gb14**：固定 M=3168、block_size=64、K=2048，覆盖 decode、混合批次及长上下文 prefill。它们是按 GLM-5 配置构造的可复现输入，**不是生产 activation trace**；14 例均使用 `has_prefill=False` 的分页映射，工作负载中的 prefill 不等于 workspace 分支。完整序列长度、query 长度、seed 和生成配方已入库；A2/A5 的请求、页表、token 数据哈希与冻结的 L20 输入包全部一致。

## 改动

| 文件 | 内容 |
| --- | --- |
| `mojo_opset/functions/indexer.py` | 公开接口与契约校验 |
| `mojo_opset/kernels/npu_a{2,5}_triton/triton_convert_req_index_to_global_index.py` | 接入两平台最新交付；保留 dispatcher/Fake/设备缓存封装 |
| `mojo_opset/kernels/torch/triton_convert_req_index_to_global_index.py` | 独立 Torch 参考 |
| `tests/_convert_req_index_balanced.py`、`tests/data/convert_req_index_balanced14.json` | 新 14 例确定性配方及完整规格 |
| `tests/functions/test_triton_convert_req_index_to_global_index.py` | 原 22 例回归、边界、图测试与新 14 例逐元素精度 |
| `tests/performance/functions/test_triton_convert_req_index_to_global_index.py` | 仓库性能入口切换到新 14 例 |
| 配置、导出、支持表和验证文档 | 注册 A2/A5/Torch provider，说明复跑口径 |

## 内核设计

- **A2**：小 T 专用路径、按行分块预载页表与 gather；单请求路径将共同页表读取移到循环外；宽页表/大 K 缩小行 tile，保留多请求和 workspace 通用路径。
- **A5**：按 T/K/页表布局分派 small、per-token、batched、cached-table、预取和 workspace/cached-starts 路径；K≥1024 且补齐页表宽度≥2048 时采用两行 tile，避免 UB 超限。
- 保留单元素页表 gather 源至少两 lane 的修复；通过 mask 保护填充元素。直接 `torch.library.Library` 注册与 metadata-only Fake 支持图捕获；按设备仅缓存硬件元数据，不缓存输入或输出。
- 已合入 upstream `4868d1f`，解决 Torch 注册表冲突并保留双方新增算子。

## 精度

2026-11-21 对提交 `9351ce9` 对应源码实板复跑，整数输出严格相等，**无失败、无跳过**。

| 覆盖 | A2 | A5 |
| --- | ---: | ---: |
| 原矩阵 / reference / basic API | 25/25 | 25/25 |
| 边界、空/单元素、尾块、连续/偏移 strided、输入不变性 | 30/30 | 30/30 |
| 显式页边界 / workspace 期望值 | 1/1 | 1/1 |
| 非法 dtype/rank/shape/device/optional/参数 | 23/23 | 23/23 |
| 新 14 例完整输出逐元素对拍 | 14/14 | 14/14 |
| FakeTensor 元数据与禁止真实 kernel 执行 | 10/10 | 10/10 |
| fullgraph：eager / aot_eager、连续 / strided、换输入重执行 | 20/20 | 20/20 |
| schema / no-mutation / Fake opcheck | 5/5 | 5/5 |
| **合计** | **128/128** | **128/128** |
| 仓库标准性能入口 | **14/14** | **14/14** |

图测试新增 M=3168、K=2048、T=129/8193，覆盖宽页表路径。这里的 fullgraph 检查整个公开调用能被捕获且执行正确；尚未验证生产编译后端、NPU Graph replay、动态 shape 或完整模型。空 strided NPU view 在 SchemaCheckMode 的克隆限制仍由连续空输入 opcheck + 独立空 strided 边界/Fake 测试覆盖。

## 性能

## 验收标准与结果

新 14 例 gb01–gb14，测试实现 `9351ce90ead79dad03b1ecd0bc0dff5b1f98f3ca`；2026-11-21 实板复跑。

| 平台 | 目标 | 本轮实测 | 结论 |
| --- | --- | --- | --- |
| A2 | 清 L2 MBU >60% | **56.94%**（沿用 Event-copy 参考）；**33.89%**（设备 kernel-copy 对齐） | 未达标 |
| A5 | 不清 L2，L20/A5 几何平均 ≥1.85× | **0.9225×**，0/14 例达到 1.85× | 未达标 |
| A5 补充 | 未达标时报告清 L2 MBU | **61.93%** | 大 shape 呈访存受限特征，不能据此认定物理带宽已饱和 |

A2 与此前 Event-copy 参考口径的对照为 **56.94%**，同样未超过 60%。本轮采用与 DUT 一致的设备 kernel 计时标定 copy，分母由 Event-copy 的 1306.128 GB/s 变为 2194.624 GB/s；这不是算子耗时退化。A5 对应分母为 1569.683 GB/s，Event-copy 参考口径为 66.17%。两种口径明确分列，不用较低分母证明验收通过。

### 计时与 MBU 口径

- 验收数据取公开 API 启动的 **device kernel duration**，不是 Python E2E，也不是 Event elapsed；整数/连续 14 例每次只有一个目标 kernel。分配、校验、分派仍真实执行，但其 host 时间不计入 device kernel duration。
- 两轮，每轮 warmup=100、40 个有效样本；每例通常 80 个样本。A2 gb08 初始两轮波动 13.35%，补测后四轮均值为 525.189/595.313/521.140/629.056 µs，**全部 160 个样本保留**，全窗口范围 20.71%；此例仍标记波动，未按速度挑选窗口。
- A5 热缓存不清 L2，使用固定确定性输入；热/冷轮间最大漂移分别 4.67% / 6.13%。A2/A5 冷缓存每次调用前均使用运行时 cache buffer 的 float sum + synchronize，192 MiB；清理 kernel 不计入目标 kernel 时间。不锁频，不停止其他卡上的工作。
- L20 对照沿用用户提供的新 14 例 hot profiler 均值（20 样本/例），`speedup_i=L20_i/A5_i`；汇总为 `exp(mean(log(speedup_i)))`。本轮未重新运行 L20，原始 L20 trace 未在本地核验，因此是给定基线对照，不宣称同日跨硬件配对验收。
- 语义字节数 `B=8*T*K + 4*T + 4*R*M`；聚合 MBU=`sum(B)/sum(case_mean_us)/1000/copy_GBps*100%`，不是各例 MBU 算术均值，也不是实测 HBM transaction 利用率。
- 同卡 int32 copy，BLOCK=8192，32/64/128M 元素；设备版每尺寸两轮各 40 样本，取各尺寸合并样本 median 对应带宽的最大值。A2 为 **2194.624 GB/s**，A5 **1569.683 GB/s**；copy 与 DUT 同样清 L2、同样按设备 kernel duration 计时。它是实测 copy 参考，不等于理论硬件峰值，且带宽对工作集大小敏感。
- 为与 A2 历史交付连续对照，同时保留原 Event-copy 标定：A2 前/后 1306.128/1291.844 GB/s，A5 1469.144/1475.682 GB/s。Event 时间可含提交间隙，不能与纯 kernel 峰值混称一个口径。A2 按该参考聚合为 56.94%，按设备对齐为 33.89%，两者都未达标。
- 标准仓库性能测试另跑 14/14，使用 upstream 的固定输入 profiler flush 策略（4×L2 或回退 768 MiB）、10 次预热与 20 样本；仅用于验证仓库测试入口，不混入上述 192 MiB 验收表。
- 下表流水线比例来自实板 `aiv_*_ratio`，已核对对应 stage_time/aiv_time；同一行最高占比加粗。不同流水线可重叠，不能相加或直接当作 MBU。每例只有一个目标 kernel，目标 kernel 采样覆盖 100%；逐 kernel 名称、各流水线时间及原始 CSV 均在证据包中。

### A2 性能（全部 14 例）

| Case | T / R | 冷 L2 MBU | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gb01 | 1 / 1 | 0.64% | **48.62%** | 33.63% | 22.08% | 9.81% |
| gb02 | 8 / 8 | 4.09% | **42.23%** | 40.06% | 20.21% | 8.76% |
| gb03 | 32 / 32 | 9.91% | **50.32%** | 33.04% | 16.62% | 6.14% |
| gb04 | 64 / 64 | 15.83% | **41.92%** | 39.74% | 17.97% | 5.31% |
| gb05 | 128 / 128 | 20.57% | **44.08%** | 33.25% | 25.12% | 5.62% |
| gb06 | 64 / 33 | 11.69% | **38.28%** | 38.13% | 22.89% | 4.94% |
| gb07 | 256 / 65 | 13.60% | 38.44% | 16.51% | **48.09%** | 14.67% |
| gb08 | 8192 / 65 | 10.84% | 26.19% | 2.53% | **75.06%** | 22.78% |
| gb09 | 8192 / 1 | 36.07% | **74.01%** | 6.18% | 24.85% | 11.38% |
| gb10 | 32768 / 1 | 36.51% | **73.95%** | 5.28% | 25.92% | 11.46% |
| gb11 | 131072 / 1 | 36.52% | **73.73%** | 5.07% | 26.39% | 11.50% |
| gb12 | 202752 / 1 | 36.49% | **73.76%** | 5.00% | 26.42% | 11.50% |
| gb13 | 8192 / 1 | 35.90% | **74.02%** | 6.17% | 24.84% | 11.28% |
| gb14 | 32768 / 4 | 26.96% | **66.20%** | 5.83% | 37.69% | 9.29% |

### A5 性能（关键 7 例；完整 14 例见折叠表与 CSV）

14 例均未达到 1.85×。大 prefill/混合批次的冷 MBU 约 58–64%，MTE2 活跃占比明显高于其他流水线，说明当前实现在这些 shape 上主要受访存路径约束；可以报告已进入带宽受限区间，但**还不能据此断言基本达到物理带宽瓶颈或已无优化空间**。小 decode 的 MBU 较低，不能套用这一结论。

| Case | T / R | 热 L2 vs L20 | 冷 L2 MBU | Vector（热） | Scalar（热） | MTE2（热） | MTE3（热） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gb01 | 1 / 1 | 0.683× | 0.65% | 17.46% | **60.71%** | 23.50% | 9.19% |
| gb05 | 128 / 128 | 0.636× | 40.40% | 19.59% | **63.18%** | 25.59% | 11.04% |
| gb07 | 256 / 65 | 0.562× | 41.38% | 19.54% | 42.84% | **46.14%** | 11.25% |
| gb08 | 8192 / 65 | 1.309× | 61.21% | 39.89% | 12.24% | **96.22%** | 16.93% |
| gb10 | 32768 / 1 | 1.625× | 64.25% | 36.26% | 11.59% | **98.01%** | 18.86% |
| gb12 | 202752 / 1 | 1.597× | 63.03% | 36.66% | 11.04% | **98.40%** | 18.49% |
| gb14 | 32768 / 4 | 1.626× | 64.19% | 40.47% | 11.59% | **98.12%** | 19.74% |

<details>
<summary>A5 完整 14 例</summary>

| Case | T / R | 热 L2 vs L20 | 冷 L2 MBU | Vector（热） | Scalar（热） | MTE2（热） | MTE3（热） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gb01 | 1 / 1 | 0.683× | 0.65% | 17.46% | **60.71%** | 23.50% | 9.19% |
| gb02 | 8 / 8 | 0.660× | 4.87% | 16.72% | **59.18%** | 28.05% | 9.35% |
| gb03 | 32 / 32 | 0.567× | 16.34% | 13.57% | **69.31%** | 21.69% | 7.56% |
| gb04 | 64 / 64 | 0.505× | 24.12% | 12.44% | **72.53%** | 18.63% | 6.99% |
| gb05 | 128 / 128 | 0.636× | 40.40% | 19.59% | **63.18%** | 25.59% | 11.04% |
| gb06 | 64 / 33 | 0.517× | 20.32% | 12.38% | **71.41%** | 20.04% | 6.97% |
| gb07 | 256 / 65 | 0.562× | 41.38% | 19.54% | 42.84% | **46.14%** | 11.25% |
| gb08 | 8192 / 65 | 1.309× | 61.21% | 39.89% | 12.24% | **96.22%** | 16.93% |
| gb09 | 8192 / 1 | 1.282× | 59.80% | 34.86% | 11.94% | **96.24%** | 17.12% |
| gb10 | 32768 / 1 | 1.625× | 64.25% | 36.26% | 11.59% | **98.01%** | 18.86% |
| gb11 | 131072 / 1 | 1.529× | 60.17% | 33.66% | 10.58% | **98.10%** | 18.34% |
| gb12 | 202752 / 1 | 1.597× | 63.03% | 36.66% | 11.04% | **98.40%** | 18.49% |
| gb13 | 8192 / 1 | 1.251× | 58.38% | 37.99% | 11.67% | **96.28%** | 17.25% |
| gb14 | 32768 / 4 | 1.626× | 64.19% | 40.47% | 11.59% | **98.12%** | 19.74% |

</details>

## 平台覆盖与复跑

| 项目 | A2 | A5 |
| --- | --- | --- |
| SoC | Ascend910B3 | Ascend950PR_957c |
| 物理卡 → 逻辑卡 | 0 → 0 | 2 → 0 |
| Python | 3.10.12 | 3.12.3 |
| Torch / torch_npu | 2.7.1+cpu / 2.7.1.post11.dev20261022 | 同左 |
| Triton | 3.6.0 | 3.6.0 |
| CANN | 9.3.0 | 9.2.0-beta.2 |

```bash
# 先 source 对应 CANN 环境，激活上述 Python 环境，选择空闲健康卡。
export ASCEND_RT_VISIBLE_DEVICES=<physical_id>
python -m pytest tests/functions/test_triton_convert_req_index_to_global_index.py \
  --mojo-target npu.a2 --mojo-implementation triton -q
python -m pytest tests/performance/functions/test_triton_convert_req_index_to_global_index.py \
  --mojo-target npu.a2 --mojo-implementation triton --perf-timers profiler \
  --perf-warmup 10 --perf-repeats 20 --perf-output current.json -q
# A5 改为 --mojo-target npu.a5。
```

验收用 hot/cold profiling 入口、copy 标定、完整输入、环境、逐例样本、流水线时间、pytest XML、源码 SHA256 及复跑说明均在 [新 14 例原始测试与 profiling 证据包](https://github.com/argtarg/mojo_opset/blob/9250ac54636d0e5bb217d549918f5972d9294772/pr535_balanced14_evidence.zip)。源码与远端上传快照逐文件哈希核对一致；Ruff lint/format 与 `git diff --check` 通过。两平台仓库测试来自手动实板复跑，不等同于 GitHub 官方 CI 已通过。

本轮未出现精度或图集成失败；修复了上游合并冲突，保留了已有 singleton gather / Fake / dispatcher 修复及最新宽页表 UB 防护。测量方面修正了 Event-copy 与设备 kernel 标定混用的问题，并保留 A2 gb08 的全部波动证据。性能目标尚未满足，PR 保持 Draft；生产图与真实模型端到端验证仍未完成。
