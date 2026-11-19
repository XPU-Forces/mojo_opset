# vision_pos_embed_interp Ascend 适配：根因分析

> 算子：Qwen3-VL 视觉位置编码双线性插值（vllm `triton_pos_embed_interpolate` 融合版）
> 平台：Ascend 910B（A2）/ CANN 9.2.0-beta.2 / triton-ascend 3.6.0 / torch 2.9.0.post9
> 基线：r57b（job 1035，2026-11-19 02:58 裁决）｜迭代：r11–r57b，67+ 轮
> 数据口径：MBU = bytes / t / fill_标定带宽（op_summary 设备时长，5 热身 + 30 次）

---

## 一、最终基线（r57b，910B4 实机）

**精度：80/81，零真数值失败**（唯一 FAIL 为 large-bf16 性能线检，非精度问题）。
位精确门（rawAB 全家族：trep2d / trep2d-m1 / pair2d / vpair / vmb / mb 等
vs 参照腿逐位对拍）每轮全量重跑，零回归。

**性能线（MBU %，验收线：常规 ≥60%、尾块 ≥50%、A 线不劣于 native）：**

| 线 | 形状 (t,h,w) | hidden | f32 | bf16 | 验收 |
|---|---|---|---|---|---|
| tb-small-A | (1,8,16) | 1152 | A 线 | A 线 | 不劣于 native |
| tb-mid | (1,32,48) | 1152 | 85.6 | **72** | ≥60% |
| tb-large | (1,60,80) | 1152 | 80.3 | 34.7 | ≥60% |
| tb-short-video | (3,16,16) | 1152 | 127.7 | 63.5 | ≥60% |
| tb-big-video | (16,100,168) | 1152 | 105.8 | 77.8 | ≥60% |
| tb-tail | (3,60,80) | 1000 | **117.7** | **57.1** | ≥50%（尾块） |
| tb-multi-merged | 3 图合并 | 1152 | 190.2 | 89.1 | ≥60% |
| raw_prod_mb | 16x100x168 | 1152 | 105.7 | 77.5 | ≥60% |

**剩余唯一缺口：large-bf16 34.7%**（r58 接线批已制备待裁决，见第五节）。

---

## 二、迭代过程中坐实的根因

### 1. 机器时钟状态回退（DVFS）——最大干扰源，五次发生

- 特征：latency/compute-bound 小 kernel 1.4–2x 变慢，带宽饱和行（big-video、
  tail）纹丝不动（1.00x）；回退幅度随程序数递减（pair 768 程序 1.95x →
  vmb4 192 程序 1.19x），符合「降核频 + 访存延迟地板」模型。
- 根因：AICore 空闲降频（idle ~800 MHz vs 额定 1650 MHz），短 kernel 会话
  自身无法触发升频。
- 对策（测量基础设施层，非 kernel 层）：每个测量会话前 `clock_warm(120ms)`
  持续 fill 负载升频；每次回退用**零改动纯复测轮**裁决（r40/r45/r52/r54b/
  r57b 四次先例，恢复率 100%）；裁决一律以**同轮 A/B 对照**为主（DVFS 免疫）。
- 教训入册：跨轮 MBU 绝对值比较在机器态污染下无效，必须同轮对照或纯复测确认。

### 2. mid 形状是 program 启动开销（S）bound，不是指令发射 bound

- 证据链：vpair 把每对元素指令砍 55%（11→5 槽），时间零变化（30.6→30.2µs
  持平 pair）；19 AIV 波 × 1.7µs/波 ≈ 30µs 与实测吻合（S 模型坐实，
  1030 健康态）。
- 含义：mid 的正解是**摊销 S**（减少程序数），不是减指令。
- 落地：**vmb BODIES=2 转生产**（r57）：768→384 程序，S 减半，双 dtype
  同步 17.8/17.3µs（**无 dtype-split**），mid-bf16 41.7→~73 过线（r57b
  实测 72）。
- 反例（证伪）：mb4-mid 出现 dtype-split（f32 -39% / bf16 +40%）——
  「mb 结构 + 非 vbody 体」的组合在 bf16 上有惩罚；vmb 用 verbatim vbody
  规避了该惩罚。

### 3. bf16 大形状缺口的结构：字节减半 + 元素速率持平

- 任务书 MBU 以**字节**计；bf16 元素速率与 f32 持平（~59–95 Gelem/s，
  元素 bound），字节减半 → MBU 直接减半。过 60% 线需 bf16 元素速率超
  f32 ~1.5x。
- big-video-bf16 77.8%（545 GB/s）证明 bf16 数据通路可达高带宽 →
  缺口在小网格（large：4800 行）的每元素成本，而非 dtype 铸造开销。
- large-bf16 已证伪方向：mb3–mb6 波次扫描（sweet spot 2–3 body，
  -6.8% 封顶，仍 <60%）、num_stages 2/3（惰性）、pair2d 2D 化（劣化）、
  num_warps 8（无变化）。

### 4. 尾块（hidden=1000）根因：六段串行链指令过多

- hidden=1000 的精确 pow2 分解 = 512+256+128+64+32+8（6 段），串行
  load/store 链 → 0.204ms 结构地板（r33–r47 八轮不变）。
- 落地：**m1 单掩码段坍缩**（r51 转生产）：BLOCK_D=1024 一次掩码轮，
  2.3% 掩码浪费，6 段链 → 1 轮，0.204→0.081ms（2.5x/3.2x 越线，
  tail-f32 117.7 / tail-bf16 57.1）。门控：n_segs≥4 且掩码浪费 ≤1/8
  （hidden=1152 两段 43.8% 浪费不触发，保持 r48 trep2d）。

### 5. 2D 存储指令合并（trep2d，r48 转生产）

- 时间帧复制（t>1）原为 FRAMES 次串行 1D store；r44 隔离出 store 侧是墙
  （fr1 0.154ms vs FRAMES=3 0.404ms）。帧分块 [FP2C, WIDTH] 2D store
  后 big-video-f32 105.4 / big-video-bf16 77.5 双过线。值与地址与 1D 路径
  逐位一致（位精确门每轮守护）。

---

## 三、位精确工程（正确性基座）

所有 kernel 变体遵循「同地址同表达式序 → 位精确同构」：
- **零发散类**（diff6：地址相等 load 复用，严格逐位）——最强；
- **±0 角点类**（vpair/vmb-la：零权重项省略，-0.0≡0.0 值相等）——
  与产线同款发散类，rawAB 门同款口径；
- 每个变体先过 rawAB 双 dtype 位精确门（机门），再进同轮 A/B 性能对照，
  预注册裁决表触发才转生产；转生产后原 kernel 以 `_DISABLE` 旗标保留为
  同轮 A/B 参照腿，防门恒真（套套逻辑防护）。

---

## 四、交付内容（mojo_opset 布局）

| 交付项 | 路径 | 说明 |
|---|---|---|
| kernel（A2） | `kernels/npu_a2_triton/vision_pos_embed_interp/` | r57b 生产基线（vmb/m1/trep2d 转产线版），bench-only 钩子默认休眠、产线行为零变化 |
| kernel（A5/950） | `kernels/npu_a5_triton/vision_pos_embed_interp/` | 同源（Triton-Ascend 源码同一份） |
| 函数接口 | `functions/vision_pos_embed_interp.py` | 公共 API（与上游 vllm 签名一致），平台分派 loader |
| 封装模块 | `modules/vision_pos_embed/` | vllm-ascend 接入改造：摘除 `patch_qwen3vl.py` native 强制 patch，改按 `HAS_TRITON` + NPU 能力分派 |
| 精度套件 | `tests/functions/`、`tests/modules/` | 上游用例 NPU 移植版（conftest 自动选平台） |
| 性能套件 | `tests/performance/functions/test_vision_pos_embed_interp.py` | 任务书口径独立实现：bytes 公式 / op_summary 设备时长（退出后收割）/ fill_ 纯写标定 / 线检；A2 实机全量约 20–30 min |
| 分派登记 | `config/npu_a2.yaml`、`config/npu_a5.yaml` | OPS 分派条目 |

---

## 五、未闭环项与下一步

1. **large-bf16（34.7% < 60%）**：r58 接线批已制备（job 1036 已排）——
   diff6（pair diff 分支地址相等 load 复用，零发散类）与 vmb-la
   （load-ahead vbody，语句序重叠段间 load 时延）两探针 + 同轮 A/B；
   预注册止损：两者均未达 0.0369ms 关线 → 宣告 large-bf16 结构性平台
   （35–47% 区间），S/B 模型归档收束。
2. **910B4 实机全量**：1036 回传后机械裁决 + 全量精度复跑（80/81 基线保持）。
3. **本机（910B1 级 vnpu）限制**（记录在案，非代码缺陷）：该 SoC 的 CANN
   op 包缺口使内建算子（Fill 等）与 torch_npu.profiler 的 PTA/acl 调用不可用
   （561103 类）；Triton kernel 路径与 CPU 参照正常，已用于 34/34 本机验证。

---

*数据溯源：npu-runner 仓 `jobs/done/1027-vpe-tb-bench-r52`、
`1035-vpe-tb-vmbprom-r57b`（precision_summary.json + bench_sweep.csv +
op_summary_evidence/）、`各版本迭代更新说明.md`（r27–r57b 全轮次表）、
`review_log.md`（逐轮裁决记录）。*
