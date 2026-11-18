# vision_pos_embed_interp Ascend 适配：验证报告

> 基线：r57b（job 1035，2026-11-19 02:58 裁决）
> 环境：Ascend 910B4（A2）/ CANN 9.2.0-beta.2 / triton-ascend 3.6.0 /
> torch 2.9.0.post9 / 驱动 25.5.5

---

## 一、精度验证（910B4 实机，r57b 全量）

**结果：80/81，零真数值失败。** 唯一 FAIL 为 `line-tb-large-bfloat16`
（MBU 34.7 < 60%，性能线检项，非精度问题）。

### 1.1 用例构成（81 项）

- **P 系列**：形状矩阵（(1,4,4)/(1,8,16)/(1,14,20)/(1,16,16)/(1,32,48)/
  (1,48,48)/(1,60,80)/(3,16,16)/(16,100,168) × f32/bf16）vs native 参照
  数值容差对拍（f32 atol 5e-5/rtol 1e-5，bf16 atol/rtol 1e-2）；
- **rawAB 位精确门**：每个 kernel 变体（trep2d / trep2d-m1 / pair2d /
  vpair / vmb / mb / 2D store 族）vs 参照腿（legacy kernel 或 native）
  逐位对拍（`torch.equal`），防门恒真（对照腿走 `_DISABLE` 旗标强制
  legacy 路径）；
- **merged 单发射**：多图合并路径 vs 逐图发射 + `torch.cat` 位精确；
- **时间维**：t=3 输出 vs t=1 输出 `repeat(3,1)` 位精确（atol=0）。

### 1.2 复跑命令（910B4 runner 作业通道）

```bash
# 作业目录（npu-runner 仓 jobs/done/1035-vpe-tb-vmbprom-r57b/）
python3 wrapper.py          # 反作弊门（seal verify）+ 精度 + bench + 收割
# 或单独跑精度套件（交付树布局）
pytest tests/functions/test_vit_bilinear_pos_embed.py -v
```

结果文件：`precision_summary.json`（81 项逐条 PASS/FAIL + 明细）、
`bench_sweep.csv`、`op_summary_evidence/*.csv`（op_summary 原始证据）。

---

## 二、性能验证（910B4 实机，r57b 全量）

**口径（任务书固定算式）：**
- MBU = bytes / t / bw_fill；
- bytes = 2 × (G²×hidden + t·h·w×hidden)（G=48，合并路径按图累加）；
- t = op_summary「Task Duration(us)」窗口内全部 NPU kernel 之和 / 30
  （5 热身 + 30 次统计；CANN parse 子进程在进程退出后落盘 CSV，
  由收割阶段按会话序 1:1 映射回行，t_dev ≤ t_host×1.2 防错配）；
- bw_fill = `fill_` 纯写探针标定（512MB bf16 块，host 口径 5+30 中位数，
  本算子写主导，取纯写带宽为分母）；r57b 实机 700–704 GB/s；
- 每个会话前 `clock_warm(120ms)` 升频（DVFS 对策，见根因分析第二节）。

**结果（详见 `bench_sweep.csv`，r57b 实机）：**

| 线 | f32 MBU | bf16 MBU | 验收线 | 判定 |
|---|---|---|---|---|
| tb-small-A | A 线（不劣于 native） | A 线 | host 口径 | PASS |
| tb-mid | 85.6 | 72 | ≥60% | PASS |
| tb-large | 80.3 | 34.7 | ≥60% | f32 PASS / **bf16 FAIL**（已知缺口） |
| tb-short-video | 127.7 | 63.5 | ≥60% | PASS |
| tb-big-video | 105.8 | 77.8 | ≥60% | PASS |
| tb-tail（尾块） | 117.7 | 57.1 | ≥50% | PASS |
| tb-multi-merged | 190.2 | 89.1 | ≥60% | PASS |
| raw_prod_mb | 105.7 | 77.5 | ≥60% | PASS |

---

## 三、本机（910B1 级 vnpu）Triton 路径验证

> 环境：CANN 9.2.0-beta.2 本机重装 / 驱动 25.5.5 未动。
> 该 SoC 的 CANN op 包仅含 Ascend910B1 目录子集：内建算子（linspace/Fill 等
> 动态 shape 内建）JSON 缺失（561103/EZ1013 类，栈-SoC 不匹配，非环境安装
> 问题），`torch_npu.profiler` 的 PTA/acl 调用不可用（`ERR00100 PTA call
> acl api failed`）。**Triton kernel 路径与 CPU 参照正常**，用于以下验证。

### 3.1 r57b/r58 生产不变性 + 位精确门（34/34 PASS，48s）

```bash
python3 vpe_local_verify.py
# 输出：== SUMMARY 34/34 pass in 48s ==，EXIT=0
```

- 30 项正确性/位精确：8 网格 × 双 dtype vs CPU native 参照（容差同官方
  套件口径）、大网格 (16,100,168)、G=14 泛化、hidden=1000 尾块（t=1 与
  t=3 m1 路径，含本机首验 m1 kernel）、时间维 t=3 vs t=1 repeat 位精确
  （atol=0）、merged vs 逐图 cat 位精确、merged vs native；
- 4 项 r58 接线门：diff6 严格位精确（vs 产线 pair，双 dtype）、vmb-la
  位精确（vs legacy pair 与产线 vmb，双 dtype）——r58 钩子默认休眠，
  生产分派行为与 r57b 零变化。

### 3.2 独立性能套件冒烟（受限，记录在案）

独立性能套件（`tests/performance/functions/test_bench_vit_pos_embed.py`）
在本机无法完成全量：fill_ 标定与 torch_npu.profiler 均依赖本机不可用的
内建算子/PTA 通道（`AclOpKernelInit failed, opType: Fill`、
`ERR00100 PTA call acl api failed`）。已验证部分：
- 套件语法、导入、降级开关（`VPE_BENCH_NATIVE_DEV=cpu`、
  `VPE_BENCH_BW_GBPS` 覆盖）工作正常；
- 本机 triton 纯写带宽探针 1271.3 GB/s（512MB bf16，host 口径 5+30，
  仅作冒烟分母，非官方口径）；
- 数据路径（randn→NPU、Triton kernel、CPU native 参照）全部正常。

**全量性能验证以 910B4 实机为准**（第二节数据 + 1036 回传）。

---

## 四、交付树与封印

- mojo_opset 交付树 kernel 与 910B4 r57b 裁决快照
  （`jobs/done/1035-vpe-tb-vmbprom-r57b/vision_pos_embed_interp.py`）
  **逐字节一致**（A2/A5 同源双份）；
- 封印门禁：harness/seal.py 重封 8 资产（测试用例/驱动/bench 脚手架/
  native oracle；被测 kernel 不封印），verify.py --job HARNESS-OK；
  r58 接线批（1036）作业副本与 staging 三文件 cmp 逐字节一致；
- 反作弊：作业副本须 hash 匹配封印原件（exit 4 = CHEAT_DETECTED）。

## 五、未闭环

- large-bf16 34.7%：r58 探针批（1036）已排，910B4 runner 恢复后回传裁决
  （预注册止损见根因分析第五节）。
