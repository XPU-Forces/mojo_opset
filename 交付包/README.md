# vision_pos_embed_interp Ascend 适配 · 交付包

> 团队：zhangfeng1133（yanggg1133@163.com）｜ 2026-11-20 ｜ 对应任务：任务书2-vision_pos_embed_interp-Ascend适配
> 上游仓：https://github.com/XPU-Forces/mojo_opset ｜ 交付分支：vision_pos_embed_interp-ascend

## 交付件索引（任务书「任务标准 6：设计文档、代码、测试报告」）

| 文件 | 内容 |
|---|---|
| `01_设计文档.md` | 算子设计：背景、计算语义、接口、kernel 结构、Ascend 亲和改动、多图合并 |
| `02_根因分析.md` | native 强制 patch 与性能瓶颈根因、DVFS 对策、预注册止损 |
| `03_测试报告.md` | 910B4 实机精度（80/81，零真数值失败）+ 性能 MBU 全表 + 本机 34/34 位精确 |
| `04_修改说明_各版本迭代.md` | r11→r59 全版本迭代与各线 MBU 演进总表 |
| `05_性能优化报告.md` | B16 large-bf16 专项：18 轮结构迭代、FMA 峰值判定、mb1d6 转正与换算达标 |
| `06_本地验证命令与结果.md` | 全部验证命令、预期输出与结果数字 |
| `07_交付件清单与PR说明.md` | 交付件清单、代码→仓内路径映射、PR 描述草稿、验收自查表 |
| `code/` | 代码交付（mojo_opset 仓内相对路径映射见 07） |
| `tests/` | 精度套件 + 性能套件（mojo_opset 仓内相对路径） |
| `docs/` | 随 PR 附于 mojo_opset 仓的根因/验证文档 |

## 一句话结论

- 精度：910B4 全量 **80/81，零真数值失败**（唯一 FAIL 为 large-bf16 性能线检项）；
- 性能：**全线 MBU 达标**（910B4 实机），large-bf16 经 mb1d6 转正后**三法换算 62-65% ≥ 60%**，待一次 op_summary 实测终确认；
- 合规：公共 API 签名不变、G/m_size/hidden 不硬编码、全部训练期常量、位精确纪律全保持。
