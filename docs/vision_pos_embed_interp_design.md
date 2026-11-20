# vision_pos_embed_interp Ascend 适配设计

## 需求背景

依据[任务网页](https://www.hiascend.com/activities/task-center/details/3e007e09740747338329cae8e0ee2696)及其下载任务书，将 Qwen3-VL 视觉位置编码插值移植至 Triton-Ascend，消除逐算子插值和多图循环拼接开销。适配覆盖 A2、A3、950，并保留 qwen3-omni 同类调用的参数泛化能力。位置编码与 patch embedding 的加法不在范围内。

源码依据为 vllm `aa31fa0` 的 `vllm/model_executor/models/qwen3_vl.py` 及 vllm-ascend `d5923bc` 的 `vllm_ascend/patch/worker/patch_qwen3vl.py`。两者是源码参考版本，下载任务书未发布自身 Git commit，不将源码 commit 用作任务书版本。

## 需求分析

单图公开签名保持 `triton_pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)`。位置表为 `[G*G, D]`，支持 FP32/BF16；输出为 `[t*h*w, D]`，按 spatial-merge 顺序排列。保留 h、w 可被 m_size 整除的断言；G、m_size、D 不硬编码。真实单行/单列边界为 h=1/w=1，要求相应 m_size=1。任务书将 h=m_size/w=m_size 与 scale=0 并列描述存在歧义，应以明确的上游 scale 公式为实现依据，不能将 h=2 误当作 scale=0。

多图入口增加 grid_thw 序列封装，输出顺序等价于按列表逐图调用单图函数再 cat，要求逐位相同。单图与多图复用同一插值运算，避免不同路径发生舍入差异。t>1 与 t=1 的 repeat 必须位精确。

性能主指标为热身5次后30次调用中全部设备 kernel 的 Duration(us) 总和除以30，排除 CPU 间隙。BF16 字节数按 `2*(G*G*D+t*h*w*D)` 计算，多图逐图累加。每轮现场标定 fill_ 与 copy_，以 fill_ 纯写带宽为分母。常规配置 MBU≥60%，hidden=1000尾块配置≥50%；小图(1,8,16)端到端不劣于 native。

## 算子分析

对图内输出空间序号 s，令 `block=s//(m*m)`、`local=s%(m*m)`，则 `row=(block//(w//m))*m+local//m`，`col=(block%(w//m))*m+local%m`。时间维通过 `s=local_output_row%(h*w)` 重计算，无需物化重复的中间表。

Host 计算 `h_scale=(G-1)/(h-1)`（h=1时0）和对应 w_scale。设备计算浮点坐标、floor 邻居及 min(floor+1,G-1) 上邻居。令小数部分为 dh、dw，则 `w11=dh*dw`、`w10=dh-w11`、`w01=dw-w11`、`w00=1-dh-w01`。依次读取四邻域，权重先转输出 dtype，再按上游表达式乘加。不得通过改成 FP32 累加调整容差；编译器融合与 BF16 运算实际舍入需结合编译结果和位精确测试确认。

计算量 O(sum(t*h*w)*D)，输出写入量随视频帧数增长，主要优化空间为连续列写入、邻域数据复用及减少 launch。native 保留为独立精度标杆及能力不足时的显式分派路径。

## Host/Tiling 设计

验证输入布局、形状及正整数尺寸，生成每图输出前缀和。采用有限行块和列块的二维 tile，列块按 D 与设备资源选择，尾列用 mask；行块调度覆盖全部图和帧。多图一次分配最终输出，将图尺寸、比例和前缀和作为编译期元数据或显式元数据传入一次 kernel。首版优先采用 CPU 列表形成不可变元组，避免额外设备元数据搬运 kernel；图数量增长时评估编译规模，再选择有界元数据调度。

单图与多图使用相同的每行数学实现及列块策略。跨图边界以各图本地行号寻址，写入全局前缀和偏移；不同 program 的写区互不重叠。空列表、非法尺寸和不支持布局需显式处理，不用错误后静默降级掩盖 kernel 失败。

## Kernel 设计

将连续 D 列映射为向量负载和连续写，四邻域按行广播索引。使用编译器支持的 floor 或对非负坐标等价的整数截断，替换前通过坐标边界测试验证等价性。每次只保留有限 tile 的四邻域及中间量，避免将全部图或视频时间维放入 UB。

A2/A3 与 A5(950)分别提供 provider，数学核心保持一致，分核和 tile 配置按硬件分别调优。在各 provider 的 OPS 和 `npu_a2.yaml`、`npu_a5.yaml` 中注册；公共入口放 functions，参数封装放 modules。多图输出直接写目标区，不再执行 torch.cat。

vllm 接入改造移除强制 native 的方法覆盖，在模型入口根据 HAS_TRITON、实际 NPU 能力和已登记 provider 分派到多图融合入口；能力不支持时保持 native。提供针对参考版本的补丁及应用命令，分别核验改造前 native、改造后融合结果和调用次数。禁止捕获任意编译/执行异常后回退 native 来伪装测试通过。

## 硬件与约束

三平台均要求 CANN≥9.2.0、torch-npu≥2.10.0、Triton-Ascend≥3.6.0，具体驱动、固件及预发布版本配套须按官方兼容表核实。各架构单独配置 toolkit/ops，Python 环境隔离。核数、可用 UB、实际驱动和库版本由执行环境采集，设计不假定某一隧道的固定资源。

tile 预算包括四邻域元素、坐标/权重、乘加临时值和编译器缓冲，不仅计算输出大小。大网格分散到足够 program，避免单核长循环导致507014/507034。最终配置以三平台编译及 profile 为依据，不以解释器或CPU验证替代NPU运行。

## 可维可测

精度计划完整覆盖8个主网格、FP32/BF16、t=3位精确重复、大视频(16,100,168)、G=14、hidden=1000、m_size泛化、h=1/w=1和h=m_size/w=m_size；多图融合与逐图拼接须逐位一致。固定seed=42，权重randn*0.25。正式套件断言与容差保持原样，CPU数学核对只作为开发验证。

任务下载包未附文中引用的testsuites及精度标准链接，正式容差以补齐后的官方套件为准；上游母本的FP32 atol=5e-5/rtol=1e-5、BF16 atol=rtol=1e-2仅作为已知参考，不宣称等价于缺失的官方套件。

性能覆盖任务书7项配置，逐项保存native、融合实现、现场带宽标定、全部kernel_details.csv和npu-smi输出。仓库通用perf计时为首kernel起点到末kernel终点，含间隙，与任务要求不一致；任务验收采用专用Duration求和路径，不直接拿通用perf结果替代。单launch由profile验证。

从最终完整commit独立检出，部署环境、运行官方套件和本任务扩展用例、现场标定及benchmark，然后由本次原始输出生成报告。随包提供必要脚本、配置、输入与实际输出对应说明；大视频原始输出的交付范围按官方套件要求确认，超出100M时不得自行遗漏必需材料。验收前重新下载任务书检查变化，设计、源码与测试报告按任务指定流程交付。
