from dataclasses import dataclass
from math import gcd
from typing import Optional
from typing import Tuple

import heapq
import torch
import triton
import triton.language as tl

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from ..utils import get_num_cores
from ..utils import is_910


TILE_BLOCK_SIZE = 128

# Q 侧加权 task list 调度的最小相对改善阈值: round-robin baseline 与 LPT 装箱
# 之间的改善率低于此值时, 认为不值得引入 task list 开销, 直接使用 round-robin。
_Q_TASKLIST_MIN_RELATIVE_IMPROVEMENT = 0.05

# DKDV 侧负载均衡阈值
_DKDV_MAX_FULL_ROUNDS_FOR_TAIL_SPLIT = 2
_DKDV_TAIL_RATIO_THRESHOLD = 0.5
# max_weight/mean_weight 超过此比例时启用 task list 调度。
# 设为 1.3 而非 1.5: 当权重变异超过 30% 时, round-robin 的核间不均衡
# 可达 10%+ (如 cross_sample mask: text/image 混合, max/mean=1.49,
# round-robin 不均衡 1.13x), LPT 装箱可将不均衡降至 ~1.01x。
_DKDV_WEIGHT_IMBALANCE_THRESHOLD = 1.3

# DKDV 装箱 heavy-light 分解阈值: weight > target * 此比例的 item 视为重 item,
# 预分配到空核后再用容量感知 fill-ratio heap 装箱轻 item, 避免 min-heap 后期
# 将重 item 堆叠到半满核上 (典型场景: SWA global-window 首块 w=512 vs target=725)
_DKDV_HEAVY_RATIO_THRESHOLD = 0.3

# fill-ratio heap 的整数化精度 (key = light_weight / light_capacity * SCALE)
_FILL_RATIO_SCALE = 100000

# 张量维度常量
_WORK_ITEM_COLUMNS = 2
_INVALID_BLOCK_INDEX = -1


def _get_num_aicore():
    try:
        return max(get_num_cores(op_type="cube"), 1)
    except Exception:
        return 1


def _persistent_launch_config(num_tasks):
    """计算 persistent kernel 的 launch grid 和实际任务数。

    Args:
        num_tasks: 原始任务总数。

    Returns:
        (grid, num_tasks): grid 为单元素 tuple (active_cores,),
            num_tasks 为钳位后的有效任务数 (至少为 1)。
    """
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks


def _get_mask_block_sizes(block_mask):
    """从 BlockMask 提取稀疏分块大小 (Q, KV)。

    BlockMask.BLOCK_SIZE 记录了掩码构建时使用的分块粒度, kernel 必须以该粒度
    解释 kv_block / q_block 索引, 否则会导致 KV 位置计算错误。

    Args:
        block_mask: 由 create_block_mask_patched / BlockMask.from_kv_blocks 构建的掩码对象。

    Returns:
        (sparse_q_block_size, sparse_kv_block_size): 掩码的 Q / KV 分块大小。
        当 block_mask 缺少 BLOCK_SIZE 属性时, 回退为 (TILE_BLOCK_SIZE, TILE_BLOCK_SIZE)。

    Raises:
        ValueError: 当分块大小不是 TILE_BLOCK_SIZE 的正整数倍时。
    """
    raw_bs = getattr(block_mask, "BLOCK_SIZE", None)
    if raw_bs is None:
        sparse_q_block_size = sparse_kv_block_size = TILE_BLOCK_SIZE
    elif isinstance(raw_bs, int):
        sparse_q_block_size = sparse_kv_block_size = raw_bs
    else:
        sparse_q_block_size, sparse_kv_block_size = raw_bs

    if (
        not isinstance(sparse_q_block_size, int)
        or not isinstance(sparse_kv_block_size, int)
        or sparse_q_block_size < TILE_BLOCK_SIZE
        or sparse_kv_block_size < TILE_BLOCK_SIZE
        or sparse_q_block_size % TILE_BLOCK_SIZE != 0
        or sparse_kv_block_size % TILE_BLOCK_SIZE != 0
    ):
        raise ValueError(
            "FlexAttention NPU kernels require Q and KV BlockMask block sizes "
            f"to be integer multiples of {TILE_BLOCK_SIZE} and at least "
            f"{TILE_BLOCK_SIZE}; received (Q, KV)=({sparse_q_block_size}, "
            f"{sparse_kv_block_size}). Block sizes such as 32, 64, or "
            "non-divisible values are unsupported by the current 128-tile kernels."
        )

    return sparse_q_block_size, sparse_kv_block_size


# ===========================================================================
# Q 侧负载均衡: LPT (Longest Processing Time first) 加权 task list 调度
#
# 设计动机:
#   原始 round-robin 调度按 task_id 顺序分配任务到各核, 当不同 Q block 的
#   KV-block 数量差异较大时 (如 SWA mask 首块 global window 导致 w 远超均值),
#   round-robin 会导致核间负载严重不均衡。LPT 贪心装箱将重任务优先分配到
#   空核或最轻核, 使各核负载逼近理论下界。
#
# 判定流程 (两级 early exit):
#   Level 1 — _may_need_q_task_schedule: 纯几何快速判定, 无需同步 device 权重
#   Level 2 — _build_q_task_schedule: 同步权重 + round-robin baseline 估算 +
#             完整 LPT 装箱 + 改善率二次校验
# ===========================================================================

def _may_need_q_task_schedule(
    Z: int, Hq: int, num_q_blocks: int,
    sparse_q_multiple: int, num_core: int,
) -> bool:
    """纯几何快速判定 Q-task 加权调度是否可能改善负载均衡。

    仅使用 host 侧已知的 launch 几何参数, 不需要同步 device 权重。
    返回 False 时可证明 round-robin 已足够均衡, 调用方跳过权重同步。

    Args:
        Z: 批次数。
        Hq: Q head 数。
        num_q_blocks: Q block 总数。
        sparse_q_multiple: 稀疏分块换算因子。
        num_core: 核数。

    Returns:
        True 表示可能需要加权调度 (需进一步同步权重判定);
        False 表示 round-robin 已足够, 可安全跳过。
    """
    # Step 1: 任务数不超过核数时, 每核最多一个任务, 无需均衡
    total_tasks = Z * Hq * num_q_blocks
    active_cores = min(max(int(num_core), 1), max(total_tasks, 1))
    if total_tasks <= num_core:
        return False

    # Step 2: 仅有一个稀疏 block 时, 所有任务等重, 无需均衡
    num_sparse_q_blocks = (num_q_blocks + sparse_q_multiple - 1) // sparse_q_multiple
    if num_sparse_q_blocks <= 1:
        return False

    # Step 3: 当 block 数与核数互素且 copy 数整除核数时,
    # round-robin 天然均衡, 无需加权
    task_copies = Z * Hq
    return not (
        gcd(num_q_blocks, active_cores) == 1
        and task_copies % active_cores == 0
    )


def _build_q_task_schedule(
    q_sparse_weights,
    Z: int,
    Hq: int,
    Hkv: int,
    num_q_blocks: int,
    sparse_q_multiple: int,
    num_core: int,
) -> Tuple[Optional[list], bool]:
    """为 forward/dQ 构建无拆分、无归约的加权 Q-task 排列。

    通过三级判定逐步决定是否需要加权调度:
      Level 1 — round-robin baseline 快速估算 (O(num_q_blocks), 无需装箱)
      Level 2 — 完整贪心装箱 + 改善率二次校验
    任一级别不满足阈值即 early exit, 返回 (None, False)。

    Args:
        q_sparse_weights: 每个 sparse Q block 的有效 KV-block 数 (partial + full 等权)。
        Z / Hq / Hkv: 批次数 / Q head 数 / KV head 数。
        num_q_blocks: Q block 总数。
        sparse_q_multiple: 稀疏分块换算因子。
        num_core: 核数。

    Returns:
        (task_ids, use_task_list): use_task_list=False 时 task_ids 为 None,
            表示 round-robin 已足够; use_task_list=True 时 task_ids 为加权排列列表。

    Raises:
        ValueError: 当权重列表长度不足以覆盖所有 Q blocks 时。
    """
    # Step 1: 展开权重并校验
    total_tasks = Z * Hq * num_q_blocks
    active_cores = min(max(int(num_core), 1), max(total_tasks, 1))
    num_sparse_q_blocks = (num_q_blocks + sparse_q_multiple - 1) // sparse_q_multiple
    weight_values = list(q_sparse_weights)
    if len(weight_values) < num_sparse_q_blocks:
        raise ValueError(
            "FlexAttention Q-task weights do not cover all Q blocks: "
            f"expected {num_sparse_q_blocks}, got {len(weight_values)}"
        )

    block_weights = [
        max(float(weight_values[q_block // sparse_q_multiple]), 1.0)
        for q_block in range(num_q_blocks)
    ]
    gqa_shared_heads = Hq // Hkv if Hq >= Hkv else 1

    # Step 2: round-robin baseline 快速计算 (Level 1 判定)
    task_copies = Z * Hq
    baseline_max = _compute_round_robin_baseline_max(
        block_weights, num_q_blocks, active_cores, task_copies,
    )
    total_weight = sum(block_weights) * task_copies
    theoretical_max = total_weight / active_cores
    quick_improvement = (baseline_max - theoretical_max) / max(baseline_max, 1.0)
    if quick_improvement < _Q_TASKLIST_MIN_RELATIVE_IMPROVEMENT:
        return None, False

    # Step 3: 完整贪心装箱 (Level 2 判定)
    per_core_tasks, capacities, core_loads = _greedy_pack_q_tasks(
        block_weights, total_tasks, active_cores, num_q_blocks,
        Z, Hq, Hkv, gqa_shared_heads,
    )
    balanced_max = max(core_loads)
    relative_improvement = (baseline_max - balanced_max) / max(baseline_max, 1.0)
    if relative_improvement < _Q_TASKLIST_MIN_RELATIVE_IMPROVEMENT:
        return None, False

    # Step 4: 轮转展平 — 按 round-robin 交替取出各核任务, 保证 persistent 调度连续性
    task_ids = _flatten_per_core_tasks(per_core_tasks, capacities)
    return task_ids, True


def _compute_round_robin_baseline_max(
    block_weights: list, num_q_blocks: int,
    active_cores: int, task_copies: int,
) -> float:
    """计算 round-robin 调度下最重核的负载。

    利用 residue_sums 预聚合 + offset 计数, 将 O(total_tasks) 降为
    O(num_q_blocks + distinct_offsets * active_cores)。

    Args:
        block_weights: 每个 Q block 的权重列表。
        num_q_blocks: Q block 总数。
        active_cores: 实际使用的核数。
        task_copies: Z * Hq, 每个 q_block 的任务副本数。

    Returns:
        round-robin 调度下最重核的负载值。
    """
    residue_sums = [0.0] * active_cores
    for q_block in range(num_q_blocks):
        residue_sums[q_block % active_cores] += block_weights[q_block]

    offset_counts = {}
    for copy_id in range(task_copies):
        offset = (copy_id * num_q_blocks) % active_cores
        offset_counts[offset] = offset_counts.get(offset, 0) + 1

    baseline_loads = [0.0] * active_cores
    for offset, count in offset_counts.items():
        for core_id in range(active_cores):
            baseline_loads[core_id] += count * residue_sums[(core_id - offset) % active_cores]

    return max(baseline_loads)


def _greedy_pack_q_tasks(
    block_weights: list, total_tasks: int, active_cores: int,
    num_q_blocks: int, Z: int, Hq: int, Hkv: int, gqa_shared_heads: int,
) -> Tuple[list, list, list]:
    """使用 LPT (Longest Processing Time first) 贪心装箱, 将 Q tasks 均衡分配到各核。

    将所有任务按权重全局降序排列后, 依次放入当前最轻的核 (min-heap)。
    相比按 head 分组的嵌套遍历, 全局 LPT 保证重任务优先占据空核,
    轻任务精确填缝, 使各核负载逼近理论下界 ceil(total_weight / num_core)。

    Args:
        block_weights: 每个 Q block 的权重列表。
        total_tasks: Z * Hq * num_q_blocks, 任务总数。
        active_cores: 实际使用的核数。
        num_q_blocks: Q block 总数。
        Z / Hq / Hkv: 批次 / Q head / KV head 数。
        gqa_shared_heads: Hq // Hkv, GQA 共享头数 (保留用于兼容签名, LPT 不依赖分组)。

    Returns:
        (per_core_tasks, capacities, core_loads):
            per_core_tasks[core_id] 为该核的任务 ID 列表;
            capacities[core_id] 为该核的容量上限;
            core_loads[core_id] 为该核的实际负载。
    """
    # Step 1: 计算每核容量上限 (前 extra_tasks 个核多分一个任务)
    tasks_per_core, extra_tasks = divmod(total_tasks, active_cores)
    capacities = [
        tasks_per_core + (1 if core_id < extra_tasks else 0)
        for core_id in range(active_cores)
    ]
    remaining_capacities = capacities.copy()
    core_loads = [0.0] * active_cores
    per_core_tasks = [[] for _ in range(active_cores)]
    core_heap = [(0.0, core_id) for core_id in range(active_cores)]
    heapq.heapify(core_heap)

    # Step 2: 枚举所有 (off_z, off_hq, q_block) 任务并按权重全局降序排列。
    #
    # LPT 策略的核心: 重任务先放置, 优先占据空核或最轻核;
    # 轻任务后放置, 精确填入剩余缝隙。全局排序消除了嵌套遍历中
    # 同一 q_block 的 head 副本被穿插放置导致的重任务堆叠问题。
    all_tasks = []
    for off_z in range(Z):
        for off_hq in range(Hq):
            for q_block in range(num_q_blocks):
                weight = block_weights[q_block]
                all_tasks.append((weight, off_z, off_hq, q_block))
    all_tasks.sort(key=lambda t: t[0], reverse=True)

    # Step 3: 依次将排序后的任务放入当前最轻的核 (min-heap 贪心)。
    for weight, off_z, off_hq, q_block in all_tasks:
        _, core_id = heapq.heappop(core_heap)
        task_id = (off_z * Hq + off_hq) * num_q_blocks + q_block
        per_core_tasks[core_id].append(task_id)
        remaining_capacities[core_id] -= 1
        core_loads[core_id] += weight
        if remaining_capacities[core_id] > 0:
            heapq.heappush(core_heap, (core_loads[core_id], core_id))

    return per_core_tasks, capacities, core_loads


def _flatten_per_core_tasks(per_core_tasks: list, capacities: list) -> list:
    """将各核任务列表按 round-robin 轮转展平为一维序列。

    persistent kernel 按展平后的顺序依次执行任务,
    轮转展平确保相邻任务来自不同核, 减少尾部空闲。

    Args:
        per_core_tasks: 各核的任务 ID 列表。
        capacities: 各核容量上限 (决定轮转轮数)。

    Returns:
        展平后的任务 ID 一维列表。
    """
    task_ids = []
    for task_round in range(max(capacities)):
        for core_id, core_tasks in enumerate(per_core_tasks):
            if task_round < len(core_tasks):
                task_ids.append(core_tasks[task_round])
    return task_ids


# ===========================================================================
# 统一负载均衡计划: Q 侧 + DKDV 侧打包为单一计划, 由 _get_or_build_load_balance_plan
# 构建并缓存到 BlockMask 上, 供 forward / bwd kernel 共享复用。
# ===========================================================================

@dataclass
class _FlexAttentionLoadBalancePlan:
    """FlexAttention 的统一负载均衡计划 (Q 侧 + DKDV 侧)。

    封装 Q 侧和 DKDV 侧两套独立的任务调度方案, 由 _get_or_build_load_balance_plan
    构建并缓存到 BlockMask 上, 供 forward / bwd kernel 复用。

    Attributes:
        q_task_ids: Q 侧任务 ID 排列张量, shape [num_q_tasks]。
            use_q_task_list=False 时退化为 bm["kv_num_blks"] (原始 round-robin 序列)。
        use_q_task_list: Q 侧是否启用加权 task list 调度。
        dkdv_work_items: DKDV work items, shape [num_work, 2],
            每行 = (hkv, kv_block)。
        dkdv_task_offsets: CSR 偏移, shape [num_core+1], 每核 work-item 起始索引。
        use_dkdv_task_list: DKDV 侧是否启用 task list 调度 (仅控制分支选择, task list 始终构建)。
    """
    q_task_ids: torch.Tensor
    use_q_task_list: bool
    dkdv_work_items: torch.Tensor
    dkdv_task_offsets: torch.Tensor
    use_dkdv_task_list: bool


@dataclass
class _DkdvSidePlan:
    """DKDV 侧负载均衡计划的中间结果。

    Attributes:
        work_items: [num_work, 2] int32 work items 张量, 每行 = (hkv, kv_block)。
        task_offsets: [num_core+1] int32 CSR 偏移张量。
        use_task_list: 是否启用 task list 调度。
    """
    work_items: torch.Tensor
    task_offsets: torch.Tensor
    use_task_list: bool


def _build_q_side_plan(
    bm: dict, Z: int, Hq: int, Hkv: int,
    num_q_blocks: int, sparse_q_multiple: int,
    num_core: int, device,
) -> Tuple[torch.Tensor, bool]:
    """构建 Q 侧负载均衡计划 (非缓存, 由上层统一缓存)。

    仅在 _may_need_q_task_schedule 返回 True 时同步权重并构建加权排列;
    否则直接使用原始 round-robin 序列 (bm["kv_num_blks"])。

    Args:
        bm: BlockMask 属性字典。
        Z / Hq / Hkv: 批次 / Q head / KV head 数。
        num_q_blocks: Q block 总数。
        sparse_q_multiple: 稀疏分块换算因子。
        num_core: 核数。
        device: 计算设备。

    Returns:
        (q_task_ids, use_q_task_list)。
    """
    if not _may_need_q_task_schedule(Z, Hq, num_q_blocks, sparse_q_multiple, num_core):
        return bm["kv_num_blks"], False

    q_weight_values = (bm["kv_num_blks"] + bm["full_kv_num_blks"]).reshape(-1).tolist()
    q_task_ids, use_q_task_list = _build_q_task_schedule(
        q_weight_values, Z, Hq, Hkv, num_q_blocks, sparse_q_multiple, num_core,
    )
    if use_q_task_list:
        q_task_ids = torch.tensor(q_task_ids, dtype=torch.int32, device=device)
    else:
        q_task_ids = bm["kv_num_blks"]
    return q_task_ids, use_q_task_list


def _build_dkdv_side_plan(
    bm: dict, Z: int, Hkv: int,
    num_kv_blocks: int, sparse_kv_multiple: int,
    num_core: int, device, build_dkdv: bool,
) -> "_DkdvSidePlan":
    """构建 DKDV 侧负载均衡计划 (非缓存, 由上层统一缓存)。

    采用两级判定:
    - Level 1: Z != 1 或不需要 dkdv 时直接返回 SIMPLE 模式, 跳过权重同步。
    - Level 2: 同步权重并检测尾部不均或权重不均衡, 确定 SIMPLE / TASKLIST 模式。
    - Level 3: 完整装箱 (仅 TASKLIST 模式触发)。

    权重不均衡检测不依赖 tail_cores 是否为 0, 因为极端 outlier
    (如 SWA mask 中 global window 集中在首个 kv_block) 在任意
    full_rounds 下都会导致 round-robin 分配不均。

    Args:
        bm: BlockMask 属性字典。
        Z: 批次数 (Z > 1 时不构建 DKDV 计划)。
        Hkv: KV head 数。
        num_kv_blocks: KV block 总数。
        sparse_kv_multiple: 稀疏分块换算因子。
        num_core: 核数。
        device: 计算设备。
        build_dkdv: 是否构建 DKDV 侧计划。

    Returns:
        _DkdvSidePlan 封装的 DKDV 侧计划。
    """
    # Step 1: 默认 SIMPLE 模式 — Z > 1 或不需要 dkdv 时直接返回
    if not build_dkdv or Z != 1:
        return _DkdvSidePlan(
            work_items=torch.empty((0, _WORK_ITEM_COLUMNS), dtype=torch.int32, device=device),
            task_offsets=torch.zeros((num_core + 1,), dtype=torch.int32, device=device),
            use_task_list=False,
        )

    # Step 2: Level 2 — 同步权重并做完整判定
    total_base = num_kv_blocks * Hkv
    full_rounds = total_base // num_core
    tail_cores = total_base % num_core

    dkdv_weight_values = (bm["q_num_blks"] + bm["full_q_num_blks"]).reshape(-1).tolist()
    total_w = Hkv * sum(dkdv_weight_values)
    mean_w = total_w / max(total_base, 1)
    max_w = max(dkdv_weight_values, default=0)

    has_significant_tail = (
        total_w > 0 and tail_cores > 0
        and full_rounds <= _DKDV_MAX_FULL_ROUNDS_FOR_TAIL_SPLIT
        and tail_cores / num_core < _DKDV_TAIL_RATIO_THRESHOLD
    )
    has_weight_imbalance = (
        mean_w > 0 and max_w / mean_w > _DKDV_WEIGHT_IMBALANCE_THRESHOLD
    )
    use_task_list = has_significant_tail or has_weight_imbalance

    if not use_task_list:
        return _DkdvSidePlan(
            work_items=torch.empty((0, _WORK_ITEM_COLUMNS), dtype=torch.int32, device=device),
            task_offsets=torch.zeros((num_core + 1,), dtype=torch.int32, device=device),
            use_task_list=False,
        )

    # Step 3: Level 3 — 完整装箱
    target = total_w / num_core
    work_items, task_offsets = _build_dkdv_task_schedule(
        dkdv_weight_values, Hkv, num_kv_blocks, sparse_kv_multiple,
        target, num_core, device,
    )
    return _DkdvSidePlan(
        work_items=work_items,
        task_offsets=task_offsets,
        use_task_list=True,
    )


def _build_load_balance_plan(
    bm: dict,
    Z: int,
    Hq: int,
    Hkv: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparse_q_multiple: int,
    sparse_kv_multiple: int,
    num_core: int,
    device,
    build_dkdv: bool,
) -> "_FlexAttentionLoadBalancePlan":
    """构建统一负载均衡计划 (Q 侧 + DKDV 侧)。

    Q 侧通过 _build_q_side_plan 构建加权排列 (含 early exit)。
    DKDV 侧先做纯算术快速判断, 能确定 SIMPLE 模式时跳过权重同步与
    task list 构建; 仅在可能不均衡时才同步权重并做完整判定。

    Args:
        bm: BlockMask 属性字典 (_prepare_block_mask_attrs 的输出)。
        Z / Hq / Hkv: 批次 / Q head 数 / KV head 数。
        num_q_blocks / num_kv_blocks: Q block / KV block 总数。
        sparse_q_multiple / sparse_kv_multiple: 稀疏分块换算因子。
        num_core: 核数。
        device: 计算设备。
        build_dkdv: 是否构建 DKDV 侧计划。

    Returns:
        _FlexAttentionLoadBalancePlan 封装的完整计划。
    """
    # Step 1: Q 侧 — 仅在几何形状可能不均衡时同步权重并构建加权排列
    q_task_ids, use_q_task_list = _build_q_side_plan(
        bm, Z, Hq, Hkv, num_q_blocks, sparse_q_multiple, num_core, device,
    )

    # Step 2: DKDV 侧 — 纯算术快速判断, 能确定 SIMPLE 则跳过所有计算
    dkdv_result = _build_dkdv_side_plan(
        bm, Z, Hkv, num_kv_blocks, sparse_kv_multiple, num_core, device, build_dkdv,
    )

    return _FlexAttentionLoadBalancePlan(
        q_task_ids=q_task_ids,
        use_q_task_list=use_q_task_list,
        dkdv_work_items=dkdv_result.work_items,
        dkdv_task_offsets=dkdv_result.task_offsets,
        use_dkdv_task_list=dkdv_result.use_task_list,
    )


def _make_load_balance_plan_cache_key(
    block_mask,
    Z: int,
    Hq: int,
    Hkv: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparse_q_multiple: int,
    sparse_kv_multiple: int,
    num_core: int,
    device,
    build_dkdv: bool,
) -> tuple:
    """构建统一负载均衡计划的缓存键。

    缓存键由 BlockMask 核心张量的身份标识 + 几何参数组成, 确保同一 BlockMask
    在相同形状参数下的计划只需构建一次。使用 BlockMask 原始属性 (而非
    _prepare_block_mask_attrs 处理后的 bm dict), 避免每次调用因 .to(int32)
    .contiguous() 产生新张量导致缓存未命中。

    Args:
        block_mask: BlockMask 对象, 提供稀疏分块索引表张量。
        Z / Hq / Hkv: 批次 / Q head / KV head 数。
        num_q_blocks / num_kv_blocks: Q block / KV block 总数。
        sparse_q_multiple / sparse_kv_multiple: 稀疏分块换算因子。
        num_core: 核数。
        device: 计算设备。
        build_dkdv: 是否构建 DKDV 侧计划 (影响计划内容)。

    Returns:
        可哈希的缓存键元组, 包含所有影响计划构建结果的参数。
    """

    def _tensor_cache_id(tensor: Optional[torch.Tensor]) -> tuple:
        if tensor is None:
            return None
        return (
            tensor.data_ptr(),
            getattr(tensor, "_version", 0),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

    return (
        _tensor_cache_id(block_mask.kv_num_blocks),
        _tensor_cache_id(getattr(block_mask, "full_kv_num_blocks", None)),
        _tensor_cache_id(block_mask.q_num_blocks),
        _tensor_cache_id(getattr(block_mask, "full_q_num_blocks", None)),
        Z, Hq, Hkv,
        num_q_blocks, num_kv_blocks,
        sparse_q_multiple, sparse_kv_multiple,
        num_core, device, build_dkdv,
    )


def _get_or_build_load_balance_plan(
    block_mask,
    bm: dict,
    Z: int,
    Hq: int,
    Hkv: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparse_q_multiple: int,
    sparse_kv_multiple: int,
    device,
    build_dkdv: bool,
) -> "_FlexAttentionLoadBalancePlan":
    """获取或构建统一负载均衡计划 (带缓存)。

    同一 BlockMask 在相同形状参数下的后续调用直接复用缓存, 避免重复装箱。
    缓存键由 BlockMask 核心张量的身份标识 + 几何参数组成。
    forward 首次调用时预构建 DKDV 计划, backward 直接复用。

    Args:
        block_mask: BlockMask 对象, 计划缓存到其 _load_balance_plan 属性上。
        bm: BlockMask 属性字典 (_prepare_block_mask_attrs 的输出)。
        Z / Hq / Hkv: 批次 / Q head / KV head 数。
        num_q_blocks / num_kv_blocks: Q / KV block 总数。
        sparse_q_multiple / sparse_kv_multiple: 稀疏分块换算因子。
        device: 计算设备。
        build_dkdv: 是否构建 DKDV 侧计划。

    Returns:
        _FlexAttentionLoadBalancePlan 封装的完整计划。
    """
    # Step 1: 计算缓存键并检查是否命中
    num_core = _get_num_aicore()
    cache_key = _make_load_balance_plan_cache_key(
        block_mask, Z, Hq, Hkv, num_q_blocks, num_kv_blocks,
        sparse_q_multiple, sparse_kv_multiple, num_core, device, build_dkdv,
    )
    if getattr(block_mask, "_load_balance_plan_cache_key", None) == cache_key:
        return block_mask._load_balance_plan

    # Step 2: 缓存未命中, 构建新计划并写入缓存
    plan = _build_load_balance_plan(
        bm, Z, Hq, Hkv, num_q_blocks, num_kv_blocks,
        sparse_q_multiple, sparse_kv_multiple, num_core, device, build_dkdv,
    )
    block_mask._load_balance_plan_cache_key = cache_key
    block_mask._load_balance_plan = plan
    return plan


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_partial_p", "stride_partial_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_out_z", "stride_out_h",
    ]
)
def flex_attention_kernel(
    Q,
    K,
    V,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    Q_TASK_IDS,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    OUT,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_out_z, stride_out_h, stride_out_m, stride_out_k,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_kv_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
    USE_Q_TASK_LIST: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_slot in range(pid, NUM_TASKS, num_core):
        task_id = task_slot
        if USE_Q_TASK_LIST:
            task_id = tl.load(Q_TASK_IDS + task_slot).to(tl.int32)

        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        out_offset = off_z * stride_out_z + off_hq * stride_out_h
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        OUT_ptr = OUT + out_offset
        LSE_ptr = LSE + lse_offset

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0
        )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        partial_mask_offset = tl.load(PARTIAL_MASK_OFFSETS + q_sparse_idx * stride_partial_offset_m)
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            if USE_PACKED_PARTIAL_MASK:
                partial_block_idx = partial_mask_offset + blk_idx_in_list
                offs_m_in_block = offs_m - q_sparse_base
                offs_n_in_block = (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = load_packed_partial_mask(
                    PARTIAL_MASK_PACKED,
                    stride_partial_p,
                    stride_partial_m,
                    stride_partial_n,
                    partial_block_idx,
                    offs_m_in_block,
                    offs_n_in_block,
                    SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                )
            else:
                mask = load_dense_mask(
                    DENSE_MASK,
                    stride_mask_m,
                    stride_mask_n,
                    offs_m,
                    offs_n_load,
                    Q_LEN=Q_LEN,
                    KV_LEN=KV_LEN,
                )

            k = tl.load(
                K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)
                k = tl.load(
                    K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                alpha = tl.math.exp(m_i - m_ij)
                p = tl.math.exp(qk - m_ij[:, None])

                pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + pv
                m_i = m_ij
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_out_m + offs_v[None, :] * stride_out_k,
            acc,
            mask=out_mask
        )

        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse, mask=offs_m < Q_LEN)


@triton.jit
def load_dense_mask(
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    offs_m,
    offs_n,
    Q_LEN,
    KV_LEN,
):
    stride_mask_m = stride_mask_m.to(tl.int64)
    ptrs = DENSE_MASK + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def load_packed_partial_mask(
    PARTIAL_MASK_PACKED,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    partial_block_idx,
    offs_m_in_block,
    offs_n_in_block,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
):
    ptrs = (
        PARTIAL_MASK_PACKED
        + partial_block_idx * stride_partial_p
        + offs_m_in_block[:, None] * stride_partial_m
        + offs_n_in_block[None, :] * stride_partial_n
    )
    valid = (
        (offs_m_in_block[:, None] < SPARSE_Q_BLOCK_SIZE)
        & (offs_n_in_block[None, :] < SPARSE_KV_BLOCK_SIZE)
    )
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def bwd_dq_block_mn(
    q, do, lse, delta,
    K_ptr, V_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    Q_LEN, KV_LEN,
    offs_m, offs_n, offs_k, offs_v,
    q_sparse_idx, kv_block, kv_sub, q_sparse_base,
    stride_kn, stride_kk, stride_vn, stride_vk,
    MATMUL_PRECISION,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    k = tl.load(
        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
        mask=(offs_n[:, None] < KV_LEN),
        other=0.0,
    )
    v = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
        mask=(offs_n[:, None] < KV_LEN),
        other=0.0,
    )

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_block * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            offs_m_in_block = offs_m - q_sparse_base
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask & (offs_n[None, :] < KV_LEN), qk, float("-inf"))
    else:
        qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
    return dq



@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dqz", "stride_dqh",
    ]
)
def flex_attention_backward_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    Q_TASK_IDS,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_kv_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
    USE_Q_TASK_LIST: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS
    MATMUL_PRECISION = Q.dtype.element_ty

    for task_slot in range(pid, NUM_TASKS, num_core):
        task_id = task_slot
        if USE_Q_TASK_LIST:
            task_id = tl.load(Q_TASK_IDS + task_slot).to(tl.int32)

        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        do_offset = off_z * stride_doz + off_hq * stride_doh
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
        delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h
        dq_offset = off_z * stride_dqz + off_hq * stride_dqh

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DO_ptr = DO + do_offset
        LSE_ptr = LSE + lse_offset
        DELTA_ptr = DELTA + delta_offset
        DQ_ptr = DQ + dq_offset

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )

        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
        lse = tl.where(lse == float("-inf"), 0.0, lse)

        dq = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)

        q_sparse_idx = q_start // sparse_q_multiple
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        for blk_idx_in_list in range(0, kv_num_blocks):
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = kv_start_full + kv_sub * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_block * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0, propagate_nan=True)
                    offs_m_in_block = offs_m - q_sparse_base
                    offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds *= SM_SCALE
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            kv_indices_f = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks_f = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            for blk_idx_in_list in range(0, kv_num_blocks_f):
                kv_block = tl.load(kv_indices_f + blk_idx_in_list)
                kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

                for kv_sub in range(NUM_KV_SUB_BLOCKS):
                    start_n = kv_start_full + kv_sub * BLOCK_N
                    offs_n = start_n + tl.arange(0, BLOCK_N)

                    k = tl.load(
                        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds *= SM_SCALE
                    dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        tl.store(
            DQ_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        COMPUTE_DQ=False,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                            COMPUTE_DQ=False,
                        )


@triton.jit
def bwd_dkdv_block_mn(
    Q, DO, DQ, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
    stride_qm, stride_qk, stride_dom, stride_dok, stride_dqm, stride_dqd,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
    COMPUTE_DQ: tl.constexpr = True,
):
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    lse = tl.load(LSE + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_sparse_idx * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
            offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask, qk, float("-inf"))
    p = tl.math.exp(qk - lse[:, None])

    dv = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
    tl.atomic_add(
        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv,
        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
    )

    Di = tl.load(DELTA + offs_m, mask=offs_m < Q_LEN, other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = (p * (dp - Di[:, None]))
    ds *= SM_SCALE

    if COMPUTE_DQ:
        dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
        tl.atomic_add(
            DQ + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


@triton.jit
def _bwd_dkdv_qblock_range(
    Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
    q_indices, q_range_start, q_range_end,
    kv_sparse_idx, kv_sub,
    stride_qm, stride_qk, stride_dom, stride_dok,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    """遍历 [q_range_start, q_range_end) 范围内的 Q-block，累加 dk/dv 梯度。

    将 partial/full Q-block 的循环逻辑统一抽取，通过 IS_FULL_BLOCKS
    编译期常量控制 mask 加载行为，消除主 kernel 中的代码重复。

    Args:
        q_indices: 当前 kv_sparse_idx 对应的 Q-block 索引列表基址
        q_range_start / q_range_end: Q-block 线性索引的迭代范围 [start, end)
        其余参数与 bwd_dkdv_block_mn 保持一致
    """
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    for start_m in range(q_range_start, q_range_end):
        blk_idx_in_list = start_m // sparse_q_multiple
        q_block = tl.load(q_indices + blk_idx_in_list)
        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)

        bwd_dkdv_block_mn(
            Q_h, DO_h, DK_OUT_ptr, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
            DENSE_MASK, stride_mask_m, stride_mask_n,
            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
            k, v, Q_LEN, KV_LEN,
            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block,
            kv_sparse_idx, kv_sub, offs_k, offs_v,
            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
            MATMUL_PRECISION,
            SM_SCALE,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            QK_HEAD_DIM=QK_HEAD_DIM,
            V_HEAD_DIM=V_HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_FULL_BLOCKS=IS_FULL_BLOCKS,
            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
            COMPUTE_DQ=False,
        )


# ===========================================================================
# Task-list dkdv kernel: 基于 host 侧装箱结果的自适应负载均衡 kernel。
#
# 设计动机:
#   原始 SIMPLE kernel 以 (z, hkv, kv_block) 为粒度 round-robin 调度, 当 KV 分块
#   间 Q-block 数差异大时核间负载严重不均衡。本 kernel 由 host 侧装箱算法
#   (heavy-light 分解 + fill-ratio heap) 生成 task list, kernel 仅按 list 执行。
#
# 数据结构 (host 侧构建, 详见 _build_dkdv_task_schedule):
#   work_items[j]  = (hkv, kv_block)                    # 2 元组, int32
#   task_offsets[c] = 核 c 的 work-item 起始索引         # CSR 偏移, len = num_core+1
#
#   - 所有 work-item 均为 direct: 处理 base task 全部 Q-blocks, atomic_add 写 DK/DV
#   - 无 split/reduce: 跨核均衡由 host 侧 heavy-light 装箱保证
#
# 写入语义:
#   - 不同 (hkv, kv_block) 写不同地址, 无竞争
#   - 同一 kv_block 的不同 hkv 写不同 head 维, 无竞争
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel_tasklist(
    Q, K, V, DO, LSE, DELTA,
    Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, PARTIAL_MASK_OFFSETS, PARTIAL_BLOCK_TABLE,
    stride_partial_p, stride_partial_m, stride_partial_n,
    stride_partial_offset_m, stride_partial_table_m, stride_partial_table_n,
    DK, DV,                                  # direct 路径写入目标 (atomic_add)
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_q_idx_m,
    WORK_ITEMS,                               # [num_work, 2]: (hkv, kv_block)
    TASK_OFFSETS,                             # [num_core+1]: CSR 偏移
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    KV_HEAD,                                  # = Hkv (Z=1)
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    """Task-list dkdv kernel (direct-only, 无 split/reduce)。

    每个 pid 处理 TASK_OFFSETS[pid] 到 TASK_OFFSETS[pid+1] 范围内的 work-item,
    遍历 partial + full Q-block 累加 dk/dv 梯度。保留两段 partial/full 循环结构,
    通过 IS_FULL_BLOCKS 编译期常量控制 mask 加载行为。
    """
    pid = tl.program_id(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    # constexpr 除法, 提升至 kernel 级避免每个 task 重复计算
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

    # ================================================================
    # 每个 pid 恰好处理 1 个核的 work-item (核数 = len(TASK_OFFSETS)-1, 无 round-robin)
    # ================================================================
    work_start = tl.load(TASK_OFFSETS + pid)
    work_end = tl.load(TASK_OFFSETS + pid + 1)

    for widx in range(work_start, work_end):
        # ---- 从 work-item 中读取任务参数 (2 元组: hkv, kv_block) ----
        off_hkv = tl.load(WORK_ITEMS + widx * 2 + 0).to(tl.int64)
        kv_block = tl.load(WORK_ITEMS + widx * 2 + 1)

        off_z = tl.zeros_like(off_hkv)              # Z=1

        # ---- 指针基址 (按 hkv 切片) ----
        k_offset  = off_z * stride_kz  + off_hkv * stride_kh
        v_offset  = off_z * stride_vz  + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh
        K_ptr = K + k_offset
        V_ptr = V + v_offset

        # ---- 输出指针: direct 路径, 写 DK/DV ----
        DK_OUT_ptr = DK + dk_offset
        DV_OUT_ptr = DV + dv_offset

        start_n_full = kv_block * KV_BLOCK_SIZE
        kv_sparse_idx = kv_block // sparse_kv_multiple
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        # ---- 计算 partial Q-block 迭代范围 [0, block_m_end_p) ----
        q_indices = Q_IDX + sparse_q_idx_offset
        q_num_blocks = tl.load(Q_NUM_BLKS + kv_sparse_idx)
        block_m_end_p = tl.minimum(
            q_num_blocks * sparse_q_multiple,
            tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )
        q_start_p = 0
        q_end_p = block_m_end_p

        # ---- 计算 full Q-block 迭代范围 (仅当 HAS_FULL_BLOCKS) ----
        q_start_f = 0
        q_end_f = 0
        q_indices_f = q_indices  # dummy; 仅 HAS_FULL_BLOCKS 时使用
        if HAS_FULL_BLOCKS:
            q_indices_f = FULL_Q_IDX + sparse_q_idx_offset
            q_num_blocks_f = tl.load(FULL_Q_NUM_BLKS + kv_sparse_idx)
            block_m_end_f = tl.minimum(
                q_num_blocks_f * sparse_q_multiple,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            q_start_f = 0
            q_end_f = block_m_end_f

        # ========================================================
        # KV sub-block 循环: 加载 k/v tile, 遍历 GQA heads
        # 复用 _bwd_dkdv_qblock_range 处理 partial + full Q-block 切片
        # ========================================================
        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            start_n = start_n_full + kv_sub * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = (off_hkv * GQA_SHARED_HEADS + off_g).to(tl.int64)

                Q_h = Q + off_z * stride_qz + off_hq * stride_qh
                DO_h = DO + off_z * stride_doz + off_hq * stride_doh
                LSE_h = LSE + off_z * stride_lse_z + off_hq * stride_lse_h
                DELTA_h = DELTA + off_z * stride_delta_z + off_hq * stride_delta_h

                # ---- partial Q-blocks 切片 ----
                _bwd_dkdv_qblock_range(
                    Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                    DENSE_MASK, stride_mask_m, stride_mask_n,
                    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                    k, v, Q_LEN, KV_LEN,
                    off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
                    q_indices, q_start_p, q_end_p,
                    kv_sparse_idx, kv_sub,
                    stride_qm, stride_qk, stride_dom, stride_dok,
                    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                    MATMUL_PRECISION,
                    SM_SCALE=SM_SCALE,
                    SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    QK_HEAD_DIM=QK_HEAD_DIM,
                    V_HEAD_DIM=V_HEAD_DIM,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    IS_FULL_BLOCKS=False,
                    USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                )

                # ---- full Q-blocks 切片 ----
                if HAS_FULL_BLOCKS:
                    _bwd_dkdv_qblock_range(
                        Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
                        q_indices_f, q_start_f, q_end_f,
                        kv_sparse_idx, kv_sub,
                        stride_qm, stride_qk, stride_dom, stride_dok,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE=SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=True,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                    )


def _prepare_block_mask_attrs(block_mask, q, num_q_blocks, sparse_q_block_size, sparse_kv_block_size):
    N = q.shape[0] if q.dim() == 4 else q.shape[2]
    kv_num_blks = block_mask.kv_num_blocks
    kv_idx = block_mask.kv_indices
    full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
    full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

    q_num_blks = getattr(block_mask, "q_num_blocks", None)
    q_idx = getattr(block_mask, "q_indices", None)
    assert q_num_blks is not None, "q_num_blocks and q_indices must be provided"
    assert q_idx is not None, "q_indices must be provided"
    full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
    full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

    kv_num_blks = kv_num_blks.to(torch.int32).contiguous()
    kv_idx = kv_idx.to(torch.int32).contiguous()
    full_kv_num_blks = full_kv_num_blks.to(torch.int32).contiguous()
    full_kv_idx = full_kv_idx.to(torch.int32).contiguous()
    q_num_blks = q_num_blks.to(torch.int32).contiguous()
    q_idx = q_idx.to(torch.int32).contiguous()
    full_q_num_blks = full_q_num_blks.to(torch.int32).contiguous()
    full_q_idx = full_q_idx.to(torch.int32).contiguous()

    dense_mask = getattr(block_mask, "dense_mask", None)
    packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
    partial_mask_offsets = getattr(block_mask, "partial_mask_offsets", None)
    partial_block_table = getattr(block_mask, "partial_block_table", None)
    use_packed_partial_mask = (
        packed_partial_mask is not None
        and partial_mask_offsets is not None
        and partial_block_table is not None
    )

    if dense_mask is None:
        dense_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
    dense_mask = dense_mask.contiguous()

    if use_packed_partial_mask:
        packed_partial_mask = packed_partial_mask.contiguous()
        partial_mask_offsets = partial_mask_offsets.to(torch.int32).contiguous()
        partial_block_table = partial_block_table.to(torch.int32).contiguous()
    else:
        packed_partial_mask = torch.zeros(
            (1, sparse_q_block_size, sparse_kv_block_size),
            dtype=torch.bool,
            device=q.device,
        )
        partial_mask_offsets = torch.zeros(
            (1, 1, max(num_q_blocks, 1)),
            dtype=torch.int32,
            device=q.device,
        )
        partial_block_table = torch.full(
            (max(num_q_blocks, 1), max((N + sparse_kv_block_size - 1) // sparse_kv_block_size, 1)),
            -1,
            dtype=torch.int32,
            device=q.device,
        )

    return {
        "kv_num_blks": kv_num_blks,
        "kv_idx": kv_idx,
        "full_kv_num_blks": full_kv_num_blks,
        "full_kv_idx": full_kv_idx,
        "q_num_blks": q_num_blks,
        "q_idx": q_idx,
        "full_q_num_blks": full_q_num_blks,
        "full_q_idx": full_q_idx,
        "dense_mask": dense_mask,
        "packed_partial_mask": packed_partial_mask,
        "partial_mask_offsets": partial_mask_offsets,
        "partial_block_table": partial_block_table,
        "use_packed_partial_mask": use_packed_partial_mask,
    }


def flex_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlexAttention 前向传播的 host 侧编排入口。

    Args:
        q / k / v: 注意力输入张量。
        block_mask: 稀疏注意力掩码 (BlockMask)。
        sm_scale: softmax 缩放因子; None 时自动取 1/sqrt(D)。

    Returns:
        (output, lse): 注意力输出及 logsumexp。
    """
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape

    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = TILE_BLOCK_SIZE
    BLOCK_N = TILE_BLOCK_SIZE
    SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE = _get_mask_block_sizes(block_mask)

    num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE
    num_kv_blocks = triton.cdiv(N, SPARSE_KV_BLOCK_SIZE)

    output = torch.empty_like(q)
    lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    # 构建统一负载均衡计划 (Q 侧 + DKDV 侧, 带缓存复用)
    # forward 预构建 DKDV 计划, backward 直接复用, 避免重复装箱
    build_dkdv = q.requires_grad or k.requires_grad or v.requires_grad
    load_balance_plan = _get_or_build_load_balance_plan(
        block_mask, bm, Z, Hq, Hkv, num_q_blocks, num_kv_blocks,
        SPARSE_Q_BLOCK_SIZE // BLOCK_M, 1,
        q.device, build_dkdv,
    )

    num_tasks = num_q_blocks * Z * Hq
    grid, num_tasks = _persistent_launch_config(num_tasks)

    flex_attention_kernel[grid](
        q, k, v,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        load_balance_plan.q_task_ids,
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        output, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_TASKS=num_tasks,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        USE_Q_TASK_LIST=load_balance_plan.use_q_task_list,
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=3,
        inter_cache_num=2,
        enable_cross_if_fusion=True,
        enable_buffer_insert_optimization=True,
        enable_ub_refine_opt = True,
    )

    return output, lse


# ============================================================================
# Host 侧负载均衡: DKDV 任务列表构建 (direct-only, heavy-light 装箱)
#
# 输入: 每个 kv_sparse_idx 的有效 Q-block 数 (partial + full 等权)
# 输出: work_items [num_work, 2] / task_offsets [num_core+1]
#
# 算法 (两阶段):
#   阶段 A: 构建模板 — 枚举 hkv=0 的所有 kv_block, weight > 0 的作为 direct item
#           (重量仅依赖 kv_block, 与 hkv 无关, 按 hkv 复制模板)
#   阶段 B: heavy-light 分解 — 重 item round-robin 预分配, 轻 item fill-ratio heap
#           - hkv 外层循环保证 L2 cache 友好
#           - 所有 work-item 直接 atomic_add 写 DK/DV, 无 split/reduce
# ============================================================================

@dataclass
class _DkdvTemplate:
    """DKDV 装箱模板 (hkv=0 的任务列表, 按 hkv 复制)。

    重量仅依赖 kv_block, 与 hkv 无关, 因此只需构建一次模板再复制。

    Attributes:
        items: 模板 item 列表, 每个元素为 2 元组:
            (kv_block, weight)
            - kv_block: KV block 索引。
            - weight: 该 item 的权重 (有效 Q-block 数)。
    """
    items: list

    @property
    def num_items(self) -> int:
        """模板 item 总数。"""
        return len(self.items)


def _build_dkdv_template(
    w_sparse_list: list,
    num_kv_blocks: int,
    sparse_kv_multiple: int,
    target: float,
) -> "_DkdvTemplate":
    """构建 hkv=0 的 DKDV 装箱模板。

    遍历所有 kv_block, 将 weight > 0 的块作为 direct item 加入模板。
    拆分策略已移除: 所有 item 均为 direct (整块处理), 由装箱算法的
    heavy-light 分解 + fill-ratio heap 实现跨核均衡, 无需 split + reduce。

    Args:
        w_sparse_list: 每个 sparse kv block 的有效 Q-block 数。
        num_kv_blocks: KV block 总数。
        sparse_kv_multiple: kv_block → sparse_idx 的换算因子。
        target: 单核目标重量 (保留参数, 供未来扩展使用)。

    Returns:
        _DkdvTemplate 封装的模板数据, items 为 (kv_block, weight) 2 元组列表。
    """
    items = []

    for kv_block in range(num_kv_blocks):
        w = w_sparse_list[kv_block // sparse_kv_multiple]
        if w == 0:
            continue
        items.append((kv_block, float(w)))

    return _DkdvTemplate(items=items)


def _make_empty_dkdv_tensors(
    device, num_core: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建空的 DKDV 调度张量 (用于无任务场景)。

    Args:
        device: 张量分配设备。
        num_core: 核数, 用于 task_offsets 长度。

    Returns:
        (work_items, task_offsets)。
    """
    return (
        torch.empty((0, _WORK_ITEM_COLUMNS), dtype=torch.int32, device=device),
        torch.zeros((num_core + 1,), dtype=torch.int32, device=device),
    )


@dataclass(frozen=True)
class _DkdvPackingPlan:
    """DKDV 装箱计划 (heavy-light 分解结果)。

    将模板 item 按重量分为重 item 和轻 item 两组, 分别采用不同装箱策略:
    - 重 item: round-robin 预分配到空核, 避免堆叠到半满核。
    - 轻 item: 容量感知 fill-ratio heap 装箱, 重核少收、轻核多收。

    Attributes:
        heavy_order: 重 item 的模板索引列表 (重量降序)。
        light_order: 轻 item 的模板索引列表 (重量降序)。
        target: 单核目标重量 (total_w / num_core), 用于计算 heavy 阈值和 light 容量。
    """

    heavy_order: list
    light_order: list
    target: float


def _classify_dkdv_items(
    template: "_DkdvTemplate",
    target: float,
) -> _DkdvPackingPlan:
    """将模板 item 按重量分为重 item 和轻 item。

    重 item 定义为 weight > target * _DKDV_HEAVY_RATIO_THRESHOLD。这些 item 在
    min-heap 装箱中容易导致后期堆叠到半满核上, 因此需要预分配。

    Args:
        template: DKDV 装箱模板, items[tid][1] 为重量。
        target: 单核目标重量。

    Returns:
        _DkdvPackingPlan 封装的 heavy/light 分组与 target。
    """
    heavy_threshold = target * _DKDV_HEAVY_RATIO_THRESHOLD
    heavy_order = []
    light_order = []

    for tid in range(template.num_items):
        weight = template.items[tid][1]
        if weight > heavy_threshold:
            heavy_order.append(tid)
        else:
            light_order.append(tid)

    # 两组均按重量降序, 保证重中重优先、轻中重优先
    get_weight = template.items.__getitem__
    heavy_order.sort(key=lambda tid: get_weight(tid)[1], reverse=True)
    light_order.sort(key=lambda tid: get_weight(tid)[1], reverse=True)

    return _DkdvPackingPlan(
        heavy_order=heavy_order,
        light_order=light_order,
        target=target,
    )


def _build_dkdv_task_schedule(
    w_sparse,
    Hkv: int,
    num_kv_blocks: int,
    sparse_kv_multiple: int,
    target: float,
    num_core: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建 DKDV task list (direct-only 统一装箱)。

    装箱策略固定为 hkv 分块连续装箱。采用模板复制优化: 先构建 hkv=0 的模板
    (重量仅依赖 kv_block, 与 hkv 无关), 再按 hkv 直接复制模板。

    所有 item 均为 direct (整块处理), 由 heavy-light 分解 + fill-ratio heap
    实现跨核均衡, 无需 split + reduce 两步流程。

    Args:
        w_sparse: 每个 kv_sparse_idx 的有效 Q-block 数 (partial + full 等权)。
        Hkv: KV head 数。
        num_kv_blocks: KV block 总数。
        sparse_kv_multiple: kv_block → sparse_kv_idx 的换算因子。
        target: 单核目标重量 (total_w / num_core)。
        num_core: 核数 (= bin 数)。
        device: 张量分配设备。

    Returns:
        (work_items, task_offsets):
        work_items: [num_work, 2] int32, (hkv, kv_block)。
        task_offsets: [num_core+1] int32, CSR 偏移。
    """
    # Step 1: 构建 hkv=0 模板
    w_sparse_list = (
        w_sparse.reshape(-1).tolist()
        if isinstance(w_sparse, torch.Tensor)
        else list(w_sparse)
    )
    template = _build_dkdv_template(w_sparse_list, num_kv_blocks, sparse_kv_multiple, target)

    if template.num_items == 0:
        return _make_empty_dkdv_tensors(device, num_core)

    # Step 2: heavy-light 分解 + 容量感知装箱
    # 重 item 预分配到空核, 轻 item 用 fill-ratio heap 按剩余容量比例装箱,
    # 避免 min-heap 后期将重 item 堆叠到半满核上
    plan = _classify_dkdv_items(template, target)

    if _HAS_NUMPY:
        return _pack_dkdv_tasks_numpy(
            template, plan, Hkv, num_core, device,
        )

    return _pack_dkdv_tasks_python(
        template, plan, Hkv, num_core, device,
    )


def _pack_dkdv_tasks_python(
    template: "_DkdvTemplate",
    plan: "_DkdvPackingPlan",
    Hkv: int,
    num_core: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """纯 Python 路径的 DKDV 装箱 (fallback, 无 numpy 时使用)。

    采用 heavy-light 分解 + 容量感知 fill-ratio heap 策略:
    1. 重 item round-robin 预分配到各核, 每核均匀承担重负载。
    2. 轻 item 用 fill-ratio = light_weight / light_capacity 作为 heap key,
       重核 (light_capacity 小) 的 fill_ratio 增长快, 自然少收轻 item。
    3. hkv 外层循环保证每核 item 列表 hkv 有序, 利于 L2 cache 共享。

    Args:
        template: DKDV 装箱模板。
        plan: heavy-light 分解的装箱计划。
        Hkv: KV head 数。
        num_core: 核数。
        device: 张量分配设备。

    Returns:
        (work_items, task_offsets)。
    """
    # 解包模板属性到局部变量, 避免热循环中重复属性查找
    items = template.items
    heavy_order = plan.heavy_order
    light_order = plan.light_order
    target = plan.target

    # Step 1: 重 item round-robin 预分配 — 每个 (hkv, tid) 固定到一个核
    heavy_assignment = {}
    core_heavy_weight = [0.0] * num_core
    heavy_core = 0
    for tid in heavy_order:
        weight = items[tid][1]
        for hkv in range(Hkv):
            heavy_assignment[(hkv, tid)] = heavy_core
            core_heavy_weight[heavy_core] += weight
            heavy_core = (heavy_core + 1) % num_core

    # Step 2: 计算每核轻 item 容量 = target - 重 item 已占重量
    light_capacity = [max(1.0, target - core_heavy_weight[c]) for c in range(num_core)]

    # Step 3: hkv-major 装箱 — 重 item 按预分配插入, 轻 item 用 fill-ratio heap
    bins = [[] for _ in range(num_core)]
    bin_counts = [0] * num_core
    light_weight = [0.0] * num_core
    bin_heap = [(0, core_id) for core_id in range(num_core)]
    heapq.heapify(bin_heap)

    for hkv in range(Hkv):
        # Step 3a: 插入本轮重 item (保持 hkv 顺序)
        for tid in heavy_order:
            core_id = heavy_assignment[(hkv, tid)]
            kv_block, _weight = items[tid]
            bins[core_id].append((hkv, kv_block))
            bin_counts[core_id] += 1

        # Step 3b: 装箱轻 item (fill-ratio heap 选择 fill 比例最低的核)
        for tid in light_order:
            _, core_id = heapq.heappop(bin_heap)
            kv_block, weight = items[tid]
            bins[core_id].append((hkv, kv_block))
            bin_counts[core_id] += 1
            light_weight[core_id] += weight
            new_key = int(light_weight[core_id] * _FILL_RATIO_SCALE / light_capacity[core_id])
            heapq.heappush(bin_heap, (new_key, core_id))

    # Step 4: 汇总 work_items 和 task_offsets
    work_items_final = [item for core_items in bins for item in core_items]
    task_offsets = [0]
    for count in bin_counts:
        task_offsets.append(task_offsets[-1] + count)

    work_items_t = torch.tensor(work_items_final, dtype=torch.int32, device=device)
    task_offsets_t = torch.tensor(task_offsets, dtype=torch.int32, device=device)

    return work_items_t, task_offsets_t


def _pack_dkdv_tasks_numpy(
    template: "_DkdvTemplate",
    plan: "_DkdvPackingPlan",
    Hkv: int,
    num_core: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """numpy 加速路径的 DKDV 装箱 (有 numpy 时使用)。

    采用 heavy-light 分解 + 容量感知 fill-ratio heap 策略 (与 python 路径逻辑一致):
    1. 重 item round-robin 预分配到各核。
    2. 轻 item 用 fill-ratio heap 按剩余容量比例装箱。
    用 flat Python lists 替代 list-of-tuples 避免 tuple 创建开销,
    最终用 np.array + torch.from_numpy 加速 H2D 传输。

    Args:
        template: DKDV 装箱模板。
        plan: heavy-light 分解的装箱计划。
        Hkv: KV head 数。
        num_core: 核数。
        device: 张量分配设备。

    Returns:
        (work_items, task_offsets)。
    """
    total_items = Hkv * template.num_items

    # 解包模板属性到局部变量, 避免热循环中重复属性查找
    items = template.items
    heavy_order = plan.heavy_order
    light_order = plan.light_order
    target = plan.target

    # Step 1: 重 item round-robin 预分配
    heavy_assignment = {}
    core_heavy_weight = [0.0] * num_core
    heavy_core = 0
    for tid in heavy_order:
        weight = items[tid][1]
        for hkv in range(Hkv):
            heavy_assignment[(hkv, tid)] = heavy_core
            core_heavy_weight[heavy_core] += weight
            heavy_core = (heavy_core + 1) % num_core

    # Step 2: 计算每核轻 item 容量
    light_capacity = [max(1.0, target - core_heavy_weight[c]) for c in range(num_core)]

    # Step 3: flat lists 装箱 — 避免 tuple 创建开销
    flat_core = [0] * total_items
    flat_hkv = [0] * total_items
    flat_kv = [0] * total_items

    light_weight = [0.0] * num_core
    bin_heap = [(0, core_id) for core_id in range(num_core)]
    heapq.heapify(bin_heap)
    idx = 0

    for hkv in range(Hkv):
        # Step 3a: 插入本轮重 item (保持 hkv 顺序)
        for tid in heavy_order:
            core_id = heavy_assignment[(hkv, tid)]
            kv_block, _weight = items[tid]
            flat_core[idx] = core_id
            flat_hkv[idx] = hkv
            flat_kv[idx] = kv_block
            idx += 1

        # Step 3b: 装箱轻 item (fill-ratio heap)
        for tid in light_order:
            _, core_id = heapq.heappop(bin_heap)
            kv_block, weight = items[tid]
            flat_core[idx] = core_id
            flat_hkv[idx] = hkv
            flat_kv[idx] = kv_block
            light_weight[core_id] += weight
            new_key = int(light_weight[core_id] * _FILL_RATIO_SCALE / light_capacity[core_id])
            heapq.heappush(bin_heap, (new_key, core_id))
            idx += 1

    # Step 4: 按 core 稳定排序得到 CSR 序, 再 column_stack 构建 [N, 2] 数组
    core_arr = _np.array(flat_core, dtype=_np.int32)
    sort_idx = _np.argsort(core_arr, kind='stable')

    work_items_np = _np.column_stack([
        _np.array(flat_hkv, dtype=_np.int32)[sort_idx],
        _np.array(flat_kv, dtype=_np.int32)[sort_idx],
    ])

    # Step 5: CSR offsets
    bin_counts = _np.bincount(core_arr, minlength=num_core)
    task_offsets_np = _np.zeros(num_core + 1, dtype=_np.int32)
    _np.cumsum(bin_counts, out=task_offsets_np[1:])

    # Step 6: numpy → torch → device
    work_items_t = torch.from_numpy(work_items_np).to(device)
    task_offsets_t = torch.from_numpy(task_offsets_np).to(device)

    return work_items_t, task_offsets_t


def flex_attention_bwd_impl(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flex Attention 反向传播的 host 侧编排入口。

    分两阶段计算梯度:
      1. DQ kernel   — 以 Q-block 为中心遍历 KV-block, 计算 dq。
      2. DKDV kernel — 以 KV-block 为中心遍历 Q-block, 计算 dk/dv。
        根据负载均衡判定, DKDV 选择两条路径之一:
        - SIMPLE kernel   : 负载均衡时走零开销的 round-robin 调度。
        - task-list kernel: 负载不均衡时走装箱 (heavy-light 分解 + fill-ratio heap) 调度。

    Args:
        grad_output: 前向输出的梯度, shape [Z, Hq, M, Dv]。
        q / k / v:   注意力输入张量。
        output:      前向注意力输出, 用于计算 delta = sum(output * grad_output)。
        lse:         前向 logsumexp, shape [Z, Hq, M]。
        block_mask:  稀疏注意力掩码 (BlockMask)。
        sm_scale:    softmax 缩放因子; None 时自动取 1/sqrt(D)。

    Returns:
        (dq, dk, dv): 梯度张量, dtype 与 q / k / v 对齐。
    """
    # ========================================================================
    # Phase 0: 预处理 — 形状解析、delta 计算、BlockMask 属性提取
    # ========================================================================
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    grad_output = grad_output.contiguous()
    delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

    SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE = _get_mask_block_sizes(block_mask)
    num_q_blocks = triton.cdiv(M, SPARSE_Q_BLOCK_SIZE)
    num_kv_blocks = triton.cdiv(N, SPARSE_KV_BLOCK_SIZE)

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

    # ========================================================================
    # 构建统一负载均衡计划 (Q 侧 + DKDV 侧, 带缓存复用)
    # forward 已预构建时直接复用, 否则首次构建并缓存
    # ========================================================================
    load_balance_plan = _get_or_build_load_balance_plan(
        block_mask, bm, Z, Hq, Hkv, num_q_blocks, num_kv_blocks,
        SPARSE_Q_BLOCK_SIZE // TILE_BLOCK_SIZE, 1,
        q.device, True,
    )

    # ========================================================================
    # Phase 1: DQ kernel — 以 Q-block 为中心, 直接写入 dq
    # ========================================================================
    BLOCK_M_DQ = TILE_BLOCK_SIZE
    BLOCK_N_DQ = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DQ
    grid_dq, num_tasks_dq = _persistent_launch_config(num_q_blocks * Z * Hq)
    flex_attention_backward_dq_kernel[grid_dq](
        q, k, v, grad_output, lse, delta,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        load_balance_plan.q_task_ids,
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
        dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M_DQ,
        BLOCK_N=BLOCK_N_DQ,
        NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
        NUM_TASKS=num_tasks_dq,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        USE_Q_TASK_LIST=load_balance_plan.use_q_task_list,
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        enable_select_analysis=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
        intra_cache_num=3,
        inter_cache_num=2,
    )

    # ========================================================================
    # Phase 2: DKDV kernel — 以 KV-block 为中心, atomic_add 写入 dk/dv
    #
    # 负载均衡分支判定已由 _build_dkdv_side_plan 完成 (封装在 load_balance_plan 中):
    #   - 均衡时走 SIMPLE kernel (零开销 round-robin)
    #   - 不均衡时走 task-list kernel (heavy-light 装箱, direct-only)
    # ========================================================================
    BLOCK_M_DKDV = TILE_BLOCK_SIZE
    BLOCK_N_DKDV = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DKDV
    total_base = num_kv_blocks * Z * Hkv
    num_core = _get_num_aicore()

    if not load_balance_plan.use_dkdv_task_list:
        # ================================================================
        # 均衡路径: 原始 SIMPLE kernel (零额外开销, 性能最优)
        # 每个 (z, hkv, kv_block) 一个 task, round-robin 调度
        # ================================================================
        grid_dkv, num_tasks_dkv = _persistent_launch_config(total_base)
        flex_attention_backward_dkdv_kernel[grid_dkv](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            bm["q_idx"].stride(2),
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            NUM_TASKS=num_tasks_dkv,
            NUM_KV_BLOCKS=num_kv_blocks,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )

    else:
        # ================================================================
        # 不均衡路径: task-list kernel (heavy-light 装箱, direct-only)
        # ================================================================
        grid_tasklist = (num_core,)
        flex_attention_backward_dkdv_kernel_tasklist[grid_tasklist](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dk, dv,
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            bm["q_idx"].stride(2),
            load_balance_plan.dkdv_work_items,
            load_balance_plan.dkdv_task_offsets,
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


# ============================================================================
# BlockMask construction: streaming stripe packed-block builders
# ============================================================================
# Two construction strategies for building a packed BlockMask:
#
#   1. Kernel-based (_build_packed_block_mask_streaming):
#      Uses custom Triton create_mask_kernel + block_classify_kernel for fused
#      mask generation and classification.  Caller supplies a mask_type_str
#      (e.g. "sparse", "full") and a *problem* dict with pre-built index tables.
#
#   2. mask_mod-based (create_block_mask_patched):
#      Evaluates an arbitrary mask_mod callable ``(b, h, q_idx, kv_idx) -> bool``
#      in streaming stripes, then classifies and packs partial blocks.
#      API mirrors torch.nn.attention.flex_attention.create_block_mask.
# ============================================================================

import torch.nn.functional as _F

from torch.nn.attention.flex_attention import _convert_mask_to_block_mask
from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask
from torch.nn.attention.flex_attention import create_mask as _torch_create_mask

from mojo_opset.utils.platform import get_torch_device as _get_torch_device


# -- Constants ---------------------------------------------------------------
_MB = 1024 ** 2

# Target memory per stripe for streaming mask build (~256 MB of bool dense mask).
_STRIPE_TARGET_BYTES = 256 * _MB

# During mask_mod evaluation, each intermediate tensor is [stripe_q, KV_LEN] in
# int64 (8 bytes).  A typical mask_mod creates ~4 simultaneous intermediates, so
# we budget ~8 bytes per (q, kv) element when sizing each stripe.
_BYTES_PER_MASK_ELEMENT = 8

# Tile size used by the kernel-based mask generation path.
MASK_BLOCK_SIZE = TILE_BLOCK_SIZE if not is_910() else 64


# -- Utility helpers ---------------------------------------------------------
def _round_up_to_multiple(x, multiple):
    """Round *x* up to the nearest multiple of *multiple*."""
    return (x + multiple - 1) // multiple * multiple


def _get_num_vector_core():
    """Return the number of vector cores on the current NPU device (fallback 1)."""
    try:
        dev = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(dev)
        return max(int(props.get("num_vectorcore", 1)), 1)
    except Exception:
        return 1


# ============================================================================
# Kernel: create_mask_kernel
# ============================================================================
@triton.jit
def create_mask_kernel(
    OUT, stride_ob, stride_oh, stride_oq, stride_ok: tl.constexpr,
    BLOCK_FLAGS, stride_bf_q, stride_bf_k,
    TABLE1, stride_t1, TABLE2, stride_t2, TABLE3, stride_t3,
    Q_LEN, KV_LEN, W, G,
    Q_OFFSET,
    MASK_TYPE: tl.constexpr, TILE: tl.constexpr,
    STORE_MASK: tl.constexpr, CLASSIFY: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int32)
    pid_k = tl.program_id(1).to(tl.int32)
    q_off = Q_OFFSET + pid_q * TILE + tl.arange(0, TILE)
    k_off = pid_k * TILE + tl.arange(0, TILE)
    q_idx = q_off[:, None]
    k_idx = k_off[None, :]

    if MASK_TYPE == 0:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=0)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-1)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        window = causal & ((q_idx - k_idx) <= W)
        ds_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        glob = causal & (k_idx >= ds_q) & (k_idx < ds_q + G)
        sparse = same_doc & (window | glob)
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = sparse | same_img
    elif MASK_TYPE == 1:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        frame_causal = fid_q >= fid_k
        result = same_doc & frame_causal
    elif MASK_TYPE == 2:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_video = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        same_frame = fid_q == fid_k
        prev_frame = fid_q > fid_k
        result = same_video & (same_frame | prev_frame)
    elif MASK_TYPE == 3:
        causal = q_idx >= k_idx
        mod_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        is_video = mod_q > 0
        same_video = is_video & (mod_q == mod_k)
        result = causal | same_video
    elif MASK_TYPE == 4:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        samedoc_causal = same_doc & causal
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = samedoc_causal | same_img
    else:
        result = tl.full([TILE, TILE], False, tl.int1)

    valid = (q_idx < Q_LEN) & (k_idx < KV_LEN)

    if STORE_MASK:
        q_store = (pid_q * TILE + tl.arange(0, TILE))[:, None]
        ptrs = OUT + q_store * stride_oq + k_idx * stride_ok
        tl.store(ptrs, result, mask=valid)

    if CLASSIFY:
        result_i = tl.where(valid, result.to(tl.int32), 0)
        has_one = tl.max(tl.max(result_i, axis=1), axis=0) != 0
        all_one = tl.min(tl.min(result_i, axis=1), axis=0) != 0
        flag = tl.where(all_one, 2, tl.where(has_one, 1, 0))
        tl.store(BLOCK_FLAGS + pid_q * stride_bf_q + pid_k * stride_bf_k, flag.to(tl.int8))


_MASK_TYPE_MAP = {
    "sparse": 0, "stair": 1, "video_stair": 2,
    "cross_sample_causal_video_bidir": 3, "full": 4,
}


def _get_mask_kernel_tables(problem, mt):
    """Extract table tensors, strides, and W/G values for create_mask_kernel based on mask type."""
    device = problem["q"].device
    t1 = t2 = t3 = torch.empty(0, device=device)
    s1 = s2 = s3 = 0
    W_val = G_val = 0

    if mt == 0:
        t1, t2, t3 = problem["segment_ids"], problem["doc_start"], problem["modality"]
        s1, s2, s3 = t1.stride(0), t2.stride(0), t3.stride(0)
        W_val = problem["sliding_window"]
        G_val = problem["global_window"]
    elif mt in (1, 2):
        t1, t2 = problem["video_ids"], problem["frame_ids"]
        s1, s2 = t1.stride(0), t2.stride(0)
    elif mt == 3:
        t1 = problem["modality"]
        s1 = t1.stride(0)
    elif mt == 4:
        t1, t3 = problem["segment_ids"], problem["modality"]
        s1, s3 = t1.stride(0), t3.stride(0)

    return t1, s1, t2, s2, t3, s3, W_val, G_val


def triton_create_mask(problem, mask_type, tile_size=MASK_BLOCK_SIZE):
    """Generate a dense ``[1, 1, SEQ_LEN, SEQ_LEN]`` bool mask via create_mask_kernel."""
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    out = torch.empty(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device)
    mt = _MASK_TYPE_MAP[mask_type]
    t1, s1, t2, s2, t3, s3, W_val, G_val = _get_mask_kernel_tables(problem, mt)

    n_tiles = (SEQ_LEN + tile_size - 1) // tile_size
    dummy_flags = torch.empty(0, dtype=torch.int8, device=device)
    create_mask_kernel[(n_tiles, n_tiles)](
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dummy_flags, 0, 0,
        t1, s1, t2, s2, t3, s3,
        SEQ_LEN, SEQ_LEN, W_val, G_val,
        Q_OFFSET=0,
        MASK_TYPE=mt, TILE=tile_size,
        STORE_MASK=True, CLASSIFY=False,
    )
    return out


def _run_create_mask_stripe(problem, mask_type, q_start, q_height, kv_len_padded,
                             out_buffer, flags_buffer=None, tile_size=MASK_BLOCK_SIZE):
    """Run create_mask_kernel for a stripe.

    If flags_buffer is provided, also classify blocks (CLASSIFY=True).
    Otherwise only generate the dense mask (CLASSIFY=False).
    """
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    mt = _MASK_TYPE_MAP[mask_type]
    t1, s1, t2, s2, t3, s3, W_val, G_val = _get_mask_kernel_tables(problem, mt)

    n_tiles_q = q_height // tile_size
    n_tiles_k = kv_len_padded // tile_size

    if flags_buffer is not None:
        create_mask_kernel[(n_tiles_q, n_tiles_k)](
            out_buffer, out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3),
            flags_buffer, flags_buffer.stride(0), flags_buffer.stride(1),
            t1, s1, t2, s2, t3, s3,
            SEQ_LEN, SEQ_LEN, W_val, G_val,
            Q_OFFSET=q_start,
            MASK_TYPE=mt, TILE=tile_size,
            STORE_MASK=True, CLASSIFY=True,
        )
    else:
        dummy_flags = torch.empty(0, dtype=torch.int8, device=device)
        create_mask_kernel[(n_tiles_q, n_tiles_k)](
            out_buffer, out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3),
            dummy_flags, 0, 0,
            t1, s1, t2, s2, t3, s3,
            SEQ_LEN, SEQ_LEN, W_val, G_val,
            Q_OFFSET=q_start,
            MASK_TYPE=mt, TILE=tile_size,
            STORE_MASK=True, CLASSIFY=False,
        )


# ============================================================================
# Kernel: block_classify_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=["stride_mq", "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "NUM_TASKS"]
)
def block_classify_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk: tl.constexpr,
    BLOCK_FLAGS, stride_fb, stride_fh, stride_fqb, stride_fkb,
    Q_LEN, KV_LEN, NUM_TASKS,
    H: tl.constexpr, Q_NUM_BLOCKS, KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    TILE_M: tl.constexpr = 64
    TILE_N: tl.constexpr = 64

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        has_one = tl.full((), 0, dtype=tl.int32)
        all_one = tl.full((), 1, dtype=tl.int32)
        mask_base = DENSE_MASK + off_b * stride_mb + off_h * stride_mh

        for m0 in range(0, Q_BLOCK_SIZE, TILE_M):
            offs_m = off_qb * Q_BLOCK_SIZE + m0 + tl.arange(0, TILE_M)
            valid_m = offs_m < Q_LEN
            for n0 in range(0, KV_BLOCK_SIZE, TILE_N):
                offs_n = off_kb * KV_BLOCK_SIZE + n0 + tl.arange(0, TILE_N)
                valid_n = offs_n < KV_LEN
                valid = valid_m[:, None] & valid_n[None, :]
                ptrs = mask_base + offs_m[:, None] * stride_mq + offs_n[None, :] * stride_mk
                vals = tl.load(ptrs, mask=valid, other=0).to(tl.int32)
                tile_any = tl.max(tl.max(tl.where(valid, vals, 0), axis=1), axis=0)
                tile_all = tl.min(tl.min(tl.where(valid, vals, 0), axis=1), axis=0)
                has_one = tl.where(tile_any != 0, 1, has_one)
                all_one = tl.where(tile_all == 0, 0, all_one)

        partial = (has_one == 1) & (all_one == 0)
        full = all_one == 1
        flag = tl.where(full, 2, tl.where(partial, 1, 0))
        out_ptr = BLOCK_FLAGS + off_b * stride_fb + off_h * stride_fh + off_qb * stride_fqb + off_kb * stride_fkb
        tl.store(out_ptr, flag.to(tl.int8))


def _classify_stripe_from_hbm(stripe_buffer, q_height, kv_len, flags_buf, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Decoupled classify: read dense mask from HBM and classify block flags.

    Uses block_classify_kernel (separate from mask generation).
    """
    Q_NUM_BLOCKS = _round_up_to_multiple(q_height, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(kv_len, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    num_tasks = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    block_classify_kernel[grid](
        stripe_buffer, stripe_buffer.stride(0), stripe_buffer.stride(1), stripe_buffer.stride(2), stripe_buffer.stride(3),
        flags_buf, 0, 0, flags_buf.stride(0), flags_buf.stride(1),
        q_height, kv_len, NUM_TASKS=num_tasks, H=1,
        Q_NUM_BLOCKS=Q_NUM_BLOCKS, KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )


# ============================================================================
# Kernel: pack_partial_blocks_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mq", "stride_mk", "stride_offset_q",
        "stride_local_q", "stride_local_k",
        "stride_flag_q", "stride_flag_k",
        "stride_table_q", "stride_table_k",
        "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "TOTAL_PARTIAL",
    ]
)
def pack_partial_blocks_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk: tl.constexpr,
    BLOCK_FLAGS, stride_flag_q, stride_flag_k: tl.constexpr,
    PARTIAL_OFFSETS, stride_offset_q,
    LOCAL_IDX, stride_local_q, stride_local_k,
    PACKED_MASK, stride_packed_p, stride_packed_m, stride_packed_n: tl.constexpr,
    BLOCK_TABLE, stride_table_q, stride_table_k: tl.constexpr,
    Q_LEN, KV_LEN, Q_NUM_BLOCKS, KV_NUM_BLOCKS, TOTAL_PARTIAL,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    if pid_q >= Q_NUM_BLOCKS:
        return

    row_offset = tl.load(PARTIAL_OFFSETS + pid_q * stride_offset_q).to(tl.int32)
    offs_m_local = tl.arange(0, Q_BLOCK_SIZE)[:, None].to(tl.int64)
    offs_n_local = tl.arange(0, KV_BLOCK_SIZE)[None, :].to(tl.int64)

    for kv_idx in range(KV_NUM_BLOCKS):
        flag = tl.load(BLOCK_FLAGS + pid_q * stride_flag_q + kv_idx * stride_flag_k).to(tl.int32)
        is_partial = flag == 1
        if is_partial:
            local_idx = tl.load(LOCAL_IDX + pid_q * stride_local_q + kv_idx * stride_local_k).to(tl.int32)
            packed_idx = (row_offset + local_idx - 1).to(tl.int64)
            offs_m = (pid_q * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE))[:, None].to(tl.int64)
            offs_n = (kv_idx * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE))[None, :].to(tl.int64)
            valid_src = (offs_m < Q_LEN) & (offs_n < KV_LEN)
            src_ptrs = DENSE_MASK + offs_m * stride_mq + offs_n * stride_mk
            block = tl.load(src_ptrs, mask=valid_src, other=0)
            dst_ptrs = PACKED_MASK + packed_idx * stride_packed_p + offs_m_local * stride_packed_m + offs_n_local * stride_packed_n
            tl.store(dst_ptrs, block)
            tl.store(BLOCK_TABLE + pid_q * stride_table_q + kv_idx * stride_table_k, packed_idx.to(tl.int32))


# ============================================================================
# Strategy 1: Kernel-based streaming packed BlockMask builder
# ============================================================================
def _build_packed_block_mask_streaming(mask_type_str, problem, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
                                                stripe_q_blocks=None, classify_strategy="fused"):
    """Build a packed BlockMask using custom Triton kernels with streaming stripes.

    Args:
        mask_type_str: Mask type string key into ``_MASK_TYPE_MAP``
            (e.g. "sparse", "full", "stair", "video_stair",
            "cross_sample_causal_video_bidir").
        problem: Problem dict containing q/k/v tensors and index tables
            (segment_ids, modality, doc_start, video_ids, frame_ids, etc.).
        SEQ_LEN: Total sequence length (Q and KV are assumed equal).
        Q_BLOCK_SIZE: Sparse block size along the query dimension.
        KV_BLOCK_SIZE: Sparse block size along the key/value dimension.
        stripe_q_blocks: Number of Q blocks per streaming stripe.  If None,
            auto-computed to target ~256 MB per stripe.
        classify_strategy: "fused" (classify inside create_mask_kernel) or
            "decoupled" (separate block_classify_kernel pass).

    Returns:
        BlockMask with ``packed_partial_mask``, ``partial_mask_offsets``,
        and ``partial_block_table`` attributes set.
    """
    device = problem["q"].device
    Q_NUM_BLOCKS = _round_up_to_multiple(SEQ_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(SEQ_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    KV_LEN_PADDED = KV_NUM_BLOCKS * KV_BLOCK_SIZE

    if stripe_q_blocks is None:
        max_rows = max(1, _STRIPE_TARGET_BYTES // KV_LEN_PADDED)
        stripe_q_blocks = max(1, max_rows // Q_BLOCK_SIZE)
    stripe_q_blocks = min(stripe_q_blocks, Q_NUM_BLOCKS)

    max_stripe_height = stripe_q_blocks * Q_BLOCK_SIZE
    stripe_buffer = torch.zeros(1, 1, max_stripe_height, KV_LEN_PADDED, dtype=torch.bool, device=device)

    block_flags = torch.zeros((1, 1, Q_NUM_BLOCKS, KV_NUM_BLOCKS), device=device, dtype=torch.int8)
    flags_stripe_buf = torch.empty((stripe_q_blocks, KV_NUM_BLOCKS), device=device, dtype=torch.int8)
    partial_block_table = torch.full((Q_NUM_BLOCKS, KV_NUM_BLOCKS), -1, dtype=torch.int32, device=device)
    global_B = torch.zeros(Q_NUM_BLOCKS, dtype=torch.int32, device=device)

    stripe_caches = []
    stripe_meta = []
    running_total = 0

    for qs_block in range(0, Q_NUM_BLOCKS, stripe_q_blocks):
        qe_block = min(qs_block + stripe_q_blocks, Q_NUM_BLOCKS)
        q_start = qs_block * Q_BLOCK_SIZE
        q_height = (qe_block - qs_block) * Q_BLOCK_SIZE
        stripe_q_nb = qe_block - qs_block

        if qe_block >= Q_NUM_BLOCKS:
            stripe_buffer.zero_()

        if classify_strategy == "fused":
            _run_create_mask_stripe(
                problem, mask_type_str, q_start, q_height, KV_LEN_PADDED,
                stripe_buffer, flags_stripe_buf, tile_size=MASK_BLOCK_SIZE
            )
        elif classify_strategy == "decoupled":
            _run_create_mask_stripe(
                problem, mask_type_str, q_start, q_height, KV_LEN_PADDED, stripe_buffer, tile_size=MASK_BLOCK_SIZE
            )
            _classify_stripe_from_hbm(
                stripe_buffer, q_height, SEQ_LEN, flags_stripe_buf, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            )
        else:
            raise ValueError(f"Unknown classify_strategy: {classify_strategy}")

        block_flags[:, :, qs_block:qe_block, :] = flags_stripe_buf[:stripe_q_nb, :].unsqueeze(0).unsqueeze(0)

        flags_stripe = flags_stripe_buf[:stripe_q_nb, :].contiguous()
        A_stripe = (flags_stripe == 1).to(torch.int32).cumsum(dim=-1)
        B_stripe = A_stripe.max(dim=-1).values
        stripe_partial_count = int(B_stripe.sum().item())
        global_B[qs_block:qe_block] = B_stripe.to(torch.int32)

        if stripe_partial_count > 0:
            stripe_cache = torch.zeros(
                (stripe_partial_count, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
            )
            row_offset_local = (B_stripe.cumsum(dim=-1) - B_stripe).to(torch.int32).contiguous()
            local_idx_stripe = A_stripe.contiguous()
            table_stripe = partial_block_table[qs_block:qe_block, :]

            pack_partial_blocks_kernel[(stripe_q_nb,)](
                stripe_buffer, stripe_buffer.stride(0), stripe_buffer.stride(1), stripe_buffer.stride(2), stripe_buffer.stride(3),
                flags_stripe, flags_stripe.stride(0), flags_stripe.stride(1),
                row_offset_local, row_offset_local.stride(0),
                local_idx_stripe, local_idx_stripe.stride(0), local_idx_stripe.stride(1),
                stripe_cache, stripe_cache.stride(0), stripe_cache.stride(1), stripe_cache.stride(2),
                table_stripe, table_stripe.stride(0), table_stripe.stride(1),
                q_height, SEQ_LEN, Q_NUM_BLOCKS=stripe_q_nb, KV_NUM_BLOCKS=KV_NUM_BLOCKS, TOTAL_PARTIAL=stripe_partial_count,
                Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            )
            stripe_caches.append(stripe_cache)
        else:
            stripe_caches.append(
                torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device)
            )

        stripe_meta.append((qs_block, qe_block, running_total))
        running_total += stripe_partial_count

    del stripe_buffer

    total_partial = running_total

    if total_partial > 0:
        packed_partial_mask = torch.cat(stripe_caches, dim=0)
        for qs_block_i, qe_block_i, cache_offset_i in stripe_meta:
            if cache_offset_i > 0:
                table_slice = partial_block_table[qs_block_i:qe_block_i, :]
                valid = table_slice >= 0
                if valid.any():
                    table_slice[valid] += cache_offset_i
    else:
        packed_partial_mask = torch.zeros(
            (0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
        )

    del stripe_caches

    partial_mask_offsets_3d = (global_B.cumsum(dim=-1) - global_B).view(1, 1, Q_NUM_BLOCKS).contiguous()

    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (SEQ_LEN, SEQ_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets_3d
    packed_block_mask.partial_block_table = partial_block_table

    del block_flags, global_B, partial_bm, full_bm
    return packed_block_mask


# ============================================================================
# Strategy 2: mask_mod-based streaming packed BlockMask builder
# ============================================================================
# Pipeline (per Q-block stripe):
#   1. _generate_stripe_mask       - evaluate mask_mod for a stripe of Q rows
#   2. _classify_stripe_blocks     - classify each (Q, KV) block as full/partial/empty
#   3. _pack_stripe_partial_blocks - extract partial blocks into packed cache + table
#   4. _assemble_packed_block_mask - merge all stripes into final BlockMask


def _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device):
    """Evaluate ``mask_mod`` for a horizontal stripe of Q rows.

    Returns a bool tensor of shape ``[B, H, actual_q, KV_LEN]``.

    For the common B=1/H=1 case we call ``mask_mod`` directly with 2-D index
    tensors, which avoids the Python overhead of ``create_mask``'s vmap stack.
    For multi-batch/multi-head we fall back to ``create_mask`` with a shifted
    mask_mod closure.
    """
    if B == 1 and H == 1:
        q_idx = torch.arange(q_start, q_start + actual_q, device=device, dtype=torch.int64)[:, None]
        kv_idx = torch.arange(0, KV_LEN, device=device, dtype=torch.int64)[None, :]
        mask_2d = mask_mod(0, 0, q_idx, kv_idx)
        return mask_2d.view(1, 1, actual_q, KV_LEN)

    def _shifted_mm(b, h, q_idx, kv_idx, _mm=mask_mod, _offset=q_start):
        return _mm(b, h, q_idx + _offset, kv_idx)

    return _torch_create_mask(_shifted_mm, B, H, actual_q, KV_LEN, device=device)


def _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Classify each (Q-block, KV-block) tile as full / partial / empty.

    Args:
        stripe_mask: bool tensor ``[B, H, stripe_q, KV_LEN_PADDED]`` whose Q and
            KV dimensions are already padded to multiples of block sizes.

    Returns:
        flags: int8 tensor ``[stripe_q_nb, KV_num_blocks]`` where
            0 = empty, 1 = partial, 2 = full.
    """
    stripe_q_nb = stripe_mask.shape[2] // Q_BLOCK_SIZE
    kv_num_blocks = stripe_mask.shape[3] // KV_BLOCK_SIZE

    partial_dense, full_dense = _convert_mask_to_block_mask(
        stripe_mask,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=True,
    )

    flags = torch.zeros((stripe_q_nb, kv_num_blocks), dtype=torch.int8, device=stripe_mask.device)
    flags[partial_dense[0, 0] == 1] = 1
    flags[full_dense[0, 0] == 1] = 2
    return flags


def _pack_stripe_partial_blocks(stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
                                 running_total, partial_block_table):
    """Extract partial blocks from a stripe into the packed cache.

    For every (Q-block, KV-block) classified as partial, copy the
    ``[Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` tile from ``stripe_mask`` into a flat list
    and record its packed index in ``partial_block_table``.

    Returns:
        packed_tiles: ``[num_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool tensor.
        num_partial:  count of partial blocks in this stripe.
    """
    partial_bool = (flags == 1)
    num_partial = int(partial_bool.sum().item())
    if num_partial == 0:
        empty = torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=stripe_mask.device)
        return empty, 0

    stripe_q_nb = flags.shape[0]
    kv_num_blocks = flags.shape[1]

    # Locate partial (q_blk, kv_blk) positions within this stripe.
    sq_idx, kv_blk_idx = partial_bool.nonzero(as_tuple=True)

    # Gather the actual [Q_BLOCK_SIZE, KV_BLOCK_SIZE] tiles.
    blocks = stripe_mask.view(stripe_q_nb, Q_BLOCK_SIZE, kv_num_blocks, KV_BLOCK_SIZE)
    packed_tiles = blocks[sq_idx, :, kv_blk_idx, :]

    # Compute the global packed index for each partial block.
    # Layout: partial blocks are packed row-major; each Q-block row's partial
    # count is cumulated so that row_offset_local[q] gives the starting index
    # of that row's partials within the stripe.
    cumsum_per_row = partial_bool.to(torch.int32).cumsum(dim=-1)
    per_row_count = cumsum_per_row.max(dim=-1).values
    row_offset_local = per_row_count.cumsum(dim=-1) - per_row_count
    local_idx = cumsum_per_row[sq_idx, kv_blk_idx] - 1
    packed_idx = (row_offset_local[sq_idx] + local_idx + running_total).to(torch.int32)

    # Record packed indices into the global table (offset by stripe's Q-block start).
    partial_block_table[qs_block + sq_idx, kv_blk_idx] = packed_idx

    return packed_tiles, num_partial


def _assemble_packed_block_mask(block_flags, packed_partial_mask, partial_block_table,
                                 global_per_row_count, Q_LEN, KV_LEN,
                                 Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Assemble the final BlockMask from streaming stripe outputs.

    Args:
        block_flags:          ``[B, H, Q_nb, KV_nb]`` int8 (0/1/2).
        packed_partial_mask:  ``[total_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool.
        partial_block_table:  ``[Q_nb, KV_nb]`` int32 (packed index or -1).
        global_per_row_count: ``[Q_nb]`` int32, partial count per Q-block row.
    """
    Q_num_blocks = block_flags.shape[2]

    # partial_mask_offsets[q] = cumulative partial count before row q.
    partial_mask_offsets = (
        (global_per_row_count.cumsum(dim=-1) - global_per_row_count)
        .view(1, 1, Q_num_blocks).contiguous()
    )

    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)

    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (Q_LEN, KV_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets
    packed_block_mask.partial_block_table = partial_block_table
    return packed_block_mask


def create_block_mask_patched(
    mask_mod,
    B=1,
    H=1,
    Q_LEN=None,
    KV_LEN=None,
    device=None,
    BLOCK_SIZE=128,
    stripe_q_blocks=None,
):
    """Build a packed BlockMask with streaming stripe processing.

    Parameters are aligned with ``torch.nn.attention.flex_attention.create_block_mask``.

    The mask is built incrementally: Q rows are processed in horizontal stripes,
    each stripe small enough to keep peak HBM bounded. For every stripe we
    evaluate ``mask_mod``, classify blocks (full/partial/empty), and immediately
    pack partial blocks into a flat cache. This avoids materialising the full
    ``[Q_LEN, KV_LEN]`` dense mask at any point.

    Args:
        mask_mod: A mask_mod callable ``(b, h, q_idx, kv_idx) -> bool``.
            Supports any flexible mask pattern, e.g. ``_full_mask_mod``,
            ``_cross_sample_causal_video_bidir_mask_mod``, ``_sparse_mask_mod``, etc.
        B: Batch size (default 1).
        H: Number of heads (default 1).
        Q_LEN: Query sequence length. If None, inferred from KV_LEN.
        KV_LEN: Key/value sequence length. If None, inferred from Q_LEN.
        device: Device for tensor allocation. If None, uses NPU/CUDA.
        BLOCK_SIZE: Block size as int (square) or ``(Q_BLOCK_SIZE, KV_BLOCK_SIZE)`` tuple.
        stripe_q_blocks: Number of Q blocks per streaming stripe. If None, auto-computed
            to target ~256MB per stripe. Controls HBM peak consumption.

    Returns:
        BlockMask with ``packed_partial_mask``, ``partial_mask_offsets``,
        and ``partial_block_table`` attributes set.
    """
    # ---- Resolve parameters --------------------------------------------------
    if device is None:
        device = _get_torch_device()
    if Q_LEN is None and KV_LEN is not None:
        Q_LEN = KV_LEN
    if KV_LEN is None and Q_LEN is not None:
        KV_LEN = Q_LEN
    assert Q_LEN is not None and KV_LEN is not None, "Q_LEN and KV_LEN must be provided"

    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE, BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    Q_num_blocks = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_num_blocks = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    KV_LEN_padded = KV_num_blocks * KV_BLOCK_SIZE

    # ---- Determine stripe size (controls HBM peak) ---------------------------
    if stripe_q_blocks is None:
        max_rows = max(1, _STRIPE_TARGET_BYTES // (KV_LEN_padded * _BYTES_PER_MASK_ELEMENT))
        stripe_q_blocks = max(1, max_rows // Q_BLOCK_SIZE)
    stripe_q_blocks = min(stripe_q_blocks, Q_num_blocks)

    # ---- Allocate accumulators ----------------------------------------------
    block_flags = torch.zeros((B, H, Q_num_blocks, KV_num_blocks), device=device, dtype=torch.int8)
    partial_block_table = torch.full((Q_num_blocks, KV_num_blocks), -1, dtype=torch.int32, device=device)
    global_per_row_count = torch.zeros(Q_num_blocks, dtype=torch.int32, device=device)
    packed_tiles_list = []
    running_total = 0

    # ---- Process stripes -----------------------------------------------------
    for qs_block in range(0, Q_num_blocks, stripe_q_blocks):
        qe_block = min(qs_block + stripe_q_blocks, Q_num_blocks)
        q_start = qs_block * Q_BLOCK_SIZE
        stripe_q = (qe_block - qs_block) * Q_BLOCK_SIZE
        actual_q = min(stripe_q, Q_LEN - q_start)

        # Step 1: generate dense mask for this stripe's Q rows.
        stripe_mask = _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device)

        # Pad Q/KV to block boundaries so classification is exact.
        pad_q = stripe_q - actual_q
        pad_kv = KV_LEN_padded - KV_LEN
        if pad_q > 0 or pad_kv > 0:
            stripe_mask = _F.pad(stripe_mask, (0, pad_kv, 0, pad_q))

        # Step 2: classify each (Q-block, KV-block) tile.
        flags = _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        block_flags[:, :, qs_block:qe_block, :] = flags

        # Record per-row partial counts for offset computation later.
        partial_bool = (flags == 1)
        per_row_count = partial_bool.to(torch.int32).cumsum(dim=-1).max(dim=-1).values
        global_per_row_count[qs_block:qe_block] = per_row_count.to(torch.int32)

        # Step 3: pack partial blocks into flat cache + update table.
        packed_tiles, num_partial = _pack_stripe_partial_blocks(
            stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            running_total, partial_block_table,
        )
        packed_tiles_list.append(packed_tiles)
        running_total += num_partial

        del stripe_mask, flags, partial_bool, per_row_count

    # ---- Merge stripe caches -------------------------------------------------
    if running_total > 0:
        packed_partial_mask = torch.cat(packed_tiles_list, dim=0)
    else:
        packed_partial_mask = torch.zeros(
            (0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
        )
    del packed_tiles_list

    # Step 4: assemble final BlockMask with packed attributes.
    packed_block_mask = _assemble_packed_block_mask(
        block_flags, packed_partial_mask, partial_block_table,
        global_per_row_count, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )

    del block_flags, global_per_row_count
    return packed_block_mask