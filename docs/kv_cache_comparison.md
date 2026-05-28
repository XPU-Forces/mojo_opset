# store_paged_kv_impl (Triton NPU) vs scatter_pa_kv_cache arch35 (AscendC A5) 比对分析

> 两个算子功能等价：将新增 KV token 写入 Paged KV Cache。前者基于 Triton NPU 后端实现，后者是华为 CANN 平台的 AscendC 原生算子。torch_npu 后端通过 `_build_slot_mapping()` 将 blockTable 转换为 slotMapping 后调用 `npu_scatter_pa_kv_cache`，证明两者的语义等价性。

---

## 1. 概述

### 1.1 算子定位

| 维度 | store_paged_kv_impl (Triton) | scatter_pa_kv_cache (AscendC) |
|------|------------------------------|-------------------------------|
| **实现语言** | Triton (Python DSL, JIT编译) | AscendC (C++ DSL, AOT编译) |
| **目标硬件** | Ascend NPU (via Triton NPU backend) | Ascend A5 (910B/C) |
| **调用路径** | `mojo_opset.backends.ttx.kernels.npu.kv_cache` | `torch_npu.npu_scatter_pa_kv_cache` |
| **地址映射模型** | blockTable 二维映射 (kernel内部解析) | slotMapping 一维线性偏移 (外部预处理) |
| **Prefill/Decode 区分** | 编译时常量 `IS_DECODE` 切换 | 不区分，统一 scatter |
| **核心操作** | 逻辑位置→物理block查表 + scatter write | 纯 scatter write (slot已预计算) |

### 1.2 功能等价性证明

torch_npu 后端 (`kv_cache.py`) 的调用链：

```
block_table + cu_q_lens + context_kv_lens + block_size
         ↓  _build_slot_mapping()
    slot_mapping[i] = phys_block * block_size + block_inner_off
         ↓  torch_npu.npu_scatter_pa_kv_cache()
    scatter_pa_kv_cache(key, value, keyCache, valueCache, slotMapping)
```

Triton 后端在 kernel 内部完成完全相同的映射：

```
block_table_idx = (kv_len + t) // block_size
block_inner_off = (kv_len + t) % block_size
physical_block_id = block_table[batch_idx, block_table_idx]
```

两者写入的物理位置完全一致，精度可对齐。

---

## 2. 逐维度比对

### 2.1 地址映射模型

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **映射模型** | blockTable: `[bsz, max_blocks]` 二维表，逻辑 block ID → 物理 block ID | slotMapping: `[numTokens]` 一维数组，每个 token 直接给出线性偏移 |
| **映射层级** | 二级：token→逻辑block→物理block | 一级：token→线性slot |
| **映射计算位置** | Kernel 内部（运行时计算） | Kernel 外部（Host端 Python 预计算） |
| **映射数据结构** | `block_table[batch_idx, logical_block_idx]` → `phys_block_id` | `slot_mapping[token_idx]` → `phys_block * block_size + offset` |
| **Kernel 内地址计算** | `phys_blk = block_table[b, log_pos // B]` + `off = log_pos % B` | `offset = slot_mapping[t] * (numHead * headSize)` |
| **无效标记** | `phys_block_id < 0` | `slot_mapping[t] < 0` |
| **额外存储** | blockTable 大小 = `bsz × max_blocks_per_seq` | slotMapping 大小 = `numTokens` |

**分析**：

- **blockTable** 是一种间接映射：每个 batch 维护一个逻辑 block 列表，kernel 需要做除法+取模来定位。优势是映射表紧凑（按 block 而非按 token），且同一 block 内的所有 token 共享一次查表。
- **slotMapping** 是一种直接映射：每个 token 的目标位置已预先展平为一维偏移，kernel 无需任何地址计算。代价是映射表较大（按 token 而非按 block），且需要 Host 端预计算开销。
- 在 **decode 场景**下（每 batch 1 token），slotMapping 的大小 = bsz，与 blockTable 的 `bsz × max_blocks` 相比通常更小。
- 在 **长序列 prefill** 场景下，slotMapping 大小 = total_tokens，可能远大于 blockTable。但 blockTable 的访问模式需要每 token 一次除法+查表，计算开销更高。

### 2.2 并行化策略

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **调度单位** | chunk (CHUNK_SIZE = block_size tokens) | token (1 token = 最小调度单位) |
| **调度策略** | Round-robin: 所有 batch 的 chunk 拉平为一维队列，program 间隔取 chunk | 连续切分: 每个 core 处理连续的 blockFactor 个 token |
| **Grid 启动** | `num_programs = get_num_cores("vector")` 个 program | `usedCoreNum = ceil(numTokens / blockFactor)` 个 core |
| **跨 batch 处理** | 每个 program 遍历所有 batch，以 `num_programs` 为步长在全局 chunk 队列中间隔取 | 每个 core 只处理连续的一段 token，不感知 batch 边界 |
| **负载均衡** | Round-robin 天然均衡：短 batch 和长 batch 的 chunk 交错分配 | 连续切分：若 token 在 batch 间分布均匀则均衡，否则尾部 core 可能空闲 |
| **调度偏移计算** | `start_chunk = (pid + num_programs - prev_chunks % num_programs) % num_programs` | 无偏移计算，直接 `blockFactorOffset = blockIdx × blockFactor` |

**分析**：

- **Round-robin** 的优势在于天然负载均衡。当 batch 间序列长度差异大时（如 batch0 有 100 个 chunk，batch1 只有 1 个），round-robin 能将长序列的 chunk 分散到不同 core，避免单个 core 被长序列阻塞。代价是需要遍历所有 batch 来维护 `prev_chunks` 计数器，引入了 `O(batch_size)` 的串行开销。
- **连续切分** 的优势是简单：无需感知 batch 边界，每个 core 独立处理一段连续 token。在 prefill 场景（token 数大）下，core 利用率通常很高。但在 **decode 场景**（token 数 = bsz，可能远小于 core 数）下，大量 core 空闲，利用率低。
- Triton 的 round-robin 策略更适合 **混合 prefill/decode** 或 **序列长度差异大** 的场景。

### 2.3 数据搬运粒度

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **搬运粒度** | 按 head 逐个 load/store: `[sub_len, head_dim]` 2D block | FullyLoad: 批量搬运 `blockFactor × (numHead × headDim)` 1D 展平；NotFullyLoad: 按 loop 搬运 `kHandleNumPerLoop` 1D 段 |
| **每次搬运大小** | `sub_len × head_dim` 元素 (sub_len ∈ [1, block_size]) | FullyLoad: `numHead × headSize` × blockFactor 元素; NotFullyLoad: `kHandleNumPerLoop` 元素 |
| **K/V 是否合并搬运** | 否，K 和 V 分两次 load/store | FullyLoad: K/V 各自批量搬运; NotFullyLoad: K/V 各自逐 loop 搬运 |
| **数据布局** | 保留 2D 结构 `[tokens, heads, dim]` | 展平为 1D `[numHead × headSize]` per token |
| **对齐处理** | Triton 编译器自动处理 | 手动 `RoundUp()` 到 32/sizeof(T) 对齐 |

**分析**：

- Triton 的按 head 搬运模式更直观，保留了 tensor 的多维结构。但每次搬运的数据量较小（`head_dim` 通常 128-256 元素），GM 访问次数 = `num_heads × num_sub_blocks × 2`（load+store），访存开销较大。
- AscendC FullyLoad 模式将所有 head 数据展平后一次搬运，GM 访问次数最少（每个 token 的 key 只需 1 次 CopyIn + 1 次 CopyOut），在 UB 足够时性能最优。
- AscendC NotFullyLoad 模式是 FullyLoad 的退化版本，当 `numHead × headSize` 超过 UB 容量时触发，需要沿 head 维度分 loop 搬运，增加了 GM 访问次数。

### 2.4 UB/Cache 管理

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **管理方式** | 隐式：Triton 编译器自动分配 UB buffer，`tl.load/store` 自动触发 DataCopy | 显式：手动 `pipe.InitBuffer()` 分配，手动 `DataCopyPad()` 搬运 |
| **Buffer 分配** | 编译器根据 `tl.arange` 推断 buffer 大小 | Tiling 阶段精确计算 UB 用量，根据 FullyLoad/NotFullyLoad 分配不同大小 |
| **同步机制** | 隐式：Triton 保证 load 完成后才执行后续计算 | 显式：`SetFlag/WaitFlag<HardEvent::MTE2_S/V_S>` 手动同步 |
| **Double Buffer** | 编译器自动决策 | 手动 `TQueBind<VECIN, VECOUT, 1>` 配置队列深度 |
| **UB 利用率** | 编译器控制，开发者无法精确调控 | 开发者根据 tiling 参数精确计算，最大化 UB 利用率 |
| **调试难度** | 低（Triton 抽象了硬件细节） | 高（需要理解 UB 分配、同步事件、Queue 机制） |

**分析**：

- Triton 的隐式管理降低了开发门槛，但牺牲了对 UB 利用的精确控制。编译器可能无法像手写 AscendC 那样充分榨取 UB 空间。
- AscendC 的显式管理虽然开发成本高，但能根据实际数据量（`blockFactor`, `numHead × headSize`）精确分配 UB，在边界场景下更可靠。
- 在 UB 紧张的场景（大 head 数 × 大 head_dim），AscendC NotFullyLoad 模式能优雅降级，而 Triton 可能因编译器 buffer 分配策略不当导致性能下降。

### 2.5 Block 边界处理

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **边界感知** | Kernel 内运行时处理 | 无需处理（slotMapping 已预处理） |
| **处理机制** | while 循环：检测 `block_inner_off ≠ 0` 时在 block 边界断开，分多次 sub_len 写入 | 不适用：slotMapping 已将每个 token 映射到正确的 slot，不存在跨 block 问题 |
| **计算开销** | 每个 chunk 需要 1-2 次迭代（取决于对齐情况），每次迭代需要 1 次除法 + 1 次取模 + 1 次查表 | 零开销 |
| **特殊情况** | 当 `kv_len_before_store % block_size ≠ 0` 时，第一个 chunk 会跨越两个物理 block | slotMapping 由 Host 端正确计算，不会出现跨 block 问题 |

**分析**：

- 这是两者最显著的架构差异之一。Triton kernel 需要处理 chunk 跨 block 边界的情况（`CHUNK_SIZE = block_size`，但写入起始位置可能不对齐），引入了 while 循环和额外的地址计算。
- AscendC kernel 完全不需要处理这个问题，因为 slotMapping 已在 Host 端将每个 token 映射到正确的线性偏移。这是一种 **计算前置** 的设计哲学：将复杂性从 kernel 转移到 Host 端。
- Host 端 `_build_slot_mapping()` 的复杂度为 `O(total_tokens)`，涉及 `block_table` 查表和除法/取模。但由于在 CPU 上执行且可向量化，通常不是性能瓶颈。
- 将边界处理前置到 Host 端的好处是 kernel 逻辑大幅简化，减少了 kernel 内的分支和计算，有利于提高 kernel 执行效率。

### 2.6 通用性/灵活性

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **Prefill/Decode** | 通过 `IS_DECODE` 编译时常量区分，两套代码路径 | 统一处理，不区分 prefill/decode |
| **Decode 优化** | `seq_start = batch_idx, seq_len = 1`，跳过 cu_seqlens 读取 | 无特殊优化，与 prefill 同路径 |
| **数据类型** | float16, bfloat16 | float16, bfloat16, float32, int8, fp4 等更多类型 |
| **K/V headDim** | 必须相同 (`head_dim`) | 可以不同 (`kHeadSize ≠ vHeadSize`) |
| **Cache 布局** | `[blocks, heads, block_size, dim]` (NHSD) | Normal: `[blocks, block_size, heads, dim]` (NSHD); NZ: 5D Nz 格式 |
| **扩展模式** | 仅 Normal scatter | Normal + NZ + Rope + Alibi + Omni 等多种模式 |

**分析**：

- Triton 实现的 `IS_DECODE` 常量允许编译器针对 decode 场景做专门优化（如消除 cu_seqlens 读取、固定 seq_len=1），理论上 decode 性能更好。但代价是需要维护两套代码路径。
- AscendC 的统一设计更简洁，且通过 tiling 机制在运行时适配不同场景，不需要编译期特化。slotMapping 的通用性使得 kernel 逻辑与上下文无关。
- AscendC 算子支持更多数据类型和 cache 布局（NZ 格式在华为平台上对 DMA 友好），通用性更强。torch_npu 后端的 `_forward_nz` 方法会尝试 NZ 模式优先，失败后回退到 Normal 模式。

### 2.7 性能特征分析

#### 2.7.1 内存访问模式

| 维度 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **Source 读取** | 连续读取 `k_states[curr_kv_pos:curr_kv_pos+sub_len, h, :]` | FullyLoad: 连续读取 `blockFactor` 个 token 的展平数据; NotFullyLoad: 连续读取 `kHandleNumPerLoop` 元素 |
| **Dest 写入** | Scatter 写入，目标地址由 blockTable 查表决定 | Scatter 写入，目标地址由 slotMapping 决定 |
| **Scatter 粒度** | 按 head × sub_len tokens (2D block) | 按 token × 完整 head 数据 (1D 展平) |
| **访问局部性** | 较差：同一 head 的连续 token 可能分布在不同物理 block | 较差：scatter 写入本质上是非连续的 |
| **block_table 读取** | 每个 sub_block 读取 1 次 | 不适用（无 block_table） |

**关键洞察**：

两个算子的目标写入都是 scatter 模式，这是 paged KV cache 的本质特征——物理 block 不保证连续。区别在于 scatter 的粒度和地址计算方式：

- **Triton**: scatter 粒度为 `[sub_len, head_dim]`（2D block），需要更频繁但更小的写入操作。
- **AscendC FullyLoad**: scatter 粒度为 `numHead × headSize`（1D 展平），每个 token 只需 1 次大块写入，DMA 效率更高。
- **AscendC NotFullyLoad**: scatter 粒度降级为 `kHandleNumPerLoop`（1D 段），需要多次写入，DMA 效率下降。

#### 2.7.2 核心利用率

| 场景 | store_paged_kv_impl | scatter_pa_kv_cache |
|------|---------------------|---------------------|
| **Decode (bsz=32, core=64)** | 32 个 program 有工作，32 空闲（但 round-robin 会分散负载） | `blockFactor=1`, 仅 32 个 core 有工作，利用率 ~50% |
| **Decode (bsz=256, core=64)** | round-robin 分散到所有 core | `blockFactor=4`, 64 个 core 全部工作，利用率 ~100% |
| **Prefill (tokens=4096, core=64)** | round-robin 充分利用 | `blockFactor=64`, 64 个 core 全部工作，利用率高 |
| **Prefill (tokens=63, core=64)** | round-robin 分散 63 个 chunk | `blockFactor=1`, 63 个 core 有工作，利用率 ~98% |
| **Mixed (长短序列混合)** | round-robin 天然均衡 | 尾部 core 可能负载不均 |

#### 2.7.3 对不同序列长度的适应性

| 序列特征 | store_paged_kv_impl | scatter_pa_kv_cache |
|----------|---------------------|---------------------|
| **短序列 decode** | 每个 batch 只有 1 个 chunk，round-robin 在 program 间均匀分配 | blockFactor=1，每个 core 处理 1 个 token，简单高效 |
| **长序列 prefill** | 每个 batch 有大量 chunk，round-robin 分散到所有 core | 连续切分充分利用所有 core |
| **极端不平衡** | round-robin 保证每个 program 的工作量差异 ≤ 1 chunk | 连续切分可能导致某些 core 空闲 |
| **KV cache 部分填充** | 需要处理 kv_len 对齐导致的跨 block | slotMapping 已处理，无需额外开销 |

### 2.8 计算流比对

#### 2.8.1 逐步骤对应关系

| 步骤 | store_paged_kv_impl (Triton) | scatter_pa_kv_cache (AscendC FullyLoad) |
|------|------------------------------|-----------------------------------------|
| **1. 并行分发** | `pid = tl.program_id(0)`<br>`num_programs = get_num_cores("vector")` | `blockIdx_ = GetBlockIdx()`<br>由 tiling 决定 `usedCoreNum` |
| **2. 初始化** | `prev_chunks = 0`<br>准备遍历所有 batch | `blockFactorOffset_ = blockIdx_ * blockFactor`<br>`pipe_.InitBuffer()` 分配 UB |
| **3. Batch 遍历** | `for batch_idx in range(batch_size)`<br>每个 program 遍历所有 batch | 无 batch 遍历<br>每个 core 只处理连续的 blockFactor 个 token |
| **4. 负载计算** | 计算 `cur_chunks` 和 `start_chunk`<br>Round-robin 分配 chunk | `curBlockFactor = (blockIdx == last) ? tail : blockFactor`<br>连续切分 token |
| **5. 地址预计算** | 无（在 per-token 处理中计算） | `CalcStartIdx()` 将 slotMapping 加载到 UB |
| **6. 同步等待** | 隐式（Triton 自动管理） | `SetFlag/WaitFlag<HardEvent::MTE2_S>` 显式同步 |
| **7. 数据加载** | `tl.load(k_ptr, ...)` per head<br>在 while 循环内逐 sub_len 加载 | `CopyIn()` 批量加载 `blockFactor` 个 token 的展平数据到 UB |
| **8. 地址解析** | `block_table_idx = curr_log_pos // block_size`<br>`physical_block_id = tl.load(block_table[...])`<br>`block_inner_off = curr_log_pos % block_size` | `kStartIdx = kSlotMappingLocal.GetValue(i)`<br>直接从 slotMapping 读取线性偏移 |
| **9. 边界处理** | `while processed < remain_chunk_len`<br>检测 block 边界，分 sub_len 写入 | 无需处理<br>slotMapping 已包含正确地址 |
| **10. Per-head 处理** | `for h in range(num_kv_heads)`<br>逐 head load/store | 隐式处理<br>数据已展平为 `[numHead × headSize]` |
| **11. 数据写入** | `tl.store(dst_k_ptr, k_val, mask=...)`<br>per head × sub_len tokens | `DataCopyPad(outputKeyCacheGm_[kStartIdx], ...)`<br>per token × 完整 head 数据 |
| **12. 完成** | 结束 batch 遍历，kernel 退出 | 结束 CopyIn/CopyOut，kernel 退出 |

#### 2.8.2 计算流对比图

```mermaid
flowchart LR
    subgraph Triton["store_paged_kv_impl (Triton)"]
        T1["Get pid and num_programs"]
        T2["Initialize prev_chunks = 0"]
        T3["For each batch_idx"]
        T4["Load cu_seqlens and kv_lens"]
        T5["Calculate cur_chunks and start_chunk"]
        T6["For each assigned chunk"]
        T7["Calculate token positions"]
        T8["While: processed < remain_chunk_len"]
        T9["Calculate block_table_idx and block_inner_off"]
        T10["Load physical_block_id from block_table"]
        T11["Calculate sub_len (space in block)"]
        T12["For each head"]
        T13["Load K data from GM (per head)"]
        T14["Calculate dst address for K"]
        T15["Store K data to cache (masked)"]
        T16["Load V data from GM (per head)"]
        T17["Calculate dst address for V"]
        T18["Store V data to cache (masked)"]
        T19["Update positions, continue while"]
        T20["End while, continue chunk loop"]
        T21["End chunk loop, continue batch loop"]
        T22["End batch loop, kernel exit"]
    end

    subgraph AscendC["scatter_pa_kv_cache (AscendC)"]
        A1["Get blockIdx"]
        A2["Calculate blockFactorOffset"]
        A3["Init UB buffers for input and slotMapping"]
        A4["Determine curBlockFactor"]
        A5["CalcStartIdx: Load slotMapping to UB"]
        A6["SetFlag and WaitFlag for sync"]
        A7["CopyIn: Load K data to UB (batch)"]
        A8["For each token in blockFactor"]
        A9["Get slot from slotMapping"]
        A10["Check if slot is valid"]
        A11["CopyOut: Write K data to cache at slot"]
        A12["Repeat for V data"]
        A13["End token loop, kernel exit"]
    end
```

**关键对比点：**

1. **并行策略差异**：
   - Triton：Round-robin 调度，所有 program 遍历所有 batch，通过 `start_chunk` 计算实现负载均衡
   - AscendC：连续切分，每个 core 处理固定的 `blockFactor` 个 token，无 batch 感知

2. **地址解析方式**：
   - Triton：运行时查表，每次迭代需要除法+取模+blockTable 读取
   - AscendC：直接读取，slotMapping 已预先计算为线性偏移

3. **边界处理复杂度**：
   - Triton：需要 while 循环处理 block 边界，可能在一次 chunk 写入中跨越多个物理 block
   - AscendC：无边界处理，slotMapping 已将每个 token 映射到正确的线性位置

4. **数据搬运粒度**：
   - Triton：按 head × sub_len 逐个小块搬运（通常 128-256 元素）
   - AscendC：按 token × 完整 head 数据批量搬运（所有 head 数据展平后一次搬运）

5. **同步机制**：
   - Triton：隐式，编译器自动管理
   - AscendC：显式，需要手动设置事件标志和等待

6. **Batch 处理**：
   - Triton：kernel 内显式遍历 batch，有 batch 语义
   - AscendC：kernel 不感知 batch，只看到一维 token 队列

---

## 3. 优势劣势总结

### 3.1 store_paged_kv_impl (Triton) 优势

1. **负载均衡优异**：Round-robin 调度保证核心间工作量差异 ≤ 1 chunk，在序列长度不均匀的场景下表现优于连续切分。
2. **开发门槛低**：Triton DSL 接近 Python，隐式 UB 管理，无需手动处理 DataCopyPad、同步事件等底层细节。
3. **Prefill/Decode 特化**：`IS_DECODE` 编译时常量允许编译器针对两种模式生成不同的优化代码，decode 路径可消除不必要的 cu_seqlens 读取。
4. **可移植性**：Triton 代码理论上可迁移到其他支持 Triton 的硬件（GPU/NPU），但当前实现依赖 Triton NPU backend。
5. **直观的 2D 数据搬运**：保留 `[tokens, head_dim]` 的 2D 结构，数据流更易理解和调试。

### 3.2 store_paged_kv_impl (Triton) 劣势

1. **Block 边界处理复杂**：while 循环处理 chunk 跨 block 边界的情况，增加了 kernel 内的分支和计算开销。
2. **访存效率低**：按 head 逐个 load/store，每次搬运数据量小（`head_dim` 通常 128-256 元素），GM 访问次数多。
3. **UB 利用不精确**：编译器隐式管理 UB，无法像手动管理那样精确控制 UB 分配和利用率。
4. **通用性不足**：不支持 K/V headDim 不同、数据类型有限（仅 fp16/bf16）、无 NZ 布局支持。
5. **batch 遍历串行开销**：每个 program 需要 `O(batch_size)` 的串行遍历来计算 round-robin 偏移，batch 数很大时成为瓶颈。

### 3.3 scatter_pa_kv_cache (AscendC) 优势

1. **Kernel 逻辑极简**：纯 scatter 操作，无需处理 block 边界、batch 遍历、prefill/decode 区分，kernel 代码高度聚焦。
2. **访存效率高**（FullyLoad）：将所有 head 数据展平后一次 CopyIn/CopyOut，GM 访问次数最少。
3. **通用性强**：支持多种数据类型（fp16/bf16/fp32/int8/fp4）、K/V headDim 可不同、NZ/Normal 两种 cache 布局、Rope/Alibi/Omni 等扩展模式。
4. **UB 精确控制**：Tiling 阶段精确计算 UB 用量，根据 FullyLoad/NotFullyLoad 自适应分配，最大化片上缓存利用。
5. **Host-GPU 协同设计**：将地址映射的复杂性前置到 Host 端（`_build_slot_mapping`），kernel 只负责纯粹的数据搬运，职责分离清晰。
6. **经过大规模验证**：作为华为 CANN 官方算子，已在生产环境中广泛部署，稳定性有保障。

### 3.4 scatter_pa_kv_cache (AscendC) 劣势

1. **负载均衡依赖数据分布**：连续切分在 batch 间 token 数不均匀时，可能导致尾部 core 空闲。decode 场景下（bsz < core 数）利用率可能较低。
2. **Host 端预计算开销**：`_build_slot_mapping()` 在 CPU 上串行执行，当 token 数极大时（如长序列 prefill）可能成为瓶颈。但实测中通常可忽略。
3. **开发维护成本高**：AscendC 是华为专用 DSL，学习曲线陡峭；多种模板（FullyLoad/NotFullyLoad/NZ/Rope 等）增加了代码复杂度和维护成本。
4. **不感知 batch 语义**：kernel 只看到一维 token 队列和 slotMapping，无法利用 batch 级别的优化（如 batch 内共享 block_table）。
5. **NotFullyLoad 性能退化**：当 UB 不足时退化为逐 token × 逐 loop 的搬运模式，GM 访问次数急剧增加。

---

## 4. 精度对齐分析

### 4.1 功能等价性

torch_npu 后端 (`kv_cache.py`) 中的 `_build_slot_mapping()` 函数证明了两个算子的功能等价性：

```python
# _build_slot_mapping 的核心逻辑
logical_pos = kv_len_start + t                    # 逻辑位置
bt_idx = logical_pos // block_size                # 逻辑 block 索引
bt_off = logical_pos % block_size                 # block 内偏移
phys_block = block_table[batch_idx, bt_idx]       # 物理块号
slot_mapping[token_start + t] = phys_block * block_size + bt_off  # 一维 slot
```

这与 Triton kernel 内部的映射逻辑完全一致：

```python
# Triton kernel 内部的映射逻辑
block_table_idx = curr_log_pos // block_size
block_inner_off = curr_log_pos % block_size
physical_block_id = block_table[batch_idx, block_table_idx]
```

### 4.2 Cache 布局转换

torch_npu 后端在调用 scatter_pa_kv_cache 前后需要进行布局转换：

| 步骤 | 操作 | 原因 |
|------|------|------|
| 调用前 | `_nhsd_to_nshd()`: `[B,H,S,D] → [B,S,H,D]` | scatter_pa_kv_cache Normal 模式要求 cache shape 为 `[blocks, blockSize, heads, dim]` |
| 调用后 | `_nshd_to_nhsd()`: `[B,S,H,D] → [B,H,S,D]` | 转回 mojo_opset 内部统一的 NHSD 布局 |
| NZ 模式 | `_nhsd_to_nz()` / `_nz_to_nhsd()`: 5D Nz 格式转换 | NZ 格式对 Ascend DMA 更友好，但需要额外的 reshape/permute 开销 |

### 4.3 精度保障

两个算子在相同输入下输出完全一致，因为：

1. **相同的映射逻辑**：两者将同一个 token 写入同一个物理位置，写后结果相同。
2. **相同的数据类型**：两者都操作 fp16/bf16 数据，无精度损失差异。
3. **原地更新**：两者都是对 key_cache/value_cache 原地更新，不存在中间精度的累积误差。
4. **padding 处理一致**：Triton 通过 `valid_block` mask 跳过无效写入，AscendC 通过 `slotMapping[i] < 0` 跳过无效 token。

### 4.4 注意事项

- torch_npu 后端的 `_build_slot_mapping()` 是纯 Python 实现，在 CPU 上串行执行。当 `total_tokens` 很大时（如 prefill 长序列），可能引入可观的 Host 端延迟。未来可考虑用 C++ extension 或 torch.compile 加速。
- 布局转换 (`_nhsd_to_nshd` / `_nhsd_to_nz`) 涉及额外的 permute 和 contiguous 操作，会引入额外的 GPU→CPU→GPU 拷贝开销。在 NZ 模式下，如果上层能直接维护 NZ 格式的 cache，可避免此开销。

---

## 5. 结论与建议

### 5.1 总体评价

| 评价维度 | store_paged_kv_impl (Triton) | scatter_pa_kv_cache (AscendC) |
|----------|------------------------------|-------------------------------|
| **开发效率** | ★★★★☆ 高 | ★★☆☆☆ 低 |
| **执行效率** | ★★★☆☆ 中 | ★★★★☆ 高（FullyLoad）/ ★★★☆☆ 中（NotFullyLoad） |
| **负载均衡** | ★★★★☆ 优（round-robin） | ★★★☆☆ 良（连续切分） |
| **通用性** | ★★☆☆☆ 有限 | ★★★★★ 广泛 |
| **可维护性** | ★★★★☆ 好 | ★★☆☆☆ 复杂 |
| **生产就绪度** | ★★★☆☆ 中 | ★★★★★ 高 |

### 5.2 场景推荐

| 场景 | 推荐选择 | 原因 |
|------|----------|------|
| **大规模推理服务** | scatter_pa_kv_cache | 经过生产验证，通用性强，支持多种数据类型和布局 |
| **快速原型开发** | store_paged_kv_impl | Triton DSL 开发效率高，迭代速度快 |
| **混合 prefill/decode** | store_paged_kv_impl | Round-robin 调度在序列长度差异大时更均衡 |
| **纯 decode 高吞吐** | scatter_pa_kv_cache | slotMapping 预处理后 kernel 极简，FullyLoad 一次搬运效率最高 |
| **大 head 数 + 大 head_dim** | scatter_pa_kv_cache | NotFullyLoad 模式能优雅处理 UB 不足的情况 |

### 5.3 改进建议

**对 store_paged_kv_impl (Triton)**：

1. **增大搬运粒度**：考虑将多个 head 的数据合并为一次 load/store，减少 GM 访问次数。
2. **借鉴 slotMapping 思路**：将 block 边界处理前置到 Host 端（生成 slotMapping），消除 kernel 内的 while 循环。
3. **增加数据类型支持**：扩展到 fp32、int8 等。
4. **优化 batch 遍历**：当 batch_size 很大时，`O(batch_size)` 的串行遍历可能成为瓶颈。可考虑将 batch 维度也并行化。

**对 scatter_pa_kv_cache (AscendC)**：

1. **优化 decode 利用率**：当 bsz < core 数时，考虑将多个 batch 的 token 合并到一个 core 处理，减少 core 空闲。
2. **加速 slotMapping 构建**：`_build_slot_mapping()` 可用 C++ extension 或 CUDA-like kernel 在 NPU 上并行执行。
3. **减少布局转换开销**：提供 NHSD 原生支持的接口，避免调用前后的 permute 操作。
4. **引入 round-robin 调度选项**：在序列长度差异大的场景下，可选的 round-robin 调度可提升负载均衡。

---

## 附录：术语对照表

| Triton 术语 | AscendC 术语 | 含义 |
|-------------|-------------|------|
| program | AI Core / blockIdx | 硬件执行单元 |
| `tl.load` | `DataCopyPad` (GM→UB) | 数据从显存搬入片上缓存 |
| `tl.store` | `DataCopyPad` (UB→GM) | 数据从片上缓存写回显存 |
| chunk (CHUNK_SIZE) | blockFactor | 调度/切分粒度 |
| `block_table` | slotMapping (外部转换) | 地址映射表 |
| `cu_seqlens` | (不需要) | 序列长度前缀和 |
| `kv_lens_before_store` | (编码在 slotMapping 中) | 历史 KV 长度 |
| GM (Global Memory) | GM (Global Memory) | 显存/HBM |
| UB (Unified Buffer) | UB (Unified Buffer) | 片上缓存 (256KB) |
| `IS_DECODE` | (不需要) | prefill/decode 模式标志 |
| N/A | Tiling | 编译期/运行期参数切分策略 |
| `get_num_cores("vector")` | `GetCoreNumAiv()` | 查询可用 core 数 |
