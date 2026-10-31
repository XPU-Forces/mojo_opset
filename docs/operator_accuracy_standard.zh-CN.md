# mojo_opset 社区生态交付算子精度验收标准

## 0. 适用范围说明

本标准适用于**浮点计算类算子**。对于整型计算类、搬运类算子需按照各算子的实际业务场景单独制定标准，不在本标准讨论范围内。

---

## 1. 用例生成规则

生态算子的输入主要由张量（Tensor）和属性（Attr）两部分组成。测试用例需全场景覆盖算子的典型场景与边界场景，确保精度评估的全面性。

### 1.1 用例规模要求

| 要求项 | 说明 |
|--------|------|
| **覆盖目标** | 对算子支持的**数据类型**、**数据格式**、**数据维度**、**属性取值范围**实现**全组合覆盖**（即所有有效组合的 100% 覆盖） |
| **数据类型覆盖** | 覆盖算子支持的所有数据类型

> 说明：用例数量不设固定下限（如 ≥1,000），强调的是**输入组合的遍历式覆盖**，而非机械地要求用例数量。

### 1.2 Tensor 生成规则

| 维度 | 生成规则 |
|------|----------|
| **数据维度** | 覆盖算子支持的维度范围（1~8维），维度值在 $[1, 2^{20}]$ 内取用（2的幂次）和（2的幂次方-1）两种取值。总元素数不超过2的34次方 |
| **数据格式** | 覆盖算子支持的所有数据格式（ND、NCHW等） |

**步长与组合生成**：各参数类型之间应进行正交组合遍历，值域覆盖策略如下：

| 分布类型 | 比例 | 值域范围 | 生成逻辑 |
|---------|------|----------|----------|
| 均匀分布 | 50% | [-5, 5] | 随机均匀采样 |
| 正态分布 | 50% | μ ∈ [-5, 5], σ ∈ [0.1, 2] | 随机采样 μ 和 σ 后生成 |

### 1.3 Attr 生成规则

| 参数类型 | 生成规则 |
|----------|----------|
| 标量参数 | 覆盖所有等价类场景 |
| 布尔参数 | 覆盖 True 和 False |
| 枚举参数 | 覆盖所有支持的枚举值 |

各类参数类型应进行组合遍历。

### 1.4 特殊场景测试

| 场景 | 说明 | 覆盖要求 |
|------|------|----------|
| 空 Tensor | 某个维度为 0 的 Tensor | 每种 dtype 每个 tensor 至少覆盖一次 |
| 标量 Tensor | shape 为 [1] 的 Tensor | 每种 dtype 覆盖 |
| 边界测试 | 下边界：shape 各维度均为 1；上边界：shape 某维度取最大值 | 全部覆盖 |
| INF/-INF/NAN | 输入元素值遍历 nan、inf、-inf、[-inf, inf] | 每种 dtype 每种值生成不同 shape 用例 |

> 说明：特殊场景用例不与常规用例正交组合。

---

## 2. 误差指标与通过标准

### 2.1 误差指标

本标准采用**混合容差**（Mixed Tolerance）指标来判断，即结合绝对容差（atol）和相对容差（rtol）的逐元素比对方式。

#### 2.1.1 逐元素通过条件

对于输出张量中的每个元素，当满足以下条件时判定该元素通过：

$$
|actual - golden| \leq atol + rtol \times |golden|
$$

其中：

- **atol**（Absolute Tolerance，绝对容差）：保证小值（golden接近0）场景下的合理误差范围，天然避免除零问题。
- **rtol**（Relative Tolerance，相对容差）：保证大值场景下的相对精度。

#### 2.1.2 整体通过条件

用例中**所有参与比对的元素**都满足逐元素条件时，判定该用例通过。

`±inf`/NaN 元素不参与逐元素比对，按 2.3 节的收窄规则处理。

### 2.2 混合容差阈值表

<table style="width: 100%; border-collapse: collapse;">
    <thead>
      <tr>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;">数据类型</th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>BFLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>HiFLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E4M3</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E5M2</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>rtol</strong></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup> (1e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-7</sup> (7.8e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-13</sup> (1.2e-4)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup> (1e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-3</sup> (0.125)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-2</sup> (0.25)</td>
      </tr>
      <tr>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>atol</strong></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup> (1e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-7</sup> (7.8e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-13</sup> (1.2e-4)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup> (1e-3)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-3</sup> (0.125)</td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-2</sup> (0.25)</td>
      </tr>
    </tbody>
</table>

**注**：
- 当前的阈值拦截在各数值精度下约为 1ULP左右，涉及到大数规约可能会引入更大精度误差导致误报，可酌情降低至2ULP。  

### 2.3 通过判定

**单标杆比对**：与更高精度的实现（CPU或昇腾小算子拼接）的单一精度标杆直接比较。

当用例中所有参与比对的元素都满足 2.1.1 的逐元素条件时，判定该用例精度通过。

比对过程中，若遇算子实际输出为 `±inf` 而 golden 为有限值时，应先将 golden 降精度（收窄）到算子实际输出 dtype 后再比对：收窄按 round-to-nearest-even 舍入，超出该 dtype 表示范围的值转换为 `±inf`（IEEE 754 语义）。收窄后双方同为同号 `±inf` 即视为该点通过。
