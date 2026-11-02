# mojo_opset Community Ecosystem Delivery Operator Accuracy Acceptance Standard

## 0. Scope

This standard applies to **floating-point compute operators**. Integer compute and data-movement operators require standards defined separately according to each operator's actual business scenario, and are outside the scope of this standard.

---

## 1. Test Case Generation Rules

Operator inputs consist mainly of two parts: tensors (Tensor) and attributes (Attr). Test cases must cover the operator's typical scenarios and boundary scenarios across all scenarios, ensuring the comprehensiveness of accuracy evaluation.

### 1.1 Test Case Scale Requirements

| Requirement | Description |
|--------|------|
| **Coverage goal** | Achieve **full combination coverage** of the **data types**, **data formats**, **data dimensions**, and **attribute value ranges** the operator supports (i.e. 100% coverage of all valid combinations) |
| **Data type coverage** | Cover all data types the operator supports |

> Note: There is no fixed lower bound on the number of test cases (e.g. ≥1,000); the emphasis is on **exhaustive coverage of input combinations**, rather than mechanically requiring a number of test cases.

### 1.2 Tensor Generation Rules

| Dimension | Generation rule |
|------|----------|
| **Data dimensions** | Cover the dimension range the operator supports (1–8 dimensions); dimension values are taken from $[1, 2^{20}]$ as (powers of 2) and (powers of 2 − 1). The total number of elements must not exceed 2^34 |
| **Data formats** | Cover all data formats the operator supports (ND, NCHW, etc.) |

**Stride and combination generation**: Orthogonal combination traversal should be performed among parameter types; the value range coverage strategy is as follows:

| Distribution type | Proportion | Value range | Generation logic |
|---------|------|----------|----------|
| Uniform distribution | 50% | [-5, 5] | Random uniform sampling |
| Normal distribution | 50% | μ ∈ [-5, 5], σ ∈ [0.1, 2] | Sample μ and σ randomly, then generate |

### 1.3 Attr Generation Rules

| Parameter type | Generation rule |
|----------|----------|
| Scalar parameters | Cover all equivalence class scenarios |
| Boolean parameters | Cover True and False |
| Enum parameters | Cover all supported enum values |

Combination traversal should be performed among parameter types.

### 1.4 Special Scenario Tests

| Scenario | Description | Coverage requirement |
|------|------|----------|
| Empty Tensor | A Tensor with one dimension equal to 0 | At least once per dtype per tensor |
| Scalar Tensor | A Tensor with shape [1] | Covered for every dtype |
| Boundary test | Lower boundary: every dimension of the shape is 1; upper boundary: one dimension of the shape takes the maximum value | Fully covered |
| INF/-INF/NAN | Input element values traverse nan, inf, -inf, [-inf, inf] | Generate test cases with different shapes for every dtype and every value |

> Note: Special scenario test cases are not orthogonally combined with regular test cases.

---

## 2. Error Metrics and Pass Criteria

### 2.1 Error Metrics

This standard judges with the **Mixed Tolerance** metric, i.e. an element-wise comparison combining absolute tolerance (atol) and relative tolerance (rtol).

#### 2.1.1 Element-wise Pass Condition

For each element in the output tensor, the element is judged as passing when the following condition is met:

$$
|actual - golden| \leq atol + rtol \times |golden|
$$

Where:

- **atol** (Absolute Tolerance): guarantees a reasonable error range for small values (golden close to 0), and inherently avoids division by zero.
- **rtol** (Relative Tolerance): guarantees relative accuracy for large values.

#### 2.1.2 Overall Pass Condition

A test case is judged as passing accuracy when all finite-valued elements satisfy the element-wise condition in Section 2.1.1 and all special-valued elements satisfy the special-value rules in Section 2.3.

### 2.2 Mixed Tolerance Threshold Table

| Data type | **FLOAT16** | **BFLOAT16** | **FLOAT32** | **HiFLOAT32** | **FLOAT8 E4M3** | **FLOAT8 E5M2** |
| --- | --- | --- | --- | --- | --- | --- |
| **rtol** | 2<sup>-10</sup> (1e-3) | 2<sup>-7</sup> (7.8e-3) | 2<sup>-13</sup> (1.2e-4) | 2<sup>-10</sup> (1e-3) | 2<sup>-3</sup> (0.125) | 2<sup>-2</sup> (0.25) |
| **atol** | 2<sup>-10</sup> (1e-3) | 2<sup>-7</sup> (7.8e-3) | 2<sup>-13</sup> (1.2e-4) | 2<sup>-10</sup> (1e-3) | 2<sup>-3</sup> (0.125) | 2<sup>-2</sup> (0.25) |

### 2.3 Pass Determination

**Single-reference comparison**: direct comparison against a single accuracy reference of the Torch reference implementation.

When all elements participating in the comparison in a test case satisfy the element-wise condition in Section 2.1.1, the test case is judged as passing accuracy.

Special-value rules:

- **NaN**: when the operator's actual output is NaN, that point fails.
- **±inf**: when either side is infinity, the point passes only if both sides are `+inf` or both are `-inf`; all other cases fail.
