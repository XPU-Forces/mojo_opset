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

This standard uses **Mixed Tolerance** for element-wise comparison, combining absolute tolerance (`atol`) and relative tolerance (`rtol`) to determine whether the operator's actual output matches the reference result.

For corresponding elements where both the actual output `actual` and the reference result `golden` are finite, the point passes if:

$$
|actual - golden| \leq atol + rtol \times |golden|
$$

Where:

- **atol** (Absolute Tolerance): controls the absolute error, primarily when `golden` is close to zero, without dividing by the reference value.
- **rtol** (Relative Tolerance): controls the allowed error that grows with the absolute value of `golden`, primarily for larger values.

If either side is NaN or infinity, the formula above does not apply. Use the special-value rules in Section 2.3 instead.

### 2.2 Mixed Tolerance Threshold Table

| Data type | **FLOAT16** | **BFLOAT16** | **FLOAT32** | **HiFLOAT32** | **FLOAT8 E4M3** | **FLOAT8 E5M2** |
| --- | --- | --- | --- | --- | --- | --- |
| **rtol** | 2<sup>-10</sup> (1e-3) | 2<sup>-7</sup> (7.8e-3) | 2<sup>-13</sup> (1.2e-4) | 2<sup>-10</sup> (1e-3) | 2<sup>-3</sup> (0.125) | 2<sup>-2</sup> (0.25) |
| **atol** | 2<sup>-10</sup> (1e-3) | 2<sup>-7</sup> (7.8e-3) | 2<sup>-13</sup> (1.2e-4) | 2<sup>-10</sup> (1e-3) | 2<sup>-3</sup> (0.125) | 2<sup>-2</sup> (0.25) |

### 2.3 Pass Determination

**Comparison reference:** Use a single reference, comparing the operator's actual output `actual` directly against the reference result `golden` produced by the Torch reference implementation.

**Element-wise rules:**

- **Finite values:** When both sides are finite, apply the mixed tolerance formula in Section 2.1.
- **NaN:** If either side is NaN, the point fails, consistent with `equal_nan=False`.
- **±inf:** If either side is infinity, the point passes only if both sides are `+inf` or both are `-inf`; all other cases fail.

**Overall pass condition:** A test case passes accuracy validation only if every corresponding pair of elements included in the comparison passes the rules above. If any pair fails, the test case fails accuracy validation.
