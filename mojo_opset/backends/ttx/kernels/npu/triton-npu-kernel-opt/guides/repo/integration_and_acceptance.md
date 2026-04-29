# Integration And Acceptance

This file maps the repo paths and concrete checks needed to land a new NPU
kernel in `mojo_opset`.

## 1. Check the semantic owner first

Determine whether the API belongs to:

- `mojo_opset/core/operators/`
- `mojo_opset/core/functions/`

Do not start from backend code alone. The `core` layer defines the public
contract and the reference path.

If the matching `core` entry does not exist, stop and ask the user for the API
contract first.

## 2. Confirm the reference implementation

Preferred reference order:

1. `torch` backend implementation
2. `torch_npu` backend implementation
3. mathematically equivalent reference code if neither backend exists

If there is no trustworthy reference, do not claim correctness. Ask the user to
provide the missing reference or approve adding one first.

Common locations:

- `mojo_opset/backends/torch_npu/operators/`
- `mojo_opset/core/operators/`
- `mojo_opset/core/functions/`

## 3. Typical code touch points

Depending on the operator, a complete change may involve:

- `mojo_opset/backends/ttx/kernels/npu/<kernel>.py`
- `mojo_opset/backends/ttx/kernels/npu/__init__.py`
- `mojo_opset/backends/ttx/operators/<family>.py`
- `mojo_opset/backends/torch_npu/operators/<family>.py`
- `mojo_opset/core/operators/<family>.py`
- `mojo_opset/core/functions/<family>.py`

Only touch `core` if the contract itself must change. For a backend-only
implementation, prefer leaving `core` semantics intact.

## 4. Accuracy validation

Use the existing test suites whenever possible:

- `mojo_opset/tests/accuracy/operators/`
- `mojo_opset/tests/accuracy/functions/`

Useful existing examples:

- `mojo_opset/tests/accuracy/operators/test_gemm.py`
- `mojo_opset/tests/accuracy/operators/test_linear.py`
- `mojo_opset/tests/accuracy/operators/test_attention.py`
- `mojo_opset/tests/accuracy/operators/test_normalization.py`

Recommended command pattern:

```bash
cd <repo-root>
MOJO_BACKEND=ttx pytest mojo_opset/tests/accuracy/operators/test_gemm.py -q
```

Adapt the file to the operator family you changed.

What to cover:

- normal production shapes
- degenerate and small shapes
- tail dimensions
- layout variants such as `trans_weight`
- fused branches such as optional bias/residual
- dtype variants that are supposed to work

## 5. Graph/API correctness

If the kernel is exposed through `torch.ops.ttx.*`, run or add opcheck tests in:

- `mojo_opset/tests/test_ttx_graph/`

Examples:

- `mojo_opset/tests/test_ttx_graph/test_activation.py`
- `mojo_opset/tests/test_ttx_graph/test_norm.py`
- `mojo_opset/tests/test_ttx_graph/test_attn.py`
- `mojo_opset/tests/test_ttx_graph/test_rope.py`

Recommended command pattern:

```bash
cd <repo-root>
pytest mojo_opset/tests/test_ttx_graph/test_activation.py -q
```

## 6. Performance validation

Use the existing perf suite first:

- `mojo_opset/tests/perf/`

Useful examples:

- `mojo_opset/tests/perf/test_gemm_dequant.py`
- `mojo_opset/tests/perf/test_attention.py`
- `mojo_opset/tests/perf/test_normalization.py`
- `mojo_opset/tests/perf/test_linear.py`

Recommended command pattern:

```bash
cd <repo-root>
MOJO_BACKEND=ttx pytest mojo_opset/tests/perf/test_gemm_dequant.py -q -s
```

The perf helpers inject the proper platform-specific timing path through
`auto_switch_platform(set_perf=True)`.

Benchmark notes:

- `mojo_opset/tests/perf/benchmark.md` is reorganized automatically at session end
- use `torch_npu` native kernels as the main baseline when they exist
- for GEMM-like kernels, also compare against any repo-specific prior `ttx`
  implementation you are replacing

## 7. Acceptance checklist

Do not mark the work complete until you can answer yes to each item:

- Is there a stable `core` contract?
- Is there a trusted reference path?
- Did accuracy tests pass on representative and boundary shapes?
- Did opcheck pass for exported `torch.ops.ttx.*` APIs?
- Did performance testing run on target NPU hardware?
- Is any regression against `torch_npu` or prior `ttx` understood and documented?

## 8. If the kernel underperforms

Do not hide it behind vague wording. Report:

- shapes where it underperforms
- baseline used
- measured gap
- likely cause
- whether the kernel should still ship for coverage or remain behind a guarded path
