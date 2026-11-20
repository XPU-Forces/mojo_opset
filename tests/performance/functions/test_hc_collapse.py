# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Case matrix adapted for Ascend NPU by the mojo_opset contributors.
import pytest
import torch

from mojo_opset import functions

# Kernel-only forward workload: the mHC collapse runs once per decoder layer
# seam on BF16 streams with FP32 pre-mix gates.
HC_COLLAPSE_CASES = [
    # Original vLLM mHC kernel shapes (tests/kernels/mhc/test_mhc_kernels.py):
    # DeepSeek V4.1 hyper-connections (hc_mult=4, hidden=5120) from single
    # decode through prefill tiles, plus tail, wide-mult and odd-hidden cases.
    pytest.param(1, 4, 5120, id="orig-decode-1"),
    pytest.param(33, 4, 5137, id="orig-tail-33x5137"),
    pytest.param(128, 4, 5120, id="orig-decode-128"),
    pytest.param(1024, 4, 5120, id="orig-prefill-1024"),
    pytest.param(7, 8, 5137, id="orig-mult8-7x5137"),
    # Hidden-size variants exercised by the vLLM mHC kernel suite.
    pytest.param(128, 4, 4096, id="orig-hidden-4096"),
    pytest.param(1024, 4, 7168, id="orig-hidden-7168"),
    # DeepSeek V4.1 serving scenarios (hc_mult=4, hidden=5120): batched
    # decode, DSpark verification (256 sequences x 5 speculative tokens) and
    # chunked prefill around and beyond the 8192-token eval context length.
    pytest.param(64, 4, 5120, id="v41-decode-64"),
    pytest.param(256, 4, 5120, id="v41-decode-256"),
    pytest.param(1280, 4, 5120, id="v41-dspark-verify-1280"),
    pytest.param(2048, 4, 5120, id="v41-chunked-prefill-2048"),
    pytest.param(8192, 4, 5120, id="v41-chunked-prefill-8192"),
    pytest.param(16384, 4, 5120, id="v41-chunked-prefill-16384"),
]


@pytest.mark.api("functions.hc_collapse")
@pytest.mark.parametrize("num_tokens,hc_mult,hidden_size", HC_COLLAPSE_CASES)
def test_hc_collapse(benchmark, perf_environment, num_tokens, hc_mult, hidden_size):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
        pre = torch.rand(num_tokens, hc_mult, dtype=torch.float32, device=device)
        return lambda: functions.hc_collapse(x, pre, implementation=implementation)

    benchmark(
        factory=factory,
        op="hc_collapse",
        num_tokens=num_tokens,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        dtype="bfloat16",
        phase="forward",
    )
