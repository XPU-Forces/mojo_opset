# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Case matrix adapted for Ascend NPU by the mojo_opset contributors.
import pytest
import torch

from mojo_opset import functions

# Kernel-only forward workload: the mHC post block runs once per decoder
# layer seam, updating BF16 residual streams from a BF16 sublayer output
# with FP32 post/comb gates from the matching mHC pre block.
MHC_POST_CASES = [
    # GLM-5.3-Flash serving scenarios (mhc=true, hc_mult=4, hidden=4096):
    # batched decode, speculative verification (256 sequences x 5 tokens) and
    # chunked prefill around and beyond the 8192-token eval context length.
    pytest.param(256, 4, 4096, id="glm53flash-decode-256"),
    pytest.param(1280, 4, 4096, id="glm53flash-verify-1280"),
    pytest.param(2048, 4, 4096, id="glm53flash-chunked-prefill-2048"),
    pytest.param(8192, 4, 4096, id="glm53flash-chunked-prefill-8192"),
    # DeepSeek V4.1 serving scenarios (hc_mult=4, hidden=5120): batched
    # decode, DSpark verification and chunked prefill, mirroring the vLLM
    # mHC suite's V4.1 shapes (tests/kernels/mhc/test_mhc_kernels.py).
    pytest.param(256, 4, 5120, id="v41-decode-256"),
    pytest.param(1280, 4, 5120, id="v41-dspark-verify-1280"),
    pytest.param(2048, 4, 5120, id="v41-chunked-prefill-2048"),
    pytest.param(8192, 4, 5120, id="v41-chunked-prefill-8192"),
    # DeepSeek V4-Pro serving scenarios (hc_mult=4, hidden=7168): batched
    # decode and chunked prefill. V4-Flash (hidden=4096) shares the GLM-5.3-
    # Flash shape class above.
    pytest.param(256, 4, 7168, id="v4pro-decode-256"),
    pytest.param(2048, 4, 7168, id="v4pro-chunked-prefill-2048"),
]


@pytest.mark.api("functions.mhc_post")
@pytest.mark.parametrize("num_tokens,hc_mult,hidden_size", MHC_POST_CASES)
def test_mhc_post(benchmark, perf_environment, num_tokens, hc_mult, hidden_size):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
        post = torch.rand(num_tokens, hc_mult, 1, dtype=torch.float32, device=device)
        comb = torch.rand(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device)
        return lambda: functions.mhc_post(x, residual, post, comb, implementation=implementation)

    benchmark(
        factory=factory,
        op="mhc_post",
        num_tokens=num_tokens,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        dtype="bfloat16",
        phase="forward",
    )
