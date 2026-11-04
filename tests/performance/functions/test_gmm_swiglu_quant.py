import pytest
import torch

import numpy as np

from mojo_opset import functions
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize_weight


# glm-5.3 / glm-5.3-flash scenarios: dense MLP 12288 and shared experts 2048
# are E=1; routed experts (top-k 8, E=256 / 288) take rows = tokens * top-k.
@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize(
    "rows,hidden,intermediate,limit,num_experts",
    [
        (1, 6144, 12288, 0.0, None),  # glm-5.3 dense MLP decode
        (8, 4096, 2048, 10.0, 288),  # glm-5.3-flash routed decode: 1 token * top-8
        (4096, 6144, 2048, 0.0, 256),  # glm-5.3 routed prefill: 512 tokens * top-8
        (32768, 6144, 2048, 0.0, 256),  # glm-5.3 routed prefill (full batch)
        (4096, 4096, 2048, 10.0, 288),  # glm-5.3-flash routed prefill
        (32768, 4096, 2048, 10.0, 288),  # glm-5.3-flash routed prefill (full batch)
        (64, 6144, 2048, 0.0, None),  # glm-5.3 shared experts
        (512, 4096, 2048, 10.0, None),  # glm-5.3-flash shared experts prefill
    ],
)
def test_gmm_swiglu_quant(benchmark, perf_environment, rows, hidden, intermediate, limit, num_experts):
    _, device, _, implementation = perf_environment

    def factory():
        x, x_scale = mxfp8_quantize(torch.randn(rows, hidden, device=device))
        if num_experts is None:
            weight, weight_scale = mxfp8_quantize_weight(
                torch.randn(hidden, 2 * intermediate, device=device) * 0.02  # GlmMoeDsaConfig.initializer_range
            )
            weight = weight.view(1, hidden, 2 * intermediate)
            weight_scale = weight_scale.view(1, 2 * intermediate, -1)
            group_list = torch.tensor([rows], dtype=torch.int64, device=device)
        else:
            # One quantized expert broadcast to E slots (perf has no value
            # dependence) plus the realistic dispatch row split.
            weight, weight_scale = mxfp8_quantize_weight(
                torch.randn(hidden, 2 * intermediate, device=device) * 0.02
            )
            weight = weight.repeat(num_experts, 1, 1)
            weight_scale = weight_scale.repeat(num_experts, 1, 1)
            if rows <= num_experts:  # decode: top-k picks, sparse experts
                counts = [1] * rows + [0] * (num_experts - rows)
            else:  # prefill: multinomial router-style imbalance
                counts = np.random.default_rng(2026).multinomial(
                    rows, np.full(num_experts, 1.0 / num_experts)).tolist()
            group_list = torch.tensor(
                [sum(counts[: g + 1]) for g in range(num_experts)],
                dtype=torch.int64, device=device)
        inputs = (x, weight, weight_scale, x_scale, group_list)
        return lambda: functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation=implementation)

    benchmark(
        factory=factory,
        op="gmm_swiglu_quant",
        shape=[rows, hidden, intermediate, num_experts or 1],
        dtype="float8e4m3fn",
        phase="forward",
        swiglu_limit=limit,
    )
