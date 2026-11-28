import pytest
import torch

from mojo_opset import modules
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize


# Module-level GmmSwigluQuant workloads at the glm-5.3 / glm-5.3-flash
# model shapes: dense MLP (12288 intermediate) and shared experts (2048).
@pytest.mark.api("modules.GmmSwigluQuant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize(
    "tokens,hidden,intermediate,limit",
    [
        (1, 6144, 12288, 0.0),  # glm-5.3 dense decode
        (512, 6144, 12288, 0.0),  # glm-5.3 dense prefill
        (1, 4096, 12288, 10.0),  # glm-5.3-flash dense decode
        (512, 4096, 12288, 10.0),  # glm-5.3-flash dense prefill
        (64, 6144, 2048, 0.0),  # glm-5.3 shared experts
        (512, 4096, 2048, 10.0),  # glm-5.3-flash shared experts prefill
    ],
)
def test_gmm_swiglu_quant(benchmark, perf_environment, tokens, hidden, intermediate, limit):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.GmmSwigluQuant(hidden, intermediate, limit, implementation=implementation, device=device)
        x, x_scale = mxfp8_quantize(torch.randn(tokens, hidden, device=device))
        return lambda: module(x, x_scale)

    benchmark(
        factory=factory,
        op="gmm_swiglu_quant",
        shape=[tokens, hidden, intermediate],
        dtype="float8e4m3fn",
        phase="forward",
        swiglu_limit=limit,
    )
