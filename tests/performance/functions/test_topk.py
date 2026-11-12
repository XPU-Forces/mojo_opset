import pytest
import torch

from mojo_opset import functions


@pytest.mark.api("functions.topk_infer")
@pytest.mark.parametrize(
    "shape,k,sorted_index,return_value",
    [
        ((8, 640), 8, False, True),
        ((64, 8192), 512, False, True),
        ((1, 131072), 512, False, False),
        ((8, 131072), 512, False, True),
        ((1, 1048576), 512, False, False),
        ((8, 65536), 1024, False, True),
        ((4, 32768), 4096, False, True),
        ((4, 65536), 513, True, True),
    ],
)
def test_topk_infer(benchmark, perf_environment, shape, k, sorted_index, return_value):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        return lambda: functions.topk_infer(
            x, k, sorted_index=sorted_index, return_value=return_value, implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="topk_infer",
        shape=list(shape),
        k=k,
        dtype=str(torch.bfloat16),
        sorted_index=sorted_index,
        return_value=return_value,
    )


@pytest.mark.api("functions.topk_infer")
@pytest.mark.parametrize("rows", [1024, 8192, 16384, 32768])
def test_m15_topk_infer(benchmark, perf_environment, rows):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(rows, 640, dtype=torch.float32, device=device).softmax(dim=-1)
        return lambda: functions.topk_infer(x, 8, sorted=True, indices_type=torch.int64, implementation=implementation)

    benchmark(
        factory=factory,
        op="topk_infer",
        shape=[rows, 640],
        k=8,
        dtype=str(torch.float32),
        sorted=True,
        indices_type=str(torch.int64),
    )


@pytest.mark.api("functions.topk_infer")
@pytest.mark.parametrize("shape,k", [((17, 33), 32), ((1024, 641), 16), ((1024, 1024), 32)])
def test_fp32_rescan(benchmark, perf_environment, shape, k):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=torch.float32, device=device)
        return lambda: functions.topk_infer(x, k, sorted=True, indices_type=torch.int64, implementation=implementation)

    benchmark(factory=factory, op="topk_infer", shape=list(shape), k=k,
              dtype=str(torch.float32), sorted=True, indices_type=str(torch.int64))
