"""Performance suite for compressor_from_projected (8 shapes from the task book).

The main metric is the per-call device time (profiler span). MBU is computed by
the suite: io_bytes uses the fixed formula below and the denominator is a
BF16 `copy_` bandwidth probe measured once per session, so no value is filled in
by hand. The two small shapes are below 16 MB and are graded against the torch
baseline instead; run the same file with `--mojo-implementation torch` to record
the eager composition numbers.
"""

import pytest
import torch

from mojo_opset import functions as F

STATE_RING_ROWS = 32


def _platform_cores():
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(properties, "vector_core_num", None) or properties.multi_processor_count

MATRIX = [
    ("8x8-decode", 8, 8, 128, None),
    ("32x16-decode", 32, 16, 128, None),
    ("8x1024-prefill", 8, 1024, 512, 0.60),
    ("32x512-prefill", 32, 512, 512, 0.60),
    ("8x8192-prefill", 8, 8192, 512, 0.60),
    ("256x128-requests", 256, 128, 512, 0.60),
    ("512x128-narrow", 512, 128, 128, 0.60),
    ("64x63-tail", 64, 63, 512, 0.50),
]

pytestmark = pytest.mark.api("functions.compressor_from_projected")


def _io_bytes(tokens, dim, requests, used, residual=0):
    """KV/SCORES reads (FP32), OUT writes (BF16) and ring tail traffic (FP32)."""
    return 2 * tokens * dim * 4 + tokens * dim * 2 + 8 * dim * requests * min(used, STATE_RING_ROWS) + (
        8 * dim * requests * residual
    )


@pytest.fixture(scope="module")
def copy_bandwidth(perf_environment):
    adapter, device, target, _ = perf_environment
    if not target.startswith("npu"):
        pytest.skip("compressor_from_projected benchmarks require NPU")
    source = torch.ones(1 << 26, dtype=torch.bfloat16, device="npu")
    destination = torch.empty_like(source)
    for _ in range(5):
        destination.copy_(source)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(30):
        destination.copy_(source)
    end.record()
    torch.npu.synchronize()
    moved = 2 * source.numel() * source.element_size() * 30
    return moved / (start.elapsed_time(end) * 1e-3)


@pytest.mark.parametrize("label,requests,used,dim,line", MATRIX, ids=[row[0] for row in MATRIX])
def test_compressor_from_projected(benchmark, accuracy_backend, copy_bandwidth, label, requests, used, dim, line):
    implementation, _, _ = accuracy_backend
    tokens = requests * used
    generator = torch.Generator().manual_seed(7)
    kv = torch.randn(tokens, dim, generator=generator).npu()
    scores = (torch.randn(tokens, dim, generator=generator) * 3).npu()
    state = torch.randn(requests + 2, STATE_RING_ROWS, 2 * dim, generator=generator).npu()
    out = torch.empty(tokens, dim, dtype=torch.bfloat16, device="npu")
    metadata = torch.zeros(5, requests, dtype=torch.int32)
    metadata[1] = used
    metadata[3] = torch.arange(requests, dtype=torch.int32) * used
    metadata[4] = torch.arange(1, requests + 1, dtype=torch.int32)
    controls = metadata.npu()
    moved = _io_bytes(tokens, dim, requests, used)
    cores = _platform_cores()

    def make_workload():
        def workload():
            return F.compressor_from_projected(
                kv,
                scores,
                state,
                controls,
                out,
                max_query_len=used,
                num_cores=cores,
                implementation=implementation,
            )

        return workload

    row = benchmark(
        factory=make_workload,
        op="compressor_from_projected",
        dim=dim,
        requests=requests,
        used=used,
        tokens=tokens,
        io_bytes=moved,
        mbu_line=line,
    )
    span_us = row["metrics"]["profiler_span_us"]["median"]
    mbu = moved / (span_us * 1e-6) / copy_bandwidth if span_us else 0.0
    row["parameters"]["copy_bandwidth_bytes_per_s"] = copy_bandwidth
    row["metrics"]["mbu"] = mbu
    if line is not None:
        row["metrics"]["mbu_pass"] = mbu >= line
