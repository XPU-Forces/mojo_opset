"""CPU contracts for the public API and framework-independent RNG reservation."""

import importlib.util

from pathlib import Path

import pytest
import torch

from mojo_opset import functions as F

pytestmark = pytest.mark.reference


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"scale_alg": True}, TypeError),
        ({"scale_alg": 1.0}, TypeError),
        ({"block_size": 32.0}, TypeError),
        ({"dst_type": 24.0}, TypeError),
        ({"dst_type_max": False}, TypeError),
        ({"max_low_bound": 1e100}, ValueError),
    ],
)
def test_reference_contract_rejects_invalid_attributes(kwargs, error):
    with pytest.raises(error):
        F.dynamic_mx_quant_sr(torch.ones(2, 64), implementation="torch", **kwargs)


@pytest.mark.parametrize(
    "x,axis",
    [
        (torch.ones(2, 64).t(), 0),
        (torch.ones(2, 16), -1),
        (torch.ones(2, 17), -1),
    ],
)
def test_reference_contract_rejects_invalid_input(x, axis):
    with pytest.raises(ValueError):
        F.dynamic_mx_quant_sr(x, axis=axis, implementation="torch")


@pytest.mark.parametrize("numel,want", [(1024, 4), (262144, 8), (2097152, 36), (33554432, 516)])
def test_rng_reserves_word_intervals_including_padded_batches(numel, want):
    path = Path(__file__).parents[2] / "mojo_opset/kernels/npu_a5_cannbotdsl/_rng_state.py"
    spec = importlib.util.spec_from_file_location("sr_rng_contract", path)
    rng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rng)
    assert rng._rng_increment(numel) == want
    # Each stream receives enough blocks for all groups, including partial SIMD batches.
    assert want // 4 * 16384 >= numel // 16 + 63


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst,code,scale", [(24, 0x78, 119), (23, 0x78, 112)])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_public_reference_encodes_exact_groups_and_pads_scale_pairs(dtype, dst, code, scale, algorithm):
    x = torch.ones(2, 48, dtype=dtype)
    y, scales = F.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=algorithm, implementation="torch")
    assert y.shape == x.shape
    assert scales.shape == (2, 1, 2)
    assert torch.equal(y.view(torch.uint8), torch.full((2, 48), code, dtype=torch.uint8))
    assert torch.equal(scales.view(torch.uint8), torch.full((2, 1, 2), scale, dtype=torch.uint8))
    if dtype == torch.bfloat16:
        _, tail_scales = F.dynamic_mx_quant_sr(
            torch.ones(2, 16, dtype=dtype), dst_type=dst, scale_alg=algorithm, implementation="torch"
        )
        assert torch.equal(tail_scales.view(torch.uint8), torch.tensor([[[scale, 0]], [[scale, 0]]], dtype=torch.uint8))


def test_invalid_input_is_rejected_before_loading_provider(monkeypatch):
    from importlib import import_module

    api = import_module("mojo_opset.functions.dynamic_mx_quant_sr")

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid input reached dispatch")

    monkeypatch.setattr(api, "load_impl", unexpected)
    with pytest.raises(ValueError, match="contiguous"):
        api.dynamic_mx_quant_sr(torch.ones(2, 64).t(), axis=0)


def test_subnormal_zero_and_invalid_block_semantics():
    x = torch.zeros(2, 64)
    x[0, 1::2] = -0.0
    x[1, 0] = float("inf")
    y, scale = F.dynamic_mx_quant_sr(x, implementation="torch")
    assert torch.equal(y[0].view(torch.uint8), torch.tensor([0, 128] * 32, dtype=torch.uint8))
    assert torch.equal(y[1, :32].view(torch.uint8), torch.full((32,), 127, dtype=torch.uint8))
    assert torch.equal(scale.view(torch.uint8), torch.tensor([[[0, 0]], [[255, 0]]], dtype=torch.uint8))


@pytest.mark.parametrize("selected", ["torch", "cannbotdsl"])
def test_performance_entry_respects_configured_provider(monkeypatch, selected):
    from mojo_opset.functions import _dispatch
    from tests.performance.functions import test_dynamic_mx_quant_sr as bench

    monkeypatch.setattr(_dispatch, "resolve_implementation", lambda op, target: selected)
    backend = bench.dynamic_mx_quant_sr_perf_backend.__wrapped__((None, "npu:0", "npu.a5", None))
    assert backend == (selected, "npu:0")
