"""Tests for functions.gmm_swiglu_quant (forward only, MXFP8 in/out)."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mojo_opset import functions
from mojo_opset import preload
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize_weight
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# Model scenarios (glm-5.3 / glm-5.3-flash lineage): dense MLP and shared
# experts are E=1; routed experts use moe_intermediate 2048 / top-k 8 with E
# scaled down for test memory. The 4th entry is the per-expert row list
# (None = dense).
GMM_SWIGLU_QUANT_SHAPES = [
    (7, 512, 320, None),  # odd tail rows/columns
    (8, 96, 32, None),  # K=96: largest pow2 factor 32 -> BLOCK_K=32 exact tiling
    (128, 512, 320, None),
    (37, 1024, 768, None),
    (1, 6144, 12288, None),  # glm-5.3 dense MLP decode
    (2048, 6144, 2048, [34, 85, 132, 135, 169, 30, 61, 157, 163, 146, 111, 150, 144, 191, 211, 129]),
    # ^ glm-5.3 routed prefill: 256 tokens * top-8, ragged dispatch over 16 experts
    (1, 4096, 12288, None),  # glm-5.3-flash dense MLP decode
    (8, 4096, 2048, [1] * 8 + [0] * 8),
    # ^ glm-5.3-flash routed decode: 1 token * top-8, sparse experts
    (7, 6144, 2048, None),  # glm-5.3 shared experts decode
    (64, 4096, 2048, None),  # glm-5.3-flash shared experts prefill
]

# 0 disables the clamp (GlmMoeDsaMLP); 10.0 is glm-5.3-flash (Glm5NextTextMLP).
GMM_SWIGLU_QUANT_LIMITS = [0.0, 10.0]

# transformers GlmMoeDsaConfig.initializer_range.
WEIGHT_STD = 0.02


def _require_fp8(target):
    """float8_e4m3fn (MX path) is A5-only on NPU; 910b (A2) has no FP8 support."""
    if target.startswith("npu.") and not target.startswith("npu.a5"):
        pytest.skip(f"gmm_swiglu_quant requires FP8 support (A5 MX path); got {target}")

# V3 numerics bound on |y * 2^e - S|, in units of the group scale 2^e
# (BF16 round-trip + one RTNE cast + no-carry saturation margin).
_QUANT_STEP_BOUND = 82.0


def _quantize(value):
    data, scales = mxfp8_quantize(value)
    return data, scales


def _gmm_inputs(tokens, hidden, intermediate, device):
    x, x_scale = _quantize(torch.randn(tokens, hidden, device=device))
    weight, weight_scale = mxfp8_quantize_weight(
        torch.randn(hidden, 2 * intermediate, device=device) * WEIGHT_STD
    )
    group_list = torch.tensor([tokens], dtype=torch.int64, device=device)
    return x, weight.view(1, hidden, 2 * intermediate), weight_scale.view(1, 2 * intermediate, -1), \
        x_scale, group_list


def _gmm_grouped_inputs(group_rows, hidden, intermediate, device):
    """E = len(group_rows) experts; group_list is the per-expert row cumsum."""
    rows = sum(group_rows)
    n = 2 * intermediate
    x, x_scale = _quantize(torch.randn(rows, hidden, device=device))
    data, scales = mxfp8_quantize(
        torch.randn(len(group_rows), n, hidden, device=device) * WEIGHT_STD)
    weight = data.transpose(1, 2).contiguous()  # [E, K, N]
    weight_scale = scales.contiguous()  # [E, N, kb] N-major
    group_list = torch.tensor([sum(group_rows[: i + 1]) for i in range(len(group_rows))],
                              dtype=torch.int64, device=device)
    return x, weight.view(len(group_rows), hidden, n), weight_scale, x_scale, group_list


def _decode_e8m0(scale_bytes):
    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def _reference_silu_half(x, weight, x_scale, weight_scale, swiglu_limit):
    """Independent oracle: FP32 dequant-GEMM + SwiGLU, pre-quantization S."""
    rows, k = x.reshape(-1, x.shape[-1]).shape
    n = weight.shape[-1]
    inter = n // 2
    xs = _decode_e8m0(x_scale.reshape(rows, -1))
    ws = _decode_e8m0(weight_scale.reshape(n, -1)).t()
    xd = x.reshape(rows, k).to(torch.float32) * xs.repeat_interleave(32, dim=1)[:, :k]
    wd = weight.reshape(k, n).to(torch.float32) * ws.repeat_interleave(32, dim=0)[:k, :]
    c = xd @ wd
    act, lin = c[:, :inter], c[:, inter:]
    if swiglu_limit > 0:
        act = act.clamp(max=swiglu_limit)
        lin = lin.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(act) * lin


def _assert_quantized_close(y, y_scale, expected):
    """Dequantized y must match the FP32 oracle within MX quantization steps."""
    rows, inter = y.shape
    scale = _decode_e8m0(y_scale.reshape(rows, -1)[:, : (inter + 31) // 32])
    # repeat_interleave, not a gather: single-group gather is wrong on torch_npu.
    scale = scale.repeat_interleave(32, dim=1)[:, :inter]
    error = (y.to(torch.float32) * scale - expected).abs()
    bound = scale * _QUANT_STEP_BOUND + 1e-6
    assert bool((error <= bound).all()), (
        f"gmm_swiglu_quant quantization error {(error / bound).max().item():.3f}x bound"
    )


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("limit", GMM_SWIGLU_QUANT_LIMITS)
@pytest.mark.parametrize("shape", GMM_SWIGLU_QUANT_SHAPES)
@pytest.mark.accuracy
def test_gmm_swiglu_quant_accuracy(accuracy_backend, limit, shape):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    tokens, hidden, intermediate, groups = shape
    if groups is None:
        inputs = _gmm_inputs(tokens, hidden, intermediate, device)
    else:
        inputs = _gmm_grouped_inputs(groups, hidden, intermediate, device)

    x, weight, weight_scale, x_scale, group_list = inputs
    starts = [0, *group_list.tolist()[:-1]]
    expected = torch.cat([
        _reference_silu_half(x[s:e], weight[g], x_scale[s:e], weight_scale[g], limit)
        for g, (s, e) in enumerate(zip(starts, group_list.tolist())) if e > s
    ], dim=0)
    y, y_scale = functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation=implementation)

    assert y.dtype is torch.float8_e4m3fn and y.shape == expected.shape
    assert y_scale.dtype is torch.uint8 and y_scale.shape == (
        tokens, (intermediate + 31) // 32,
    )
    _assert_quantized_close(y, y_scale, expected)


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize(
    "shape,limit",
    [((128, 512, 320), 0.0), ((128, 512, 320), 10.0), ((64, 4096, 2048), 10.0)],
)
@pytest.mark.bitwise
def test_gmm_swiglu_quant_bitwise(accuracy_backend, shape, limit):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    inputs = _gmm_inputs(*shape, device)
    options = {"swiglu_limit": limit}
    # Forward-only op: no differentiable outputs, so skip the backward phase.
    assert_repeatable(partial(functions.gmm_swiglu_quant, implementation=implementation, **options),
                      inputs, backward=False)


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gmm_swiglu_quant_fake_metadata(accuracy_backend, limit):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("NPU A5 Triton fake metadata")
    preload("gmm_swiglu_quant", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    inputs = _gmm_inputs(14, 32, 16, device)  # K/I tails inside one scale block
    torch.library.opcheck(ops.gmm_swiglu_quant_fwd.default, (*inputs, limit), test_utils=("test_faketensor",))


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_gmm_swiglu_quant_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("gmm_swiglu_quant", implementation=implementation)
    inputs = _gmm_inputs(18, 32, 16, device)

    def run(*args):
        outputs = functions.gmm_swiglu_quant(*args, swiglu_limit=10.0, implementation=implementation)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        _, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        expected = _reference_silu_half(inputs[0], inputs[1], inputs[3], inputs[2], 10.0)
        _assert_quantized_close(actual[0], actual[1], expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
def test_gmm_swiglu_quant_contracts(accuracy_backend):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    x, weight, weight_scale, x_scale, group_list = _gmm_inputs(5, 64, 32, device)
    with pytest.raises(ValueError):  # weight must be [1, K, N]
        functions.gmm_swiglu_quant(x, weight.view(64, 64), weight_scale, x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # group_list length must match the expert count
        functions.gmm_swiglu_quant(x, weight, weight_scale, x_scale,
                                   torch.tensor([3, 5], dtype=torch.int64, device=device),
                                   implementation=implementation)
    with pytest.raises(ValueError):  # odd N has no SwiGLU halves
        functions.gmm_swiglu_quant(x, weight[..., :63].contiguous(),
                                   weight_scale[..., :63].contiguous(), x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # e8m0 carriers are uint8/int8 only
        functions.gmm_swiglu_quant(x, weight, weight_scale.float(), x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # x must be fp8
        functions.gmm_swiglu_quant(x.to(torch.bfloat16), weight, weight_scale, x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):
        functions.gmm_swiglu_quant(x, weight, weight_scale, x_scale, group_list,
                                   swiglu_limit=-1.0, implementation=implementation)
    if implementation in (None, "triton") and target.startswith("npu.a5"):
        with pytest.raises(ValueError):  # K=100: partial trailing MX group -> invalid k-axis
            functions.gmm_swiglu_quant(*_gmm_inputs(5, 100, 32, device),
                                       implementation=implementation)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("limit", GMM_SWIGLU_QUANT_LIMITS)
def test_gmm_swiglu_quant_reference_matches_cpu_fp64(accuracy_backend, limit):
    # A CPU FP64 oracle catches vendor-specific dispatch bugs in the reference itself.
    _, target, device = accuracy_backend
    _require_fp8(target)
    tokens, hidden, intermediate = 9, 64, 48
    inputs = _gmm_inputs(tokens, hidden, intermediate, device)

    actual = functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation="torch")
    cpu_inputs = tuple(t.cpu() for t in inputs)
    x64 = cpu_inputs[0].to(torch.float64)
    w64 = cpu_inputs[1].to(torch.float64).reshape(hidden, 2 * intermediate)
    xs64 = _decode_e8m0(cpu_inputs[3].to(torch.float64)).reshape(tokens, -1)
    ws64 = _decode_e8m0(cpu_inputs[2].to(torch.float64)).reshape(2 * intermediate, -1).t()
    xd = x64 * xs64.repeat_interleave(32, dim=1)[:, :hidden]
    wd = w64 * ws64.repeat_interleave(32, dim=0)[:hidden, :]
    c = xd @ wd
    act, lin = c[:, :intermediate], c[:, intermediate:]
    if limit > 0:
        act = act.clamp(max=limit)
        lin = lin.clamp(min=-limit, max=limit)
    expected = F.silu(act) * lin

    _assert_quantized_close(actual[0].cpu(), actual[1].cpu(), expected)
