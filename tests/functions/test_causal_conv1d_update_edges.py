"""Edge coverage for functions.causal_conv1d_update.

Complements the 700-case matrix with the dimensions that only the A2 delivery
exercised: widths above 4, the cache-only path (``weight=None``), extreme
hidden sizes and single-sequence batches.

The rotation state is ``cache[W, N, D]``; ``head`` names the plane that
receives ``x`` and ``cache[(head + k) % W]`` holds the k-th oldest token.
"""

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32

LAYOUTS = ["2d", "packed", "batch"]


def _require_triton(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("causal_conv1d_update has only an NPU Triton provider")
    return implementation, target, device


def _build_cases():
    """(N, D, W, dtype, activation, bias, residual, head, layout, weighted)."""
    cases = []

    # width sweep above the production W in {2, 3, 4}
    for w in (5, 6, 8):
        for dt in (BF16, FP16):
            cases.append((4, 256, w, dt, None, False, False, 0, "2d", True))

    # cache-only path: no conv weight, the window is summed
    for dt in (BF16, FP16, FP32):
        for act in (None, "silu"):
            cases.append((4, 256, 4, dt, act, False, False, 0, "2d", False))

    # extreme hidden sizes (masked BD tail / single row)
    for d in (1, 3, 7, 513, 1023, 2047, 4096, 16384):
        cases.append((1, d, 4, BF16, "silu", False, False, 0, "2d", True))

    # sequence-count edges
    for n in (1, 2, 3, 5, 7, 17, 65):
        cases.append((n, 512, 4, FP16, None, False, False, 0, "2d", True))

    # non-2d layouts with the wider ring
    for w in (5, 8):
        for layout in LAYOUTS:
            cases.append((3, 128, w, BF16, "silu", False, False, 0, layout, True))

    # bias / residual on the wider ring
    for w in (5, 6):
        for dt in (BF16, FP16):
            cases.append((8, 320, w, dt, "swish", True, True, w - 1, "packed", True))

    # fp32 on the widest ring
    cases.append((2, 96, 8, FP32, "silu", True, False, 3, "batch", True))

    return cases


CASES = _build_cases()


def _make_layout(tokens, channels, layout, device, dtype):
    flat = torch.randn(tokens, channels, device=device, dtype=dtype)
    if layout == "2d":
        return flat
    if layout == "packed":
        return flat.unsqueeze(0)
    return flat.unsqueeze(1)


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.parametrize(
    "tokens,channels,width,dtype,activation,bias,residual,head,layout,weighted",
    CASES,
    ids=[
        f"N{n}_D{d}_W{w}_{str(dt).split('.')[-1]}_{act}_{int(b)}_{int(r)}_{h}_{lay}{'' if wt else '_noweight'}"
        for n, d, w, dt, act, b, r, h, lay, wt in CASES
    ],
)
@pytest.mark.accuracy
def test_causal_conv1d_update_edges(
    accuracy_backend, tokens, channels, width, dtype, activation, bias, residual, head, layout, weighted
):
    implementation, _, device = _require_triton(accuracy_backend)
    x = _make_layout(tokens, channels, layout, device, dtype)
    weight = torch.randn(channels, width, device=device, dtype=dtype) if weighted else None
    b = torch.randn(channels, device=device, dtype=dtype) if bias else None
    r = torch.randn_like(x) if residual else None
    planes = torch.randn(width, tokens, channels, device=device, dtype=dtype)
    reference_planes = planes.clone()
    ptr = planes.data_ptr()

    actual, actual_planes = F.causal_conv1d_update(
        x,
        planes,
        residual=r,
        weight=weight,
        bias=b,
        activation=activation,
        head=head,
        implementation=implementation,
    )
    expected, expected_planes = F.causal_conv1d_update(
        x,
        reference_planes,
        residual=r,
        weight=weight,
        bias=b,
        activation=activation,
        head=head,
        implementation="torch",
    )

    assert actual.shape == x.shape
    assert_close(actual, expected, dtype)
    assert_close(actual_planes, expected_planes, rtol=0, atol=0)
    assert actual_planes.data_ptr() == ptr
    assert planes.data_ptr() == ptr
