"""700-case accuracy matrix for causal_conv1d_update (rotation-plane contract).

Composition: 28 (N, D, W) shapes x 25 variants = 700 cases.

Each shape keeps one dtype (cycled over fp16/bf16/fp32) and exercises the full
12-way (activation x bias x residual) matrix at two head cursors, plus one
strided-input case. Layout (2d / packed / batch) is cycled across variants.
Only (N, D, W, dtype, HEAD, ACTIVATION, HAS_BIAS, HAS_RESIDUAL) affect
compilation; layout and strides do not, which keeps the autotune key count at
28 and the build count bounded.
"""

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
ACTIVATIONS = [None, "silu", "swish"]
LAYOUTS = ["2d", "packed", "batch"]


def _require_triton(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("causal_conv1d_update has only an NPU Triton provider")
    return implementation, target, device


# (N, D, W): PoT and non-PoT D, edge N, decode and prefill-sized batches.
SHAPES = [
    (1, 64, 4),
    (2, 128, 4),
    (3, 17, 2),
    (7, 33, 3),
    (16, 100, 4),
    (64, 256, 4),
    (256, 512, 4),
    (1024, 1024, 4),
    (2, 600, 3),
    (4, 768, 4),
    (8, 1000, 2),
    (16, 1200, 4),
    (32, 2048, 3),
    (128, 4096, 4),
    (1, 8192, 4),
    (2, 8192, 3),
    (8, 4096, 4),
    (32, 2048, 4),
    (128, 1024, 3),
    (256, 512, 2),
    (8192, 512, 4),
    (4096, 1024, 4),
    (2048, 2048, 3),
    (1024, 4096, 4),
    (2048, 4096, 4),
    (4096, 2048, 3),
    (8192, 2048, 3),
    (4096, 1024, 3),
]


def _build_cases():
    cases = []
    for si, (n, d, w) in enumerate(SHAPES):
        dtype = DTYPES[si % len(DTYPES)]
        vi = 0
        for activation in ACTIVATIONS:
            for bias in (False, True):
                for residual in (False, True):
                    for head in (0, w - 1):
                        cases.append(
                            (
                                n,
                                d,
                                w,
                                dtype,
                                activation,
                                bias,
                                residual,
                                head,
                                False,
                                LAYOUTS[(si * 25 + vi) % len(LAYOUTS)],
                            )
                        )
                        vi += 1
        cases.append((n, d, w, dtype, None, False, False, 0, True, LAYOUTS[(si * 25 + vi) % len(LAYOUTS)]))
    return cases


CASES = _build_cases()
assert len(CASES) == 700, len(CASES)


def make_layout(tokens, channels, layout, device, dtype, strided):
    if strided:
        base = torch.randn(tokens, channels * 2, device=device, dtype=dtype)
        flat = base[:, ::2]
    else:
        flat = torch.randn(tokens, channels, device=device, dtype=dtype)
    if layout == "2d":
        return flat
    if layout == "packed":
        return flat.unsqueeze(0)
    return flat.unsqueeze(1)


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.parametrize(
    "tokens,channels,width,dtype,activation,bias,residual,head,strided,layout",
    CASES,
    ids=[
        f"{n}_{d}_{w}_{str(dt).split('.')[-1]}_{act}_{int(b)}_{int(r)}_{h}_{int(s)}_{lay}"
        for n, d, w, dt, act, b, r, h, s, lay in CASES
    ],
)
@pytest.mark.accuracy
def test_causal_conv1d_update_accuracy_700(
    accuracy_backend, tokens, channels, width, dtype, activation, bias, residual, head, strided, layout
):
    implementation, _, device = _require_triton(accuracy_backend)
    x = make_layout(tokens, channels, layout, device, dtype, strided)
    weight = torch.randn(channels, width, device=device, dtype=dtype)
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
    # only plane[head] changes, and it is an exact copy of x
    assert_close(actual_planes, expected_planes, rtol=0, atol=0)
    assert actual_planes.data_ptr() == ptr
    assert planes.data_ptr() == ptr
