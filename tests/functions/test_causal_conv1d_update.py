import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

# tokens, channels, width, activation, bias, residual
UPDATE_CASES = [
    (1, 64, 3, None, False, False),
    (2, 128, 4, "silu", True, True),
    (4, 256, 4, "swish", True, False),
    (8, 1024, 3, None, True, True),
    (16, 4096, 4, "silu", False, True),
    (1, 8192, 4, None, False, False),
    (2, 17, 2, "silu", True, True),
    (3, 33, 5, None, False, False),
]

LAYOUTS = ["2d", "packed", "batch"]


def _require_triton(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("causal_conv1d_update has only an NPU Triton provider")
    return implementation, target, device


def make_layout(tokens, channels, layout, device, dtype):
    flat = torch.randn(tokens, channels, device=device, dtype=dtype)
    if layout == "2d":
        return flat
    if layout == "packed":
        return flat.unsqueeze(0)
    return flat.unsqueeze(1)


def make_planes(tokens, channels, width, device, dtype):
    """Rotation planes [W, N, D] built from a natural-order [N, D, W] window."""
    cache = torch.randn(tokens, channels, width, device=device, dtype=dtype)
    return cache.permute(2, 0, 1).contiguous()


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,channels,width,activation,bias,residual", UPDATE_CASES)
@pytest.mark.accuracy
def test_causal_conv1d_update_accuracy(
    accuracy_backend, layout, dtype, tokens, channels, width, activation, bias, residual
):
    implementation, _, device = _require_triton(accuracy_backend)
    x = make_layout(tokens, channels, layout, device, dtype)
    weight = torch.randn(channels, width, device=device, dtype=dtype)
    b = torch.randn(channels, device=device, dtype=dtype) if bias else None
    r = torch.randn_like(x) if residual else None
    planes = make_planes(tokens, channels, width, device, dtype)
    reference_planes = planes.clone()
    ptr = planes.data_ptr()

    actual, actual_planes = F.causal_conv1d_update(
        x,
        planes,
        residual=r,
        weight=weight,
        bias=b,
        activation=activation,
        head=0,
        implementation=implementation,
    )
    expected, expected_planes = F.causal_conv1d_update(
        x,
        reference_planes,
        residual=r,
        weight=weight,
        bias=b,
        activation=activation,
        head=0,
        implementation="torch",
    )

    assert actual.shape == x.shape
    assert_close(actual, expected, dtype)
    # only plane[head] changes, and it is an exact copy of x
    assert_close(actual_planes, expected_planes, rtol=0, atol=0)
    assert actual_planes.data_ptr() == ptr
    assert planes.data_ptr() == ptr


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.accuracy
def test_causal_conv1d_update_hand_computed(accuracy_backend):
    implementation, _, device = _require_triton(accuracy_backend)
    x = torch.tensor([[1.0, 2.0]], device=device)
    cache = torch.tensor([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]], device=device)
    planes = cache.permute(2, 0, 1).contiguous()  # [3, 1, 2]
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

    y, new_planes = F.causal_conv1d_update(x, planes, weight=weight, head=0, implementation=implementation)

    # head=0 receives x; the window is plane[1], plane[2], x.
    expected_planes = torch.tensor([[[1.0, 2.0]], [[20.0, 50.0]], [[30.0, 60.0]]], device=device)
    expected_y = torch.tensor([[20.0 + 60.0 + 3.0, 200.0 + 300.0 + 12.0]], device=device)
    assert_close(new_planes, expected_planes, rtol=0, atol=0)
    assert_close(y, expected_y, rtol=0, atol=0)


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_causal_conv1d_update_streaming_matches_conv(accuracy_backend, dtype):
    implementation, _, device = _require_triton(accuracy_backend)
    channels, width, tokens = 17, 4, 9
    x = torch.randn(1, tokens, channels, device=device, dtype=dtype)
    weight = torch.randn(channels, width, device=device, dtype=dtype)
    planes = torch.zeros(width, 1, channels, device=device, dtype=dtype)
    head = 0

    outputs = []
    for t in range(tokens):
        y, planes = F.causal_conv1d_update(
            x[:, t : t + 1], planes, weight=weight, head=head, implementation=implementation
        )
        outputs.append(y)
        head = (head + 1) % width
    streamed = torch.cat(outputs, dim=1)

    expected = torch.nn.functional.conv1d(
        x.float().transpose(1, 2), weight.float()[:, None], padding=width - 1, groups=channels
    )[..., :tokens].transpose(1, 2)
    assert_close(streamed, expected.to(dtype), dtype)


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.accuracy
def test_causal_conv1d_update_strided_input(accuracy_backend):
    implementation, _, device = _require_triton(accuracy_backend)
    channels, width, tokens = 32, 4, 5
    base = torch.randn(tokens, channels * 2, device=device)
    x = base[:, ::2]  # [tokens, channels] with stride (2 * channels, 2)
    assert not x.is_contiguous()
    weight = torch.randn(channels, width, device=device)
    planes = torch.randn(width, tokens, channels, device=device)
    reference_planes = planes.clone()

    actual, _ = F.causal_conv1d_update(x, planes, weight=weight, implementation=implementation)
    expected, _ = F.causal_conv1d_update(x, reference_planes, weight=weight, implementation="torch")
    assert_close(actual, expected)
    assert_close(planes, reference_planes, rtol=0, atol=0)


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.accuracy
def test_causal_conv1d_update_rejects_grad(accuracy_backend):
    implementation, _, device = _require_triton(accuracy_backend)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    planes = torch.randn(3, 2, 8, device=device)
    weight = torch.randn(8, 3, device=device)
    with pytest.raises(RuntimeError):
        F.causal_conv1d_update(x, planes, weight=weight, implementation=implementation)
