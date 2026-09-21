from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._compile import compile_fullgraph

ROPE_HEADS = [
    (torch.float16, 32, 8, True, 96, 1.0),
    (torch.bfloat16, 8, 2, False, 96, 1 / 3),
    (torch.float16, 16, 8, True, 128, 1.0),
    (torch.bfloat16, 64, 8, False, 88, 1.0),
    (torch.float16, 64, 4, True, 128, 0.375),
]


ROPE_MODES = ["padding_prefill_pos2d", "padding_prefill_pos3d", "varlen_prefill", "decode"]


ROPE_SHAPES = [(1, 124), (6, 555), (2, 2048)]


VISION_GRIDS = [((4, 4),), ((8, 6),), ((8, 8), (4, 6))]


def make_rope(shape, heads, mode, device):
    batch, length = shape
    dtype, q_heads, k_heads, head_first, head_dim, fraction = heads
    dim = int(head_dim * fraction)
    if mode.startswith("padding"):
        q = torch.randn(batch, length, q_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, length, k_heads, head_dim, dtype=dtype, device=device)
        pos = torch.arange(length, device=device)
        if mode.endswith("pos3d"):
            pos = pos[None, :] + torch.randint(0, 32768 - length - 1, (batch, 1), device=device)
        if head_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
    else:
        if mode == "decode":
            tokens = batch
            pos = torch.randint(0, 32767, (batch,), device=device)
        else:
            lengths = torch.randint(1, length + 1, (batch,))
            tokens = int(lengths.sum())
            offsets = torch.randint(0, 32768 - length, (batch,))
            pos = torch.cat(
                [torch.arange(int(size)) + int(offset - size) for size, offset in zip(lengths, offsets)]
            ).to(device)
            pos %= 32768
        q = torch.randn(tokens, q_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(tokens, k_heads, head_dim, dtype=dtype, device=device)
        if head_first:
            q, k = q.transpose(0, 1), k.transpose(0, 1)
    inv = 1 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    angles = pos[..., None] * inv
    angles = torch.cat((angles, angles), dim=-1)
    return q, k, angles.cos(), angles.sin()


def make_vision(grid, dtype, device):
    # Original ViT profile: 20 heads, head_dim=64, 2x2 adapooling regrouping.
    positions = []
    for height, width in grid:
        h = torch.arange(height).unsqueeze(1).expand(-1, width)
        w = torch.arange(width).unsqueeze(0).expand(height, -1)
        h = h.reshape(height // 2, 2, width // 2, 2).permute(0, 2, 1, 3).flatten()
        w = w.reshape(height // 2, 2, width // 2, 2).permute(0, 2, 1, 3).flatten()
        positions.append(torch.stack((h, w), dim=-1))
    pos = torch.cat(positions).to(device)
    inv = 1 / (10000 ** (torch.arange(0, 32, 2, device=device, dtype=torch.float32) / 32))
    angles = (pos[..., None] * inv).flatten(-2)
    angles = torch.cat((angles, angles), dim=-1)
    q = torch.randn(pos.shape[0], 20, 64, dtype=dtype, device=device)
    return q, torch.randn_like(q), angles.cos(), angles.sin()


@pytest.mark.api("functions.rope_infer")
@pytest.mark.accuracy
@pytest.mark.parametrize("shape", ROPE_SHAPES)
@pytest.mark.parametrize("heads", ROPE_HEADS)
@pytest.mark.parametrize("mode", ROPE_MODES)
def test_rope(accuracy_backend, shape, heads, mode):
    impl, _, device = accuracy_backend
    inputs = make_rope(shape, heads, mode, device)
    head_first = heads[3]
    dim = inputs[2].shape[-1]
    if (
        impl == "torch_npu"
        and shape[0] > 1
        and mode.startswith("padding")
        and head_first
        and dim // 2 * inputs[0].element_size() % 32
    ):
        pytest.skip("Original torch_npu BNSD rotary half-dimension alignment restriction")
    op = partial(F.rope_infer, head_first=head_first, implementation=impl)
    expected = F.rope_infer(*inputs, head_first=head_first, implementation="torch_reference")
    actual = op(*inputs)
    for a, e in zip(actual, expected):
        assert_close(a, e, heads[0], rtol=5e-2, atol=5e-2)


@pytest.mark.api("functions.rope_infer")
@pytest.mark.accuracy
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize(
    "head_first,contiguous",
    [
        pytest.param(False, (True, True), id="sequence-first"),
        pytest.param(True, (True, True), id="head-first-contiguous"),
        pytest.param(True, (False, False), id="head-first-transposed"),
        pytest.param(True, (True, False), id="head-first-mixed-q"),
        pytest.param(True, (False, True), id="head-first-mixed-k"),
    ],
)
def test_a5_rope_infer_fake_and_compiled_layout(accuracy_backend, ndim, head_first, contiguous):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("A5 Triton inference RoPE layout contract")
    preload("rope_infer", implementation="triton")
    inputs = []
    for heads, make_contiguous in zip((4, 2), contiguous):
        shape = (2, 7, heads, 64) if ndim == 4 else (7, heads, 64)
        value = torch.randn(shape, dtype=torch.bfloat16, device=device)
        if head_first:
            value = value.transpose(-3, -2)
            if make_contiguous:
                value = value.contiguous()
        inputs.append(value)
    angles = torch.randn((2, 7, 32) if ndim == 4 else (7, 32), device=device)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos(), angles.sin()
    args = (*inputs, cos, sin)
    torch.library.opcheck(
        torch.ops.mojo_npu_triton_a5.rope_infer_fwd.default,
        (*args, head_first, False),
        test_utils=("test_schema", "test_faketensor"),
    )

    def run(q, k, cos, sin):
        outputs = F.rope_infer(q, k, cos, sin, head_first=head_first, implementation="triton")
        return outputs, tuple(output.reshape(-1) for output in outputs), tuple(
            (output.stride(), output.is_contiguous()) for output in outputs
        )

    try:
        eager, flattened, metadata = run(*args)
        assert metadata == tuple((value.stride(), value.is_contiguous()) for value in inputs)
        compiled = compile_fullgraph(run)
        actual, actual_flattened, captured_metadata = compiled(*args)
        assert captured_metadata == metadata
        assert captured_metadata == tuple((value.stride(), value.is_contiguous()) for value in actual)
        for result, expected in zip((*actual, *actual_flattened), (*eager, *flattened)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.vision_rope_2d_infer")
@pytest.mark.accuracy
@pytest.mark.parametrize("grid", VISION_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vision(accuracy_backend, grid, dtype):
    impl, _, device = accuracy_backend
    inputs = make_vision(grid, dtype, device)
    op = partial(F.vision_rope_2d_infer, implementation=impl)
    expected = F.vision_rope_2d_infer(*inputs, implementation="torch_reference")
    actual = op(*inputs)
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype, rtol=5e-2, atol=5e-2)
