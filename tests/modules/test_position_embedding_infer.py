import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

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


@pytest.mark.api("modules.ApplyRoPEInfer", ops=["apply_rope_infer"])
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
    op = M.ApplyRoPEInfer(head_first=head_first, implementation=impl)
    expected = F.apply_rope_infer(*inputs, head_first=head_first, implementation="torch_reference")
    actual = op(*inputs)
    for a, e in zip(actual, expected):
        assert_close(a, e, heads[0], rtol=5e-2, atol=5e-2)


@pytest.mark.api("modules.ApplyVisionRoPE2DInfer", ops=["apply_vision_rope2d_infer"])
@pytest.mark.accuracy
@pytest.mark.parametrize("grid", VISION_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vision(accuracy_backend, grid, dtype):
    impl, _, device = accuracy_backend
    inputs = make_vision(grid, dtype, device)
    op = M.ApplyVisionRoPE2DInfer(implementation=impl)
    expected = F.apply_vision_rope2d_infer(*inputs, implementation="torch_reference")
    actual = op(*inputs)
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype, rtol=5e-2, atol=5e-2)
