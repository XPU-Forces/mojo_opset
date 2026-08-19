import pytest
import torch
from mojo_opset import MojoApplyVisionRoPE2D
from mojo_opset import MojoVisionRotaryEmbedding2D
from mojo_opset import MojoRotaryEmbedding
from mojo_opset import MojoApplyRoPE
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented

VISION_VIT_CONFIG = {
    "img_size": 448,
    "patch_size": 14,
    "embed_dim": 1280,
    "depth": 27,
    "num_heads": 20,
    "head_dim": 64,
    "mlp_ratio": 4,
    "mlp_intermediate_dim": 5120,
    "rope_theta": 10000.0,
    "adapooling_factor": 2,
}


@pytest.mark.parametrize("bs", [32])
@pytest.mark.parametrize("seqlen", [8192])
@pytest.mark.parametrize(
    "q_heads, k_heads",
    [
        (32, 32),
        (32, 8),
        (16, 1),
        (1, 1),
    ],
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, dtype):
    device = get_torch_device()
    x = torch.randn(bs, seqlen, q_heads * head_dim, device=device, dtype=dtype)
    rot_pos_emb = MojoRotaryEmbedding(
        rope_theta=10000.0, rope_dim=head_dim, init_max_length=seqlen,
    ).to(device)
    cos, sin = rot_pos_emb(x)

    # [B, S, N, D] -> [B, N, S, D]
    q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype).transpose(1, 2)
    k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype).transpose(1, 2)

    rope = MojoApplyRoPE()

    perf(lambda: rope(q, k, cos, sin, head_first=True))  # noqa: F821

@pytest.mark.parametrize(
    "grid",
    [
        ((4, 4),),
        ((8, 6),),
        ((8, 8), (4, 6)),
    ],
)
@pytest.mark.parametrize(
    "vision_config",
    [
        pytest.param(VISION_VIT_CONFIG, id="vision_448_27l_20h_h64"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_apply_vision_rope_2d(grid, vision_config, dtype):
    device = get_torch_device()
    rope_theta = vision_config["rope_theta"]
    q_heads = vision_config["num_heads"]
    k_heads = vision_config["num_heads"]
    head_dim = vision_config["head_dim"]
    adapooling_factor = vision_config["adapooling_factor"]

    grid_hw = torch.tensor(grid, device=device, dtype=torch.int32)
    total_tokens = int(grid_hw.to(dtype=torch.int64).prod(dim=-1).sum().item())
    q = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, k_heads, head_dim, device=device, dtype=dtype)
    rot_pos_emb_ref = MojoVisionRotaryEmbedding2D._registry.get("torch")(
        rope_theta=rope_theta,
        rope_dim=head_dim,
        adapooling_factor=adapooling_factor,
    ).to(device)
    cos, sin = rot_pos_emb_ref(grid_hw)

    rope = MojoApplyVisionRoPE2D()

    perf(lambda: rope(q, k, cos, sin))
