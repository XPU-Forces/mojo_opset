import pytest
import torch
import torch.nn.functional as F

from mojo_opset import modules


@pytest.mark.api("modules.FusedQKNormRope", ops=["fused_qk_norm_rope"])
@pytest.mark.accuracy
@pytest.mark.parametrize("tokens,heads", [(17, 8), (257, 57)])
def test_fused_qk_norm_rope_module(accuracy_backend, tokens, heads):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    packed = torch.randn(tokens, heads * 3, 128, device=device, dtype=torch.bfloat16)
    q = packed[:, :heads]
    k = packed[:, heads : heads * 2]
    angles = torch.randn(tokens, 48, device=device)
    rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    module = modules.FusedQKNormRope(dtype=torch.bfloat16, implementation=implementation).to(device)

    actual_q, actual_k = module(q, k, rope_table)
    expected_q = _reference(q, module.q_weight, rope_table)
    expected_k = _reference(k, module.k_weight, rope_table)
    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.reference
def test_qk_norm_rope_alias():
    assert modules.QKNormRope is modules.FusedQKNormRope


def _reference(x, weight, rope_table):
    x = F.rms_norm(x, (128,), weight, 1e-5)
    first = x[..., :48]
    second = x[..., 48:96]
    cos = rope_table[:, :48].unsqueeze(1)
    sin = rope_table[:, 48:].unsqueeze(1)
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., 96:],
        ),
        dim=-1,
    )
