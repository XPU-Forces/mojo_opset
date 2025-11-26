import pytest
import torch
import torch.nn.functional as F

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGatedDeltaRuleFunction

EPS = 1e-3


@pytest.mark.parametrize(
    "q, k, v, beta, g, cu_seqlens",
    [
        (
            torch.randn(1, 128, 32, 128, dtype=torch.float16, requires_grad=True),
            torch.randn(1, 128, 32, 128, dtype=torch.float16, requires_grad=True),
            torch.randn(1, 128, 32, 256, dtype=torch.float16, requires_grad=True),
            torch.rand(1, 128, 32, dtype=torch.float32, requires_grad=True).sigmoid() * EPS,
            F.logsigmoid(torch.rand(1, 128, 32, dtype=torch.float32, requires_grad=True)) * EPS,
            torch.tensor([0, 128], dtype=torch.long),
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gated_delta_rule_forward_backward_diff(monkeypatch, q, k, v, beta, g, cu_seqlens):
    monkeypatch.setenv("MOJOGATEDDELTARULEFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOGATEDDELTARULEFUNCTION_BWD_MODE", "DIFF")

    beta.retain_grad()
    g.retain_grad()

    scale = k.shape[-1] ** -0.5

    y = MojoGatedDeltaRuleFunction.apply(q, k, v, beta, g, cu_seqlens, scale, True)
    loss = y.sum()
    loss.backward()
