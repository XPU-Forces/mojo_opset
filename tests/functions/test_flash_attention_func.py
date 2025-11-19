import pytest
import torch
import math
from tests.utils import auto_switch_platform, bypass_not_implemented
from mojo_opset import MojoFlashAttnFunction


@pytest.mark.parametrize(
    "q, k, v, cu_seqlens_q, cu_seqlens_k",
    [
        (
            torch.randn(1000, 32, 128, dtype=torch.float16, requires_grad=True),
            torch.randn(1000, 16, 128, dtype=torch.float16, requires_grad=True),
            torch.randn(1000, 16, 128, dtype=torch.float16, requires_grad=True),
            torch.Tensor([0, 100, 384, 1000]).to(torch.int32),
            torch.Tensor([0, 100, 384, 1000]).to(torch.int32),
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_flash_attention_forward_backward_diff(monkeypatch, q, k, v, cu_seqlens_q, cu_seqlens_k):
    monkeypatch.setenv("MOJOFLASHATTNFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOFLASHATTNFUNCTION_BWD_MODE", "DIFF")

    dropout_p = 0.0
    causal = True

    # FIXME: delete me.
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    y = MojoFlashAttnFunction.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p, causal, sm_scale)

    grad_output = torch.rand_like(y)
    y.backward(grad_output)
