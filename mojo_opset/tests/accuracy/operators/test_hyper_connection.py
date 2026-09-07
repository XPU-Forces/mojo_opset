import pytest
import torch
import torch.nn.functional as F

from mojo_opset import MojoHcPost
from mojo_opset import MojoHcPre


def _torch_op(operator_cls, **kwargs):
    return operator_cls._registry.get("torch")(**kwargs, device="cpu")


def _hc_pre_reference(x, hc_fn, hc_scale, hc_base, *, iterations, norm_eps, eps):
    hc_mult = x.shape[-2]
    x_fp = x.float()
    flat = x_fp.flatten(-2)
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_fn.float()) * inverse_rms
    pre, post, combine = mixes.split((hc_mult, hc_mult, hc_mult * hc_mult), dim=-1)
    combine = combine.unflatten(-1, (hc_mult, hc_mult))
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    combine = combine * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    combine = combine.softmax(-1) + eps
    combine = combine / (combine.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        combine = combine / (combine.sum(-1, keepdim=True) + eps)
        combine = combine / (combine.sum(-2, keepdim=True) + eps)
    y = (pre.unsqueeze(-1) * x_fp).sum(dim=-2)
    return y.to(x.dtype), post, combine


@pytest.mark.parametrize("hc_mult", [2, 4])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_hc_pre_matches_sinkhorn_formula(hc_mult, batch_shape):
    torch.manual_seed(17)
    x = torch.randn(*batch_shape, hc_mult, 3, device="cpu")
    mix_dim = 2 * hc_mult + hc_mult * hc_mult
    hc_fn = torch.randn(mix_dim, hc_mult * 3, device="cpu")
    hc_scale = torch.tensor([0.7, 1.1, 0.4], device="cpu")
    hc_base = torch.randn(mix_dim, device="cpu")
    expected = _hc_pre_reference(x, hc_fn, hc_scale, hc_base, iterations=3, norm_eps=0.1, eps=0.02)

    op = _torch_op(MojoHcPre, hidden_size=3, hc_mult=hc_mult, hc_sinkhorn_iters=3, norm_eps=0.1, hc_eps=0.02)
    op.load_state_dict({"weight": hc_fn, "scale": hc_scale, "bias": hc_base}, strict=True)
    actual = op(hidden_states=x)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6)


def test_hc_post_matches_residual_mix_formula():
    torch.manual_seed(23)
    x = torch.randn(2, 3, device="cpu")
    residual = torch.randn(2, 4, 3, device="cpu")
    post = torch.randn(2, 4, device="cpu")
    combine = torch.randn(2, 4, 4, device="cpu")
    expected = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
    expected += (combine.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=1)

    actual = _torch_op(MojoHcPost)(
        hidden_states=x, residual=residual, post_weights=post, combination_weights=combine
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
