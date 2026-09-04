import pytest
import torch

from mojo_opset.tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4096),
        (16, 7168),
        (4, 4, 4096),
        (4, 4, 7168),
        (8, 16, 4096),
        (8, 16, 7168),
    ],
)
@auto_switch_platform()
def test_hc_post_opcheck(shape):
    HC = 2
    D = shape[-1]

    if len(shape) == 2:
        BS = shape[0]
        x = torch.randn(BS, D, device='npu')
        residual = torch.randn(BS, HC, D, device='npu')
        post = torch.randn(BS, HC, device='npu')
        comb = torch.randn(BS, HC, HC, device='npu')
    else:
        B, S = shape[0], shape[1]
        x = torch.randn(B, S, D, device='npu')
        residual = torch.randn(B, S, HC, D, device='npu')
        post = torch.randn(B, S, HC, device='npu')
        comb = torch.randn(B, S, HC, HC, device='npu')

    torch.library.opcheck(torch.ops.ttx.hc_post, (x, residual, post, comb))


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4096),
        (4, 4, 4096),
        (8, 16, 7168),
    ],
)
@auto_switch_platform()
def test_hc_post_compile(shape):
    HC = 2
    D = shape[-1]

    if len(shape) == 2:
        BS = shape[0]
        x = torch.randn(BS, D, device='npu')
        residual = torch.randn(BS, HC, D, device='npu')
        post = torch.randn(BS, HC, device='npu')
        comb = torch.randn(BS, HC, HC, device='npu')
    else:
        B, S = shape[0], shape[1]
        x = torch.randn(B, S, D, device='npu')
        residual = torch.randn(B, S, HC, D, device='npu')
        post = torch.randn(B, S, HC, device='npu')
        comb = torch.randn(B, S, HC, HC, device='npu')

    @torch.compile(fullgraph=True)
    def compiled_hc_post(x, residual, post, comb):
        return torch.ops.ttx.hc_post(x, residual, post, comb)

    out = compiled_hc_post(x, residual, post, comb)
    assert out.shape == residual.shape, f"Expected {residual.shape}, got {out.shape}"
