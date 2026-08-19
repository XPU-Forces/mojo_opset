import pytest
import torch

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoRMSNormFunction
from mojo_opset.tests.utils import MockFunctionCtx
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device

device = get_torch_device()

@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),

    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_residual_add_rmsnorm(shape, dtype, norm_pos, eps):
    torch.manual_seed(43)
    x = torch.randn(size=shape, dtype=dtype, device=device)
    residual = torch.randn(size=shape, dtype=dtype, device=device)
    weight = torch.randn(size=(shape[-1],), dtype=dtype, device=device)
    add_norm = MojoResidualAddRMSNorm(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    ).to(x.device)
    add_norm.weight = torch.nn.Parameter(weight)

    perf(lambda: add_norm(x, residual))  # noqa: F821


@pytest.mark.parametrize(
    "x, residual, weight, bias",
    [
        (
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_residual_add_layernorm(x, residual, weight, bias, norm_pos, eps):
    add_norm = MojoResidualAddLayerNorm(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm.weight = torch.nn.Parameter(weight)
    add_norm.bias = torch.nn.Parameter(bias)

    perf(lambda: add_norm(x, residual))  # noqa: F821


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=shape, dtype=dtype),
            torch.nn.Parameter(torch.randn(size=(shape[-1],), dtype=torch.float32)),
        )
        for shape in [
                        (1, 32, 2048),
                        (1, 32, 1024),
                        (1, 64, 8192),
                        (1, 57, 7338),
                        (1, 77, 489),
                        (1, 2, 256),
                        (1, 763, 8777),
                        (1, 7762, 18778),
                        (32, 1024),
                        (64, 8192),
                        (57, 7338),
                        (2, 256),
                        (7762, 18778),
    ]
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_rmsnorm(x, weight, eps):
    rmsnorm = MojoRMSNorm(
        weight.size(0),
        eps,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))

    perf(lambda: rmsnorm(x))  # noqa: F821

@pytest.mark.parametrize(
    "x, weight, group_dims",
    [
        (
            torch.randn(
                size=(bsz, sum(group_dims), hidden_size),
                dtype=dtype,
            ),
            torch.randn(
                size=(len(group_dims), hidden_size),
                dtype=dtype,
            ),
            group_dims,
        )
        for bsz, group_dims, hidden_size in [
            (1024, (16, 4), 96),
            (798, (16, 4, 8, 2), 128),
            (8000, (48, 8, 16, 4), 128),
            (17, (3, 5), 128),
            (33, (2, 7, 1), 128),
            (65, (4, 4, 4, 4), 128),
            (129, (1, 3, 5, 7), 128),
            (257, (6, 2), 192),
            (513, (8, 8, 8), 256),
            (1025, (12, 6, 3, 1), 128),
            (2049, (5, 9, 7, 3), 64),
        ]
        for dtype in [torch.float32, torch.float16, torch.bfloat16] # , torch.float16, torch.bfloat16
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_grouprmsnorm(x, weight, group_dims, eps):
    # x_groups = tuple(
    #     t.contiguous()
    #     for t in torch.split(x, group_dims, dim=1)
    # )
    x_groups = torch.split(x, group_dims, dim=1)

    from mojo_opset import MojoGroupRMSNorm
    rmsnorm = MojoGroupRMSNorm(
        num_groups=len(group_dims),
        eps=eps,
        norm_size=x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    ).to(x.device)
    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))

    perf(lambda: rmsnorm(x_groups)) # noqa: F821

@pytest.mark.parametrize(
    "shape",
    [
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),
        (7762, 18778),
        (256, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_layernorm(shape, dtype, eps):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    weight = torch.randn(size=(shape[-1],), dtype=dtype, device=device)
    bias = torch.randn(size=(shape[-1],), dtype=dtype, device=device)
    layernorm = MojoLayerNorm(
        norm_size=weight.size(0),
        eps=eps,
    ).to(x.device)

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))

    perf(lambda: layernorm(x))  # noqa: F821


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=shape, dtype=dtype),
            torch.randn(size=(shape[-1],), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.bfloat16]
        for shape in  [
                        (32, 1024),
                        (64, 8192),
                        (57, 7338),
                        (77, 489),
                        (763, 8777),
                        (7762, 18778),
                    ]
    ],
)
@auto_switch_platform()
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_rmsnorm_forward_backward_diff(x, weight):
    ctx = MockFunctionCtx()
    y = MojoRMSNormFunction.forward(ctx, x, weight, 1e-6)
    dy = torch.rand_like(y)
    perf(lambda: MojoRMSNormFunction.forward(ctx, x, weight, 1e-6))
    perf(lambda: MojoRMSNormFunction.backward(ctx, dy))
