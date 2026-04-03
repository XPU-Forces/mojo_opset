import pytest
import torch
import torch.nn.functional as F

from mojo_opset.utils.platform import get_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoChannelRMSNorm
from mojo_opset import MojoLayerNorm
from mojo_opset import MojoLayerNormQuant
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddLayerNormQuant
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoResidualAddRMSNormQuant
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoRMSNormQuant
from mojo_opset import MojoGroupRMSNorm
from mojo_opset import MojoGroupLayerNorm

torch.manual_seed(43)


dtypes = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize(
    "shape",
    [
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),
        (7762, 18778),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@bypass_not_implemented
def test_rmsnorm(shape, dtype, eps):
    x = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=dtype)
    rmsnorm = MojoRMSNorm(eps=eps, norm_size=shape[-1], device=x.device, dtype=x.dtype)

    rmsnorm_ref = (
        MojoRMSNorm._registry.get("torch")(
            eps=eps,
            norm_size=weight.size(0),
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))
        rmsnorm_ref.weight.copy_(weight.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    rmsnorm.forward_diff_with(rmsnorm_ref, x, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),
        (7762, 18778),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@bypass_not_implemented
def test_layernorm(shape, dtype, eps):
    x = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=dtype)
    bias = torch.randn(size=(shape[-1],), dtype=dtype)
    layernorm = MojoLayerNorm(eps=eps, norm_size=weight.size(0), dtype=weight.dtype, device=x.device)

    layernorm_ref = (
        MojoLayerNorm._registry.get("torch")(
            eps=eps,
            norm_size=weight.size(0),
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))
        layernorm_ref.weight.copy_(weight.to(torch.float32))
        layernorm_ref.bias.copy_(bias.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5
    else:
        atol, rtol = 5e-2, 1e-2
    layernorm.forward_diff_with(layernorm_ref, x, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz,group_dims,hidden_size",
    [
        (1024, (16, 4), 96),
        (798, (16, 4, 8, 2), 128),
        (8000, (48, 8, 16, 4), 128),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@bypass_not_implemented
def test_grouprmsnorm(bsz, group_dims, hidden_size, dtype, eps):
    x = torch.randn(size=(bsz, sum(group_dims), hidden_size), dtype=dtype)
    x_groups = torch.split(x, group_dims, dim=1)
    weight = torch.randn(size=(len(group_dims), hidden_size), dtype=dtype)
    rmsnorm = MojoGroupRMSNorm(num_groups = len(group_dims), eps=eps, norm_size=hidden_size, device=x.device, dtype=x.dtype)

    rmsnorm_ref = (
        MojoGroupRMSNorm._registry.get("torch")(
            num_groups = len(group_dims), 
            eps=eps,
            norm_size=hidden_size,
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))
        rmsnorm_ref.weight.copy_(weight.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3

    # Placeholder for ttx implementation
    x_normed_outputs = rmsnorm(x_groups)
    # rmsnorm.forward_diff_with(rmsnorm_ref, x_groups, atol=atol, rtol=rtol)

@pytest.mark.parametrize(
    "bsz,group_dims,hidden_size",
    [
        (1024, (16, 4), 96),
        (798, (16, 4, 8, 2), 128),
        (8000, (48, 8, 16, 4), 128),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@bypass_not_implemented
def test_grouplayernorm(bsz, group_dims, hidden_size, dtype, eps):
    x = torch.randn(size=(bsz, sum(group_dims), hidden_size), dtype=dtype)
    x_groups = torch.split(x, group_dims, dim=1)
    weight = torch.randn(size=(len(group_dims), hidden_size), dtype=dtype)
    bias = torch.randn(size=(len(group_dims), hidden_size), dtype=dtype)
    # Placeholder for ttx implementation
    layernorm = MojoGroupLayerNorm(num_groups = len(group_dims), eps=eps, norm_size=hidden_size, device=x.device, dtype=x.dtype)

    layernorm_ref = (
        MojoGroupLayerNorm._registry.get("torch")(
            num_groups = len(group_dims), 
            eps=eps,
            norm_size=hidden_size,
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm_ref.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))
        layernorm_ref.bias.copy_(bias.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    x_normed_outputs = layernorm(x_groups)
    # layernorm.forward_diff_with(layernorm_ref, x_groups, atol=atol, rtol=rtol)



@pytest.mark.parametrize(
    "shape",
    [
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@bypass_not_implemented
def test_residual_add_rms_norm(shape, dtype, norm_pos, eps):
    torch.manual_seed(43)
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=dtype)
    add_norm = MojoResidualAddRMSNorm(
        norm_size=weight.size(0), eps=eps, norm_pos=norm_pos, device=x.device, dtype=weight.dtype
    )
    add_norm_ref = MojoResidualAddRMSNorm._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )

    add_norm_ref.weight = add_norm.weight = torch.nn.Parameter(weight)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 5e-2, 1e-2

    add_norm.forward_diff_with(
        add_norm_ref,
        x,
        residual,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32, 1024),
        (64, 8192),
        (57, 7338),
        (2, 256),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@bypass_not_implemented
def test_residual_add_layernorm(shape, dtype, norm_pos, eps):
    torch.manual_seed(43)
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=dtype)
    bias = torch.randn(size=(shape[-1],), dtype=dtype)
    add_norm = MojoResidualAddLayerNorm(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm_ref = MojoResidualAddLayerNorm._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )

    add_norm.weight = torch.nn.Parameter(weight)
    add_norm.bias = torch.nn.Parameter(bias)
    add_norm_ref.weight = torch.nn.Parameter(weight)
    add_norm_ref.bias = torch.nn.Parameter(bias)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 5e-2, 1e-2

    add_norm.forward_diff_with(
        add_norm_ref,
        x,
        residual,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "x, norm_size, channel_first, images",
    [
        (torch.randn(size=(1, 1024, 30, 52), dtype=dtype), 1024, True, True)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ]
    + [
        (torch.randn(size=(1, 256, 4, 240, 416), dtype=dtype), 256, True, False)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ]
    + [
        (torch.randn(size=(1, 1024, 30, 52), dtype=dtype), 52, False, True)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ]
    + [
        (torch.randn(size=(1, 512, 4, 120, 208), dtype=dtype), 208, False, False)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@bypass_not_implemented
def test_channel_rmsnorm(x, norm_size, channel_first, images):
    norm = MojoChannelRMSNorm(
        norm_size=norm_size,
        channel_first=channel_first,
        images=images,
        device=x.device,
        dtype=torch.float32,
    )
    norm_ref = (
        MojoChannelRMSNorm._registry.get("torch")(
            norm_size=norm_size,
            channel_first=channel_first,
            images=images,
        )
        .to(x.device)
        .to(torch.float32)
    )

    with torch.no_grad():
        weight_data = torch.randn(norm.weight.shape, dtype=torch.float32, device=x.device)
        norm.weight.copy_(weight_data)
        norm_ref.weight.copy_(weight_data)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    norm.forward_diff_with(norm_ref, x, atol=atol, rtol=rtol)


# ===========================================================================
# NormQuant tests
# ===========================================================================

norm_quant_shapes = [
    (32, 1024),
    (64, 8192),
    (2, 256),
]


@pytest.mark.parametrize("shape", norm_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_rmsnorm_quant(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=torch.float32)

    op = MojoRMSNormQuant(norm_size=shape[-1])
    op_ref = MojoRMSNormQuant._registry.get("torch")(norm_size=shape[-1])
    with torch.no_grad():
        op.weight.copy_(weight)
        op_ref.weight.copy_(weight)

    # NOTE(liuyuan): The difference between torch_npu and torch natvie is caused by the computation of normalization, leading to -1.3342->32 (torch natvie) and -1.3340->31 (torch_npu).
    op.forward_diff_with(op_ref, x, atol=(1, 1e-3), rtol=(0, 1e-3))

    # Semantic check: fp32 RMSNorm matches MojoRMSNormQuant forward
    normed = F.rms_norm(x.float(), [x.shape[-1]], weight=weight, eps=1e-5)
    normed_fp = normed
    scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    expected = torch.clamp(torch.round(normed_fp / scale), -128, 127).to(torch.int8)
    out, out_scale = op_ref(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_scale, scale, atol=0, rtol=0)


@pytest.mark.parametrize("shape", norm_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_layernorm_quant(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=torch.float32)
    bias = torch.randn(size=(shape[-1],), dtype=torch.float32)

    op = MojoLayerNormQuant(norm_size=shape[-1])
    op_ref = MojoLayerNormQuant._registry.get("torch")(norm_size=shape[-1])
    with torch.no_grad():
        op.weight.copy_(weight)
        op.bias.copy_(bias)
        op_ref.weight.copy_(weight)
        op_ref.bias.copy_(bias)

    # NOTE(liuyuan): The difference between torch_npu and torch natvie is caused by the computation of normalization.
    op.forward_diff_with(op_ref, x, atol=(1, 1e-3), rtol=(0, 1e-3))

    # Semantic check (fp32 LN matches MojoLayerNormQuant forward)
    normed = F.layer_norm(x.float(), [x.shape[-1]], weight=weight, bias=bias, eps=1e-5)
    normed_fp = normed
    scale = normed_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    expected = torch.clamp(torch.round(normed_fp / scale), -128, 127).to(torch.int8)
    out, out_scale = op_ref(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_scale, scale, atol=0, rtol=0)


@pytest.mark.parametrize("shape", norm_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@bypass_not_implemented
def test_residual_add_rmsnorm_quant(shape, dtype, norm_pos):
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=torch.float32)

    op = MojoResidualAddRMSNormQuant(norm_size=shape[-1], norm_pos=norm_pos)
    op_ref = MojoResidualAddRMSNormQuant._registry.get("torch")(
        norm_size=shape[-1], norm_pos=norm_pos
    )
    with torch.no_grad():
        op.weight.copy_(weight)
        op_ref.weight.copy_(weight)

    op.forward_diff_with(
        op_ref,
        x,
        residual,
        atol=(2, 1e-3, 1e-3),
        rtol=(0, 1e-3, 1e-3),
    )


@pytest.mark.parametrize("shape", norm_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@bypass_not_implemented
def test_residual_add_layernorm_quant(shape, dtype, norm_pos):
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    weight = torch.randn(size=(shape[-1],), dtype=torch.float32)
    bias = torch.randn(size=(shape[-1],), dtype=torch.float32)

    op = MojoResidualAddLayerNormQuant(
        norm_size=shape[-1], norm_pos=norm_pos
    )
    op_ref = MojoResidualAddLayerNormQuant._registry.get("torch")(
        norm_size=shape[-1], norm_pos=norm_pos
    )
    with torch.no_grad():
        op.weight.copy_(weight)
        op.bias.copy_(bias)
        op_ref.weight.copy_(weight)
        op_ref.bias.copy_(bias)

    # NOTE(liuyuan): The difference between torch_npu and torch natvie is caused by the computation of normalization.
    op.forward_diff_with(
        op_ref,
        x,
        residual,
        atol=(1, 1e-3, 1e-3),
        rtol=(0, 1e-3, 1e-3),
    )


# ===========================================================================
# NormQuant: float8_e4m3fn quantization dtype
# ===========================================================================

_requires_cpu = pytest.mark.skipif(
    get_platform() != "cpu",
    reason="float8_e4m3fn not supported on NPU; reference-only test requires CPU",
)


@_requires_cpu
@pytest.mark.parametrize("shape", [(32, 1024), (64, 8192)])
@pytest.mark.parametrize("dtype", dtypes)
def test_rmsnorm_quant_fp8(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    op = MojoRMSNormQuant._registry.get("torch")(norm_size=shape[-1], quant_dtype=torch.float8_e4m3fn)
    out, scale = op(x)
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == x.shape
    assert scale.shape == (*x.shape[:-1], 1)
    out2, scale2 = op(x)
    torch.testing.assert_close(out.float(), out2.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, scale2, atol=0, rtol=0)


@_requires_cpu
@pytest.mark.parametrize("shape", [(32, 1024), (64, 8192)])
@pytest.mark.parametrize("dtype", dtypes)
def test_layernorm_quant_fp8(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    op = MojoLayerNormQuant._registry.get("torch")(norm_size=shape[-1], quant_dtype=torch.float8_e4m3fn)
    out, scale = op(x)
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == x.shape
    assert scale.shape == (*x.shape[:-1], 1)
    out2, scale2 = op(x)
    torch.testing.assert_close(out.float(), out2.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, scale2, atol=0, rtol=0)


@_requires_cpu
@pytest.mark.parametrize("shape", [(32, 1024)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
def test_residual_add_rmsnorm_quant_fp8(shape, dtype, norm_pos):
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    op = MojoResidualAddRMSNormQuant._registry.get("torch")(
        norm_size=shape[-1], norm_pos=norm_pos, quant_dtype=torch.float8_e4m3fn
    )
    out, updated_residual, scale = op(x, residual)
    assert out.dtype == torch.float8_e4m3fn
    assert scale.shape[-1] == 1


@_requires_cpu
@pytest.mark.parametrize("shape", [(32, 1024)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
def test_residual_add_layernorm_quant_fp8(shape, dtype, norm_pos):
    x = torch.randn(size=shape, dtype=dtype)
    residual = torch.randn(size=shape, dtype=dtype)
    op = MojoResidualAddLayerNormQuant._registry.get("torch")(
        norm_size=shape[-1], norm_pos=norm_pos, quant_dtype=torch.float8_e4m3fn
    )
    out, updated_residual, scale = op(x, residual)
    assert out.dtype == torch.float8_e4m3fn
    assert scale.shape[-1] == 1


# ===========================================================================
# NormQuant: unsupported quant_dtype → NotImplementedError
# ===========================================================================

def test_normquant_unsupported_dtype_raises():
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoRMSNormQuant._registry.get("torch")(norm_size=64, quant_dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoLayerNormQuant._registry.get("torch")(norm_size=64, quant_dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoResidualAddRMSNormQuant._registry.get("torch")(norm_size=64, quant_dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoResidualAddLayerNormQuant._registry.get("torch")(norm_size=64, quant_dtype=torch.float32)
