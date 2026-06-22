import pytest
import torch

from mojo_opset import MojoRMSNormDynamicQuant
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _requantize_compare(golden: torch.Tensor, output: torch.Tensor) -> bool:
    output_i8 = output.view(torch.int8)
    golden_i8 = golden.view(torch.int8)

    diff_results = torch.abs(output_i8.view(-1).to(torch.int16) - golden_i8.view(-1).to(torch.int16))
    diff_indices = torch.where(diff_results > 1)[0]

    output_nan, golden_nan = torch.isnan(output.view(-1)), torch.isnan(golden.view(-1))
    diff_nan = torch.logical_and(output_nan, golden_nan)
    both_nan_idx = torch.where(diff_nan)[0]

    if both_nan_idx.numel() > 0 and diff_indices.numel() > 0:
        mask = torch.isin(diff_indices, both_nan_idx)
        diff_indices = diff_indices[~mask]

    golden_size = golden.numel()
    diff_size = diff_indices.numel()
    precision = (golden_size - diff_size) / golden_size
    return (1 - precision) <= 0.001


def _rms_norm_and_dynamic_quant_ref(
    x: torch.Tensor,
    gamma: torch.Tensor,
    smooth_scale: torch.Tensor,
    epsilon: float,
):
    # Reference uses torch_npu primitives to match upstream comparison.
    import torch_npu

    y, _ = torch_npu.npu_rms_norm(x, gamma, epsilon=epsilon)
    out, scale_out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scale)
    return out, scale_out


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [4096, 7168])
@bypass_not_implemented
def test_rms_norm_dynamic_quant_different_dtypes(dtype, d):
    device = get_torch_device()
    if device != "npu":
        pytest.skip("This op is Ascend NPU only.")

    torch.manual_seed(0)
    bs = 128
    n = 1
    epsilon = 1e-6

    x = (torch.rand(bs, n, d, device=device, dtype=dtype) * 9.0 + 1.0).contiguous()
    gamma = torch.rand(d, device=device, dtype=dtype).contiguous()
    smooth_scale = torch.rand(d, device=device, dtype=dtype).contiguous()

    cpu_y_out, cpu_scale_out = _rms_norm_and_dynamic_quant_ref(x, gamma, smooth_scale, epsilon)

    mojo_op = MojoRMSNormDynamicQuant()
    print(f"{type(mojo_op)=}")
    npu_y_out, npu_scale_out = mojo_op.forward(
        x,
        gamma,
        smooth_scale=smooth_scale,
        beta=None,
        epsilon=epsilon,
    )

    assert _requantize_compare(cpu_y_out.cpu(), npu_y_out.cpu())
    torch.testing.assert_close(npu_scale_out.cpu().float(), cpu_scale_out.cpu().float(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_rms_norm_dynamic_quant_different_dtypes_graph(dtype):
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Graph-mode test only runs on Ascend NPU.")

    try:
        import torchair  # noqa: F401
        from torchair.configs.compiler_config import CompilerConfig
    except Exception as e:
        pytest.skip(f"torchair is not available: {e}")

    try:
        import custom_ops  # noqa: F401
    except Exception as e:
        pytest.skip(f"custom_ops is not available: {e}")

    if not hasattr(torch.ops.custom, "npu_rms_norm_dynamic_quant"):
        pytest.skip("torch.ops.custom.npu_rms_norm_dynamic_quant is not registered in this environment.")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch build.")

    torch.manual_seed(0)
    bs = 128
    n = 1
    d = 4096
    epsilon = 1e-6

    x = (torch.rand(bs, n, d, device=device, dtype=dtype) * 9.0 + 1.0).contiguous()
    gamma = torch.rand(d, device=device, dtype=dtype).contiguous()
    smooth_scale = torch.rand(d, device=device, dtype=dtype).contiguous()

    import torch_npu

    cpu_y_out, cpu_scale_out = _rms_norm_and_dynamic_quant_ref(x, gamma, smooth_scale, epsilon)

    class Network(torch.nn.Module):
        def forward(self, x_in, gamma_in, smooth_in, eps_in):
            y_out, scale_out = torch.ops.custom.npu_rms_norm_dynamic_quant(
                x_in,
                gamma_in,
                smooth_scale=smooth_in,
                beta=None,
                epsilon=eps_in,
            )
            return y_out, scale_out

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    npu_model = Network().to(device)
    try:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
    except TypeError:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend)
    except Exception as e:
        pytest.skip(f"torch.compile failed on this environment: {e}")

    npu_y_out, npu_scale_out = npu_model(x, gamma, smooth_scale, epsilon)

    assert _requantize_compare(cpu_y_out.cpu(), npu_y_out.cpu())
    torch.testing.assert_close(npu_scale_out.cpu().float(), cpu_scale_out.cpu().float(), atol=1e-3, rtol=1e-2)

