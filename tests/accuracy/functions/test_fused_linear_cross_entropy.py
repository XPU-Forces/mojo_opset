import os

import pytest
import torch
import torch_npu

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoFusedLinearCrossEntropyFunction


def device_perf_npu(executor, profiling_dir="./npu_profiling", active=5):
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )

    executor()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()

        for _ in range(10):
            executor()
            prof.step()

        torch.npu.synchronize()


@pytest.mark.parametrize(
    "input_tensor, weight, target, bias",
    [
        (
            torch.randn(2048, 1024, dtype=torch.bfloat16, requires_grad=True),
            torch.randn(4096, 1024, dtype=torch.bfloat16, requires_grad=True),
            torch.randint(0, 4096, (2048,), dtype=torch.long),
            None,
        )
    ],
)
@pytest.mark.parametrize(
    "has_bias, has_ce_weight, ignore_index, label_smoothing, lse_square_scale, reduction, return_z_loss",
    [
        (False, False, -100, 0.0, 0.0, "mean", False),
        (False, False, -100, 0.0, 0.0, "sum", False),
        (False, False, -100, 0.0, 0.0, "none", False),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_fused_ce_forward_backward_diff(
    monkeypatch,
    input_tensor,
    weight,
    target,
    bias,
    has_bias,
    has_ce_weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    return_z_loss,
):
    monkeypatch.setenv("MOJOFUSEDLINEARCROSSENTROPYFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOFUSEDLINEARCROSSENTROPYFUNCTION_BWD_MODE", "DIFF")

    ce_weight = None
    if has_ce_weight:
        ce_weight = torch.rand(weight.shape[0], device=weight.device, dtype=torch.float32) + 0.1

    output = MojoFusedLinearCrossEntropyFunction.apply(
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        None,
        return_z_loss,
        None,
    )

    device_perf_npu(
        lambda: MojoFusedLinearCrossEntropyFunction.apply(
            input_tensor,
            weight,
            target,
            bias,
            ce_weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            None,
            return_z_loss,
            None,
        )
    )

    if return_z_loss:
        loss, z_loss = output
    else:
        loss = output
        z_loss = None

    if reduction == "mean":
        grad_output = torch.rand_like(loss)
    else:
        grad_output = torch.rand_like(loss) / input_tensor.shape[0]

    if return_z_loss:
        grad_z_loss = torch.rand_like(z_loss)

        torch.autograd.backward([loss, z_loss], [grad_output, grad_z_loss])

    else:
        loss.backward(grad_output)
