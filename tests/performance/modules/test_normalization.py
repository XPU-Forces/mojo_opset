import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.RMSNorm", ops=["rms_norm"])
@pytest.mark.parametrize("shape", [(128, 1024), (1024, 4096)], ids=["small", "large"])
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_rms_norm(benchmark, perf_environment, shape, phase):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.RMSNorm(shape[-1], device=device, implementation=implementation)
        module.weight.requires_grad_(phase != "forward")
        x = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
        grad = torch.randn_like(x) if phase != "forward" else None
        saved_output = module(x) if phase == "backward" else None

        def run():
            output = saved_output if phase == "backward" else module(x)
            if phase != "forward":
                return torch.autograd.grad(
                    output, (x, module.weight), grad_outputs=grad, retain_graph=phase == "backward"
                )
            return output

        return run

    benchmark(factory=factory, op="rms_norm", shape=list(shape), dtype="bfloat16", weight_dtype="float32", phase=phase)


# Original module inference matrices (perf and perf_new).
@pytest.mark.api("modules.RMSNormInfer", ops=["rms_norm_infer"])
@pytest.mark.parametrize(
    "shape",
    [
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
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_infer(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.RMSNormInfer(
            shape[-1], eps=1e-5, device=device, dtype=torch.float32, implementation=implementation
        ).requires_grad_(False)
        module.weight.copy_(torch.randn_like(module.weight))
        x = torch.randn(shape, dtype=dtype, device=device)
        return lambda: module(x)

    benchmark(factory=factory, op="rms_norm_infer", shape=list(shape), dtype=str(dtype), phase="forward")


@pytest.mark.api("modules.LayerNormInfer", ops=["layer_norm_infer"])
@pytest.mark.parametrize("shape", [(32, 1024), (64, 8192), (57, 7338), (2, 256), (7762, 18778), (256, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layer_norm_infer(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.LayerNormInfer(
            shape[-1], eps=1e-5, device=device, dtype=torch.float32, implementation=implementation
        ).requires_grad_(False)
        # Original inputs are generated in input dtype, then copied to FP32 parameters.
        module.weight.copy_(torch.randn(shape[-1], dtype=dtype, device=device))
        module.bias.copy_(torch.randn(shape[-1], dtype=dtype, device=device))
        x = torch.randn(shape, dtype=dtype, device=device)
        return lambda: module(x)

    benchmark(factory=factory, op="layer_norm_infer", shape=list(shape), dtype=str(dtype), phase="forward")


@pytest.mark.parametrize(
    "op,shape",
    [
        pytest.param(
            op, shape,
            marks=pytest.mark.api(
                "modules.ResidualAddLayerNormInfer"
                if op == "residual_add_layer_norm_infer"
                else "modules.ResidualAddRMSNormInfer",
                ops=[op],
            ),
        )
        for op, shape in [
            *[
                ("residual_add_rms_norm_infer", shape)
                for shape in [(128, 128), (32, 1024), (64, 8192), (57, 7338), (2, 256)]
            ],
            ("residual_add_layer_norm_infer", (128, 128)),
        ]
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
def test_residual_norm(benchmark, perf_environment, op, shape, dtype, norm_pos):
    _, device, _, implementation = perf_environment

    def factory():
        cls = (
            modules.ResidualAddLayerNormInfer
            if op == "residual_add_layer_norm_infer"
            else modules.ResidualAddRMSNormInfer
        )
        module = cls(
            shape[-1], eps=1e-5, norm_pos=norm_pos, device=device, dtype=dtype, implementation=implementation
        ).requires_grad_(False)
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter))
        x, residual = (torch.randn(shape, dtype=dtype, device=device) for _ in range(2))
        return lambda: module(x, residual)

    benchmark(factory=factory, op=op, shape=list(shape), dtype=str(dtype), norm_pos=norm_pos, phase="forward")


@pytest.mark.api("modules.GroupRMSNormInfer", ops=["group_rms_norm_infer"])
@pytest.mark.parametrize(
    "batch,groups,hidden",
    [
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
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_group_rms_norm(benchmark, perf_environment, batch, groups, hidden, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.GroupRMSNormInfer(
            len(groups), hidden, eps=1e-5, device=device, dtype=dtype, implementation=implementation
        ).requires_grad_(False)
        module.weight.copy_(torch.randn_like(module.weight))
        x = torch.randn(batch, sum(groups), hidden, dtype=dtype, device=device)
        inputs = torch.split(x, groups, dim=1)
        return lambda: module(inputs)

    benchmark(
        factory=factory,
        op="group_rms_norm_infer",
        batch=batch,
        groups=list(groups),
        hidden=hidden,
        dtype=str(dtype),
        phase="forward",
    )
