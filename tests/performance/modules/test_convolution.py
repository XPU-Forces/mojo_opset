import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.CausalConv1dUpdateStateInfer", ops=["causal_conv1d_update_state_infer"])
@pytest.mark.parametrize(
    "batch,tokens,channels,width,activation",
    [
        (1, 12291, 8192, 4, "swish"),
        (1, 5000, 2048, 4, "swish"),
        (2, 64, 128, 3, "swish"),
        (2, 128, 128, 4, "swish"),
        (2, 64, 128, 3, None),
        (3, 1446, 256, 4, None),
        (1, 32, 32, 4, None),
    ],
)
def test_causal_conv1d_update_state(benchmark, perf_environment, batch, tokens, channels, width, activation):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.CausalConv1dUpdateStateInfer(implementation=implementation)
        x = torch.randn(batch, channels, tokens, dtype=torch.float16, device=device)
        state = torch.randn(batch, channels, width, dtype=torch.float16, device=device)
        weight = torch.randn(channels, width, dtype=torch.float16, device=device)
        # tokens >= width in all inherited cases: after warmup state is the same
        # input tail on every call, so repeated updates cannot drift or grow.
        return lambda: module(x, state, weight, activation=activation)

    benchmark(
        factory=factory,
        op="causal_conv1d_update_state_infer",
        batch=batch,
        tokens=tokens,
        channels=channels,
        width=width,
        activation=activation,
        dtype="float16",
        phase="forward",
    )
