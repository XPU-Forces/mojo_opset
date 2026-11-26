import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions
from tests.performance._workload import training

EPS = 1e-6

# The shape matrix of the task statement, one row per (feature combination, scale). Keys are
# spelled out instead of positional so a reader can match a line against the contract table.
MATRIX = [
    # label, M, N, gate (None | "2d" | "3d"), activation, norm_before_gate, is_rms_norm, group, quant_heads
    ("gdn_32x8", 256, 128, "3d", "swish", True, True, None, 0),
    ("gdn_320x8", 2560, 128, "3d", "swish", True, True, None, 0),
    ("gdn_4096x8", 32768, 128, "3d", "swish", True, True, None, 0),
    ("gdn_4096x8_sigmoid", 32768, 128, "3d", "sigmoid", True, True, None, 0),
    ("gdn_16384x8", 131072, 128, "3d", "swish", True, True, None, 0),
    ("qk_norm_4096x16", 65536, 128, None, "swish", True, True, None, 0),
    ("groupnorm_1024", 1024, 2048, "2d", "swish", False, True, 128, 0),
    ("groupnorm_8192", 8192, 2048, "2d", "swish", False, True, 128, 0),
    ("quant_4096x8", 32768, 512, "2d", "swish", True, True, None, 8),
    ("gdn_65536x8", 524288, 128, "3d", "swish", True, True, None, 0),
    ("tail_32771x192", 32771, 192, "2d", "swish", True, True, None, 0),
]

CASE_IDS = [row[0] for row in MATRIX]


def make_case(m, n, gate, device):
    """Inputs outside the timed region; the 3-D gate is a strided view, as in the model."""
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(m, n, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    weight = (torch.rand(n, generator=gen) + 0.5).to(torch.bfloat16).to(device)
    bias = (torch.randn(n, generator=gen) * 0.1).to(torch.bfloat16).to(device)
    z = None
    if gate == "2d":
        z = (torch.randn(m, n, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    elif gate == "3d":
        # (T, H, D) with a padded feature stride: T * H == M, D == N, stride(-1) == 1.
        t, h, pad = m // 8, 8, 16
        buf = torch.randn(t, h, n + pad, generator=gen, dtype=torch.bfloat16).to(device)
        z = buf[:, :, :n]
    return x, weight, bias, z


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("case", MATRIX, ids=CASE_IDS)
@pytest.mark.parametrize("phase", ["forward"])
def test_layernorm_gated(benchmark, perf_environment, case, phase):
    label, m, n, gate, activation, norm_before_gate, is_rms_norm, group, quant_heads = case
    _, device, target, implementation = perf_environment

    # The FP8 epilogue is only required on 950 (the A2/A3 parts have no FP8 storage path).
    if quant_heads and Target.parse(target).arch != "a5":
        pytest.skip("FP8 quant epilogue is required on 950 only")

    x, weight, bias, z = make_case(m, n, gate, device)
    kwargs = dict(
        z=z,
        group_size=group,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
    )

    def factory():
        call = lambda: functions.layernorm_gated_infer(x, weight, bias, EPS, **kwargs, implementation=implementation)
        return training(call, (x, weight, bias, z), phase)

    benchmark(
        factory=factory,
        op="layernorm_gated_infer",
        case=label,
        rows=m,
        hidden=n,
        gate=gate or "none",
        activation=activation,
        group_size=group or n,
        quant_heads=quant_heads,
        dtype="bfloat16",
        phase=phase,
    )
