"""Gated LayerNorm performance over the task shape matrix (BF16).

Each case records its fixed read/write byte count (``io_bytes``, computed from
the task formulas: gated ``(3*M*N + 2*N)*es``, ungated ``(2*M*N + 2*N)*es``,
fused FP8 ``2*M*N*es + M*N + 4*M/quant_heads + 2*N*es``) and the applicable
MBU line. MBU = io_bytes / profiler time / the machine's on-site calibrated
copy_ bandwidth; cases without an MBU line (io_bytes < 16 MB) use the relative
line: no worse than the torch eager composition.
"""

import pytest
import torch

from mojo_opset import functions

Z_PAD = 32  # token-major strided-view padding of 3D gate branches

# (name, tokens, heads, N, z_mode, activation, norm_before_gate, group_size, quant_heads, mbu_line)
LAYERNORM_GATED_CASES = [
    ("gdn_decode_swish", 32, 8, 128, "3d", "swish", True, 0, 0, None),
    ("gdn_short_seq_swish", 320, 8, 128, "3d", "swish", True, 0, 0, None),
    ("gdn_prefill_swish", 4096, 8, 128, "3d", "swish", True, 0, 0, 60.0),
    ("gdn_prefill_sigmoid", 4096, 8, 128, "3d", "sigmoid", True, 0, 0, 60.0),
    ("gdn_long_seq_swish", 16384, 8, 128, "3d", "swish", True, 0, 0, 60.0),
    ("qk_norm_nogate", 4096, 16, 128, "none", "swish", True, 0, 0, 60.0),
    ("group_gate_first_short", 1024, 1, 2048, "2d", "swish", False, 128, 0, None),
    ("group_gate_first_prefill", 8192, 1, 2048, "2d", "swish", False, 128, 0, 60.0),
    ("quant_fused_fp8", 4096, 8, 512, "3d", "swish", True, 0, 8, 60.0),
    ("gdn_xlong_seq_swish", 65536, 8, 128, "3d", "swish", True, 0, 0, 60.0),
    ("tail_2d_swish", 32771, 1, 192, "2d", "swish", True, 0, 0, 50.0),
]


def io_bytes(M, N, *, gated, quant_heads=0, es=2):
    """Fixed per-call read/write byte formulas from the task test standard."""
    if quant_heads:
        return 2 * M * N * es + M * N + 4 * M // quant_heads + 2 * N * es
    return (3 * M * N + 2 * N) * es if gated else (2 * M * N + 2 * N) * es


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize(
    "name,tokens,heads,N,z_mode,activation,norm_before_gate,group_size,quant_heads,mbu_line",
    LAYERNORM_GATED_CASES,
)
def test_layer_norm_gated(
    benchmark, perf_environment, name, tokens, heads, N, z_mode, activation,
    norm_before_gate, group_size, quant_heads, mbu_line,
):
    _, device, _, implementation = perf_environment
    M = tokens * heads

    def factory():
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        weight = torch.randn(N, dtype=torch.bfloat16, device=device)
        if z_mode == "3d":
            z = torch.randn(tokens, heads, N + Z_PAD, dtype=torch.bfloat16, device=device)[:, :, :N]
        elif z_mode == "2d":
            z = torch.randn(M, N + Z_PAD, dtype=torch.bfloat16, device=device)[:, :N]
        else:
            z = None
        return lambda: functions.layer_norm_gated(
            x, weight, None, z=z, group_size=group_size or None,
            norm_before_gate=norm_before_gate, is_rms_norm=True,
            activation=activation, quant_heads=quant_heads,
            implementation=implementation,
        )

    total_bytes = io_bytes(M, N, gated=z_mode != "none", quant_heads=quant_heads)
    benchmark(
        factory=factory,
        op="layer_norm_gated",
        case=name,
        shape=[M, N],
        tokens=tokens,
        heads=heads,
        dtype="bfloat16",
        z=z_mode,
        activation=activation,
        norm_before_gate=norm_before_gate,
        group_size=group_size or None,
        quant_heads=quant_heads,
        io_bytes=total_bytes,
        io_megabytes=round(total_bytes / 1e6, 2),
        mbu_line=mbu_line,
    )
