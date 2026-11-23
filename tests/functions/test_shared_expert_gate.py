"""Accuracy tests for functions.shared_expert_gate (700 deterministic cases).

Reference: ``sigmoid(F.linear(x, gate_weight)) * shared_out``.

Coverage
--------
* the 10 production shapes, bf16 + fp16
* 22 token counts around every tile boundary (1, 7, 8, 9, ..., 257)
* 18 hidden sizes that are not multiples of 16 / 256 (17, 31, ..., 8191)
* fp32 (halved BLOCK_K branch)
* 3-D ``[B, S, H]`` inputs
* non-contiguous strided inputs
* special values: zeros / ones / large (sigmoid saturation) / tiny / sparse
* seeded pseudo-random fill up to exactly 700 cases
"""

import random

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32

SHAPES = [
    (8192, 2048), (16384, 2048), (32768, 2048),
    (4096, 3584), (8192, 3584),
    (4096, 4096), (8192, 4096),
    (4096, 5120), (8192, 5120),
    (8192, 7168),
]
N_EDGES = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
           63, 64, 65, 127, 128, 129, 255, 256, 257]
D_ODD = [17, 31, 33, 63, 65, 127, 129, 255, 257, 511,
         513, 1023, 1025, 3585, 4097, 5119, 7169, 8191]
T_3D = [(1, 1, 2048), (1, 7, 2048), (2, 3, 16), (3, 5, 128),
        (7, 1, 3584), (1, 64, 5120), (4, 8, 2048), (2, 2, 17),
        (5, 3, 65), (8, 1, 4096)]
N_STRIDED = [(1, 2048), (7, 128), (64, 3584), (257, 5120),
             (1024, 2048), (33, 65), (128, 4096), (17, 17),
             (256, 257), (65, 1023)]
SPECIAL = [(1, 2048), (7, 128), (64, 3584), (257, 5120), (1024, 2048)]
MODES = ["zeros", "ones", "large", "tiny", "sparse"]


def _build_cases():
    """Every case is (a, b, dtype, mode, extra); see ``test_shared_expert_gate_accuracy``."""
    cases = []

    for t, h in SHAPES:
        for dt in (BF16, FP16):
            cases.append((t, h, dt, "randn", None))

    for t in N_EDGES:
        for h in (16, 128, 2048):
            for dt in (BF16, FP16):
                cases.append((t, h, dt, "randn", None))

    for t in (1, 7, 64, 257):
        for h in D_ODD:
            for dt in (BF16, FP16):
                cases.append((t, h, dt, "randn", None))

    for t in (1, 7, 64, 257, 1024, 4096):
        for h in (16, 128, 2048, 3584):
            cases.append((t, h, FP32, "randn", None))

    # a=batch, b=seq_len, extra=hidden
    for batch, seq, h in T_3D:
        for dt in (BF16, FP16):
            cases.append((batch, seq, dt, "3d", h))

    for t, h in N_STRIDED:
        for dt in (BF16, FP16):
            cases.append((t, h, dt, "strided", None))

    for t, h in SPECIAL:
        for mode in MODES:
            for dt in (BF16, FP16):
                cases.append((t, h, dt, mode, None))

    rng = random.Random(20261118)
    seen = set(cases)
    t_pool = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96,
              127, 128, 129, 192, 255, 256, 257, 384, 512, 1000, 1024, 2048,
              4096]
    h_pool = [16, 17, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 256,
              257, 384, 512, 513, 768, 1024, 1025, 1536, 2048, 2049, 3072,
              3584, 3585, 4096, 5120, 7168, 8191]
    while len(cases) < 700:
        t = rng.choice(t_pool)
        h = rng.choice(h_pool)
        dt = rng.choice([BF16, FP16, FP16, BF16, FP32])
        c = (t, h, dt, "randn", None)
        if c in seen:
            continue
        seen.add(c)
        cases.append(c)
    return cases


CASES = _build_cases()
assert len(CASES) == 700, len(CASES)


def _seed_of(case):
    a, b, dt, mode, extra = case
    return (a * 7919 + b * 104729 + len(mode) * 31
            + (extra or 0) * 13 + int(dt == FP32) * 5) % (2 ** 31)


def _randn(shape, dtype, g, device):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype)


def _make(case, device):
    tokens, hidden, dtype, mode, _ = case
    g = torch.Generator(device="cpu").manual_seed(_seed_of(case))
    x = _randn((tokens, hidden), dtype, g, device)
    w = _randn((1, hidden), dtype, g, device)
    s = _randn((tokens, hidden), dtype, g, device)

    if mode == "zeros":
        x = torch.zeros_like(x)
        s = torch.zeros_like(s)
    elif mode == "ones":
        x = torch.ones_like(x)
        s = torch.ones_like(s)
    elif mode == "large":
        # Scale only x: the score saturates sigmoid, but keeping s at O(1)
        # avoids amplifying the inherent fp32 reduction-order difference on
        # the rare rows whose dot product nearly cancels.
        x = x * 1e2
    elif mode == "tiny":
        x = x * 1e-3
        s = s * 1e-3
    elif mode == "sparse":
        mask = (torch.rand(tokens, hidden, generator=g) > 0.9).to(
            device=device, dtype=dtype)
        x = x * mask
        s = s * mask
    return x, w, s


def _make_3d(case, device):
    batch, seq, dtype, _, hidden = case
    g = torch.Generator(device="cpu").manual_seed(_seed_of(case))
    return (_randn((batch, seq, hidden), dtype, g, device),
            _randn((1, hidden), dtype, g, device),
            _randn((batch, seq, hidden), dtype, g, device))


def _make_strided(case, device):
    tokens, hidden, dtype, _, _ = case
    g = torch.Generator(device="cpu").manual_seed(_seed_of(case))
    xb = _randn((tokens, hidden * 2), dtype, g, device)
    sb = _randn((tokens, hidden * 2), dtype, g, device)
    wb = _randn((1, hidden * 2), dtype, g, device)
    return xb[:, ::2], wb[:, ::2], sb[:, ::2]


def _id(c):
    return f"{c[0]}x{c[1]}-{str(c[2]).split('.')[-1]}-{c[3]}"


@pytest.mark.api("functions.shared_expert_gate")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_shared_expert_gate_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    tokens, hidden, dtype, mode, _ = case
    if mode == "3d":
        x, w, s = _make_3d(case, device)
    elif mode == "strided":
        x, w, s = _make_strided(case, device)
    else:
        x, w, s = _make(case, device)

    expected = F.shared_expert_gate(x, w, s, implementation="torch")
    actual = F.shared_expert_gate(x, w, s, implementation=implementation)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert_close(actual, expected, dtype=dtype)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("shape", [(7, 513), (128, 4096), (1024, 2048)])
@pytest.mark.parametrize("dtype", [BF16, FP16, FP32])
def test_shared_expert_gate_reference_matches_cpu_fp64(accuracy_backend, shape, dtype):
    # A CPU FP64 oracle catches bugs in the torch reference itself.
    _, _, device = accuracy_backend
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=dtype)
    w = torch.randn(1, shape[-1], device=device, dtype=dtype)
    s = torch.randn(shape, device=device, dtype=dtype)

    actual = F.shared_expert_gate(x, w, s, implementation="torch")
    expected = (
        torch.sigmoid(torch.nn.functional.linear(
            x.detach().cpu().double(), w.detach().cpu().double()))
        * s.detach().cpu().double()
    ).to(dtype)
    assert_close(actual, expected.to(device), dtype=dtype)
