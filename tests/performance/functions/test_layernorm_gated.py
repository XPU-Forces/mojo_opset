"""MBU performance suite for ``layer_norm_gated`` (task-book measurement contract).

Definitions, all fixed here rather than hand-entered per run:

* per-call NPU time = sum of every NPU kernel duration inside one call, averaged
  over a 30-call window after 5 warmup calls (``reduction="sum"`` on the profiler
  trace; no ``target_kernel_name`` filter);
* bytes: gated ``(3*M*N + 2*N)*es``, no gate ``(2*M*N + 2*N)*es``, fused quant
  ``2*M*N*es + M*N + 4*M/quant_heads + 2*N*es``;
* ``MBU = bytes / time / B_copy`` where ``B_copy`` is calibrated on site from a
  BF16 ``copy_`` probe (5 warmup + 30 effective), with ``fill_`` reported too.
"""

import json
import os
import subprocess
import tempfile

from pathlib import Path

import pytest
import torch

from tests.performance.timing import kernel_spans, summarize

SAMPLE_PREFIX = "mojo_perf_sample_"
MBU_LINE = 0.60
TAIL_MBU_LINE = 0.50
SMALL_SHAPE_BYTES = 16 * 1024 * 1024
CALIBRATION_ELEMENTS = 192 * 1024 * 1024  # 384 MiB of BF16: working set (768 MiB) well above L2
CSV_ENV = "MOJO_LN_GATED_PERF_CSV"
DEFAULT_CSV = Path("tests/performance/results/layernorm_gated_mbu.csv")


def _npu_smi():
    try:
        return subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False).stdout
    except Exception as error:  # pragma: no cover - diagnostics only
        return f"npu-smi unavailable: {error}"


TRANSIENT_TRACE_ERRORS = ("Incomplete NPU flow", "has no matching device event")


def _window_samples(adapter, call, warmup=5, repeats=30, attempts=3):
    """Return per-call totals (us) of all NPU kernels, one value per call.

    A profiler trace can lose the closing flow event of the last launch; the
    parser rejects such a trace. That is an instrument artifact, so re-take the
    window a bounded number of times and say so, rather than failing the case.
    """

    def run():
        for _ in range(warmup):
            call()
        adapter.runtime.synchronize()
        for index in range(repeats):
            with torch.autograd.profiler.record_function(f"{SAMPLE_PREFIX}{index}"):
                call()
        adapter.runtime.synchronize()

    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory(prefix="mojo-ln-gated-") as directory:
            trace_path = adapter.collect_trace(run, Path(directory))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        try:
            return kernel_spans(
                trace,
                adapter.is_kernel,
                repeats,
                reduction="sum",
                kernel_owners=getattr(adapter, "kernel_owners", None),
            )
        except ValueError as error:
            if attempt == attempts or not any(text in str(error) for text in TRANSIENT_TRACE_ERRORS):
                raise
            print(f"\ntrace parse retry {attempt}/{attempts - 1}: {error}")
    raise AssertionError("unreachable")


CALIBRATION_PAIRS = 5


@pytest.fixture(scope="session")
def calibration_buffers(perf_environment):
    """Four co-existing buffers, allocated once.

    On 910B3 the copy_ probe depends on where the buffers land: four pairs in the
    same process measure 1240 / 1246 / 1430 / 1474 GB/s and each pair is stable
    to +-0.3%. A single fresh pair therefore puts a 19% lottery factor into every
    MBU, so the probe reports the observed range and each case records both ends.
    """
    _, device, _, _ = perf_environment
    return [
        (torch.randn(CALIBRATION_ELEMENTS, dtype=torch.bfloat16, device=device),
         torch.empty(CALIBRATION_ELEMENTS, dtype=torch.bfloat16, device=device))
        for _ in range(CALIBRATION_PAIRS)
    ]


def _copy_bandwidth(adapter, buffers):
    """In-window BF16 copy_ probe over every pair (5 warmup + 30 effective each).

    Buffer size matters more than placement: on 910B3 a 128 MiB buffer pair reads
    ~1470 GB/s (L2 participates) while a 512 MiB pair reads ~1263 GB/s, each
    stable to +-0.3% within a size. Mixing sizes across sessions is what produced
    a 15% denominator swing, so the probe deliberately uses a working set above
    L2 and reports min/median/max for transparency.
    """
    values = []
    for source, destination in buffers:
        samples = _window_samples(adapter, lambda: destination.copy_(source))
        values.append(2 * CALIBRATION_ELEMENTS * 2 / (summarize(samples)["median"] * 1e-6) / 1e9)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    ordered = sorted(values)
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]


@pytest.fixture(scope="session")
def calibrated_bandwidth(perf_environment, request):
    adapter, device, target, _ = perf_environment
    source = torch.randn(CALIBRATION_ELEMENTS, dtype=torch.bfloat16, device=device)
    destination = torch.empty_like(source)
    fill_bytes = source.numel() * source.element_size()
    copy_bytes = 2 * fill_bytes

    fill_us = _window_samples(adapter, lambda: destination.fill_(0.5))
    copy_us = _window_samples(adapter, lambda: destination.copy_(source))
    del source, destination
    result = {
        "target": target,
        "device": device,
        "fill_gbs": fill_bytes / (summarize(fill_us)["median"] * 1e-6) / 1e9,
        "copy_gbs": copy_bytes / (summarize(copy_us)["median"] * 1e-6) / 1e9,
        "fill_us": summarize(fill_us)["median"],
        "copy_us": summarize(copy_us)["median"],
        "samples": copy_us,
    }
    print(f"\ncalibration target={target} fill={result['fill_gbs']:.1f} GB/s copy={result['copy_gbs']:.1f} GB/s")
    request.config._ln_gated_calibration = result
    return result


def _strided_gate(device, dtype, M, N, heads=8, pad=32):
    base = torch.randn(M // heads, heads, N + pad, device=device, dtype=dtype)
    return base[..., :N]


def _strided_x(device, dtype, M, N, pad=32):
    base = torch.randn(M, N + pad, device=device, dtype=dtype)
    return base[:, :N]


# label, M, N, group_size, z_mode, activation, norm_before_gate, is_rms_norm, quant_heads, line
CONFIGS = [
    pytest.param("gdn-decode-32x8", 256, 128, None, "3d", "swish", True, True, 0, None, id="gdn-32x8-n128"),
    pytest.param("gdn-320x8", 2560, 128, None, "3d", "swish", True, True, 0, None, id="gdn-320x8-n128"),
    pytest.param("gdn-4096x8", 32768, 128, None, "3d", "swish", True, True, 0, MBU_LINE, id="gdn-4096x8-n128"),
    pytest.param(
        "gdn-4096x8-sigmoid", 32768, 128, None, "3d", "sigmoid", True, True, 0, MBU_LINE, id="gdn-4096x8-sigmoid"
    ),
    pytest.param("gdn-16384x8", 131072, 128, None, "3d", "swish", True, True, 0, MBU_LINE, id="gdn-16384x8-n128"),
    pytest.param("qk-norm-x3d", 65536, 128, None, "none", "swish", True, True, 0, MBU_LINE, id="qk-4096x16-n128"),
    pytest.param(
        "groupnorm-1024", 1024, 2048, 128, "2d", "swish", False, True, 0, None, id="groupnorm-1024-n2048"
    ),
    pytest.param(
        "groupnorm-8192", 8192, 2048, 128, "2d", "swish", False, True, 0, MBU_LINE, id="groupnorm-8192-n2048"
    ),
    pytest.param("fp8-quant", 32768, 512, None, "none", "swish", True, True, 8, MBU_LINE, id="fp8-quant-qh8"),
    pytest.param("gdn-65536x8", 524288, 128, None, "3d", "swish", True, True, 0, MBU_LINE, id="gdn-65536x8-n128"),
    pytest.param("tail", 32771, 192, None, "2d", "swish", True, True, 0, TAIL_MBU_LINE, id="tail-m32771-n192"),
]


def _byte_count(M, N, es, z_mode, quant_heads):
    if quant_heads:
        return 2 * M * N * es + M * N + 4 * M // quant_heads + 2 * N * es
    streams = 3 if z_mode != "none" else 2
    return (streams * M * N + 2 * N) * es


def _write_csv(report, calibrated):
    path = Path(os.getenv(CSV_ENV, DEFAULT_CSV))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# layernorm_gated MBU report",
        f"# target={calibrated['target']} device={calibrated['device']}",
        f"# B_copy={calibrated['copy_gbs']:.3f} GB/s B_fill={calibrated['fill_gbs']:.3f} GB/s",
        f"# torch={torch.__version__}",
        "# npu-smi:",
        *[f"# {line}" for line in _npu_smi().splitlines()],
    ]
    columns = "case,M,N,dtype,bytes,triton_us,torch_us,MBU,line,pass,copy_low,copy_median,copy_high,gbs,mbu_low".split(",")
    lines = header + [",".join(columns)]
    for row in report:
        lines.append(
            ",".join(
                str(row[key])
                for key in ("case", "M", "N", "dtype", "bytes", "triton_us", "torch_us", "MBU", "line", "pass", "copy_low", "copy_median", "copy_high", "gbs", "mbu_low")
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMBU report: {path}")


@pytest.fixture(scope="session", autouse=True)
def _mbu_report(request):
    request.config._ln_gated_rows = []
    yield
    rows = request.config._ln_gated_rows
    calibrated = getattr(request.config, "_ln_gated_calibration", None)
    if rows and calibrated:
        _write_csv(rows, calibrated)


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize(
    "label,M,N,group_size,z_mode,activation,norm_before_gate,is_rms_norm,quant_heads,line", CONFIGS
)
def test_layernorm_gated_mbu(
    perf_environment,
    calibrated_bandwidth,
    calibration_buffers,
    label,
    M,
    N,
    group_size,
    z_mode,
    activation,
    norm_before_gate,
    is_rms_norm,
    quant_heads,
    line,
    request,
):
    from mojo_opset import functions as F

    adapter, device, target, implementation = perf_environment
    if quant_heads and not target.startswith("npu.a5") and implementation != "torch":
        pytest.skip("fused FP8 epilogue is scoped to Ascend 950")

    dtype = torch.bfloat16
    es = 2  # bfloat16
    torch.manual_seed(1234)
    x = _strided_x(device, dtype, M, N) if label == "qk-norm-x3d" else torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    z = None
    if z_mode == "3d":
        z = _strided_gate(device, dtype, M, N)
    elif z_mode == "2d":
        z = torch.randn(M, N, device=device, dtype=dtype)

    def call_triton():
        return F.layer_norm_gated(
            x,
            weight,
            None,
            eps=1e-6,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            activation=activation,
            quant_heads=quant_heads,
            implementation=implementation,
        )

    def call_torch():
        return F.layer_norm_gated(
            x,
            weight,
            None,
            eps=1e-6,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            activation=activation,
            quant_heads=quant_heads,
            implementation="torch",
        )

    # Calibrate around the case: the copy_ probe itself drifts between windows
    # (1252 vs 1460 GB/s observed), so the denominator must be taken in-window.
    low_before, mid_before, high_before = _copy_bandwidth(adapter, calibration_buffers)
    triton_samples = _window_samples(adapter, call_triton)
    triton_us = summarize(triton_samples)["median"]
    torch_us = summarize(_window_samples(adapter, call_torch))["median"]
    low_after, mid_after, high_after = _copy_bandwidth(adapter, calibration_buffers)
    copy_low = min(low_before, low_after)
    copy_high = max(high_before, high_after)
    copy_gbs = 0.5 * (mid_before + mid_after)

    byte_count = _byte_count(M, N, es, z_mode, quant_heads)
    gbs = byte_count / (triton_us * 1e-6) / 1e9
    mbu = gbs / copy_gbs
    mbu_low = gbs / copy_low

    if byte_count >= SMALL_SHAPE_BYTES:
        passed = mbu >= line
        verdict = f"MBU {mbu:.1%} (median cal, slow end {mbu_low:.1%}), {gbs:.0f} GB/s vs line {line:.0%}"
    else:
        # Small shapes are judged against the torch eager composition instead of MBU.
        passed = triton_us <= torch_us
        verdict = f"triton {triton_us:.2f}us vs torch {torch_us:.2f}us"

    row = {
        "case": label,
        "M": M,
        "N": N,
        "dtype": "bfloat16",
        "bytes": byte_count,
        "triton_us": f"{triton_us:.3f}",
        "torch_us": f"{torch_us:.3f}",
        "MBU": f"{mbu:.4f}",
        "line": "" if line is None else f"{line:.2f}",
        "pass": passed,
        "copy_low": f"{copy_low:.1f}",
        "copy_median": f"{copy_gbs:.1f}",
        "copy_high": f"{copy_high:.1f}",
        "gbs": f"{gbs:.1f}",
        "mbu_low": f"{mbu_low:.4f}",
    }
    request.config._ln_gated_rows.append(row)

    print(f"\n{label:22s} M={M:<7d} N={N:<5d} bytes={byte_count/1e6:8.2f}MB {verdict}")
    assert passed, f"{label}: {verdict}"
