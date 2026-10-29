"""Benchmark the fused QK RMSNorm+RoPE kernels with the task's MBU formula."""

import argparse
import csv
import json
import os
import statistics
import subprocess

from pathlib import Path

import torch
import torch_npu

if os.getenv("MOJO_ASCEND_ARCH", "a2").startswith("a5"):
    from mojo_opset.kernels.npu_a5_triton import qk_norm_rope as kernel
else:
    from mojo_opset.kernels.npu_a2_triton import qk_norm_rope as kernel

DEFAULT_SHAPES = (
    (32, 8),
    (320, 8),
    (320, 56),
    (1024, 56),
    (4096, 56),
    (16384, 56),
    (69888, 56),
    (4096, 57),
)
COPY_BYTES = 256 * 1024 * 1024


def _profile_kernel_samples(
    launch,
    profile_dir: Path,
    warmup: int,
    repeats: int,
) -> tuple[list[float], int]:
    """Return per-call device times and the number of kernels in each call."""
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ):
        for _ in range(repeats):
            launch()
        torch.npu.synchronize()

    kernel_files = list(profile_dir.rglob("kernel_details.csv"))
    if len(kernel_files) != 1:
        raise RuntimeError(f"Expected one kernel_details.csv under {profile_dir}, got {kernel_files}")
    with kernel_files[0].open(newline="", encoding="utf-8") as stream:
        durations = [float(row["Duration(us)"]) for row in csv.DictReader(stream)]
    if not durations or len(durations) % repeats:
        raise RuntimeError(f"Kernel rows are not divisible by repeats: rows={len(durations)}, repeats={repeats}")
    kernels_per_call = len(durations) // repeats
    samples = [
        sum(durations[index * kernels_per_call : (index + 1) * kernels_per_call]) / 1e3 for index in range(repeats)
    ]
    return samples, kernels_per_call


def _copy_bandwidth_bytes_per_ms(
    warmup: int,
    repeats: int,
    output_dir: Path,
) -> tuple[float, float]:
    elements = COPY_BYTES // torch.bfloat16.itemsize
    src = torch.empty(elements, device="npu", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    samples, _ = _profile_kernel_samples(
        lambda: dst.copy_(src),
        output_dir / "copy_probe",
        warmup,
        repeats,
    )
    copy_ms = statistics.median(samples)
    traffic_bytes = 2 * COPY_BYTES
    return traffic_bytes / copy_ms, copy_ms


def _measure_shape(
    tokens: int,
    heads: int,
    output_dir: Path,
    warmup: int,
    repeats: int,
) -> dict:
    packed = torch.randn(tokens, heads * 3, 128, device="npu", dtype=torch.bfloat16)
    q = packed[:, :heads]
    k = packed[:, heads : heads * 2]
    q_weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
    k_weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(tokens, 48, device="npu")
    rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return kernel.fused_qk_norm_rope_fwd(q, k, q_weight, k_weight, rope_table, 1e-5)

    profile_dir = output_dir / f"tokens_{tokens}_heads_{heads}"
    samples, kernels_per_call = _profile_kernel_samples(
        launch,
        profile_dir,
        warmup,
        repeats,
    )
    kernel_ms = sum(samples) / repeats
    traffic_bytes = 2 * (4 * tokens * heads * 128 + tokens * 96 + 2 * 128)
    return {
        "tokens": tokens,
        "heads": heads,
        "total_kernel_ms": kernel_ms,
        "kernels_per_call": kernels_per_call,
        "traffic_bytes": traffic_bytes,
        "profile_dir": str(profile_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/performance/results/fused_qk_norm_rope"))
    parser.add_argument(
        "--shapes",
        default=",".join(f"{tokens}x{heads}" for tokens, heads in DEFAULT_SHAPES),
        help="Comma-separated TOKENSxHEADS values.",
    )
    args = parser.parse_args()

    torch_npu.npu.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_bw, copy_ms = _copy_bandwidth_bytes_per_ms(args.warmup, args.repeats, args.output_dir)
    rows = []
    for item in args.shapes.split(","):
        tokens_text, heads_text = item.lower().split("x", 1)
        row = _measure_shape(
            int(tokens_text),
            int(heads_text),
            args.output_dir,
            args.warmup,
            args.repeats,
        )
        row["mbu"] = row["traffic_bytes"] / (row["total_kernel_ms"] * copy_bw)
        rows.append(row)
        print(
            f"tokens={row['tokens']:5d} heads={row['heads']:2d} "
            f"kernel={row['total_kernel_ms'] * 1000:8.2f} us MBU={row['mbu']:.2%}"
        )

    report = {
        "device": args.device,
        "copy_probe_ms": copy_ms,
        "copy_probe_bytes": 2 * COPY_BYTES,
        "copy_bandwidth_gbps": copy_bw * 1e3 / 1e9,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
        "npu_smi": subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=False).stdout,
    }
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
