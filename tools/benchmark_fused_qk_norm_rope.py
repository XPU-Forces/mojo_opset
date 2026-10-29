"""Benchmark the fused QK RMSNorm+RoPE kernels with the task's MBU formula."""

import argparse
import json
import os
import subprocess

from pathlib import Path

import torch
import torch_npu

from triton.backends.ascend.testing import do_bench_npu

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
TARGET_KERNEL = "_qk_norm_rope_kernel"


def _copy_bandwidth_bytes_per_ms(warmup: int, repeats: int) -> tuple[float, float]:
    elements = COPY_BYTES // torch.bfloat16.itemsize
    src = torch.empty(elements, device="npu", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    copy_ms = do_bench_npu(
        lambda: dst.copy_(src),
        warmup=warmup,
        active=repeats,
        clear_l2_cache=True,
    )
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
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)

    def launch_q() -> None:
        kernel._launch_qk_norm_rope(q, q_weight, rope_table, q_out, 1e-5)

    def launch_k() -> None:
        kernel._launch_qk_norm_rope(k, k_weight, rope_table, k_out, 1e-5)

    profile_dir = output_dir / f"tokens_{tokens}_heads_{heads}"
    q_ms, k_ms = do_bench_npu(
        [launch_q, launch_k],
        warmup=warmup,
        active=repeats,
        prof_dir=str(profile_dir),
        keep_res=True,
        target_kernel_name=TARGET_KERNEL,
    )
    kernel_ms = q_ms + k_ms
    traffic_bytes = 2 * (4 * tokens * heads * 128 + tokens * 96 + 2 * 128)
    return {
        "tokens": tokens,
        "heads": heads,
        "q_kernel_ms": q_ms,
        "k_kernel_ms": k_ms,
        "total_kernel_ms": kernel_ms,
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
    copy_bw, copy_ms = _copy_bandwidth_bytes_per_ms(args.warmup, args.repeats)
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
