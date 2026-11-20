"""Compare small-shape end-to-end latency with the native NPU operator chain."""

import csv
import json
import shutil
import time

from pathlib import Path

import torch
import torch_npu

from mojo_opset import functions

SHAPES = ((32, 8), (320, 8))
WARMUP = 5
REPEATS = 30


def _native_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps):
    q_norm = torch_npu.npu_rms_norm(q, q_weight, epsilon=eps)[0]
    k_norm = torch_npu.npu_rms_norm(k, k_weight, epsilon=eps)[0]

    def rotate(x):
        cos = rope_table[:, :48].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = rope_table[:, 48:].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        rotated = torch_npu.npu_rotary_mul(x[..., :96].unsqueeze(0), cos, sin).squeeze(0)
        return torch.cat((rotated, x[..., 96:]), dim=-1)

    return rotate(q_norm), rotate(k_norm)


def _event_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS


def _profile_kernel_ms(fn, profile_dir: Path):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=config,
    ):
        for _ in range(REPEATS):
            fn()
        torch.npu.synchronize()
    kernel_files = list(profile_dir.rglob("kernel_details.csv"))
    if len(kernel_files) != 1:
        raise RuntimeError(f"Expected one kernel_details.csv under {profile_dir}, got {kernel_files}")
    with kernel_files[0].open(newline="", encoding="utf-8") as stream:
        durations = [float(row["Duration(us)"]) for row in csv.DictReader(stream)]
    return sum(durations) / REPEATS / 1000, sum(durations)


def main():
    output_dir = Path("tests/performance/results/fused_qk_norm_rope_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_npu.npu.set_device(0)
    torch.manual_seed(17)
    rows = []
    for tokens, heads in SHAPES:
        packed = torch.randn(tokens, heads * 3, 128, device="npu", dtype=torch.bfloat16)
        q = packed[:, :heads]
        k = packed[:, heads : heads * 2]
        q_weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
        k_weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
        angles = torch.randn(tokens, 48, device="npu")
        rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

        def triton_call():
            return functions.fused_qk_norm_rope(
                q,
                k,
                q_weight,
                k_weight,
                rope_table,
                1e-5,
            )

        def native_call():
            return _native_qk_norm_rope(q, k, q_weight, k_weight, rope_table, 1e-5)

        expected_q, expected_k = native_call()
        actual_q, actual_k = triton_call()
        torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
        torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)
        triton_kernel_ms, triton_kernel_sum_us = _profile_kernel_ms(
            triton_call,
            output_dir / f"triton_{tokens}x{heads}",
        )
        native_kernel_ms, native_kernel_sum_us = _profile_kernel_ms(
            native_call,
            output_dir / f"native_{tokens}x{heads}",
        )
        row = {
            "tokens": tokens,
            "heads": heads,
            "triton_kernel_ms": triton_kernel_ms,
            "native_kernel_ms": native_kernel_ms,
            "triton_kernel_sum_us": triton_kernel_sum_us,
            "native_kernel_sum_us": native_kernel_sum_us,
            "triton_e2e_ms": _event_ms(triton_call),
            "native_e2e_ms": _event_ms(native_call),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        rows.append(row)
        print(row)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"report={output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
