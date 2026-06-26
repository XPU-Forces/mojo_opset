"""Lightweight single-process perf runner for mojo_opset benchmark op_defs.

Reuses xpu-perf's micro_perf machinery (the real ``BackendNPU.perf`` timing loop,
``BasicOp.summary`` metrics, and ``export_reports`` report format) but runs every
case in-process -- no server, engine, or subprocess spawning. Each case is run for
every provider (``base`` torch reference + registered vendors such as
``torch_npu``) and a side-by-side comparison table is printed.

Examples:
    # default: run benchmark/workloads/mojo_quant_gemm.json on device 0
    python benchmark/run_perf.py

    # custom workload + write micro_perf-style reports
    python benchmark/run_perf.py --workload benchmark/workloads/mojo_quant_gemm.json \
        --device 0 --report_dir benchmark/reports

    # restrict providers
    python benchmark/run_perf.py --providers base,torch_npu
"""
import argparse
import json
import pathlib
import sys

import prettytable

FILE_DIR = pathlib.Path(__file__).parent.absolute()
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

from xpu_perf.micro_perf.backends.NPU.backend_npu import BackendNPU
from xpu_perf.micro_perf.core.common_utils import export_reports, parse_workload

from runner_common import (
    BASE_PROVIDER,
    DEFAULT_WORKLOAD,
    OP_DEFS,
    VENDOR_OPS,
    build_provider_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", type=str, default=str(DEFAULT_WORKLOAD))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--providers",
        type=str,
        default="base,torch_npu",
        help="comma-separated provider order; 'base' is the torch reference baseline",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="if set, also export micro_perf-style jsonl/csv reports here",
    )
    return parser.parse_args()


def run_case(backend, op_name, op_provider, op_cls, case):
    """Build + perf a single (provider, case); returns the summary target dict."""
    try:
        op_instance = op_cls(case, backend)
        op_instance.is_concurrent = False
    except Exception as err:  # noqa: BLE001
        print(f"[{op_provider}] failed to build op: {err}")
        return {}

    try:
        return backend.perf(op_instance)
    except Exception as err:  # noqa: BLE001
        print(f"[{op_provider}] failed to perf op: {err}")
        return {}


def print_case_report(op_name, op_provider, device_id, case, target_dict):
    """Per-case block in micro_perf style (key/value table + args + targets)."""
    pt = prettytable.PrettyTable()
    pt.field_names = ["key", "value"]
    pt.align = "l"
    pt.add_row(["op_name", op_name])
    pt.add_row(["op_provider", op_provider])
    pt.add_row(["device_id", str(device_id)])
    print(pt)
    print(json.dumps(case))
    print(json.dumps(target_dict, indent=4))
    print("")


def print_comparison(op_name, case, provider_targets):
    """Side-by-side comparison table; speedup is relative to 'base'."""
    label_keys = ("num_tokens", "in_features", "out_features", "output_dtype", "trans_weight")
    label = ", ".join(f"{k}={case[k]}" for k in label_keys if k in case)
    print(f"=== comparison | {op_name} | {label} ===")

    pt = prettytable.PrettyTable()
    pt.field_names = ["provider", "latency(us)", "tflops", "mem_bw(GB/s)", "speedup_vs_base"]
    pt.align = "r"
    pt.align["provider"] = "l"

    base_latency = None
    if BASE_PROVIDER in provider_targets and provider_targets[BASE_PROVIDER].get("latency(us)"):
        base_latency = provider_targets[BASE_PROVIDER]["latency(us)"]

    for provider, target in provider_targets.items():
        latency = target.get("latency(us)")
        if not latency:
            pt.add_row([provider, "-", "-", "-", "-"])
            continue
        tflops = target.get("calc_flops_power(tflops)", "-")
        mem_bw = target.get("mem_bw(GB/s)", "-")
        speedup = f"{base_latency / latency:.2f}x" if base_latency else "-"
        pt.add_row([provider, f"{latency:.3f}", tflops, mem_bw, speedup])

    print(pt)
    print("")


def build_info_dict(backend, device_id):
    return {
        "backend_type": backend.backend_type,
        "common": backend.common_info,
        "provider": backend.provider_info,
        "backend": backend.backend_info,
        "runtime": {
            "device_mapping": [device_id],
            "device_ids": [device_id],
            "numa_num": 1,
            "numa_order": [0],
            "node_world_size": 1,
            "node_rank": 0,
            "all_numa_num": 1,
        },
    }


def main():
    args = parse_args()
    requested_providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    backend = BackendNPU(backend="NPU", op_defs=OP_DEFS, vendor_ops=[VENDOR_OPS])
    backend.set_device(args.device)
    backend.load_all_ops()

    test_cases = parse_workload(args.workload)
    if not test_cases:
        raise SystemExit(f"no test cases found in {args.workload}")

    bench_results = {}
    for op_name, cases in test_cases.items():
        provider_map = build_provider_map(backend, op_name, requested_providers)
        if not provider_map:
            print(f"[skip] op '{op_name}' has no registered providers")
            bench_results[op_name] = [{} for _ in cases]
            continue

        print("#" * 100)
        print(f"# op: {op_name} | providers: {list(provider_map)} | cases: {len(cases)}")
        print("#" * 100)

        op_results = []
        for case in cases:
            provider_targets = {}
            for provider, op_cls in provider_map.items():
                target = run_case(backend, op_name, provider, op_cls, case)
                provider_targets[provider] = target
                print_case_report(op_name, provider, args.device, case, target)
            print_comparison(op_name, case, provider_targets)
            op_results.append(provider_targets)
        bench_results[op_name] = op_results

    if args.report_dir:
        info_dict = build_info_dict(backend, args.device)
        export_reports(args.report_dir, info_dict, test_cases, bench_results)


if __name__ == "__main__":
    main()
