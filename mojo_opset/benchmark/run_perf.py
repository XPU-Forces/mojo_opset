"""Lightweight single-process perf runner for mojo_opset benchmark op_defs.

Reuses xpu-perf's micro_perf machinery (the real ``BackendNPU.perf`` timing loop,
``BasicOp.summary`` metrics, and ``export_reports`` report format) but runs every
case in-process -- no server, engine, or subprocess spawning. Each case is run for
every provider (``base`` torch reference + registered vendors such as
``torch_npu``) and a side-by-side comparison table is printed.

Examples:
    # default: run descriptor cases tagged "smoke" on device 0
    python -m mojo_opset.benchmark.run_perf

    # run full descriptor cases + write micro_perf-style reports
    python -m mojo_opset.benchmark.run_perf --preset full --report_dir benchmark_reports

    # restrict providers
    python -m mojo_opset.benchmark.run_perf --providers base,torch_npu
"""
import argparse
import json

import prettytable

from xpu_perf.micro_perf.backends.NPU.backend_npu import BackendNPU
from xpu_perf.micro_perf.core.common_utils import export_reports

from .runner_common import (
    BASE_PROVIDER,
    DEFAULT_PRESET,
    build_provider_map,
    case_provider_support,
    resolve_test_cases,
    select_plugin_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        help="descriptor case tag to run (for example smoke, full, or all)",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="optional comma-separated descriptor op names",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--providers",
        type=str,
        default="base,torch_npu,ttx",
        help="comma-separated provider order; unavailable providers are skipped per target",
    )
    parser.add_argument(
        "--timing",
        choices=("profiler", "event"),
        default=None,
        help="override descriptor timing; default is profiler unless the descriptor says otherwise",
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
        raise RuntimeError(f"failed to build {op_name}/{op_provider}: {err}") from err

    try:
        target = backend.perf(op_instance)
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"failed to perf {op_name}/{op_provider}: {err}") from err
    if not target:
        raise RuntimeError(
            f"xpu-perf returned no result for {op_name}/{op_provider}; "
            "inspect the preceding backend traceback"
        )
    return target


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
    label_keys = (
        "__case_id__",
        "m",
        "k",
        "n",
        "num_tokens",
        "in_features",
        "out_features",
        "output_dtype",
        "trans_weight",
    )
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
    requested_ops = [name.strip() for name in args.ops.split(",") if name.strip()] if args.ops else None

    op_defs, vendor_ops = select_plugin_paths()
    backend = BackendNPU(backend="NPU", op_defs=op_defs, vendor_ops=[vendor_ops])
    backend.set_device(args.device)
    backend.load_all_ops()

    test_cases = resolve_test_cases(args.preset, requested_ops, args.timing)
    if not test_cases:
        raise SystemExit("no benchmark cases matched the requested preset/ops")

    report_cases = {}
    bench_results = {}
    for op_name, cases in test_cases.items():
        provider_map = build_provider_map(backend, op_name, requested_providers)
        if not provider_map:
            raise RuntimeError(f"op {op_name!r} has no registered requested providers")

        print("#" * 100)
        print(f"# op: {op_name} | providers: {list(provider_map)} | cases: {len(cases)}")
        print("#" * 100)

        provider_run_counts = {provider: 0 for provider in provider_map}
        executed_cases = []
        op_results = []
        for case in cases:
            runnable_providers = {}
            for provider, op_cls in provider_map.items():
                supported, reason = case_provider_support(op_name, provider, case)
                if supported:
                    runnable_providers[provider] = op_cls
                else:
                    print(
                        f"SKIP {op_name}/{case.get('__case_id__')}/{provider}: {reason}"
                    )

            if not runnable_providers:
                continue

            provider_targets = {}
            for provider, op_cls in runnable_providers.items():
                target = run_case(backend, op_name, provider, op_cls, case)
                provider_targets[provider] = target
                provider_run_counts[provider] += 1
                print_case_report(op_name, provider, args.device, case, target)
            print_comparison(op_name, case, provider_targets)
            executed_cases.append(case)
            op_results.append(provider_targets)

        if not executed_cases:
            raise RuntimeError(f"op {op_name!r} has no cases supported by the requested providers")
        inactive = [provider for provider, count in provider_run_counts.items() if count == 0]
        if inactive:
            raise RuntimeError(
                f"op {op_name!r} has no selected cases supported by providers {inactive}"
            )
        report_cases[op_name] = executed_cases
        bench_results[op_name] = op_results

    if args.report_dir:
        info_dict = build_info_dict(backend, args.device)
        export_reports(args.report_dir, info_dict, report_cases, bench_results)


if __name__ == "__main__":
    main()
