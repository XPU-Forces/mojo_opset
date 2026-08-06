"""Multi-process launcher for descriptor-based mojo_opset benchmarks.

This wrapper keeps xpu-perf's subprocess, multi-device, NUMA, profiling and
reporting machinery. Provider capability checks are applied before dispatch;
each provider receives only the descriptor cases it supports.

Examples:
    # run all smoke cases on all devices (backend is inferred)
    python -m mojo_opset.benchmark.launch

    # restrict provider and devices
    python -m mojo_opset.benchmark.launch --device 0,1 \
        --providers torch_npu --preset full
"""

import argparse
import pathlib
from functools import partial

import torch.multiprocessing as mp

from xpu_perf.micro_perf.core.common_utils import existing_dir_path
from xpu_perf.micro_perf.core.common_utils import export_reports
from xpu_perf.micro_perf.core.common_utils import get_submodules
from xpu_perf.micro_perf.core.common_utils import logger
from xpu_perf.micro_perf.core.common_utils import setup_logger
from xpu_perf.micro_perf.core.common_utils import valid_file
from xpu_perf.micro_perf.core.perf_engine import XpuPerfServer

from mojo_opset.benchmark import build_test_cases

from .runner_common import build_provider_map
from .runner_common import case_provider_support
from .runner_common import detect_xpu_backend
from .runner_common import parse_requested_providers
from .runner_common import select_plugin_paths

FILE_DIR = pathlib.Path(__file__).parent.absolute()

mp.set_start_method("spawn", force=True)


def parse_args():
    setup_logger("INFO")

    backend_name_list, backend_mod_list = get_submodules("xpu_perf.micro_perf.backends")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=backend_name_list,
        help="xpu-perf backend; default is inferred from the current Mojo platform",
    )
    parser.add_argument("--op_defs", type=existing_dir_path, default=None)
    parser.add_argument("--vendor_ops", type=existing_dir_path, default=None, action="append")
    parser.add_argument(
        "--env",
        type=partial(valid_file, required_suffix=".json"),
        default=None,
        help="Path to env JSON file (default: vendor_ops/{backend}/env.json)",
    )

    parser.add_argument("--numa", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="e.g. '0' or '0,1'; default all devices")
    parser.add_argument("--node_world_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    parser.add_argument("--host_port", type=int, default=49372)
    parser.add_argument("--device_port", type=int, default=49373)

    parser.add_argument("--preset", type=str, default="smoke", help="descriptor tag: smoke, full, or all")
    parser.add_argument("--ops", type=str, default=None, help="optional comma-separated descriptor op names")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="comma-separated providers; default selects base and all available target providers",
    )
    parser.add_argument(
        "--timing",
        choices=("profiler", "event"),
        default=None,
        help="override descriptor timing; default is profiler unless the descriptor says otherwise",
    )
    parser.add_argument("--report_dir", type=str, default=str(pathlib.Path.cwd() / "benchmark_reports"))

    args = parser.parse_args()
    default_op_defs, default_vendor_ops = select_plugin_paths()
    args.op_defs = args.op_defs or default_op_defs
    args.vendor_ops = args.vendor_ops or [default_vendor_ops]
    args.backend = args.backend or detect_xpu_backend(backend_name_list)
    args.script_dir = FILE_DIR
    args.backend_name_list = backend_name_list
    args.backend_mod_list = backend_mod_list
    return args


def load_test_cases(args):
    op_names = [name.strip() for name in args.ops.split(",") if name.strip()] if args.ops else None
    return build_test_cases(preset=args.preset, op_names=op_names, timing=args.timing)


def log_test_cases(test_cases):
    print("*" * 100)
    logger.info("test cases:")
    for op_name, op_cases in test_cases.items():
        logger.info(f"{op_name} has {len(op_cases)} test cases")
    print("*" * 100)


def prepare_provider_matrix(server, test_cases, requested_providers):
    """Select cases supported by at least one available requested provider."""

    available_by_op = {}
    selected_cases = {}
    for op_name, cases in test_cases.items():
        provider_map = build_provider_map(server.backend_instance, op_name, requested_providers)
        if not provider_map:
            if requested_providers is None:
                logger.info(f"skip {op_name}: no provider is available in this environment")
                continue
            raise RuntimeError(f"op {op_name!r} has no registered requested providers")

        inactive = [
            provider
            for provider in provider_map
            if not any(case_provider_support(op_name, provider, case)[0] for case in cases)
        ]
        if inactive and requested_providers is not None:
            raise RuntimeError(
                f"op {op_name!r} has no selected cases supported by providers {inactive}"
            )
        if inactive:
            provider_map = {
                provider: op_cls
                for provider, op_cls in provider_map.items()
                if provider not in inactive
            }
        if not provider_map:
            logger.info(f"skip {op_name}: no provider supports the selected cases")
            continue

        available_by_op[op_name] = provider_map

        selected = []
        for case in cases:
            supported = [
                provider
                for provider in provider_map
                if case_provider_support(op_name, provider, case)[0]
            ]
            if supported:
                selected.append(case)
                continue
            reasons = {
                provider: case_provider_support(op_name, provider, case)[1]
                for provider in provider_map
            }
            logger.info(f"skip {op_name}/{case.get('__case_id__')}: {reasons}")

        if not selected:
            raise RuntimeError(
                f"op {op_name!r} has no cases supported by the requested providers"
            )
        selected_cases[op_name] = selected
    if not selected_cases:
        raise RuntimeError("no benchmark target has a runnable provider in this environment")

    return selected_cases, available_by_op


def run_provider_matrix(server, test_cases, available_by_op, requested_providers):
    """Run provider-specific case batches through xpu-perf and merge results."""

    bench_results = {
        op_name: [{} for _ in cases]
        for op_name, cases in test_cases.items()
    }
    original_mapping = server.backend_instance.op_mapping
    try:
        if requested_providers is None:
            provider_order = list(dict.fromkeys(
                provider for provider_map in available_by_op.values() for provider in provider_map
            ))
        else:
            provider_order = requested_providers
        for provider in provider_order:
            provider_cases = {}
            provider_indices = {}
            provider_mapping = {}

            for op_name, cases in test_cases.items():
                op_provider_map = available_by_op[op_name]
                if provider not in op_provider_map:
                    continue

                selected = []
                indices = []
                for case_index, case in enumerate(cases):
                    supported, reason = case_provider_support(op_name, provider, case)
                    if supported:
                        selected.append(case)
                        indices.append(case_index)
                    else:
                        logger.info(
                            f"skip {op_name}/{case.get('__case_id__')}/{provider}: {reason}"
                        )

                if selected:
                    provider_cases[op_name] = selected
                    provider_indices[op_name] = indices
                    provider_mapping[op_name] = {provider: op_provider_map[provider]}

            if not provider_cases:
                continue

            server.backend_instance.op_mapping = provider_mapping
            provider_results = server.normal_bench(provider_cases)
            for op_name, case_results in provider_results.items():
                for original_index, result in zip(provider_indices[op_name], case_results):
                    bench_results[op_name][original_index].update(result)
    finally:
        server.backend_instance.op_mapping = original_mapping

    return bench_results


def raise_for_failed_results(test_cases, bench_results):
    """Turn xpu-perf child-process failures into a launcher error."""

    failures = []
    for op_name, cases in test_cases.items():
        op_results = bench_results.get(op_name)
        if op_results is None:
            failures.append(f"{op_name}: no results")
            continue
        if len(op_results) != len(cases):
            failures.append(f"{op_name}: expected {len(cases)} cases, got {len(op_results)}")
            continue
        for case_index, (case, provider_results) in enumerate(zip(cases, op_results)):
            case_id = case.get("__case_id__", case_index)
            if not provider_results:
                failures.append(f"{op_name}/{case_id}: no provider results")
                continue
            for provider, target in provider_results.items():
                if not target:
                    failures.append(f"{op_name}/{case_id}/{provider}: empty result")
    if failures:
        raise RuntimeError("benchmark failed: " + "; ".join(failures))


def run_bench(args):
    raw_test_cases = load_test_cases(args)
    if not raw_test_cases:
        logger.error("No valid test cases found. Exiting.")
        raise SystemExit(1)
    requested_providers = parse_requested_providers(args.providers)

    with XpuPerfServer(args) as server_instance:
        test_cases, available_by_op = prepare_provider_matrix(
            server_instance,
            raw_test_cases,
            requested_providers,
        )
        log_test_cases(test_cases)
        info_dict = server_instance.get_info()
        bench_results = run_provider_matrix(
            server_instance,
            test_cases,
            available_by_op,
            requested_providers,
        )

    raise_for_failed_results(test_cases, bench_results)
    export_reports(args.report_dir, info_dict, test_cases, bench_results)


def main():
    run_bench(parse_args())


if __name__ == "__main__":
    main()
