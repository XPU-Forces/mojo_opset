"""Multi-process launcher for the mojo_opset benchmark, reusing xpu-perf micro_perf.

This is a thin wrapper around xpu-perf's ``XpuPerfServer`` (the original
multi-process / multi-device flow: per-device subprocesses, NUMA pinning, queue
dispatch) with defaults pointing at this benchmark's ``op_defs`` / ``vendor_ops``
/ ``workloads``. Reports are written via the standard ``export_reports``.

Examples:
    # run the demo workload on all NPUs
    python benchmark/launch.py --backend NPU \
        --workload benchmark/workloads/mojo_quant_gemm.json

    # restrict to device 0
    python benchmark/launch.py --backend NPU --device 0 \
        --workload benchmark/workloads/mojo_quant_gemm.json
"""
import argparse
import pathlib
from functools import partial

import torch.multiprocessing as mp

from xpu_perf.micro_perf.core.common_utils import (
    existing_dir_path,
    export_reports,
    get_submodules,
    logger,
    parse_tasks,
    parse_workload,
    setup_logger,
    valid_file,
)
from xpu_perf.micro_perf.core.perf_engine import XpuPerfServer

FILE_DIR = pathlib.Path(__file__).parent.absolute()
OP_DEFS_DIR = FILE_DIR.joinpath("op_defs")
VENDOR_OPS_DIR = FILE_DIR.joinpath("vendor_ops")
WORKLOADS_DIR = FILE_DIR.joinpath("workloads")
DEFAULT_WORKLOAD = WORKLOADS_DIR.joinpath("mojo_quant_gemm.json")

mp.set_start_method("spawn", force=True)


def parse_args():
    setup_logger("INFO")

    backend_name_list, backend_mod_list = get_submodules("xpu_perf.micro_perf.backends")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--backend", type=str, default="NPU", choices=backend_name_list)
    parser.add_argument("--op_defs", type=existing_dir_path, default=OP_DEFS_DIR)
    parser.add_argument("--vendor_ops", type=existing_dir_path, default=[], action="append")
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

    parser.add_argument("--task_dir", type=str, default=str(WORKLOADS_DIR))
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--workload", type=str, default=str(DEFAULT_WORKLOAD))

    parser.add_argument("--report_dir", type=str, default=str(FILE_DIR.joinpath("reports")))

    args = parser.parse_args()
    args.script_dir = FILE_DIR
    args.backend_name_list = backend_name_list
    args.backend_mod_list = backend_mod_list
    return args


def load_test_cases(args):
    if args.workload is not None:
        return parse_workload(args.workload)
    return parse_tasks(args.task_dir, args.task)


def log_test_cases(test_cases):
    print("*" * 100)
    logger.info("test cases:")
    for op_name, op_cases in test_cases.items():
        logger.info(f"{op_name} has {len(op_cases)} test cases")
    print("*" * 100)


def run_bench(args):
    test_cases = load_test_cases(args)
    if not test_cases:
        logger.error("No valid test cases found. Exiting.")
        raise SystemExit(1)

    log_test_cases(test_cases)

    with XpuPerfServer(args) as server_instance:
        info_dict = server_instance.get_info()
        bench_results = server_instance.normal_bench(test_cases)

    export_reports(args.report_dir, info_dict, test_cases, bench_results)


def main():
    run_bench(parse_args())


if __name__ == "__main__":
    main()
