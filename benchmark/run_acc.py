"""Lightweight single-process accuracy runner for mojo_opset benchmark op_defs.

Reuses xpu-perf's ``BasicOp`` tensor creation and ``core_run`` so ``base`` and
vendor providers see identical inputs. Compares vendor outputs against the
``base`` torch reference with preset accuracy metrics.

Examples:
    python benchmark/run_acc.py

    python benchmark/run_acc.py \
        --workload benchmark/workloads/mojo_paged_prefill_gqa.json \
        --providers torch_npu --device 0

    python benchmark/run_acc.py \
        --workload benchmark/workloads/mojo_quant_gemm.json \
        --atol 1e-1 --rtol 1e-2 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import prettytable
import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

from xpu_perf.micro_perf.backends.NPU.backend_npu import BackendNPU
from xpu_perf.micro_perf.core.common_utils import parse_workload
from xpu_perf.micro_perf.core.op import ProviderRegistry

from acc_compare import AccuracyMetrics, clone_tensor_mapping, compare_outputs
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
        default="torch_npu",
        help="comma-separated vendors to compare against base",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="if set, export per-case accuracy results as JSON",
    )
    return parser.parse_args()


def _is_oom_error(err: BaseException) -> bool:
    if isinstance(err, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _build_op(op_cls, case, backend):
    op_instance = op_cls(case, backend)
    op_instance.is_concurrent = False
    return op_instance


def _run_reference(base_op, shared_mapping, backend):
    mapping = clone_tensor_mapping(shared_mapping)
    out = base_op.core_run(mapping)
    backend.device_synchronize()
    return out


def _run_vendor(vendor_op, shared_mapping, backend):
    mapping = clone_tensor_mapping(shared_mapping)
    out = vendor_op.core_run(mapping)
    backend.device_synchronize()
    return out


def _skip_metrics(note: str) -> AccuracyMetrics:
    return AccuracyMetrics(
        max_abs_diff=None,
        mean_abs_diff=None,
        max_rel_diff=None,
        mean_rel_diff=None,
        rmse=None,
        match_ratio=None,
        status="SKIP",
        note=note,
    )


def run_accuracy_case(backend, base_cls, vendor_map, case, seed, atol, rtol):
    """Run one workload case; return reference output and per-vendor metrics."""
    try:
        base_op = _build_op(base_cls, case, backend)
    except Exception as err:  # noqa: BLE001
        note = f"failed to build base op: {err}"
        return None, {provider: _skip_metrics(note) for provider in vendor_map}

    torch.manual_seed(seed)
    try:
        shared_mapping = base_op.create_tensors(1)[0]
    except Exception as err:  # noqa: BLE001
        note = f"failed to create tensors: {err}"
        return None, {provider: _skip_metrics(note) for provider in vendor_map}

    try:
        ref_out = _run_reference(base_op, shared_mapping, backend)
    except Exception as err:  # noqa: BLE001
        note = "reference OOM" if _is_oom_error(err) else f"reference run failed: {err}"
        return None, {provider: _skip_metrics(note) for provider in vendor_map}

    provider_metrics = {}
    for provider, vendor_cls in vendor_map.items():
        try:
            vendor_op = _build_op(vendor_cls, case, backend)
            vendor_out = _run_vendor(vendor_op, shared_mapping, backend)
            provider_metrics[provider] = compare_outputs(
                vendor_out,
                ref_out,
                atol=atol,
                rtol=rtol,
            )
        except Exception as err:  # noqa: BLE001
            if _is_oom_error(err):
                note = "vendor OOM"
            else:
                note = f"vendor run failed: {err}"
            provider_metrics[provider] = _skip_metrics(note)

    return ref_out, provider_metrics


def _fmt_metric(value):
    if value is None:
        return "-"
    return f"{value:.6e}"


def print_case_header(op_name, device_id, case):
    pt = prettytable.PrettyTable()
    pt.field_names = ["key", "value"]
    pt.align = "l"
    pt.add_row(["op_name", op_name])
    pt.add_row(["reference", BASE_PROVIDER])
    pt.add_row(["device_id", str(device_id)])
    print(pt)
    print(json.dumps(case))
    print("")


def print_accuracy_table(provider_metrics):
    pt = prettytable.PrettyTable()
    pt.field_names = [
        "provider",
        "max_abs",
        "mean_abs",
        "max_rel",
        "match_ratio",
        "status",
        "note",
    ]
    pt.align = "r"
    pt.align["provider"] = "l"
    pt.align["status"] = "l"
    pt.align["note"] = "l"

    for provider, metrics in provider_metrics.items():
        pt.add_row(
            [
                provider,
                _fmt_metric(metrics.max_abs_diff),
                _fmt_metric(metrics.mean_abs_diff),
                _fmt_metric(metrics.max_rel_diff),
                _fmt_metric(metrics.match_ratio),
                metrics.status,
                metrics.note,
            ]
        )
    print(pt)
    print("")


def export_report(report_dir: str, op_name: str, case, provider_metrics, seed, atol, rtol):
    out_dir = pathlib.Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "op_name": op_name,
        "reference": BASE_PROVIDER,
        "case": case,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "providers": {
            provider: {
                "max_abs_diff": metrics.max_abs_diff,
                "mean_abs_diff": metrics.mean_abs_diff,
                "max_rel_diff": metrics.max_rel_diff,
                "mean_rel_diff": metrics.mean_rel_diff,
                "rmse": metrics.rmse,
                "match_ratio": metrics.match_ratio,
                "status": metrics.status,
                "note": metrics.note,
            }
            for provider, metrics in provider_metrics.items()
        },
    }
    case_tag = hashlib.sha1(
        json.dumps(case, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    out_path = out_dir / f"{op_name}-{case_tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    requested_vendors = [name.strip() for name in args.providers.split(",") if name.strip()]
    if not requested_vendors:
        raise SystemExit("no vendors requested; set --providers torch_npu")

    backend = BackendNPU(backend="NPU", op_defs=OP_DEFS, vendor_ops=[VENDOR_OPS])
    backend.set_device(args.device)
    backend.load_all_ops()

    test_cases = parse_workload(args.workload)
    if not test_cases:
        raise SystemExit(f"no test cases found in {args.workload}")

    has_failure = False
    for op_name, cases in test_cases.items():
        provider_map = build_provider_map(backend, op_name, requested_vendors)
        provider_map = {
            name: cls for name, cls in provider_map.items() if name != BASE_PROVIDER
        }
        if op_name not in ProviderRegistry.BASE_IMPL_MAPPING:
            print(f"[skip] op '{op_name}' has no base reference")
            continue
        if not provider_map:
            print(f"[skip] op '{op_name}' has no requested vendors")
            continue

        base_cls = build_provider_map(backend, op_name, [BASE_PROVIDER])[BASE_PROVIDER]

        print("#" * 100)
        print(f"# op: {op_name} | reference: {BASE_PROVIDER} | vendors: {list(provider_map)}")
        print("#" * 100)

        for case in cases:
            _, provider_metrics = run_accuracy_case(
                backend,
                base_cls,
                provider_map,
                case,
                args.seed,
                args.atol,
                args.rtol,
            )
            print_case_header(op_name, args.device, case)
            print_accuracy_table(provider_metrics)
            if args.report_dir:
                export_report(
                    args.report_dir,
                    op_name,
                    case,
                    provider_metrics,
                    args.seed,
                    args.atol,
                    args.rtol,
                )
            for metrics in provider_metrics.values():
                if metrics.status == "FAIL":
                    has_failure = True

    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
