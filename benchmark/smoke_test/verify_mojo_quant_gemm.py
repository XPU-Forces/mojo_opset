"""Quick smoke test for benchmark harness and NPU environment.

Checks op_def / vendor_ops registration, runs a small mojo_quant_gemm case on
NPU, and compares base vs torch_npu outputs. Not a full accuracy suite; use
``run_acc.py`` for workload-driven accuracy testing.

    python benchmark/smoke_test/verify_mojo_quant_gemm.py
"""
from __future__ import annotations

import pathlib
import sys

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent.parent
OP_DEFS = BENCHMARK_DIR / "op_defs"
VENDOR_OPS = BENCHMARK_DIR / "vendor_ops" / "NPU" / "ops"


class _NpuBackend:
    def get_torch_device_name(self):
        return "npu"


def _clone_mapping(mapping: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in mapping.items()
    }


def main() -> int:
    ProviderRegistry.load_all_vendor_impls(OP_DEFS, [VENDOR_OPS])

    op_name = "mojo_quant_gemm"
    if op_name not in ProviderRegistry.BASE_IMPL_MAPPING:
        print("FAIL: base impl missing")
        return 1
    if op_name not in ProviderRegistry.OP_MAPPING:
        print(f"FAIL: {op_name} not registered")
        return 1

    vendor_providers = ProviderRegistry.OP_MAPPING[op_name]
    print("base impl:", ProviderRegistry.BASE_IMPL_MAPPING[op_name].__name__)
    print("vendor providers:", sorted(vendor_providers.keys()))
    if "torch_npu" not in vendor_providers:
        print("FAIL: torch_npu impl missing")
        return 1

    providers = {
        "base": ProviderRegistry.BASE_IMPL_MAPPING[op_name],
        "torch_npu": vendor_providers["torch_npu"],
    }

    backend = _NpuBackend()
    args_dict = {
        "arg_type": "llm",
        "num_tokens": 128,
        "in_features": 4096,
        "out_features": 4096,
        "trans_weight": False,
        "sp_size": 1,
    }

    for provider in ["base", "torch_npu"]:
        op_cls = providers[provider]
        op = op_cls(args_dict, backend)
        tensors = op.create_tensors(1)[0]
        out = op.core_run(_clone_mapping(tensors))
        torch.npu.synchronize()
        summary = op.summary(latency_us=1000.0)
        print(
            f"[{provider}] out shape={tuple(out.shape)} dtype={out.dtype} "
            f"flops={op.calc_flops} io_bytes={op.io_bytes} "
            f"tflops@1ms={summary.get('calc_flops_power(tflops)')}"
        )

    base_op = providers["base"](args_dict, backend)
    npu_op = providers["torch_npu"](args_dict, backend)
    torch.manual_seed(42)
    shared = base_op.create_tensors(1)[0]
    base_out = base_op.core_run(_clone_mapping(shared)).float()
    npu_out = npu_op.core_run(_clone_mapping(shared)).float()
    torch.npu.synchronize()
    max_abs = (base_out - npu_out).abs().max().item()
    print(f"max_abs_diff(base vs torch_npu) = {max_abs:.6e}")

    if max_abs > 1e-2:
        print("FAIL: base vs torch_npu diff exceeds atol=1e-2")
        return 1

    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
