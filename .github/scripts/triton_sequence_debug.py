"""Replay original worker prefixes and check each autotune candidate numerically."""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TASKS = (
    ("3.2.2", "swa-prefix"),
    ("3.2.2", "quant-prefix"),
    ("3.2.2", "configs-0"),
    ("3.2.2", "configs-1"),
    ("3.2.2.post2", "configs-0"),
    ("3.2.2.post2", "configs-1"),
    ("3.2.2.post2", "swa-prefix"),
    ("3.2.2", "swa-isolated"),
)


class Replay:
    def __init__(self, mode):
        filename = "quant-prefix" if mode == "quant-prefix" else "swa-prefix"
        self.nodes = json.loads((HERE / f"{filename}.json").read_text())
        if mode == "swa-isolated":
            self.nodes = [node for node in self.nodes if "test_swa_backward_varlen_matches_fp64" in node]
        self.phase = 0

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        by_id = {item.nodeid: item for item in items}
        missing = set(self.nodes) - by_id.keys()
        assert not missing, f"Replay collection changed: {sorted(missing)}"
        retained = {by_id[node] for node in self.nodes}
        config.hook.pytest_deselected(items=[item for item in items if item not in retained])
        items[:] = [by_id[node] for node in self.nodes]
        module = importlib.import_module("tests.functions.test_attention")
        original = module.assert_close

        def tagged_close(*args, **kwargs):
            if self.current_swa:
                kwargs["name"] = ("output", "dQ", "dK", "dV")[self.phase % 4]
                print(f"SWA comparison: {kwargs['name']}", flush=True)
                self.phase += 1
            return original(*args, **kwargs)

        module.assert_close = tagged_close

    def pytest_runtest_setup(self, item):
        self.current_swa = "test_swa_backward_varlen_matches_fp64" in item.nodeid
        self.phase = 0

    def pytest_runtest_teardown(self, item):
        module = sys.modules.get("mojo_opset.kernels.npu_a2_triton.int8_gemm")
        if module is not None and "test_quant_gemm_accuracy" in item.nodeid:
            kernel = module._int8_gemm_dequant_kernel
            print("GEMM selected config:", getattr(kernel, "best_config", None), flush=True)


def replay(mode, label):
    return int(pytest.main([
        "-q", "-s", "tests/functions", "tests/modules", "--mojo-target", "npu.a2",
        "--mojo-implementation", "triton", "--check", "all", "-ra",
        f"--junitxml=test-results/{label}.xml",
    ], plugins=[Replay(mode)]))


def candidates(mode, label):
    import torch
    from mojo_opset import functions as F
    from tests.functions.test_gemm import _make_int8_gemm_data

    module = importlib.import_module("mojo_opset.kernels.npu_a2_triton.int8_gemm")
    configs = module._get_autotune_configs()
    torch.npu.matmul.allow_hf32 = False
    results = []
    for index, config in enumerate(configs):
        if index % 2 != int(mode[-1]):
            continue
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(42)
            data = _make_int8_gemm_data(32, 4096, 11008, False)
            x, weight, xs, ws = [tensor.npu() for tensor in data]
            expected = F.quant_gemm(x, weight, xs, ws, output_dtype=dtype, implementation="torch")
            a = torch.zeros((128, 4096), dtype=torch.int8, device="npu")
            a[:32] = x
            bt = module.prepare_b_impl(weight)
            ins = torch.zeros(128, device="npu")
            ins[:32] = xs
            wts = torch.zeros(bt.shape[0], device="npu")
            wts[:11008] = ws.float()
            for multibuffer in (True, False):
                entry = dict(index=index, config=str(config), dtype=str(dtype), multibuffer=multibuffer)
                try:
                    counts = []
                    for repeat in range(3):
                        output = torch.full((128, bt.shape[0]), float("nan"), device="npu", dtype=dtype)
                        module._int8_gemm_dequant_kernel.fn[(20,)](
                            a, bt, output, ins, wts, wts, 128, bt.shape[0], 4096,
                            a.stride(0), a.stride(1), bt.stride(0), bt.stride(1),
                            output.stride(0), output.stride(1), HAS_BIAS=False,
                            multibuffer=multibuffer, **config.all_kwargs(),
                        )
                        actual = output[:32, :11008]
                        bad = ~torch.isclose(actual, expected, atol=0.01, rtol=0.01)
                        counts.append(int(bad.sum().item()))
                        if counts[-1] and "first_bad_column" not in entry:
                            entry["first_bad_column"] = int(bad.any(dim=0).nonzero()[0].item())
                            entry["max_abs_error"] = float((actual.float()-expected.float()).abs().max().item())
                    entry["mismatches"] = counts
                except Exception as error:
                    entry["error"] = repr(error)
                results.append(entry)
                print(json.dumps(entry), flush=True)
                Path(f"test-results/{label}.json").write_text(json.dumps(results, indent=2))
    return 0


def run_all():
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in visible.split(",")] if visible else [str(i) for i in range(8)]
    assert len(devices) == len(set(devices)) == 8
    processes = {}

    def cancel(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    started = last_update = time.monotonic()
    with ExitStack() as files:
        try:
            for rank, (version, mode) in enumerate(TASKS):
                label = f"{version}-{mode}"
                cache = f"/tmp/triton-sequence/{label}"
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=devices[rank], PYTHONUNBUFFERED="1",
                           TRITON_PRINT_AUTOTUNING="1", XDG_CACHE_HOME=cache,
                           TRITON_CACHE_DIR=cache+"/triton", TORCHINDUCTOR_CACHE_DIR=cache+"/inductor")
                log = files.enter_context(Path(f"test-results/{label}.log").open("w"))
                command = [f"/tmp/triton-ab/{version}/bin/python", str(Path(__file__).resolve()),
                           "--mode", mode, "--label", label]
                processes[label] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"Started {label} on device {devices[rank]}", flush=True)
            pending = dict(processes)
            while pending:
                for label, process in list(pending.items()):
                    if process.poll() is not None:
                        print(f"{label}: exit={process.returncode}, elapsed={time.monotonic()-started:.0f}s", flush=True)
                        # Put a compact result in the live job log before all other lanes finish.
                        lines = Path(f"test-results/{label}.log").read_text(errors="replace").splitlines()
                        for line in lines[-20:]:
                            print(f"[{label}] {line}", flush=True)
                        del pending[label]
                if pending:
                    if time.monotonic()-last_update >= 60:
                        print(f"Still running: {sorted(pending)}", flush=True)
                        last_update = time.monotonic()
                    time.sleep(1)
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return int(any(p.returncode not in (0, 1) for p in processes.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.mode:
        import torch
        import torch_npu  # noqa: F401

        assert torch.npu.device_count() == 1
        torch.npu.set_device(0)
        status = candidates(args.mode, args.label) if args.mode.startswith("configs-") else replay(args.mode, args.label)
    else:
        status = run_all()
    raise SystemExit(status)
