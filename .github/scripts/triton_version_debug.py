"""Compare compiler builds with unchanged operator implementations and tests."""

import os
import signal
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from xml.etree import ElementTree

VERSIONS = ("3.2.2", "3.2.2.post2")
SCENARIOS = (
    ("function-quant", "tests/functions/test_gemm.py::test_quant_gemm_accuracy"),
    ("module-quant", "tests/modules/test_gemm.py::test_quant_gemm_accuracy"),
    ("swa", "tests/functions/test_attention.py::test_swa_backward_varlen_matches_fp64"),
    # Reproduce the three SWA failures in a fresh process without preceding cases.
    ("swa-isolated", "tests/functions/test_attention.py::test_swa_backward_varlen_matches_fp64", "False-dtype0"),
)


def compare():
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in visible.split(",")] if visible else [str(i) for i in range(8)]
    assert len(devices) == 8 and len(set(devices)) == 8
    reports = Path("test-results")
    reports.mkdir(exist_ok=True)
    processes = {}

    def cancel(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    started = last_update = time.monotonic()
    with ExitStack() as files:
        try:
            for version_index, version in enumerate(VERSIONS):
                for index, (name, node, *selection) in enumerate(SCENARIOS):
                    rank = version_index * len(SCENARIOS) + index
                    label = f"{version}-{name}"
                    report = reports / f"{label}.xml"
                    report.unlink(missing_ok=True)
                    cache = Path("/tmp/triton-ab/cache") / label
                    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=devices[rank], PYTHONUNBUFFERED="1",
                               XDG_CACHE_HOME=str(cache), TRITON_CACHE_DIR=str(cache / "triton"),
                               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
                    command = [f"/tmp/triton-ab/{version}/bin/python", "-m", "pytest", "-q", node,
                               "--mojo-target", "npu.a2", "--mojo-implementation", "triton", "--check", "all",
                               "-ra", f"--junitxml={report}"]
                    if selection:
                        command.extend(["-k", selection[0]])
                    log = files.enter_context((reports / f"{label}.log").open("w"))
                    processes[label] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    print(f"Started {label} on device {devices[rank]}", flush=True)
            pending = dict(processes)
            while pending:
                for label, process in list(pending.items()):
                    status = process.poll()
                    if status is not None:
                        print(f"{label}: exit={status}, elapsed={time.monotonic()-started:.0f}s", flush=True)
                        del pending[label]
                if pending:
                    if time.monotonic() - last_update >= 60:
                        print(f"Still running: {sorted(pending)}, elapsed={time.monotonic()-started:.0f}s", flush=True)
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

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write("| Triton / test group | Passed | Failed | Errors | Skipped |\n| --- | --- | --- | --- | --- |\n")
        for label, process in processes.items():
            report = reports / f"{label}.xml"
            if not report.is_file():
                summary.write(f"| {label} (exit={process.returncode}) | missing report | | | |\n")
                continue
            counts = Counter(
                next((s for s in ("failure", "error", "skipped") if c.find(s) is not None), "passed")
                for c in ElementTree.parse(report).iter("testcase")
            )
            summary.write(
                f"| {label} | {counts['passed']} | {counts['failure']} | {counts['error']} | {counts['skipped']} |\n"
            )
    # Pytest assertion failures are expected measurements in this diagnostic run.
    return int(any(p.returncode not in (0, 1) or not (reports / f"{label}.xml").is_file()
                   for label, p in processes.items()))


if __name__ == "__main__":
    try:
        raise SystemExit(compare())
    except KeyboardInterrupt:
        raise SystemExit(130)
