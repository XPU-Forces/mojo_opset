"""Run whole pytest parameter groups on one worker per allocated NPU."""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import pytest


class GroupedShard:
    def __init__(self, rank, cards):
        self.rank, self.cards = rank, cards
        self.total = self.selected = 0

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        groups = defaultdict(list)
        for item in items:
            function = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
            groups[(item.parent.nodeid, function)].append(item)
        loads = [0] * self.cards
        retained = set()
        for _, cases in sorted(groups.items(), key=lambda entry: (-len(entry[1]), entry[0])):
            owner = min(range(self.cards), key=lambda rank: (loads[rank], rank))
            loads[owner] += len(cases)
            if owner == self.rank:
                retained.update(cases)
        self.total, self.selected = len(items), len(retained)
        config.hook.pytest_deselected(items=[item for item in items if item not in retained])
        items[:] = [item for item in items if item in retained]
        print(f"NPU shard {self.rank + 1}/{self.cards}: {self.selected}/{self.total} cases", flush=True)


def worker(args):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.device_count() == 1, "Each worker must see exactly one allocated NPU"
    torch.npu.set_device(0)
    print(
        f"Worker {args.worker}: visible={os.environ['ASCEND_RT_VISIBLE_DEVICES']} "
        f"device={torch.npu.get_device_name(0)}", flush=True,
    )
    shard = GroupedShard(args.worker, args.cards)
    status = pytest.main([
        "-q", "tests/functions", "tests/modules", "--mojo-target", "npu.a2",
        "--mojo-implementation", args.implementation, "--check", "all", "-ra",
        f"--junitxml=test-results/worker-{args.worker}.xml",
    ], plugins=[shard])
    if status == pytest.ExitCode.NO_TESTS_COLLECTED and shard.total and not shard.selected:
        return 0
    return int(status)


def run_all(args):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.device_count() == args.cards, "Unexpected NPU allocation"
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [device.strip() for device in visible.split(",")] if visible else [str(rank) for rank in range(args.cards)]
    assert len(devices) == args.cards and len(set(devices)) == args.cards and all(devices)
    reports = Path("test-results")
    reports.mkdir(exist_ok=True)
    processes = {}
    started = last_update = time.monotonic()
    failed = False

    def cancel(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    with ExitStack() as files:
        try:
            for rank, device in enumerate(devices):
                (reports / f"worker-{rank}.xml").unlink(missing_ok=True)
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                cache = Path(env["MOJO_CI_WORK_DIR"]) / "cache" / str(rank)
                env.update(XDG_CACHE_HOME=str(cache), TRITON_CACHE_DIR=str(cache / "triton"),
                           TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
                log = files.enter_context((reports / f"worker-{rank}.log").open("w"))
                command = [sys.executable, str(Path(__file__).resolve()), "--implementation", args.implementation,
                           "--cards", str(args.cards), "--worker", str(rank)]
                processes[rank] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"Started worker {rank} on device {device}", flush=True)
            pending = dict(processes)
            while pending:
                for rank, process in list(pending.items()):
                    status = process.poll()
                    if status is not None:
                        print(f"Worker {rank} finished: exit={status}, elapsed={time.monotonic()-started:.0f}s", flush=True)
                        failed |= status != 0 or not (reports / f"worker-{rank}.xml").is_file()
                        del pending[rank]
                if pending:
                    if time.monotonic() - last_update >= 60:
                        print(
                            f"Still running: workers {sorted(pending)}, elapsed={time.monotonic()-started:.0f}s",
                            flush=True,
                        )
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
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("triton", "torch_npu", "native"), required=True)
    parser.add_argument("--cards", type=int, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.cards < 1 or (args.worker is not None and not 0 <= args.worker < args.cards):
        parser.error("invalid card count or worker index")
    try:
        sys.exit(worker(args) if args.worker is not None else run_all(args))
    except KeyboardInterrupt:
        sys.exit(130)
