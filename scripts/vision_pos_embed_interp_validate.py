"""Run task checks and generate reports exclusively from this run's outputs.

Activate the platform's CANN/Python environment first, install this repository
with `python -m pip install -e . --no-deps`, then run from its root directory.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

from datetime import datetime
from datetime import timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory outside the repository")
    parser.add_argument("--reference-dir", required=True, type=Path, help="Pinned qwen3_vl.py and patch_qwen3vl.py")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    references = args.reference_dir.resolve()
    if output.is_relative_to(root):
        parser.error("put output outside the repository so the checked source stays unchanged")
    output.mkdir(parents=True, exist_ok=False)
    state = {"started_at": timestamp(), "scope": "development checks from task description", "stages": []}

    def save():
        (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")

    def run(label, command):
        log = output / f"{label}.log"
        stage = {"name": label, "command": command, "started_at": timestamp(), "log": log.name}
        state["stages"].append(stage)
        save()
        print(f"{label}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode, finished_at=timestamp())
        save()
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    state["source"] = {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "clean": not dirty.stdout.strip() if dirty.returncode == 0 else None,
    }
    source_manifest = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build"} for part in rel.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".yaml", ".patch", ".json", ".sh"}:
            source_manifest[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    state["environment"] = {
        "python": sys.version,
        "architecture": platform.machine(),
        "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "torch-npu", "triton-ascend")},
    }
    save()
    python = sys.executable
    try:
        run("pip-check", [python, "-m", "pip", "check"])
        run("pip-freeze", [python, "-m", "pip", "freeze"])
        run("npu-smi", ["npu-smi", "info"])
        run(
            "accuracy",
            [
                python,
                "-m",
                "pytest",
                "tests/functions/test_vision_pos_embed_interp.py",
                "tests/functions/test_vision_pos_embed_rounding.py",
                "tests/modules/test_vision_pos_embed_interp.py",
                "tests/functions/test_vision_pos_embed_integration.py",
                "--mojo-implementation=triton",
                "-x",
                "-q",
                f"--junitxml={output / 'accuracy.xml'}",
            ],
        )
        run(
            "integration",
            [
                python,
                "integrations/vision_pos_embed_interp/validate_patches.py",
                "--reference-dir",
                str(references),
                "--output",
                str(output / "integration.json"),
            ],
        )
        run(
            "benchmark",
            [
                python,
                "-m",
                "tests.performance.functions.bench_vision_pos_embed_interp",
                "--output",
                str(output / "benchmark"),
            ],
        )
        run(
            "report",
            [
                python,
                "scripts/vision_pos_embed_interp_report.py",
                "--junit",
                str(output / "accuracy.xml"),
                "--benchmark",
                str(output / "benchmark/benchmark.json"),
                "--output",
                str(output / "测试报告.md"),
            ],
        )
        run(
            "profile-analysis",
            [
                python,
                "scripts/vision_pos_embed_interp_profile.py",
                "--benchmark",
                str(output / "benchmark/benchmark.json"),
                "--output",
                str(output / "性能剖析.md"),
            ],
        )
        cases = list(ET.parse(output / "accuracy.xml").iter("testcase"))
        accuracy_ok = bool(cases) and all(
            case.find("failure") is None and case.find("error") is None and case.find("skipped") is None
            for case in cases
        )
        benchmark = json.loads((output / "benchmark/benchmark.json").read_text())
        state["accuracy_passed"] = accuracy_ok
        state["performance_passed"] = benchmark.get("complete", False) and benchmark.get("passed", False)
        state["passed"] = accuracy_ok and state["performance_passed"]
        state["status"] = "passed" if state["passed"] else "requirements_not_met"
        return_code = 0 if state["passed"] else 2
    except subprocess.CalledProcessError as error:
        state.update(passed=False, status="stage_failed", failed_stage=state["stages"][-1]["name"])
        return_code = error.returncode
    state["finished_at"] = timestamp()
    save()
    print(json.dumps(state, indent=2))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
