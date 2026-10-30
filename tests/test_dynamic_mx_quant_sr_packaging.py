"""Exercise the distributable wheel rather than the editable checkout."""

import os
import shutil
import subprocess
import sys
import zipfile

from pathlib import Path


def test_wheel_contains_sr_rng_bridge(tmp_path):
    root = Path(__file__).parents[1]
    source = tmp_path / "source"
    source.mkdir()
    # Build from the current working tree, including uncommitted integration changes.
    for name in ("setup.py", "_packaging.py", "packaging.json", "README.md", "LICENSE"):
        shutil.copy2(root / name, source / name)
    for name in ("mojo_opset", "native"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"))
    for override in root.glob("packaging_*.json"):
        shutil.copy2(override, source / override.name)
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(tmp_path / "dist")],
        cwd=source,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    resource = "mojo_opset/kernels/npu_a5_cannbotdsl/csrc/rng_bridge.cpp"
    with zipfile.ZipFile(wheel) as archive:
        assert resource in archive.namelist()
        assert archive.read(resource) == (root / resource).read_bytes()
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-compile",
            "--target",
            str(tmp_path / "installed"),
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    # Isolated import from the wheel payload, with no checkout on sys.path.
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from importlib.resources import files; "
            "p=files('mojo_opset.kernels.npu_a5_cannbotdsl').joinpath('csrc/rng_bridge.cpp'); "
            "assert p.is_file() and 'reserve_philox_state' in p.read_text()",
            str(tmp_path / "installed"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
