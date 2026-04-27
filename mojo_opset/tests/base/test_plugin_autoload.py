import sys
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = Path(__file__).resolve().parent / "plugin"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_import_plugins_loads_fake_ext(tmp_path):
    install_target = tmp_path / "plugin_install"

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--target",
        str(install_target),
        str(PLUGIN_DIR),
    ]
    install_result = subprocess.run(install_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert install_result.returncode == 0, install_result.stderr

    env = os.environ.copy()
    pythonpath_parts = [str(install_target), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.pop("MOJO_OPSET_TEST_PLUGIN_IMPORTED", None)

    run_cmd = [
        sys.executable,
        "-c",
        (
            "import os; "
            "import mojo_opset; "
            "assert os.environ.get('MOJO_OPSET_TEST_PLUGIN_IMPORTED') == '1'"
        ),
    ]
    run_result = subprocess.run(run_cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr
