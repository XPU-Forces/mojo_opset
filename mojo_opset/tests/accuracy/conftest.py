import logging
import os
import sys
from pathlib import Path

import pytest
import torch

from mojo_opset.core.backend_registry import MojoBackendRegistry
from mojo_opset.utils.platform import get_platform, get_torch_device
from mojo_opset.tests.utils import BackendNotImplementedForTest
from mojo_opset.tests.utils import resolve_backend_for_accuracy_test


def _candidate_ext_roots():
    env_root = os.environ.get("MOJO_OPSET_EXT_PATH")
    if env_root:
        yield Path(env_root)

    repo_root = Path(__file__).resolve().parents[3]
    yield repo_root.parent / "mojo_opset_gitlab"


def _load_xops_backend_for_accuracy():
    if get_platform() != "mlu":
        return

    if os.environ.get("MOJO_OPSET_ACCURACY_LOAD_XOPS", "1") != "1":
        return

    for ext_root in _candidate_ext_roots():
        if (ext_root / "mojo_opset_ext_autoload.py").is_file():
            ext_root_str = str(ext_root)
            if ext_root_str not in sys.path:
                sys.path.insert(0, ext_root_str)
            break

    try:
        import mojo_opset_ext_autoload
    except ModuleNotFoundError as exc:
        if exc.name != "mojo_opset_ext_autoload":
            raise
        logging.warning("mojo_opset_ext_autoload is not available, xops backend will not be loaded.")
        return

    mojo_opset_ext_autoload._autoload()


_load_xops_backend_for_accuracy()


@pytest.fixture(scope="session", autouse=True)
def setup_session_device(request):
    platform = get_platform()
    torch.set_default_device(get_torch_device())

    worker_id = 0
    if hasattr(request.config, "workerinput"):
        worker_id = request.config.workerinput["workerid"]
        worker_id = int(worker_id.replace("gw", ""))

    if platform == "npu":
        device_num = torch.npu.device_count()
        if worker_id >= device_num:
            logging.warning(
                f"worker_id {worker_id} is greater than device_num {device_num}, "
                f"set worker_id to {worker_id % device_num}"
            )
        torch.npu.set_device(worker_id % device_num)
    elif platform == "mlu":
        device_num = torch.mlu.device_count()
        if worker_id >= device_num:
            logging.warning(
                f"worker_id {worker_id} is greater than device_num {device_num}, "
                f"set worker_id to {worker_id % device_num}"
            )
        torch.mlu.set_device(worker_id % device_num)
    elif platform == "ilu":
        device_num = torch.cuda.device_count()
        if worker_id >= device_num:
            logging.warning(
                f"worker_id {worker_id} is greater than device_num {device_num}, "
                f"set worker_id to {worker_id % device_num}"
            )
        torch.cuda.set_device(worker_id % device_num)
    else:
        pass


@pytest.fixture(autouse=True)
def enable_strict_backend_resolution_for_accuracy(monkeypatch):
    if not os.environ.get("MOJO_BACKEND"):
        return

    original_get = MojoBackendRegistry.get

    def _patched_get(self, backend_name=None):
        requested_backend = resolve_backend_for_accuracy_test(self, backend_name)
        return original_get(self, requested_backend)

    monkeypatch.setattr(MojoBackendRegistry, "get", _patched_get)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    del item
    try:
        yield
    except BackendNotImplementedForTest as exc:
        pytest.skip(str(exc))
