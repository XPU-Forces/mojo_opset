import logging
import os

import pytest
import torch

from mojo_opset.core.backend_registry import MojoBackendRegistry
from mojo_opset.utils.platform import get_platform, get_torch_device
from tests.utils import BackendNotImplementedForTest
from tests.utils import resolve_backend_for_accuracy_test


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
