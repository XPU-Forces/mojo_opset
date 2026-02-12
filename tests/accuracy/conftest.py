import logging

import pytest
import torch

from tests.utils import get_platform


@pytest.fixture(scope="session", autouse=True)
def setup_session_device(request):
    platform = get_platform()
    torch.set_default_device(platform)

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
    else:
        pass
