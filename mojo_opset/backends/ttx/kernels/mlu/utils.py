import torch
import triton.backends.mlu.driver as driver
from functools import lru_cache

@lru_cache(maxsize=None)
def get_mlu_total_cores() -> int:
    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    return _devprob['cluster_num'] * _devprob['core_num_per_cluster']