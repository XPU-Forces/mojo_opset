# utils.py
import torch
import triton.backends.mlu.driver as driver

_CACHED_CORE_NUM = None

def get_mlu_total_cores_cached() -> int:
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
        _CACHED_CORE_NUM = _devprob['cluster_num'] * _devprob['core_num_per_cluster']
    return _CACHED_CORE_NUM