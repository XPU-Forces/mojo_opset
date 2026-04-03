from functools import lru_cache

import triton
import triton.language as tl
import torch

VEC_ALIGN_BYTES = 256


@lru_cache(maxsize=1)
def get_num_cores(op_type="vector"):
    
    assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    return (
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        if op_type == "vector"
        else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
    )

def get_num_cores():
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()).multi_processor_count
    return processor_count

# npu triton only
exp = tl.exp
exp2 = tl.math.exp2
log = tl.log
log2 = tl.log2
gather = tl.gather
