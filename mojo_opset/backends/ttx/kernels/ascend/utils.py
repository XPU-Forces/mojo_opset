from functools import lru_cache

import triton
import triton.language as tl

VEC_ALIGN_BYTES = 256


@lru_cache(maxsize=1)
def get_num_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


# npu triton only
exp = tl.exp
exp2 = tl.math.exp2
log = tl.log
log2 = tl.log2
gather = tl.gather
