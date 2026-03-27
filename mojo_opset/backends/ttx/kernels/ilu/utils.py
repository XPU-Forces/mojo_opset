from functools import lru_cache

import triton
import triton.language as tl

VEC_ALIGN_BYTES = 256

# FIXME: adapt ilu.

# @lru_cache(maxsize=1)
# def get_num_cores(op_type="vector"):
#     assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
#     return (
#         triton.runtime.driver.active.utils.get_device_properties("ilu")["num_vectorcore"]
#         if op_type == "vector"
#         else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
#     )


