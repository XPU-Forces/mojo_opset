"""NPU tensor-info helpers, mirroring xpu-perf's vendor_ops torch_npu_utils.

Provides an ``NpuOpTensorInfo`` dataclass and dtype/size utilities backed by an
NPU-aware dtype mapping, so vendor impls allocate tensors with the correct
torch_npu dtypes.
"""
from functools import partial
from typing import List
from dataclasses import dataclass, field

import torch
import torch_npu

from xpu_perf.micro_perf.core.utils import TorchDtypeInfo
from xpu_perf.micro_perf.core.utils import bool_creator, int_creator, uint_creator, float_creator


NPU_TORCH_DTYPE_MAPPING = {
    "bool": TorchDtypeInfo(torch.bool, 1, bool_creator),

    "int64": TorchDtypeInfo(torch.int64, 8, partial(int_creator, cast_dtype=True)),
    "int32": TorchDtypeInfo(torch.int32, 4, partial(int_creator, cast_dtype=True)),
    "int16": TorchDtypeInfo(torch.int16, 2, partial(int_creator, cast_dtype=True)),
    "int8": TorchDtypeInfo(torch.int8, 1, partial(int_creator, cast_dtype=True)),

    "uint64": TorchDtypeInfo(torch.uint64, 8, partial(uint_creator, cast_dtype=True)),
    "uint32": TorchDtypeInfo(torch.uint32, 4, partial(uint_creator, cast_dtype=True)),
    "uint16": TorchDtypeInfo(torch.uint16, 2, partial(uint_creator, cast_dtype=True)),
    "uint8": TorchDtypeInfo(torch.uint8, 1, partial(uint_creator, cast_dtype=True)),

    "float": TorchDtypeInfo(torch.float32, 4, float_creator),
    "float32": TorchDtypeInfo(torch.float32, 4, float_creator),
    "tfloat32": TorchDtypeInfo(torch.float32, 4, float_creator),
    "half": TorchDtypeInfo(torch.float16, 2, float_creator),
    "float16": TorchDtypeInfo(torch.float16, 2, float_creator),
    "bfloat16": TorchDtypeInfo(torch.bfloat16, 2, float_creator),
}


if hasattr(torch, "float8_e4m3fn"):
    NPU_TORCH_DTYPE_MAPPING.update({
        "float8": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "float8_e4m3": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "float8_e5m2": TorchDtypeInfo(torch.float8_e5m2, 1, float_creator),
    })


NPU_DTYPE_SIZE_MAPPING = {}
for _dtype_str, _dtype_info in NPU_TORCH_DTYPE_MAPPING.items():
    NPU_DTYPE_SIZE_MAPPING[_dtype_info.torch_dtype] = _dtype_info.dtype_size

NPU_CREATOR_MAPPING = {}
for _dtype_str, _dtype_info in NPU_TORCH_DTYPE_MAPPING.items():
    NPU_CREATOR_MAPPING[_dtype_info.torch_dtype] = _dtype_info.creator


def npu_get_torch_dtype(dtype: str) -> torch.dtype:
    return NPU_TORCH_DTYPE_MAPPING[dtype].torch_dtype


def npu_get_torch_dtype_size(dtype: torch.dtype):
    return NPU_DTYPE_SIZE_MAPPING[dtype]


def npu_default_creator(size, dtype, device):
    return NPU_CREATOR_MAPPING[dtype](size, dtype, device)


@dataclass
class NpuOpTensorInfo:
    shape: List[int] = field(default_factory=list)
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    creator: callable = npu_default_creator


def npu_calc_tensor_size(tensor_info: NpuOpTensorInfo):
    element_count = 1
    for dim in tensor_info.shape:
        element_count *= dim
    dtype_size = npu_get_torch_dtype_size(tensor_info.dtype)
    return element_count * dtype_size
