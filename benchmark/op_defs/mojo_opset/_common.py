"""Shared imports for llm_ops op_def submodules."""
from functools import partial

import torch

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
