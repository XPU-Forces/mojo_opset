# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu

import os

import triton
import triton.language as tl


exp = tl.exp
exp2 = tl.math.exp2
log = tl.log
log2 = tl.log2
gather = tl.gather


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
