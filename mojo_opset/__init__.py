import os

from mojo_opset.backends import init_mojo_backend

# from mojo_opset.backends.ttx.kernels import register_ttx_to_torch_ops
from mojo_opset.core import *

_SUPPORT_BACKEND_LIST = ["REFERENCE", "TTX"]

"""
NOTICE: init_mojo_backend should be called before importing any mojo operator.
"""
mojo_backend = os.getenv("MOJO_BACKEND", "+ALL")
if mojo_backend[0] != "+":
    raise RuntimeError("Wrong backend format!")
backend_list = mojo_backend[1:].split(",")
if "ALL" in backend_list:
    for backend in _SUPPORT_BACKEND_LIST:
        init_mojo_backend(backend.lower())
else:
    # Always init reference backend.
    init_mojo_backend("reference")

    for backend in backend_list:
        if backend not in _SUPPORT_BACKEND_LIST:
            raise RuntimeError(f"Unsupport backend[{backend}]!")
        else:
            init_mojo_backend(backend.lower())
