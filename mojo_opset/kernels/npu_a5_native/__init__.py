"""A5 native wrappers backed by architecture and SKU extensions."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {}

OVERRIDES = {"npu.a5.950pr": {"native_swa_infer": "sku_950pr.swa"}}

_extend_bindings(__name__, OPS)
del _extend_bindings
