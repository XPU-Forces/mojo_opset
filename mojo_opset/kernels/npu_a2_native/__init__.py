"""A2 native provider; importing this map never loads shared libraries."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {
    "flash_attention": "flash_attention",
    "group_gemm": "group_gemm",
    "varlen_fa_infer": "varlen_fa",
    "rope": "rope",
    "swiglu": "swiglu",
}

_extend_bindings(__name__, OPS)
del _extend_bindings
