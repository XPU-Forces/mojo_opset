"""A5 native wrappers shared by Ascend 950PR and 950DT."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {
    "gemm": "dense_gemm",
    "mxfp8_dequant_gemm": "dense_gemm",
    "mxfp4_gemm": "dense_gemm",
    "native_swa_infer": "swa",
    "swa": "swa",
    "topk_infer": "topk",
}

_extend_bindings(__name__, OPS)
del _extend_bindings
