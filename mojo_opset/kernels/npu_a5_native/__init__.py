"""A5 native wrappers shared by Ascend 950PR and 950DT."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {"native_swa_infer": "swa", "swa": "swa", "topk_infer": "topk"}

_extend_bindings(__name__, OPS)
del _extend_bindings
