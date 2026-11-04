"""UNIC provider declarations; optional bindings load without runtime imports."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {}

_extend_bindings(__name__, OPS)
del _extend_bindings
