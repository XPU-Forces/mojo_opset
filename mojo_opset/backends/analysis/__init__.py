"""Analysis backend for MojoOpSet."""

from mojo_opset.utils.platform import get_impl_by_platform

_op_map = get_impl_by_platform()
globals().update(_op_map)
__all__ = list(_op_map.keys())
