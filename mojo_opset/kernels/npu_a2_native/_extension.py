"""Lazy A2 native symbol loading; provider imports never initialize the NPU."""

from functools import lru_cache


@lru_cache(maxsize=None)
def launcher(symbol: str):
    import torch_npu  # noqa: F401 -- initialize dependent runtime libraries first

    try:
        from mojo_opset_lib.npu_a2 import _native

        function = getattr(_native, symbol)
        if not callable(function):
            raise AttributeError(symbol)
        return function
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Native extension mojo_opset_lib.npu_a2._native could not be loaded "
            f"or does not export {symbol}. Build and install it for your runtime: "
            "bash build.sh npu_a2"
        ) from error
