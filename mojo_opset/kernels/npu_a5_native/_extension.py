"""Lazy loading of A5 C++ dispatcher registrations."""

from importlib import import_module

import torch


def load_ops(provider, namespace, symbols):
    import torch_npu  # noqa: F401 -- load dependent runtime libraries first

    module = "mojo_opset_lib." + provider.replace("/", ".") + "._native"
    try:
        import_module(module)
        ops = getattr(torch.ops, namespace)
        for symbol in symbols:
            getattr(ops, symbol).default
        return ops
    except (ImportError, AttributeError) as error:
        raise ImportError(
            f"Native extension {module} could not be loaded or does not register "
            f"{namespace} operators {', '.join(symbols)}. Build and install it: bash build.sh {provider}"
        ) from error
