import importlib

from mojo_opset.utils.platform import get_platform

platform = get_platform()

try:
    uc_backend_module = importlib.import_module(f".{platform}", package=__name__)
except ImportError as e:
    raise RuntimeError(f"Unsupported UC Platform '{platform}': {e}") from e


def _get_kernel_impl(uc_backend_module, kernel_name):
    def _not_impl(*args, **kwargs):
        raise NotImplementedError(f"Kernel '{kernel_name}' not implemented for platform '{platform}'.")

    return getattr(uc_backend_module, kernel_name, _not_impl)


gelu_fwd_impl = _get_kernel_impl(uc_backend_module, "gelu_fwd_impl")


def gelu_fwd(x):
    return gelu_fwd_impl(x)
