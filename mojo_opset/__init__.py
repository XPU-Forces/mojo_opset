import os

from mojo_opset.backends import init_mojo_backend

# from mojo_opset.backends.ttx.kernels import register_ttx_to_torch_ops
from mojo_opset.core import *

_AVAILABLE_BACKENDS = set()


_BUILTIN_BACKEND_PATH = os.path.join(os.path.dirname(__file__), "backends")
if os.path.isdir(_BUILTIN_BACKEND_PATH):
    for name in os.listdir(_BUILTIN_BACKEND_PATH):
        if os.path.isdir(os.path.join(_BUILTIN_BACKEND_PATH, name)) and not name.startswith("__"):
            _AVAILABLE_BACKENDS.add(name.upper())


try:
    from importlib.metadata import entry_points
except ImportError:
    try:
        from importlib_metadata import entry_points
    except ImportError:
        entry_points = None

if entry_points:
    eps = entry_points(group="mojo_opset.backends")
    for ep in eps:
        _AVAILABLE_BACKENDS.add(ep.name.upper())


# NOTE: init_mojo_backend is called before importing any mojo operator.
# This setup allows dynamic backend loading via the MOJO_BACKEND environment variable.
default_backend_str = "+ALL"
mojo_backend_env = os.getenv("MOJO_BACKEND", default_backend_str)

if not mojo_backend_env.startswith("+"):
    raise RuntimeError(f"Wrong backend format for MOJO_BACKEND: '{mojo_backend_env}'. It must start with '+'.")

user_requested_backends = mojo_backend_env[1:].upper().split(",")


if "ALL" in user_requested_backends:
    if not _AVAILABLE_BACKENDS:
        print("Warning: MOJO_BACKEND=+ALL, but no backends were found (neither built-in nor plugin).")

    for backend_name in sorted(list(_AVAILABLE_BACKENDS)):
        init_mojo_backend(backend_name)
else:
    for backend_name in user_requested_backends:
        if not backend_name:
            continue

        if backend_name in _AVAILABLE_BACKENDS:
            init_mojo_backend(backend_name)

        else:
            raise RuntimeError(
                f"Unsupported backend '{backend_name}' requested. "
                f"Available backends found: {sorted(list(_AVAILABLE_BACKENDS)) or 'None'}"
            )
