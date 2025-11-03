import importlib


def init_mojo_backend(backend_name: str):
    """
    Initialize the mojo backend.
    """
    importlib.import_module(f"mojo_opset.backends.{backend_name}")
