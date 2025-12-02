import importlib

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from importlib.metadata import entry_points
except ImportError:
    try:
        from importlib_metadata import entry_points
    except ImportError:

        def entry_points(group=None):
            return []


def init_mojo_backend(backend_name: str):
    logger.info(f"Attempting to initialize backend: {backend_name}")

    try:
        eps = entry_points(group="mojo_opset.backends")
    except Exception as e:
        logger.info(f"Warning: Could not query entry points. Error: {e}")
        eps = []

    registered_backends_map = {ep.name: ep for ep in eps}

    backend_key = backend_name.upper()
    backend_entry = registered_backends_map.get(backend_key)

    if backend_entry:
        importlib.import_module(backend_entry.module)
        logger.info(
            f"Successfully initialized backend '{backend_name}' by importing plugin module '{backend_entry.module}'."
        )
    else:
        logger.info(f"Backend '{backend_name}' not found via entry point, trying legacy import...")
        importlib.import_module(f"mojo_opset.backends.{backend_name.lower()}")
        logger.info(f"Successfully initialized backend '{backend_name}' as a built-in module.")
