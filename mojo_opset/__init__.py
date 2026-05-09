import os

if os.environ.get("MOJO_DEBUG") == "1":
    from mojo_opset.utils.debugger import MojoDebugger

    MojoDebugger.enable()

from mojo_opset.utils.patching import rewrite_assertion

with rewrite_assertion(__name__):
    from mojo_opset.backends import *
    from mojo_opset.core import *


def _is_plugin_autoload_enabled() -> bool:
    return os.getenv("MOJO_OPSET_PLUGIN_AUTOLOAD", "1") == "1"


def _import_plugins() -> None:
    from importlib.metadata import entry_points

    from mojo_opset.utils.logging import get_logger

    logger = get_logger(__name__)

    try:
        backend_plugins = entry_points(group="mojo_opset.plugins")
    except TypeError:
        backend_plugins = entry_points().get("mojo_opset.plugins", ())

    for plugin in backend_plugins:
        try:
            entrypoint = plugin.load()
            if not callable(entrypoint):
                raise TypeError(
                    f"Plugin entry point '{plugin.name}' is not callable: {entrypoint!r}"
                )
            entrypoint()
            logger.info("Loaded mojo_opset plugin: %s", plugin.name)
        except Exception as err:
            logger.warning("Skipping mojo_opset plugin '%s' due to: %s", plugin.name, err)


if _is_plugin_autoload_enabled():
    _import_plugins()
