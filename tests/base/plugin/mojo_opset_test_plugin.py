import os


def _autoload():
    os.environ["MOJO_OPSET_TEST_PLUGIN_IMPORTED"] = "1"
