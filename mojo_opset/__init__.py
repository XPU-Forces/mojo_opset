import os

if os.environ.get("MOJO_DEBUG") == "1":
    from mojo_opset.utils.debugger import MojoDebugger

    MojoDebugger.enable()

from mojo_opset.utils.patching import rewrite_assertion

with rewrite_assertion(__name__):
    from mojo_opset.backends import *
    from mojo_opset.core import *
