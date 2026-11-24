"""Exercise the public compile entry, with a fallback for the known vendor stub."""

import torch

from torch._dynamo.backends.registry import lookup_backend


def compile_fullgraph(function, *, backend="aot_eager"):
    compiler = lookup_backend(backend) if isinstance(backend, str) else backend
    captured_graphs = 0

    def tracked_backend(graph, inputs):
        nonlocal captured_graphs
        captured_graphs += 1
        return compiler(graph, inputs)

    entry = torch.compile
    if (getattr(entry, "__module__", None), getattr(entry, "__name__", None)) == (
        "torch._xpu_adaptor", "torch_compile"
    ):
        # This vendor entry ignores the backend and directly invokes the model.
        compiled = torch._dynamo.optimize(tracked_backend, nopython=True)(function)
    else:
        compiled = entry(function, backend=tracked_backend, fullgraph=True)

    def checked(*args, **kwargs):
        result = compiled(*args, **kwargs)
        assert captured_graphs, (
            "The compile test did not capture a graph; check whether compilation "
            "is disabled or torch.compile was replaced by an unrecognized eager stub"
        )
        return result

    return checked
