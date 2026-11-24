"""Build a repeatable public-autograd workload outside the timed region."""

import torch

from torch.utils._pytree import tree_leaves


def training(call, inputs, phase, *, grad_factory=torch.randn_like):
    if phase == "forward":
        return call
    if phase not in ("backward", "forward_backward"):
        raise ValueError(f"Unknown training phase: {phase}")
    inputs = tuple(x for x in inputs if isinstance(x, torch.Tensor) and x.requires_grad)

    def outputs():
        return tuple(x for x in tree_leaves(call()) if isinstance(x, torch.Tensor) and x.requires_grad)

    saved = outputs()
    grads = tuple(grad_factory(x) for x in saved)
    if phase == "backward":
        return lambda saved=saved: torch.autograd.grad(saved, inputs, grads, retain_graph=True)
    del saved
    return lambda: torch.autograd.grad(outputs(), inputs, grads)
