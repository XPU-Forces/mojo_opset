import torch

from mojo_opset.core import MojoSiluFunction

from ._utils import run_unary_kernel


class UCSiluFunction(MojoSiluFunction):
    """UC backend autograd wrapper for ``MojoSiluFunction``.

    Forward
    -------
    Reuses the existing ``mojo_silu`` UC wheel kernel (single-pass UB vector
    op).  Equivalent to the eager ``input * sigmoid(input)`` reference.

    Backward
    --------
    The analytic SiLU backward is

        dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    which is purely element-wise / vector-path.  The UC wheel does **not**
    currently ship a ``mojo_silu_backward`` kernel, and per task constraint
    we are not adding/compiling a new kernel in this commit.  We therefore
    inherit ``MojoSiluFunction.backward`` (torch-native fused multiply path
    that runs on NPU through torch_npu).  Accuracy ≡ reference; perf will
    match torch_npu baseline.  A dedicated UC bwd kernel is tracked as a
    follow-up (see ``docs/project-ops/lessons-learned.md`` §C — vector-only
    fused bwd is a strong ROI candidate once we have a stable wrapper).
    """

    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return run_unary_kernel("mojo_silu", input)

    # backward intentionally inherited from MojoSiluFunction (torch-native
    # fallback).  Re-declaring would just rebind the same staticmethod and
    # adds no value; keeping a single source of truth avoids drift if the
    # reference formula ever changes.
