import torch

from mojo_opset.core import MojoRMSNormFunction

from ._utils import _matrix_shape
from ._utils import run_kernel


class UCRMSNormFunction(MojoRMSNormFunction):
    """UC backend autograd wrapper for ``MojoRMSNormFunction``.

    Forward
    -------
    Reuses the master ``mojo_rmsnorm`` UC wheel kernel (``kernels/mojo_rmsnorm.py``),
    which currently ships ``fp16`` and ``bf16`` variants.  The kernel runs the
    full RMSNorm fused path: per-row ``mean(x^2)`` reduce across hidden dim,
    ``rsqrt(mean + eps)``, ``y = x * scale * weight``.  The autograd context
    saves the *raw* (non-contiguous) input + weight so that backward sees the
    same tensors a torch-native ``F.rms_norm`` would, keeping gradient
    semantics identical to the reference path.

    Note on P-01 in-flight rewrite
    ------------------------------
    The forward kernel is being rewritten from 3-pass (cast → sum-sq → norm)
    to 1-pass + non-scalar reduce (see lessons-learned §C.1 / §C.2 — current
    bf16 path is ~3× slower than TTX, mostly tile + 3-pass overhead).  This
    wrapper does **not** pin a kernel version; it dispatches to whatever
    ``mojo_rmsnorm_{bf16,fp16}`` symbol the wheel currently exposes, so the
    P-01 rewrite is picked up transparently after wheel rebuild.

    Backward
    --------
    Analytically, RMSNorm backward is

        rms     = sqrt(mean(x^2) + eps)
        y_hat   = x / rms
        dW      = sum_b ( dy * y_hat ),   reduced over batch dim
        dx_hat  = dy * weight
        dx      = (1 / rms) * ( dx_hat - y_hat * mean(dx_hat * y_hat) )

    The dominant cost is the same ``reduce-across-hidden`` traversal as
    forward, so a UC bwd kernel would have a very similar UB layout (one
    reduce per row to compute the inner-product term, then a per-element
    update).  The wheel does **not** currently ship ``mojo_rmsnorm_backward``
    and per task constraint we do not add / compile a new kernel here.

    We therefore inherit ``MojoRMSNormFunction.backward`` (re-runs
    ``F.rms_norm`` through torch autograd on saved tensors).  Accuracy ≡
    reference; perf matches torch_npu native bwd.  A dedicated 1-pass UC bwd
    kernel — sharing the row-reduce traversal pattern with the P-01 forward
    rewrite — is tracked as a follow-up (see ``docs/project-ops/
    lessons-learned.md`` §C; high ROI once P-01 lands the fused row-reduce
    template).
    """

    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        normalized_shape = (input.shape[-1],)

        # Save raw tensors for the inherited torch-native backward.  These
        # are the same arguments parent ``MojoRMSNormFunction.forward``
        # would have saved, so backward semantics are identical.
        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        if input.numel() == 0:
            return torch.empty_like(input)

        # UC wheel currently exposes only fp16 / bf16 ``mojo_rmsnorm``.  Per
        # project rule "wheel 没实现的就直接给报错" (2026-06-08), unsupported
        # dtypes raise NotImplementedError instead of silently using torch's
        # ``F.rms_norm`` — use TTX / torch_npu / torch_native backend for fp32.
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"UCRMSNormFunction supports bf16/fp16 only, got {input.dtype}. "
                "No UC kernel registered for this dtype."
            )

        kernel_input = input.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)

        run_kernel(
            "mojo_rmsnorm",
            kernel_input.dtype,
            kernel_input,
            weight.contiguous(),
            kernel_output,
            rows,
            cols,
            float(eps),
        )
        return kernel_output.reshape(input.shape)

    # backward intentionally inherited from MojoRMSNormFunction (torch-native
    # autograd over F.rms_norm on saved input + weight).  Re-declaring would
    # just rebind the same staticmethod and add no value; keeping a single
    # source of truth avoids drift if the reference formula ever changes.
    # When a UC bwd kernel ships, override here with the same dispatch shape
    # as `forward` (run_kernel("mojo_rmsnorm_backward", ...)).
