import torch

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU

from ._utils import run_binary_kernel
from ._utils import run_unary_kernel


# Element-wise activations have a ~10-15us NPU launch / scope-setup floor on
# the UC kernel (`T.Kernel(NUM_PROGRAMS=48, threads=1)` per-program scope +
# per-launch sync).  torch_npu / torch_native run their fused single-launch
# sigmoid+mul at ~7-9us regardless of shape, so on tiny inputs UC can never
# close the launch-floor gap.
#
# Empirical measurements on 910B 2026-06-11 (bf16, MOJO_BACKEND=uc, median of
# 5 x 100-iter runs via torch.npu.Event):
#
#     shape           UC us   torch_native us   verdict
#     (1, 1024)       13.2    7.7                fall back (numel = 1 K)
#     (4, 1024)       11.0    8.3                fall back (numel = 4 K)
#     (256, 128)      13.7    7.4                fall back (numel = 32 K)
#     (1024, 10240)   57.5    25.9               UC kernel (DRAM-bound regime)
#     (4096, 4096)    87.2    39.8               UC kernel
#
# Crossover sits around numel ~ 64 K elements — same threshold as
# `dequant.py:_UC_MIN_NUMEL`.  Below the threshold we surface
# `NotImplementedError` so the dispatcher selects torch_native / torch_npu
# (matches the project rule "wheel 没实现的就直接给报错" — UC kernel cannot
# beat the baseline in this regime).
#
# At and above the threshold the UC kernel still trails torch_npu / torch_native
# by ~2.2x (F.4 launch + F.5 CCE codegen — see lessons-learned §F + perf-debug
# op-MojoSilu-2026-06-11.md).  Worker-side schedule is tile-sweep-optimal
# (P2-17 X=4, Y=1024, NPROG=48); further gains require lifter element-wise
# chain fusion (P0 backlog) or a single-launch path for pure element-wise
# kernels (P1 backlog).
_UC_MIN_NUMEL = 64 * 1024


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_unary_kernel("mojo_gelu", x)


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.1 launch-overhead floor: tiny inputs fall back to
        # torch_native / torch_npu via NotImplementedError. UC kernel
        # needs ~10-15us scope setup + sync; the 1-pass element-wise
        # body itself is sub-microsecond at numel < 64 K. See
        # `_UC_MIN_NUMEL` block comment above and silu-p2-17.md §5
        # cannot-optimize for the launch-floor root cause.
        if x.numel() < _UC_MIN_NUMEL:
            raise NotImplementedError(
                "UC backend mojo_silu has a ~10-15us launch-floor that loses to "
                "torch_npu's fused silu (~7-9us) on tiny inputs. Per project rule "
                "'wheel 没实现的就直接给报错' (2026-06-08), this wrapper does not "
                "silently fall back to torch — use TTX / torch_npu / torch_native "
                f"backend for inputs with numel < {_UC_MIN_NUMEL} (got {x.numel()})."
            )
        return run_unary_kernel("mojo_silu", x)


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]
    # P1-G5 (2026-06-11): kernel retuned to (X=4, Y=2048) wins at >=64K elements
    # but pays ~3 µs Device launch-overhead at <32K shapes. Below ``_UC_MIN_NUMEL``
    # torch native (silu+mul) takes ~5 µs Device on (256, 128) bf16 — strictly
    # faster than UC at that size — so fall back per best-practices §C.1.
    # See ``docs/project-ops/perf-debug/op-MojoSwiGLU-2026-06-11.md`` for full
    # 4-backend per-shape table.
    _UC_MIN_NUMEL = 64 * 1024  # 64K elements

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            raise NotImplementedError(
                f"UCSwiGLU does not implement swiglu_limit > 0 (got {self.swiglu_limit}); "
                "the UC kernel only supports the unclipped variant. "
                "Per project rule 'wheel 没实现的就直接给报错', this wrapper does not "
                "silently fall back to torch — use TTX / torch_npu / torch_native instead."
            )
        if gate_out.numel() < self._UC_MIN_NUMEL:
            # Launch-overhead-floor fallback (best-practices §C.1).
            return super().forward(gate_out, up_out)
        return run_binary_kernel("mojo_swiglu", gate_out, up_out)
