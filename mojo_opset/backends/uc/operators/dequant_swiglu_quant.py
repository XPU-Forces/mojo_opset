import torch

from mojo_opset.core import MojoDequantSwiGLUQuant

from ._utils import _uc_kernels


_KERNEL_API = "mojo_dequant_swiglu_quant_bf16"
# Inner-tile column width of mojo_dequant_swiglu_quant_bf16.  Shapes with
# ``H < _KERNEL_INNER_Y`` enter the ``H_TILES == 1`` regime where the
# kernel reproducibly hangs ``torch.npu.synchronize()`` (P2-10 finding;
# see ``docs/project-ops/perf-debug/dequant-swiglu-quant-p2-10.md`` §5).
# Until the SetFlag/WaitFlag imbalance is fixed in the kernel, we fence
# small-H shapes back to the torch reference (which is the same path the
# wrapper already uses for any other capability-fence violation).
_KERNEL_INNER_Y = 128


class UCDequantSwiGLUQuant(MojoDequantSwiGLUQuant):
    """UC backend for the fused dequant + SwiGLU + dynamic-quant op.

    The wheel kernel handles the most common W8A8 MLP fast path:

        - 2D contiguous int8 input ``x`` of shape ``(tokens, 2H)``.
        - Single-group ``weight_scale`` ``(2H,)`` fp32 and ``quant_scale``
          ``(H,)`` fp32 (i.e. ``expert_num == 1``).
        - No ``activation_scale``, no ``bias``, no ``quant_offset`` and no
          grouped ``token_count``.
        - Default ``activate_left=False`` (mojo's ``silu(right) * left``).
        - Dynamic int8 quant (``quant_dtype=int8``, ``quant_mode=1``).
        - ``H >= 128`` (smaller H trips the kernel's H_TILES==1 hang; see
          P2-10 perf-debug doc).

    Anything outside that envelope falls back to ``MojoDequantSwiGLUQuant``'s
    torch reference forward, so accuracy parity is preserved.
    """

    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        activation_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        token_count: torch.Tensor = None,
    ):
        # Capability fence: anything the kernel cannot model must drop back to
        # the torch reference implementation.
        if (
            x.dtype != torch.int8
            or x.dim() != 2
            or x.shape[-1] % 2 != 0
            or self.hidden_size < _KERNEL_INNER_Y  # H_TILES==1 hang (P2-10)
            or activation_scale is not None
            or bias is not None
            or quant_offset is not None
            or token_count is not None
            or self.activate_left
            or self.quant_dtype != torch.int8
            or self.quant_mode != 1
            or self.weight_scale.shape[0] != 1
            or self.quant_scale.shape[0] != 1
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        try:
            kernels = _uc_kernels()
            kernel = kernels[_KERNEL_API]
        except (KeyError, ImportError):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        x_c = x.contiguous()
        rows, cols = x_c.shape  # rows = M (tokens), cols = N = 2*H
        half = cols // 2  # H

        # weight_scale Parameter shape is (expert_num, 2H); take the single
        # expert row and force fp32 contiguous to match the prim_func ABI.
        ws = self.weight_scale[0].to(torch.float32).contiguous()
        qs = self.quant_scale[0].to(torch.float32).contiguous()

        y = torch.empty((rows, half), dtype=torch.int8, device=x_c.device)
        scale_1d = torch.empty((rows,), dtype=torch.float32, device=x_c.device)

        # Trailing INT32 scalars follow the first-occurrence order of dim
        # names in the prim_func tensor annotations: M (from x), N (from x),
        # H (from quant_scale).
        kernel(x_c, ws, qs, y, scale_1d, rows, cols, half)

        # Mojo contract: scale shape == input.shape[:-1] + (1,) (amax keepdim).
        return y, scale_1d.unsqueeze(-1)
