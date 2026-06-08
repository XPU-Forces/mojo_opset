"""UC backend for ``MojoChannelRMSNorm``.

ChannelRMSNorm semantics (see :class:`mojo_opset.experimental.operators.normalization.MojoChannelRMSNorm`)::

    y[..., c] = F.normalize(x, dim=channel_axis)[..., c] * sqrt(C) * weight[c]
                [+ bias[c]]

By algebraic equivalence (see kernel docstring) this is identical to
RMSNorm with ``eps_effective = 1e-12 / C`` plus an optional bias term:

    y[..., c] = x[..., c] * rsqrt(mean(x[..., :]**2) + 1e-12/C)
                * weight_flat[c] [+ bias_flat[c]]

The dedicated kernel ``mojo_channel_rmsnorm_{bf16,fp16}`` always accepts a
4-tensor argument list ``(x, weight, bias, y)`` over a 2-D ``(M, N=C)``
view of the input.  For the no-bias case (``op.bias is None``) the wrapper
synthesises a zeros bias tensor so a single kernel variant suffices.

For ``channel_first=True`` layouts (NCHW / NCTHW) the wrapper permutes the
channel dimension to the trailing axis, flattens everything else into
``M``, and inverts the permutation on the output.  For
``channel_first=False`` layouts (NHWC / NTHWC) no permutation is needed.

Anything outside the fast path defers to ``MojoChannelRMSNorm.forward``.
"""

import torch

from mojo_opset.experimental.operators.normalization import MojoChannelRMSNorm

from ._utils import _typed_api
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
# ``torch.nn.functional.normalize`` default eps (raw, before /C rescaling).
_NORMALIZE_DEFAULT_EPS = 1e-12


def _flat_weight(weight: torch.Tensor, norm_size: int) -> torch.Tensor:
    """Return a contiguous ``(norm_size,)`` view of ``weight``.

    Accepts ``(C,)``, ``(C, 1, 1)``, ``(C, 1, 1, 1)`` shapes (the two latter
    are the channel-first broadcasting layouts; the former is channel-last).
    """
    if weight is None:
        return None
    flat = weight.reshape(-1)
    if flat.numel() != norm_size:
        return None
    return flat.contiguous()


def _can_use_kernel(op: MojoChannelRMSNorm, hidden_state: torch.Tensor) -> bool:
    if not isinstance(hidden_state, torch.Tensor):
        return False
    if hidden_state.dtype not in _SUPPORTED_DTYPES:
        return False
    if op.norm_size <= 0:
        return False
    if hidden_state.numel() == 0:
        return False

    if op.channel_first:
        # Expect (N, C, H, W) for 2D images or (N, C, T, H, W) for video.
        if hidden_state.dim() < 3:
            return False
        if hidden_state.shape[1] != op.norm_size:
            return False
    else:
        # Expect (N, H, W, C) for 2D images or (N, T, H, W, C) for video.
        if hidden_state.dim() < 2:
            return False
        if hidden_state.shape[-1] != op.norm_size:
            return False

    flat_w = _flat_weight(op.weight, op.norm_size)
    if flat_w is None:
        return False
    if op.weight.device != hidden_state.device:
        return False
    if op.bias is not None:
        flat_b = _flat_weight(op.bias, op.norm_size)
        if flat_b is None:
            return False
        if op.bias.device != hidden_state.device:
            return False

    return True


class UCChannelRMSNorm(MojoChannelRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if not _can_use_kernel(self, hidden_state):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        try:
            api = _typed_api("mojo_channel_rmsnorm", hidden_state.dtype)
        except NotImplementedError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        dtype = hidden_state.dtype
        device = hidden_state.device
        N = self.norm_size

        # Move channel dim to the last axis if needed so the kernel always
        # reduces over the trailing contiguous dim.
        if self.channel_first:
            ndim = hidden_state.dim()
            # (N, C, *) -> (N, *, C)
            perm = [0] + list(range(2, ndim)) + [1]
            inv_perm = [0, ndim - 1] + list(range(1, ndim - 1))
            x_ch_last = hidden_state.permute(*perm).contiguous()
            channel_last_shape = x_ch_last.shape
        else:
            x_ch_last = hidden_state.contiguous()
            channel_last_shape = x_ch_last.shape
            perm = None
            inv_perm = None

        flat_x = x_ch_last.reshape(-1, N)
        M = flat_x.shape[0]
        if M == 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        weight_flat = _flat_weight(self.weight, N)
        if weight_flat.dtype != dtype:
            weight_flat = weight_flat.to(dtype)
        weight_flat = weight_flat.contiguous()

        if self.bias is not None:
            bias_flat = _flat_weight(self.bias, N)
            if bias_flat.dtype != dtype:
                bias_flat = bias_flat.to(dtype)
            bias_flat = bias_flat.contiguous()
        else:
            bias_flat = torch.zeros(N, dtype=dtype, device=device)

        # F.normalize default eps is 1e-12 added to sum(x^2); rewriting in
        # RMSNorm form gives eps_effective = eps_raw / norm_size added to
        # mean(x^2).
        eps_effective = float(_NORMALIZE_DEFAULT_EPS) / float(N)

        y_flat = torch.empty_like(flat_x)
        _uc_kernels()[api](
            flat_x.contiguous(),
            weight_flat,
            bias_flat,
            y_flat,
            M,
            N,
            eps_effective,
        )

        y_ch_last = y_flat.reshape(channel_last_shape)
        if self.channel_first:
            return y_ch_last.permute(*inv_perm).contiguous()
        return y_ch_last
