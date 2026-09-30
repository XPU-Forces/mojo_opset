"""Reference implementation for the dynamic MX quant SR function.

Thin device wrapper over the NumPy byte-exact golden (Philox RNG, bit
reversal, SR Algorithm 19, E8M0 scales).  Requires numpy and ml_dtypes.
"""

import torch

from .dynamic_mx_quant_sr_golden import dynamic_mx_quant_golden

_DST_NAMES = {23: "float8_e5m2", 24: "float8_e4m3fn"}


def _to_numpy(tensor, dtype):
    return tensor.detach().cpu().float().numpy().astype(dtype)


def dynamic_mx_quant_sr_fwd(
    input,
    axis,
    round_mode,
    dst_type,
    block_size,
    scale_alg,
    dst_type_max,
    max_low_bound,
):
    if round_mode != "stochastic":
        raise NotImplementedError("reference supports round_mode='stochastic' only")
    if dst_type not in _DST_NAMES:
        raise NotImplementedError("reference supports dst_type 23 (E5M2) or 24 (E4M3FN)")
    if block_size != 32:
        raise NotImplementedError("reference supports block_size=32 only")
    if dst_type_max != 0.0:
        raise NotImplementedError("reference supports dst_type_max=0.0 only")
    import numpy as np
    from ml_dtypes import bfloat16

    values = _to_numpy(input, bfloat16 if input.dtype == torch.bfloat16 else np.float32)
    codes, scale = dynamic_mx_quant_golden(
        values,
        _DST_NAMES[dst_type],
        scale_alg=scale_alg,
        max_low_bound=max_low_bound,
        axis=axis,
    )
    dst_dtype = torch.float8_e5m2 if dst_type == 23 else torch.float8_e4m3fn
    y = torch.from_numpy(np.ascontiguousarray(codes)).view(dst_dtype).to(input.device)
    mx_scale = (
        torch.from_numpy(np.ascontiguousarray(scale))
        .view(torch.float8_e8m0fnu)
        .to(input.device)
    )
    return y, mx_scale
