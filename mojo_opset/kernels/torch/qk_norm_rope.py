"""Reference QK RMSNorm + half-split RoPE, mirroring the vLLM-Omni eager chain.

This is the accuracy yardstick mandated by the task statement: ``F.rms_norm`` followed by
``_apply_rope_table``. It is deliberately written as the upstream eager path (BF16 rotation
in the activation dtype) rather than as a re-statement of the Triton kernel, so the two
implementations share no formula. The Triton kernel is allowed to differ from it by BF16
rounding noise; the bit-exact cases target the kernel's internal ordering instead.
"""

import torch
import torch.nn.functional as F

__all__ = ["qk_norm_rope_infer_fwd"]


def _apply_rope_table(x, rope_table, rotary_dim):
    """Apply the packed ``[cos | sin]`` table to a ``[tokens, heads, head_dim]`` tensor.

    Upstream's ``_apply_rope_table`` additionally takes ``dtype``/``output_dtype``, which
    select the precision the rotation arithmetic runs in; its interleaved (Boogu-Image)
    caller passes ``dtype=torch.float32`` to match that kernel's composition. The half-split
    path ported here leaves both at their defaults — rotation in the activation dtype, one
    cast back — so neither knob is carried over. The table is cast down to the activation
    dtype first, which is what upstream's default does to a float32 table.
    """
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    first = x[..., :half]
    second = x[..., half:rotary_dim]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    return torch.cat((rotated_first, rotated_second, x[..., rotary_dim:]), dim=-1)


def qk_norm_rope_infer_fwd(q, k, q_weight, k_weight, rope_table, eps):
    head_dim = q.shape[-1]
    rotary_dim = rope_table.shape[-1]
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim),
        _apply_rope_table(k_norm, rope_table, rotary_dim),
    )
