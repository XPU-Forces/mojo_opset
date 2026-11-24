"""Standalone AscendC RoPE; extension launchers submit through the NPU queue."""

import torch

from ._extension import launcher


def _rotate_leaf(input, cos, sin, inverse):
    rotary = cos.shape[-1]
    if rotary <= 0 or rotary % 2 or rotary > input.shape[-1]:
        raise ValueError("RoPE rotation width must be positive, even and fit the input")
    half = rotary // 2
    padded_half = ((half + 7) // 8) * 8
    prefix = input.shape[-1] - rotary

    def pad_pair(value, fill):
        left, right = value.chunk(2, -1)
        padding = padded_half - half
        return torch.cat(
            (
                torch.nn.functional.pad(left, (0, padding), value=fill),
                torch.nn.functional.pad(right, (0, padding), value=fill),
            ),
            -1,
        )

    if padded_half != half:
        # Align each rotary half separately. Padding the suffix as one block
        # would change which channels are paired (for example for R=88).
        output = torch.cat((input[..., :prefix], pad_pair(input[..., prefix:], 0)), -1)
        cos, sin = pad_pair(cos, 1), pad_pair(sin, 0)
    else:
        output = input.clone(memory_format=torch.contiguous_format)
    launcher("rope_backward" if inverse else "rope_forward")(
        output.reshape(-1, output.shape[-2], output.shape[-1]),
        sin.contiguous(),
        cos.contiguous(),
        cos.shape[-1],
    )
    if padded_half != half:
        output = torch.cat(
            (
                output[..., :prefix],
                output[..., prefix : prefix + half],
                output[..., prefix + padded_half : prefix + padded_half + half],
            ),
            -1,
        )
    return output


def _rotate(q, k, cos, sin, unsqueeze_dim, inverse):
    if q.ndim not in (3, 4) or k.ndim != q.ndim or cos.ndim not in (2, 3):
        raise ValueError("native RoPE expects 3D/4D Q,K and 2D/3D tables")
    if unsqueeze_dim < 0:
        unsqueeze_dim += cos.ndim + 1
    if unsqueeze_dim not in (cos.ndim - 2, cos.ndim - 1):
        raise ValueError("unsqueeze_dim must identify the head axis")
    head_first = unsqueeze_dim == cos.ndim - 2
    outputs = []
    for x in (q, k):
        original_shape = x.shape
        normalized = x.transpose(-2, -3) if head_first else x
        if normalized.ndim == 3:
            if cos.ndim != 2 or cos.shape[0] != normalized.shape[0]:
                raise ValueError("RoPE table sequence length must match input")
            c, s = cos.contiguous(), sin.contiguous()
        else:
            batch, seq = normalized.shape[:2]
            c = cos.expand(batch, seq, cos.shape[-1]).contiguous().reshape(batch * seq, cos.shape[-1])
            s = sin.expand(batch, seq, sin.shape[-1]).contiguous().reshape(batch * seq, sin.shape[-1])
        y = _rotate_leaf(normalized, c, s, inverse)
        outputs.append(y.transpose(-2, -3).contiguous() if head_first else y.reshape(original_shape))
    return tuple(outputs)


@torch.library.custom_op("mojo_npu_native_a2::rope_fwd", mutates_args=())
def rope_fwd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rotate(q, k, cos, sin, unsqueeze_dim, False)


@rope_fwd.register_fake
def _fake_fwd(q, k, cos, sin, unsqueeze_dim):
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(
        k, memory_format=torch.contiguous_format
    )


@torch.library.custom_op("mojo_npu_native_a2::rope_bwd", mutates_args=())
def rope_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rotate(q, k, cos, sin, unsqueeze_dim, True)


@rope_bwd.register_fake
def _fake_bwd(
    q,
    k,
    cos,
    sin,
    unsqueeze_dim,
):
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(
        k, memory_format=torch.contiguous_format
    )
