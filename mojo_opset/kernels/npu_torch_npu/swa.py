"""Causal SWA as a disjoint prefix/local union of native CANN attention calls.

The prefix is counted once, including where it overlaps the local window.
Backward uses the *union* output and normalization for both tail branches.
No KVC layout or backend selection policy belongs in this provider.
"""

import torch
import torch_npu

_MASKS = {}
_CU = {}


def _kwargs(q, k, mode, window):
    mask = None
    if mode:
        if q.device not in _MASKS:
            _MASKS[q.device] = torch.ones(2048, 2048, device=q.device, dtype=torch.bool).triu_(1)
        mask = _MASKS[q.device]
    return dict(
        head_num=q.shape[1],
        input_layout="TND",
        atten_mask=mask,
        keep_prob=1.0,
        actual_seq_qlen=(q.shape[0],),
        actual_seq_kvlen=(k.shape[0],),
        sparse_mode=mode,
        pre_tockens=window,
        next_tockens=0,
        softmax_layout="TND",
    )


def _fwd(q, k, v, mode, window, scale):
    return torch_npu.npu_fusion_attention(q, k, v, scale=scale, **_kwargs(q, k, mode, window))[:3]


def _bwd(q, k, v, do, out, mx, sm, mode, window, scale):
    return torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        do.contiguous(),
        softmax_max=mx.contiguous(),
        softmax_sum=sm.contiguous(),
        attention_in=out.contiguous(),
        scale_value=scale,
        **_kwargs(q, k, mode, window),
    )[:3]


def _merge(a, b):
    key = (a[0].device, a[0].shape[0])
    if key not in _CU:
        _CU[key] = torch.tensor([0, a[0].shape[0]], device=a[0].device, dtype=torch.int64)
    return torch_npu.npu_ring_attention_update(
        *a,
        *b,
        actual_seq_qlen=_CU[key],
        input_layout="TND",
        input_softmax_layout="TND",
    )


def _bounds(cuq, cuk, qtotal, ktotal):
    # CP owns suffix queries of each packed KV prefix. Native band masks are
    # right-aligned, so every sample may have its own nonnegative Q offset.
    a = cuq.tolist()
    b = a if cuq is cuk else cuk.tolist()
    if len(a) < 2 or len(a) != len(b) or a[0] != 0 or b[0] != 0 or a[-1] != qtotal or b[-1] != ktotal:
        raise ValueError("torch_npu SWA requires matching packed sample counts and valid cumulative lengths")
    if any(q1 <= q0 or k1 <= k0 or q1 - q0 > k1 - k0 for q0, q1, k0, k1 in zip(a, a[1:], b, b[1:])):
        raise ValueError("torch_npu SWA requires nonempty samples with Q length <= KV length")
    return tuple(zip(a, a[1:], b, b[1:]))


def _validate(q, k, v, causal, window, global_size, interleave, output_f32=False):
    if not causal or interleave or output_f32:
        raise NotImplementedError("torch_npu SWA requires causal=True, gqa_interleave=False, output_f32=False")
    if window is None or window <= 0 or global_size is None or global_size < 0:
        raise NotImplementedError("torch_npu SWA requires a positive local window and nonnegative global window")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("torch_npu SWA requires TND Q/K/V with matching K/V shapes")
    if k.shape[1] <= 0 or q.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("torch_npu SWA requires valid GQA head counts")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or q.shape[-1] != 128:
        raise NotImplementedError("torch_npu SWA currently supports BF16 head_dim=128")
    if q.device.type != "npu" or q.device != k.device or q.device != v.device:
        raise ValueError("torch_npu SWA requires Q/KV on the same NPU")


@torch.library.custom_op("mojo_npu_torch_npu::swa_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cuq: torch.Tensor,
    cuk: torch.Tensor,
    window: int,
    global_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bounds = _bounds(cuq, cuk, q.shape[0], k.shape[0])
    outputs, lses = [], []
    for qa, qb, ka, kb in bounds:
        qi, ki, vi = q[qa:qb].contiguous(), k[ka:kb].contiguous(), v[ka:kb].contiguous()
        nq, nk = qb - qa, kb - ka
        g = min(global_size, nk)
        if g == 0 or g == nk:
            out, mx, sm = _fwd(qi, ki, vi, 4 if g == 0 else 3, window, scale)
        else:
            # Only queries whose absolute compact positions precede g belong
            # to the global triangle; later queries see disjoint global/local KV.
            split = max(0, g - (nk - nq))
            tail = _merge(
                _fwd(qi[split:], ki[:g], vi[:g], 0, window, scale),
                _fwd(qi[split:], ki[g:], vi[g:], 4, window, scale),
            )
            if split:
                prefix = _fwd(qi[:split], ki[:g], vi[:g], 3, window, scale)
                out, mx, sm = (torch.cat((p, t), dim=0) for p, t in zip(prefix, tail))
            else:
                out, mx, sm = tail
        outputs.append(out)
        lses.append(mx[..., 0] + sm[..., 0].log())
    return torch.cat(outputs), torch.cat(lses).transpose(0, 1).contiguous(), q.new_empty(0)


@_forward.register_fake
def _forward_fake(q, k, v, cuq, cuk, window, global_size, scale):
    return (torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32), q.new_empty(0))


@torch.library.custom_op("mojo_npu_torch_npu::swa_bwd", mutates_args=())
def _backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cuq: torch.Tensor,
    cuk: torch.Tensor,
    window: int,
    global_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bounds = _bounds(cuq, cuk, q.shape[0], k.shape[0])
    grads = []
    for qa, qb, ka, kb in bounds:
        qi, doi, oi = (t[qa:qb].contiguous() for t in (q, do, out))
        ki, vi = (t[ka:kb].contiguous() for t in (k, v))
        # Both branches use the union probabilities, including dQ and shared dKV.
        mx = lse[:, qa:qb].transpose(0, 1).unsqueeze(-1).expand(-1, -1, 8).contiguous()
        sm = torch.ones_like(mx)
        nq, nk = qb - qa, kb - ka
        g = min(global_size, nk)
        if g == 0 or g == nk:
            grad = _bwd(qi, ki, vi, doi, oi, mx, sm, 4 if g == 0 else 3, window, scale)
        else:
            split = max(0, g - (nk - nq))
            dqg, dkg, dvg = _bwd(
                qi[split:], ki[:g], vi[:g], doi[split:], oi[split:], mx[split:], sm[split:], 0, window, scale
            )
            dql, dkl, dvl = _bwd(
                qi[split:], ki[g:], vi[g:], doi[split:], oi[split:], mx[split:], sm[split:], 4, window, scale
            )
            dq = dqg + dql
            if split:
                dqp, dkp, dvp = _bwd(
                    qi[:split], ki[:g], vi[:g], doi[:split], oi[:split], mx[:split], sm[:split], 3, window, scale
                )
                dq, dkg, dvg = torch.cat((dqp, dq)), dkp + dkg, dvp + dvg
            grad = dq, torch.cat((dkg, dkl)), torch.cat((dvg, dvl))
        grads.append(grad)
    return tuple(torch.cat(parts) for parts in zip(*grads))


@_backward.register_fake
def _backward_fake(do, q, k, v, out, lse, cuq, cuk, window, global_size, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v))


def swa_fwd(
    q, k, v, cuq, cuk, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave, output_f32
):
    _validate(q, k, v, is_causal, local_window_size, global_window_size, gqa_interleave, output_f32)
    return _forward(q, k, v, cuq, cuk, local_window_size, global_window_size, softmax_scale)


def swa_bwd(
    do, q, k, v, out, lse, cuq, cuk, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
):
    _validate(q, k, v, is_causal, local_window_size, global_window_size, gqa_interleave)
    return _backward(do, q, k, v, out, lse, cuq, cuk, local_window_size, global_window_size, softmax_scale)
