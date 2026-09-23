"""Cached-prefix SWA gradients, checked against independent CPU FP64 autograd."""

import pytest
import torch

from mojo_opset import functions as F


@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize("lengths", [((257, 257),), ((128, 384), (129, 386)), ((256, 2048),)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleave", [False, True])
def test_swa_backward_varlen(accuracy_backend, lengths, dtype, interleave):
    implementation, _, device = accuracy_backend
    generator = torch.Generator().manual_seed(17)
    q_heads, kv_heads, dim = 4, 2, 128
    local_window, global_window = 128, 32
    total_q = sum(q_len for q_len, _ in lengths)
    total_kv = sum(kv_len for _, kv_len in lengths)
    inputs = [
        torch.randn(tokens, heads, dim, generator=generator, dtype=dtype, device="cpu")
        for tokens, heads in [(total_q, q_heads), (total_kv, kv_heads), (total_kv, kv_heads)]
    ]
    grad_out = torch.randn(total_q, q_heads, dim, generator=generator, dtype=dtype, device="cpu")
    refs = [tensor.double().requires_grad_() for tensor in inputs]
    cu_q, cu_kv = [0], [0]
    outputs, unattended = [], []
    head_indices = torch.arange(q_heads, device="cpu")
    head_indices = head_indices % kv_heads if interleave else head_indices // (q_heads // kv_heads)
    for q_len, kv_len in lengths:
        q = refs[0][cu_q[-1] : cu_q[-1] + q_len].transpose(0, 1)
        k = refs[1][cu_kv[-1] : cu_kv[-1] + kv_len][:, head_indices].transpose(0, 1)
        v = refs[2][cu_kv[-1] : cu_kv[-1] + kv_len][:, head_indices].transpose(0, 1)
        q_pos = torch.arange(q_len, device="cpu")[:, None] + kv_len - q_len
        k_pos = torch.arange(kv_len, device="cpu")[None, :]
        mask = (k_pos <= q_pos) & ((k_pos < global_window) | (k_pos >= q_pos - local_window))
        scores = (q @ k.transpose(-1, -2)) * dim**-0.5
        probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        outputs.append((probabilities @ v).transpose(0, 1))
        unattended.append(~mask.any(dim=0))
        cu_q.append(cu_q[-1] + q_len)
        cu_kv.append(cu_kv[-1] + kv_len)
    expected = torch.cat(outputs)
    expected.backward(grad_out.double())
    unattended = torch.cat(unattended)
    cu_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_kv = torch.tensor(cu_kv, dtype=torch.int32, device=device)
    grad_out = grad_out.to(device)
    # Repeat to expose stale accumulator contents left by earlier invocations.
    for _ in range(2):
        actual_inputs = [tensor.to(device).requires_grad_() for tensor in inputs]
        actual = F.swa(
            *actual_inputs,
            cu_q,
            cu_kv,
            is_causal=True,
            local_window_size=local_window,
            global_window_size=global_window,
            softmax_scale=dim**-0.5,
            gqa_interleave=interleave,
            output_f32=True,
            implementation=implementation,
        )
        actual.backward(grad_out)
        pairs = [(actual, expected)] + [(got.grad, ref.grad) for got, ref in zip(actual_inputs, refs)]
        for got, ref in pairs:
            got = got.detach().cpu().double()
            ref = ref.detach()
            tolerance = 0.02 if dtype == torch.bfloat16 else 0.006
            torch.testing.assert_close(got, ref, atol=tolerance, rtol=tolerance)
            assert (got - ref).norm() / ref.norm() < 0.01
        for got in actual_inputs[1:]:
            assert torch.count_nonzero(got.grad.cpu()[unattended]).item() == 0
