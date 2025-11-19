import unittest
import torch
import torch_npu
import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
)
from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.gated_delta_rule.torch_impl import (
    torch_chunk_gated_delta_rule,
)
from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.utils import prepare_chunk_indices_and_offsets
from datetime import datetime

from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,  # 关键：生成 Perfetto 格式的处理器
)


def get_diff(name, real, custom, is_print=True):
    real = real.reshape(-1)
    custom = custom.reshape(-1)
    max_index = torch.argmax(torch.abs(real - custom)).item()
    real_val = real[max_index].item()
    custom_val = custom[max_index].item()
    max_diff = abs(real_val - custom_val)
    if is_print:
        print(name + " max diff", max_diff, "index@", max_index, real_val, custom_val)
    return max_diff


def perf(warm_up, itr, func):
    for ii in range(warm_up):
        func()
    torch.cuda.synchronize()
    a = datetime.now()
    for ii in range(itr):
        with torch.cuda.nvtx.range("kernel"):
            func()

    torch.cuda.synchronize()
    b = datetime.now()
    cost = (b - a).total_seconds() * 1000 / itr
    return cost


head_dim = 128
head_dim_v = 256
num_heads = 6
num_heads_kv = 1
use_torch_profiler = False


seqlens = torch.LongTensor([1024] * 32).int().cuda()
cu_seqlens = torch.cat(
    [
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seqlens, dim=0),
    ],
    dim=0,
).to(torch.int32)
max_seqlen = seqlens.max().item()
batch_size = len(seqlens)

q = torch.empty(cu_seqlens[-1], num_heads, head_dim, device="cuda").uniform_(-1, 1).to(torch.float16)
k = torch.empty(cu_seqlens[-1], num_heads_kv, head_dim, device="cuda").uniform_(-1, 1).to(torch.float16)
v = torch.empty(cu_seqlens[-1], num_heads_kv, head_dim_v, device="cuda").uniform_(-1, 1).to(torch.float16)

g = torch.empty(cu_seqlens[-1], num_heads_kv, device="cuda").uniform_(0, 1).to(torch.float16)
g = F.logsigmoid(g).to(torch.float16)

b = torch.empty(cu_seqlens[-1], num_heads_kv, device="cuda").uniform_(-1, 1).to(torch.float16)

scale = head_dim**-0.5

q = q.unsqueeze(0)
k = k.unsqueeze(0)
v = v.unsqueeze(0)
g = g.unsqueeze(0)
b = b.unsqueeze(0)

q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
k = k / torch.linalg.norm(k, dim=-1, keepdim=True).clamp(min=1e-6)

q_full = q.detach().clone()
k_full = (
    k.unsqueeze(3)
    .repeat(1, 1, 1, num_heads // num_heads_kv, 1)
    .reshape(1, k.shape[1], q.shape[2], k.shape[3])
    .detach()
    .clone()
)
v_full = (
    v.unsqueeze(3)
    .repeat(1, 1, 1, num_heads // num_heads_kv, 1)
    .reshape(1, v.shape[1], q.shape[2], v.shape[3])
    .detach()
    .clone()
)
g_full = g.unsqueeze(3).repeat(1, 1, 1, num_heads // num_heads_kv).reshape(1, g.shape[1], q.shape[2]).detach().clone()
b_full = b.unsqueeze(3).repeat(1, 1, 1, num_heads // num_heads_kv).reshape(1, g.shape[1], q.shape[2]).detach().clone()

q_full.requires_grad_()
q_full.retain_grad()
k_full.requires_grad_()
k_full.retain_grad()
v_full.requires_grad_()
v_full.retain_grad()
g_full.requires_grad_()
g_full.retain_grad()
b_full.requires_grad_()
b_full.retain_grad()

chunk_size = 64
host_cu_seqlens = cu_seqlens.cpu()
chunk_indices, chunk_offsets = prepare_chunk_indices_and_offsets(cu_seqlens, host_cu_seqlens, chunk_size)

torch_func = partial(
    torch_chunk_gated_delta_rule,
    q=q_full,
    k=k_full,
    v=v_full,
    g=g_full,
    beta=b_full,
    cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=False,
    scale=scale,
)
triton_fwd_func = partial(
    chunk_gated_delta_rule_fwd,
    q=q,
    k=k,
    v=v,
    g=g,
    beta=b,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_offsets=chunk_offsets,
    chunk_size=chunk_size,
    scale=scale,
    initial_state=None,
    output_final_state=False,
)

torch_out, _ = torch_func()
gamma, triton_out, A, w, u, h, v_new, final_state = triton_fwd_func()

do = torch.rand_like(triton_out)
triton_bwd_func = partial(
    chunk_gated_delta_rule_bwd,
    q=q,
    k=k,
    v=v,
    g=gamma,
    beta=b,
    A=A,
    w=w,
    u=u,
    h=h,
    v_new=v_new,
    scale=scale,
    do=do,
    dht=None,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_offsets=chunk_offsets,
    chunk_size=chunk_size,
    initial_state=None,
)

torch_out.backward(do, retain_graph=True)
dq, dk, dv, db, dg, _ = triton_bwd_func()

if num_heads > num_heads_kv:
    ref_dk = k_full.grad.reshape(
        k_full.grad.shape[0],
        k_full.grad.shape[1],
        k.shape[2],
        q.shape[2] // k.shape[2],
        k_full.grad.shape[3],
    ).sum(dim=3, keepdim=False)
    ref_dv = v_full.grad.reshape(
        v_full.grad.shape[0],
        v_full.grad.shape[1],
        v.shape[2],
        q.shape[2] // v.shape[2],
        v_full.grad.shape[3],
    ).sum(dim=3, keepdim=False)
    ref_db = b_full.grad.reshape(b_full.grad.shape[0], b_full.grad.shape[1], k.shape[2], q.shape[2] // k.shape[2]).sum(
        dim=3, keepdim=False
    )
    ref_dg = g_full.grad.reshape(g_full.grad.shape[0], g_full.grad.shape[1], g.shape[2], q.shape[2] // k.shape[2]).sum(
        dim=3, keepdim=False
    )


get_diff("fwd diff ", torch_out, triton_out)
get_diff("dq ", dq, q_full.grad)
get_diff("dk ", dk, ref_dk)
get_diff("dv ", dv, ref_dv)
get_diff("db ", db, ref_db)
get_diff("dg ", dg, ref_dg)

prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./linear_ut_perf", use_gzip=True),
    record_shapes=True,
    with_stack=True,
)

if use_torch_profiler:
    prof.start()

triton_fwd_cost = perf(10, 10, triton_fwd_func)
print("triton_fwd_cost %.3f ms" % triton_fwd_cost)

triton_bwd_cost = perf(10, 10, triton_bwd_func)
print("triton_bwd_cost %.3f ms" % triton_bwd_cost)

if use_torch_profiler:
    prof.stop()
