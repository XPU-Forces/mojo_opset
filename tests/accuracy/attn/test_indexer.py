import math
import pytest
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoIndexer
from mojo_opset import MojoLightningIndexer
from mojo_opset.backends.ttx.kernels.ascend.indexer import (
    rope,
    linear,
    layer_norm,
    hadamard_triton,
    act_quant
)

from mojo_opset.backends.ttx.kernels.ascend.rope import (
    rome_fwd
)

def indexer_result_compare(triton_out: torch.Tensor, torch_out: torch.Tensor, topk = 2048):
    triton_out = triton_out.cpu()
    torch_out = torch_out.cpu()
    if triton_out.shape != torch_out.shape:
        print(f"shape not equal: triton shape {triton_out.shape},torch shape: {torch_out.shape}")
        return False

    B, M, _ = triton_out.shape
    all_match = True

    for i in range(B):
        for j in range(M):
            pred_indices = triton_out[i, j]
            target_indices = torch_out[i, j]
            pred_sorted = torch.sort(pred_indices)[0]
            target_sorted = torch.sort(target_indices)[0]
            match = torch.equal(pred_sorted, target_sorted)
            if not match: 
                all_match = False

    if all_match:
        print(f"{B} batch、{M} seqlen topk={topk} index is same")
    else:
        print(f"{B} batch、{M} seqlen topk={topk} index is not same")

    return all_match

@pytest.mark.parametrize(
    "batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor",
    [
        (
            batch,
            q_seq_len,
            8192 if q_seq_len==1 else q_seq_len,  #
            q_head_num,
            1,
            128,
            torch.randn(1),
        )
        for batch in [1, 2, 8, 16,] #32, 128
        for q_seq_len in [1, 1024, 4096, 8192]
        for q_head_num in [128, 64]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_indexer(batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor):
    device = dummy_tensor.device
    query = torch.randn(
        batch,
        q_seq_len,
        q_head_num,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query_scale = torch.randn(batch, q_seq_len, q_head_num, device=device, dtype=torch.float32)
    key = torch.randn(
        batch,
        k_seq_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mask = torch.full((q_seq_len, q_seq_len), float("-inf"), device=device).triu_(1) if q_seq_len > 1 else None
    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    op = MojoLightningIndexer(top_k = topk)
    triton_topk_result,triton_index_score=op.forward_std(query, query_scale, key, None, mask)
    torch_topk_result,torch_index_score=op.forward_ref(query, query_scale, key, None, mask)
    torch.testing.assert_close(torch_index_score, triton_index_score, atol=1e-4, rtol=1e-4)
    ## Due to the order of index_score in the entire collection may diff of torch
    ## we use topk result to compare precision， not forward_diff                
    assert indexer_result_compare(triton_topk_result,torch_topk_result,topk)
      
    #atol, rtol = 1e-3, 1e-3
    #op.forward_diff(query, query_scale, key, None, mask,atol=atol, rtol=rtol)

# Component test
#    rope,
#    linear,
#    layer_norm,
#    hadamard_triton,
#    act_quant

# rope start
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)

def precompute_freqs_cis(seqlen, dim, device="npu") -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis

@pytest.mark.parametrize(
    "bs, seq_len, n_qh, hd, rope_head_dim",
    [(1, 1024, 16, 128, 64),
    (1, 1024, 16, 16, 16)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_rope(bs, seq_len, n_qh, hd, rope_head_dim):
    device=torch.device("npu:0")
    # create input tensor
    q = torch.randn(bs, seq_len, n_qh, hd, device=device, dtype=torch.float32)
    k = torch.randn(bs, seq_len, n_qh, hd, device=device, dtype=torch.float32)

    freqs_cis = precompute_freqs_cis(seq_len, rope_head_dim)
    emb = torch.cat((freqs_cis, freqs_cis), dim=-1)
    mojo_cos = emb.real[None, None, :, :].contiguous() # torch.Size([1, 1, 32, 32])
    mojo_sin = emb.imag[None, None, :, :].contiguous() # torch.Size([1, 1, 32, 32])

    cos = freqs_cis.real.unsqueeze(0).expand(bs, -1, -1)
    sin = freqs_cis.imag.unsqueeze(0).expand(bs, -1, -1)

    q_rope_part,_ = torch.split(q, [rope_head_dim, hd - rope_head_dim], dim=-1)
    # indexer rope
    indexer_rope = apply_rotary_emb(q_rope_part, freqs_cis, False)
    # rome_fwd need BNSD input
    mojo_rope, _ = rome_fwd(q_rope_part.transpose(1,2), k.transpose(1,2), mojo_cos, mojo_sin)
    mojo_rope = mojo_rope.transpose(1,2)

    triton_rope = rope(q, cos, sin, rope_head_dim)
    torch.testing.assert_close(indexer_rope, mojo_rope, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(indexer_rope, triton_rope[:,:,:,:rope_head_dim], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q[:,:,:,rope_head_dim:], triton_rope[:,:,:,rope_head_dim:], rtol=1e-2, atol=1e-2)
# rope end

# linear start
@pytest.mark.parametrize("M,N,K,batch_size", [
    (256, 128, 512, 8),
    (16, 8, 32, 1),
    (8, 4, 16, 1),
    (2, 4, 3, 2),
    (64, 32, 128, 2),
])
@auto_switch_platform()
@bypass_not_implemented
def test_linear(M, N, K, batch_size):
    device=torch.device("npu:0")
    x = torch.ones(batch_size, M, K, device=device, dtype=torch.float32)
    w = torch.ones(K, N, device=device, dtype=torch.float32)

    # torch ref
    ref_output = torch.matmul(x, w)
    # triton
    triton_output = linear(x,w)
    # compare
    torch.testing.assert_close(triton_output, ref_output)
# linear end

# layer_norm start
def ref_layer_norm(x, weight, bias, eps):
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    y = (x - mu) / torch.sqrt(var + eps)
    y = y * weight
    return y

@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (
            batch,
            seq_len,
        )
        for batch in [256, 2, 32]
        for seq_len in [128, 3, 64]
    ],
)
def test_layer_norm(batch, seq_len):
    torch.set_default_device('npu:0')
    x = torch.randn(batch, seq_len, dtype=torch.float32)  # (batch, seq, dim)
    dim = x.size(-1)
    # test weight && bias
    weight = torch.ones(dim)
    bias = torch.zeros(dim)
    eps = 1e-5
    shape = x.shape
    output, x_2d, mean, rstd = layer_norm(x, weight, bias, eps=eps)
    assert output.shape == x.shape
    assert output.device == x.device
    assert output.dtype == torch.float32
    ref_output = ref_layer_norm(x, weight, bias, eps=eps)
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    assert torch.equal(output, ref_output)
# layer_norm end

# hadamard_triton start
def ref_hadamard(n: int, dtype, device):
    """Torch version hadamard matrix generation"""
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    
    if 2**lg2 != n:
        raise ValueError(f"n must be a power of 2, but got {n}")
    
    H = torch.tensor([1], dtype=dtype, device=device)
    for i in range(0, lg2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))
    return H

@pytest.mark.parametrize(
    "dim, dtype",
    [
        (16,torch.float32),
        (32,torch.float32),
        (64,torch.float32),
    ],
)
def test_hadamard(dim, dtype):
    torch.set_default_device('npu:0')
    device = torch.device("npu:0")
    triton_H = hadamard_triton(dim, dtype=dtype, device=device)
    # scipy.linalg hadamard
    ref_H = torch.tensor(hadamard(dim, dtype=float), dtype=dtype)
    # torch ref
    torch_ref_H = ref_hadamard(dim, dtype, device)
    torch.testing.assert_close(torch_ref_H, ref_H)
    torch.testing.assert_close(triton_H, ref_H)
# hadamard_triton end

# act_quant start
def ref_quant(input_tensor: torch.Tensor, scale_tensor: torch.Tensor):
    scaled = input_tensor * scale_tensor  # shape: [batch, seq_len, rows, cols]
    max_abs = scaled.abs().amax(dim=-1)  # shape: [batch, seq_len, rows]
    quant_scale = max_abs / 127.0
    quantized = (scaled / quant_scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return quantized, quant_scale

def test_quant():
    batch = 2
    seq_len = 3
    rows = 4
    cols = 5
    
    input_tensor = torch.randn(batch, seq_len, rows, cols, dtype=torch.float32, device="npu")
    scale_tensor = torch.randn(cols, dtype=torch.float32, device="npu")

    output_triton, quant_scale_triton = act_quant(input_tensor, scale_tensor)
    output_ref, quant_scale_ref = ref_quant(input_tensor, scale_tensor)

    torch.testing.assert_close(output_triton, output_ref, atol=1, rtol=0)
    torch.testing.assert_close(quant_scale_triton, quant_scale_ref, atol=1e-4, rtol=1e-3)
# act_quant end

# testindexer
@pytest.mark.parametrize(
    "batch, q_seq_len, q_head_num, head_dim, dim, q_lora_rank, dummy_tensor",
    [
        (
            batch,
            q_seq_len,
            q_head_num,
            128,
            7168,
            1536,
            torch.randn(1),
        )
        # for batch in [1, 2, 8, 16,] #32, 128
        # for q_seq_len in [1, 1024, 4096, 8192]
        # for q_head_num in [128, 64]
        for batch in [1] #32, 128
        for q_seq_len in [1024]
        for q_head_num in [64]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_indexer_with_freqs_cis(batch, q_seq_len, q_head_num, head_dim, dim, q_lora_rank, dummy_tensor):
    rope_head_dim = 64
    device = dummy_tensor.device
    x = torch.randn(
        batch,
        q_seq_len,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query_scale = torch.randn(batch, q_seq_len, q_lora_rank, device=device, dtype=torch.float32)
    freqs_cis = precompute_freqs_cis(q_seq_len, rope_head_dim)

    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    op = MojoIndexer(topk = topk)
    #triton_topk_result,triton_index_score=op.forward_std(query, query_scale, 0,freqs_cis)
    #torch_topk_result,torch_index_score=op.forward_ref(query, query_scale, 0, freqs_cis)
    #torch.testing.assert_close(torch_index_score, triton_index_score, atol=1e-4, rtol=1e-4)
                           
    #compare_result(triton_topk_result,torch_topk_result,topk)
    atol, rtol = 1e-3, 1e-3
    op.forward_diff(x, query_scale, 0,freqs_cis, atol=atol, rtol=rtol)