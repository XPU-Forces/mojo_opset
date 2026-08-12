import pytest
import torch
import torch.nn.functional as F

from mojo_opset.experimental import mojo_flex_attention
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.backends.ttx.kernels.npu.utils import is_910
import torch_npu._inductor
from mojo_opset.backends.ttx.kernels.npu.flex_attention import _build_packed_block_mask_streaming
from mojo_opset.backends.ttx.kernels.npu.flex_attention import create_block_mask_patched
from mojo_opset.backends.ttx.kernels.npu.flex_attention import triton_create_mask
from mojo_opset.backends.ttx.kernels.npu.flex_attention import MASK_BLOCK_SIZE
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


# NPU device validation monkey-patch (same as original test)
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass

GEN_MASK_TRITON = False
USE_MOJO_FLEX_ATTENTION = True
FULL_MASK_MODALITIES = ("image_gen", "image_vae")

SEED = 0
APPLY_Q_CHUNK = 2048
Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128

def _device():
    return get_torch_device()
def _sync():
    if _device() == "npu":
        torch.npu.synchronize()
    elif _device() == "cuda":
        torch.cuda.synchronize()

# ============================================================================
# Mask function definitions
# ============================================================================
def _sparse_mask_mod(problem):
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= W)
        glob = causal & (kv_idx >= doc_start[q_idx]) & (kv_idx < doc_start[q_idx] + G)
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img
    return mask_mod


def _stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = video_ids[q_idx] == video_ids[kv_idx]
        frame_causal = frame_ids[q_idx] >= frame_ids[kv_idx]
        return same_doc & frame_causal
    return mask_mod

def _video_stair_mask_mod(problem):
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_video = video_ids[q_idx] == video_ids[kv_idx]
        same_frame = frame_ids[q_idx] == frame_ids[kv_idx]
        prev_frame = frame_ids[q_idx] > frame_ids[kv_idx]
        return same_video & (same_frame | prev_frame)
    return mask_mod

def _cross_sample_causal_video_bidir_mask_mod(problem):
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_video = modality[q_idx] > 0
        same_video = is_video & (modality[q_idx] == modality[kv_idx])
        return causal | same_video
    return mask_mod

def _full_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img
    return mask_mod

_MASK_FUNCS = [
    ("sparse", _sparse_mask_mod),
    ("full", _full_mask_mod),
    ("stair", _stair_mask_mod),
    ("video_stair", _video_stair_mask_mod),
    ("cross_sample_causal_video_bidir", _cross_sample_causal_video_bidir_mask_mod),
]

_MASK_FUNC_TO_TYPE = {id(fn): name for name, fn in _MASK_FUNCS}

def _build_dense_mask(mask_func, problem):
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
    return triton_create_mask(problem, mask_type_str, tile_size=MASK_BLOCK_SIZE)

# ============================================================================
# Attention wrappers
# ============================================================================
def _flex_attention_mojo(q, k, v, mask, block_mask, dropout_rate=0.0, input_format=None):
    if input_format == "head-first":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if mask is not None:
        block_mask.dense_mask = mask
    if USE_MOJO_FLEX_ATTENTION:
        output = mojo_flex_attention(q, k, v, block_mask=block_mask)
    else:
        flex_compiled = torch.compile(flex_attention, backend="inductor")
        output = flex_compiled(q, k, v, block_mask=block_mask,
                               enable_gqa=True, return_lse=False)                            
    return output.transpose(1, 2)

def _build_block_mask(mask_func, problem):
    SEQ_LEN = problem["total_s"]
    device=problem["q"].device
    if GEN_MASK_TRITON:
        classify_strategy= "fused" if not is_910() else "decoupled"
        mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
        packed_block_mask = _build_packed_block_mask_streaming(mask_type_str, problem, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, classify_strategy=classify_strategy)
    else:
        if USE_MOJO_FLEX_ATTENTION:
            packed_block_mask = create_block_mask_patched(
                mask_func(problem), B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN,
                device=device, BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
            )
        else:
            packed_block_mask = create_block_mask(mask_func(problem),B=1, H=1, Q_LEN=SEQ_LEN, 
                KV_LEN=SEQ_LEN,device=device, BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE))
    return packed_block_mask

def _sdpa_with_dense_mask(query_states, key_states, value_states, attention_mask, dropout_rate, input_format):
    if input_format == "head-first":
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    mask_2d = attention_mask
    while mask_2d.dim() > 2:
        mask_2d = mask_2d[0]

    q_len_total = query_states.size(2)
    block_q = APPLY_Q_CHUNK if APPLY_Q_CHUNK is not None else q_len_total
    chunks = []
    for qs in range(0, q_len_total, block_q):
        qe = min(qs + block_q, q_len_total)
        row = mask_2d[qs:qe]
        col_any = row.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            chunks.append(query_states.new_zeros((query_states.size(0), query_states.size(1), qe - qs, query_states.size(3))))
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        chunks.append(
            F.scaled_dot_product_attention(
                query_states[:, :, qs:qe], key_states[:, :, kmin:kmax], value_states[:, :, kmin:kmax],
                attn_mask=row[None, None, :, kmin:kmax], dropout_p=dropout_rate, enable_gqa=False,
            )
        )
    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous()

# ============================================================================
# Data building
# ============================================================================
def _build_video_indicators(device,frame_lens):
    segment_ids, doc_start, video_ids, frame_ids, modality = [], [], [], [], []
    sample_start = 0
    next_video_id = 0

    for sample_id, sample_videos in enumerate(frame_lens):
        for frame_lens in sample_videos:
            cur_video_id = next_video_id
            next_video_id += 1
            for frame_id, frame_len in enumerate(frame_lens):
                segment_ids.append(torch.full((frame_len,), sample_id, dtype=torch.long))
                doc_start.append(torch.full((frame_len,), sample_start, dtype=torch.long))
                video_ids.append(torch.full((frame_len,), cur_video_id, dtype=torch.long))
                frame_ids.append(torch.full((frame_len,), frame_id, dtype=torch.long))
                modality.append(torch.full((frame_len,), cur_video_id + 1, dtype=torch.long))
        sample_start += sum(sum(fl) for fl in sample_videos)

    return {
        "segment_ids": torch.cat(segment_ids).to(device),
        "doc_start": torch.cat(doc_start).to(device),
        "video_ids": torch.cat(video_ids).to(device),
        "frame_ids": torch.cat(frame_ids).to(device),
        "modality": torch.cat(modality).to(device),
    }

def _build_modality_indicators(device, data_length=None, data_input_type=None, image_modalities=None):
    indicator = []
    iidx = 1
    for sample_types, sample_lens in zip(data_input_type, data_length):
        for sample_type, sample_len in zip(sample_types, sample_lens):
            if sample_type in image_modalities:
                indicator.append(torch.full((sample_len,), iidx, dtype=torch.long))
                iidx += 1
            else:
                indicator.append(torch.full((sample_len,), -1, dtype=torch.long))
    return torch.cat(indicator).to(device)

def build_problem(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func):
    device = _device()
    torch.manual_seed(SEED)

    num_q_heads = q_head
    num_kv_heads = kv_head

    sample_lens = [sum(s) for s in data_lens]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int32),
        torch.tensor(sample_lens, device=device),
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)

    q = torch.rand(batch_size, num_q_heads, total_s, head_dim, device=device, dtype=dtype)
    k = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)
    v = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)

    if mask_func in [_video_stair_mask_mod, _stair_mask_mod]:
        meta = _build_video_indicators(device,data_types)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": meta["segment_ids"], "doc_start": meta["doc_start"],
            "video_ids": meta["video_ids"], "frame_ids": meta["frame_ids"], "modality": meta["modality"],
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": sliding_windows, "global_window": global_windows,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        }
    else:
        modality = _build_modality_indicators(device=device,data_length=data_lens, 
                                              data_input_type=data_types, image_modalities=FULL_MASK_MODALITIES,)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": segment_ids.long(), "modality": modality, "doc_start": doc_start,
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": sliding_windows, "global_window": global_windows,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        }


# ============================================================================
# Shared parametrize decorator for mask functions
# ============================================================================
_mask_func_param = pytest.mark.parametrize(
    "mask_func",
    [fn for _, fn in _MASK_FUNCS],
    ids=[name for name, _ in _MASK_FUNCS],
)

@pytest.mark.parametrize(
    "batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func,",
    [     
        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"], 
            ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _sparse_mask_mod,id="sparse_2000_22000"), 

        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"], 
                    ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _full_mask_mod,id="full_2000_22000"), 

        pytest.param(1, 16, 8, 128, [[2000, 22000, 2000], [2000, 22000, 2000]],[["text", "image_gen", "text"], 
                            ["text", "image_gen", "text"]], 1024, 4, torch.bfloat16, _cross_sample_causal_video_bidir_mask_mod,id="cross_2000_22000"), 

        pytest.param(1, 16, 8, 128, [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],[
                        [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
                        [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],]
                    , 1024, 4, torch.bfloat16, _video_stair_mask_mod,id="video_stair_6500"), 

        pytest.param(1, 16, 8, 128, [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],[
                        [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
                        [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],], 1024, 4, torch.bfloat16, _stair_mask_mod,id="stair_6500"), 
      
    ]
   
)
@pytest.mark.skipif(get_platform() != "npu", reason="FlexAttention TTX backend requires NPU")
@bypass_not_implemented
def test_flex_attention(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func):
    problem = build_problem(batch_size,q_head, kv_head, head_dim, data_lens, data_types, sliding_windows, global_windows, dtype, mask_func)

    q_base = problem["q"]
    k_base = problem["k"]
    v_base = problem["v"]

    q_mojo = q_base.detach().clone().requires_grad_(True)
    k_mojo = k_base.detach().clone().requires_grad_(True)
    v_mojo = v_base.detach().clone().requires_grad_(True)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)

    SEQ_LEN = problem["total_s"]

    packed_block_mask = _build_block_mask(mask_func, problem)
                                                  
    _sync()

    return_grid = torch.tensor(SEQ_LEN, dtype=dtype, device=torch.device(_device()))

    mojo_output = _flex_attention_mojo(q_mojo, k_mojo, v_mojo, None, packed_block_mask, 0.0, None)
    _sync()

    dense_mask = _build_dense_mask(mask_func, problem)
    _sync()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>dense_mask.sum().item()", dense_mask.to("cpu").sum().item())

    ref_output = _sdpa_with_dense_mask(q_ref, k_ref, v_ref, dense_mask, 0.0, None)
    _sync()

    assert mojo_output.shape == ref_output.shape
    torch.testing.assert_close(mojo_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    _sync()

    mojo_output.float().mean().backward(return_grid)
    _sync()

    ref_output.float().mean().backward(return_grid)
    _sync()

    torch.testing.assert_close(q_mojo.grad.cpu(), q_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k_mojo.grad.cpu(), k_ref.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v_mojo.grad.cpu(), v_ref.grad.cpu(), atol=5e-3, rtol=5e-3)



if __name__ == "__main__":
    import sys

    mask_map = dict(_MASK_FUNCS)
    name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    if name == "all":
        for n, fn in _MASK_FUNCS:
            print(f"\n{'=' * 60}")
            print(f"Testing: {n}")
            print(f"{'=' * 60}")
            test_flex_attention(fn)
    else:
        test_flex_attention(mask_map[name])

