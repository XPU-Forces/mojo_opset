from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.utils.flex_attention_mask import create_flex_block_mask
from tests._checks import assert_close
from tests._checks import assert_repeatable


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


def _sparse_mask_mod(problem):
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & (q_idx - kv_idx <= W)
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


BUSINESS_FLEX_CASES = [
    pytest.param(
        1,
        16,
        8,
        128,
        [[2000, 22000, 2000], [2000, 22000, 2000]],
        [["text", "image_gen", "text"], ["text", "image_gen", "text"]],
        1024,
        4,
        torch.bfloat16,
        _sparse_mask_mod,
        id="sparse_2000_22000",
    ),
    pytest.param(
        1,
        16,
        8,
        128,
        [[2000, 22000, 2000], [2000, 22000, 2000]],
        [["text", "image_gen", "text"], ["text", "image_gen", "text"]],
        1024,
        4,
        torch.bfloat16,
        _full_mask_mod,
        id="full_2000_22000",
    ),
    pytest.param(
        1,
        16,
        8,
        128,
        [[2000, 22000, 2000], [2000, 22000, 2000]],
        [["text", "image_gen", "text"], ["text", "image_gen", "text"]],
        1024,
        4,
        torch.bfloat16,
        _cross_sample_causal_video_bidir_mask_mod,
        id="cross_2000_22000",
    ),
    pytest.param(
        1,
        16,
        8,
        128,
        [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],
        [
            [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
            [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],
        ],
        1024,
        4,
        torch.bfloat16,
        _video_stair_mask_mod,
        id="video_stair_6500",
    ),
    pytest.param(
        1,
        16,
        8,
        128,
        [[6500, 6500, 6500, 6500], [6500, 6500, 6500, 6500]],
        [
            [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
            [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],
        ],
        1024,
        4,
        torch.bfloat16,
        _stair_mask_mod,
        id="stair_6500",
    ),
]


FULL_MASK_MODALITIES = ("image_gen", "image_vae")


SEED = 0


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


def _build_video_indicators(device, frame_lens):
    segment_ids, doc_start, video_ids, frame_ids, modality = ([], [], [], [], [])
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
        sample_start += sum((sum(fl) for fl in sample_videos))
    return {
        "segment_ids": torch.cat(segment_ids).to(device),
        "doc_start": torch.cat(doc_start).to(device),
        "video_ids": torch.cat(video_ids).to(device),
        "frame_ids": torch.cat(frame_ids).to(device),
        "modality": torch.cat(modality).to(device),
    }


def build_problem(
    batch_size,
    q_head,
    kv_head,
    head_dim,
    data_lens,
    data_types,
    sliding_windows,
    global_windows,
    dtype,
    mask_func,
    device,
):
    torch.manual_seed(SEED)
    num_q_heads = q_head
    num_kv_heads = kv_head
    sample_lens = [sum(s) for s in data_lens]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int32), torch.tensor(sample_lens, device=device)
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)
    q = torch.rand(batch_size, num_q_heads, total_s, head_dim, device=device, dtype=dtype)
    k = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)
    v = torch.rand(batch_size, num_kv_heads, total_s, head_dim, device=device, dtype=dtype)
    if mask_func in [_video_stair_mask_mod, _stair_mask_mod]:
        meta = _build_video_indicators(device, data_types)
        return {
            "q": q,
            "k": k,
            "v": v,
            "segment_ids": meta["segment_ids"],
            "doc_start": meta["doc_start"],
            "video_ids": meta["video_ids"],
            "frame_ids": meta["frame_ids"],
            "modality": meta["modality"],
            "cu_seqlens": cu_seqlens,
            "total_s": total_s,
            "sliding_window": sliding_windows,
            "global_window": global_windows,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
    else:
        modality = _build_modality_indicators(
            device=device, data_length=data_lens, data_input_type=data_types, image_modalities=FULL_MASK_MODALITIES
        )
        return {
            "q": q,
            "k": k,
            "v": v,
            "segment_ids": segment_ids.long(),
            "modality": modality,
            "doc_start": doc_start,
            "cu_seqlens": cu_seqlens,
            "total_s": total_s,
            "sliding_window": sliding_windows,
            "global_window": global_windows,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }


def _require_flex(backend):
    impl, _, device = backend
    if device != "npu" or impl not in (None, "triton", "torch_reference"):
        pytest.skip("The migrated FlexAttention providers are NPU Triton and torch_reference")
    return impl, device


def _business_inputs(
    backend,
    batch_size,
    q_head,
    kv_head,
    head_dim,
    data_lens,
    data_types,
    sliding_windows,
    global_windows,
    dtype,
    mask_func,
):
    impl, device = _require_flex(backend)
    problem = build_problem(
        batch_size,
        q_head,
        kv_head,
        head_dim,
        data_lens,
        data_types,
        sliding_windows,
        global_windows,
        dtype,
        mask_func,
        device,
    )
    mask = create_flex_block_mask(
        mask_func(problem), B=1, H=1, Q_LEN=problem["total_s"], KV_LEN=problem["total_s"], device=device, BLOCK_SIZE=128
    )
    inputs = tuple(problem[name].requires_grad_(True) for name in ("q", "k", "v"))
    return impl, inputs, mask


@pytest.mark.api("functions.flex_attention")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "batch_size,q_head,kv_head,head_dim,data_lens,data_types,sliding_windows,global_windows,dtype,mask_func",
    BUSINESS_FLEX_CASES,
)
def test_business(
    accuracy_backend,
    batch_size,
    q_head,
    kv_head,
    head_dim,
    data_lens,
    data_types,
    sliding_windows,
    global_windows,
    dtype,
    mask_func,
):
    impl, inputs, mask = _business_inputs(
        accuracy_backend,
        batch_size,
        q_head,
        kv_head,
        head_dim,
        data_lens,
        data_types,
        sliding_windows,
        global_windows,
        dtype,
        mask_func,
    )
    reference_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    actual = F.flex_attention(*inputs, block_mask=mask, implementation=impl)
    expected = F.flex_attention(*reference_inputs, block_mask=mask, implementation="torch_reference")
    assert_close(actual, expected, dtype, rtol=5e-3, atol=5e-3)
    length = inputs[0].shape[2]
    seed_scale = torch.tensor(length, dtype=dtype).item() / actual.numel()
    upstream = torch.full_like(actual, seed_scale)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)
    for a, e in zip(actual_grads, expected_grads):
        assert_close(a, e, dtype, rtol=5e-3, atol=5e-3)


@pytest.mark.api("functions.flex_attention")
@pytest.mark.bitwise
@pytest.mark.parametrize(
    "batch_size,q_head,kv_head,head_dim,data_lens,data_types,sliding_windows,global_windows,dtype,mask_func",
    BUSINESS_FLEX_CASES,
)
def test_business_bitwise(
    accuracy_backend,
    batch_size,
    q_head,
    kv_head,
    head_dim,
    data_lens,
    data_types,
    sliding_windows,
    global_windows,
    dtype,
    mask_func,
):
    impl, inputs, mask = _business_inputs(
        accuracy_backend,
        batch_size,
        q_head,
        kv_head,
        head_dim,
        data_lens,
        data_types,
        sliding_windows,
        global_windows,
        dtype,
        mask_func,
    )
    assert_repeatable(partial(F.flex_attention, block_mask=mask, implementation=impl), inputs)
