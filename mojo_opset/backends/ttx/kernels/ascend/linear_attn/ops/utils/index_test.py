import torch
import torch_npu
from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.utils.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_chunk_indices_and_offsets,
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


def test_helper(cu_seqlens: torch.LongTensor, chunk_size: int):
    host_cu_seqlens = cu_seqlens.cpu()
    org_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    org_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)

    new_indices, new_offsets = prepare_chunk_indices_and_offsets(cu_seqlens, host_cu_seqlens, chunk_size)

    get_diff("indices", org_indices, new_indices)
    get_diff("offsets", org_offsets, new_offsets)
    print("org_indices.shape", org_indices.shape, "new_indices.shape", new_indices.shape)
    # print("org_offsets.shape", org_offsets.shape)
    print("org_offsets.shape", org_offsets.shape, "new_offsets.shape", new_offsets.shape)


cu_seqlens = (
    torch.Tensor(
        [
            0,
            819,
            1028,
            2890,
            12890,
            29082,
            32781,
            32789,
            39789,
            49789,
            59789,
            60000,
            60911,
            61911,
            62911,
            63911,
            64911,
            66911,
        ]
    )
    .npu()
    .int()
)
chunk_size = 64

test_helper(cu_seqlens, chunk_size)
