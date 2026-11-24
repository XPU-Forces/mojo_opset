import torch

from mojo_opset.kernels._nf4 import get_nf4_codebook


def n_gram_impl_torch(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offset: torch.Tensor,
    n_grams: torch.Tensor,
    ori_vocab_size: int,
):
    """_summary_

    Args:
        input_ids (torch.Tensor): _description_
        oe_history_inputs (torch.Tensor): _description_
        oe_vocab_sizes (torch.Tensor): _description_
        oe_vocab_offset (torch.Tensor): _description_
        n_grams (torch.Tensor): _description_
        ori_vocab_size (int): _description_

    Returns:
        _type_: _description_
    """
    n_gram_ids = []
    complete_input_ids = torch.cat([oe_history_inputs, input_ids], dim=-1)
    for gram_idx, gram in map(lambda val: (val[0], val[1].item()), enumerate(n_grams)):
        oe_carry = ori_vocab_size
        n_gram_id = input_ids

        # TODO(liuyuan): make it recomputation free.
        for i in range(1, gram):
            prev_input_ids = complete_input_ids[..., -i - n_gram_id.size(-1) : -i]

            # NOTE(liuyuan): the fowllowing modulo operators are designed for big decimals.
            n_gram_id = (n_gram_id + prev_input_ids * oe_carry) % oe_vocab_sizes[gram_idx]
            oe_carry = oe_carry * ori_vocab_size % oe_vocab_sizes[gram_idx]

        n_gram_id = n_gram_id + oe_vocab_offset[gram_idx]
        n_gram_ids.append(n_gram_id)
    n_gram_ids_tensor = torch.stack(n_gram_ids, dim=-1)

    return n_gram_ids_tensor


def unpack_nf4_int8_to_uint4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f"`packed` must be 2D, got shape={tuple(packed.shape)}")
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"`packed` must have dtype torch.int8 or torch.uint8, got {packed.dtype}.")

    q_u8 = packed.to(torch.uint8)
    low = q_u8 & 0x0F
    high = (q_u8 >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def dequantize_nf4_rows(
    nf4_qweight: torch.Tensor,
    nf4_scale: torch.Tensor,
    nf4_mean: torch.Tensor,
    *,
    group_size: int,
    codebook: torch.Tensor = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if nf4_qweight.ndim != 2:
        raise ValueError(f"`nf4_qweight` must be 2D, got shape={tuple(nf4_qweight.shape)}.")
    if nf4_scale.ndim != 2 or nf4_mean.ndim != 2:
        raise ValueError(
            "`nf4_scale` and `nf4_mean` must both be 2D, "
            f"got scale={tuple(nf4_scale.shape)}, mean={tuple(nf4_mean.shape)}."
        )
    if nf4_scale.shape != nf4_mean.shape:
        raise ValueError(
            "`nf4_scale` and `nf4_mean` must have the same shape, "
            f"got scale={tuple(nf4_scale.shape)}, mean={tuple(nf4_mean.shape)}."
        )
    if group_size <= 0:
        raise ValueError(f"`group_size` must be > 0, got {group_size}.")

    num_rows = nf4_scale.shape[0]
    num_groups = nf4_scale.shape[1]
    embedding_dim = num_groups * group_size

    if nf4_qweight.shape[0] != num_rows:
        raise ValueError(
            "`nf4_qweight` row count must match scale/mean, "
            f"got qweight={tuple(nf4_qweight.shape)}, scale={tuple(nf4_scale.shape)}."
        )
    if nf4_qweight.shape[1] * 2 != embedding_dim:
        raise ValueError(
            "`nf4_qweight` column count must be embedding_dim / 2, "
            f"got qweight={tuple(nf4_qweight.shape)}, embedding_dim={embedding_dim}."
        )

    if codebook is None:
        codebook = get_nf4_codebook(device=nf4_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=nf4_qweight.device, dtype=torch.float16)

    unpack_input = nf4_qweight.cpu() if nf4_qweight.device.type == "npu" else nf4_qweight
    q_idx = (
        unpack_nf4_int8_to_uint4(unpack_input)
        .to(device=nf4_qweight.device)
        .reshape(
            num_rows,
            num_groups,
            group_size,
        )
        .to(torch.long)
    )
    values = (
        codebook.index_select(0, q_idx.reshape(-1))
        .reshape(
            num_rows,
            num_groups,
            group_size,
        )
        .to(torch.float32)
    )

    scale = nf4_scale.to(torch.float32).reshape(num_rows, num_groups, 1)
    mean = nf4_mean.to(torch.float32).reshape(num_rows, num_groups, 1)
    return (values * scale + mean).reshape(num_rows, embedding_dim).to(output_dtype)


def n_gram_decode_fwd(input_ids, history, sizes, offsets, grams, vocab_size):
    return n_gram_impl_torch(input_ids, history, sizes, offsets, grams, vocab_size)


def n_gram_prefill_fwd(input_ids, q_lens, history, sizes, offsets, grams, vocab_size):
    outputs = []
    start = 0
    for row, length in enumerate(q_lens.cpu().tolist()):
        outputs.append(
            n_gram_impl_torch(input_ids[start : start + length], history[row], sizes, offsets, grams, vocab_size)
        )
        start += length
    return torch.cat(outputs, dim=0)


def embedding_nf4_dequant_fwd(input_ids, qweight, scale, mean, group_size, codebook, vocab_start_id, output_dtype):
    dim = scale.shape[1] * group_size
    flat = input_ids.contiguous().view(-1)
    output = torch.zeros(flat.numel(), dim, device=input_ids.device, dtype=output_dtype)
    local = flat.long() - vocab_start_id
    valid = (local >= 0) & (local < qweight.shape[0])
    if valid.any():
        indices = local[valid]
        output[valid] = dequantize_nf4_rows(
            qweight.index_select(0, indices),
            scale.index_select(0, indices),
            mean.index_select(0, indices),
            group_size=group_size,
            codebook=codebook,
            output_dtype=output_dtype,
        )
    return output.view(*input_ids.shape, dim)


def over_encoding_decode_fwd(
    input_ids,
    history,
    sizes,
    offsets,
    grams,
    qweight,
    scale,
    mean,
    ori_vocab_size,
    group_size,
    codebook,
    vocab_start_id,
    output_dtype,
):
    ids = n_gram_decode_fwd(input_ids, history, sizes, offsets, grams, ori_vocab_size)
    return embedding_nf4_dequant_fwd(ids, qweight, scale, mean, group_size, codebook, vocab_start_id, output_dtype)
