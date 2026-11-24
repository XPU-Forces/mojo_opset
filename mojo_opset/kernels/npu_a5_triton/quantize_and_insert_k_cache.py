"""UE8M0 (MXFP8) K quantization + paged-cache insertion for DeepSeek-V4 on A5.

Ported from vLLM ``vllm/models/deepseek_v41/common/ops/cache_utils.py``
(commit ``5be4266f5c7c10c69e4be5f3ec01d6a1d8bae9a1``), A5 record layouts only.

Two paged-cache records are supported; ``bytes_per_token`` selects one:

* V4   (584 B/token): dims ``[0, 448)`` as fp8 e4m3 with 7 UE8M0 scales of 64
  dims each, dims ``[448, 512)`` copied verbatim as bf16 (RoPE), plus one pad
  scale byte. A page stores its whole data region before its whole scale
  region.
* V4.1 (528 B/token): all 512 dims as fp8 e4m3 with 16 UE8M0 scales of 32 dims
  each. Each token's scales immediately follow its own data.

Both kernels run one token per program with 1-D loads and stores.

Implementation notes:

* ``tl.float8e4nv`` casts and ``bitcast`` to uint8 are supported on A5 and
  match ``torch.float8_e4m3fn`` bit-exactly.
* The power-of-two scale is built from its exponent bits rather than
  ``tl.exp2``, whose runtime approximation can shift exact fp8 rounding ties.
* Ascend FP8 is OCP E4M3, so ``use_fnuz`` is rejected by the host wrapper.
"""

import torch
import triton
import triton.language as tl

V4_BYTES_PER_TOKEN = 584
V41_BYTES_PER_TOKEN = 528
V41_QUANT_BLOCK = 32
V41_NUM_SCALES = 512 // V41_QUANT_BLOCK  # 16

V4_FP8_DIM = 448
V4_BF16_DIM = 64
V4_SCALE_DIM = 8
V4_QUANT_BLOCK = 64
V4_N_QUANT_BLOCKS = 8  # 7 real + 1 padding
V4_DATA_SIZE = V4_FP8_DIM + V4_BF16_DIM * 2  # 576 bytes

_FP8_MAX_OCP = 448.0

@triton.jit
def _quantize_and_insert_k_v4_kernel(
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    k_cache_ptr,  # [num_blocks, block_bytes] uint8
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,  # 576
    block_stride: tl.constexpr,  # bytes per page, padded
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 8
):
    # One token per program with 1-D loads/stores.
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    slot = tl.load(slot_mapping_ptr + pid)
    if slot == -1:
        return

    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    # int64: block_idx * block_stride can exceed 2^31 for large caches.
    block_off = block_idx.to(tl.int64) * block_stride
    data_off = block_off + pos_in_block * token_data_size
    scale_off = block_off + cache_block_size * token_data_size + pos_in_block * scale_dim

    d = tl.arange(0, input_dim)
    x = tl.load(k_ptr + pid.to(tl.int64) * input_dim + d).to(tl.float32)

    # ===== FP8 portion: 7 blocks of 64 dims, UE8M0 scale per block =====
    # Block 7 covers the bf16 tail; it is quantized only to keep the tile
    # rectangular and its scale is dropped.
    tiles = tl.reshape(x, (n_quant_blocks, quant_block))
    block_max = tl.maximum(tl.max(tl.abs(tiles), 1), 1e-4)
    # scale = 2^ceil(log2(amax / fp8_max)), rounded up. Build the power of two
    # from its exponent bits: runtime ``tl.exp2`` is approximate and perturbs
    # exact fp8 rounding ties. Scaling by the reciprocal (2^-e) is exact.
    exponent = tl.ceil(tl.log2(block_max / fp8_max))
    e_int = exponent.to(tl.int32)
    inv_scale = ((127 - e_int) << 23).to(tl.float32, bitcast=True)
    scaled = tl.clamp(
        tiles * tl.reshape(inv_scale, (n_quant_blocks, 1)), -fp8_max, fp8_max
    )
    fp8_u8 = tl.reshape(
        scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (input_dim,)
    )
    tl.store(k_cache_ptr + data_off + d, fp8_u8, mask=d < fp8_dim)

    # ===== BF16 tail: dims [448, 512), copied verbatim =====
    t = tl.arange(0, bf16_dim)
    tail = tl.load(k_ptr + pid.to(tl.int64) * input_dim + fp8_dim + t)
    # Cast the scalar base pointer once (a 2-D pointer cast is rejected by the
    # Ascend backend) and address the tail in bf16 elements.
    cache_bf16_ptr = k_cache_ptr.to(tl.pointer_type(tl.bfloat16))
    tl.store(cache_bf16_ptr + (data_off + fp8_dim) // 2 + t, tail)

    # ===== UE8M0 scales: 7 real bytes + one zero pad =====
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    s = tl.arange(0, scale_dim)
    encoded = tl.where(s < n_quant_blocks - 1, encoded, 0)
    tl.store(k_cache_ptr + scale_off + s, encoded)


@triton.jit
def _quantize_and_insert_k_v41_kernel(
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    k_cache_ptr,  # [num_blocks, block_bytes] uint8
    num_tokens,
    head_dim: tl.constexpr,  # 512, RoPE dims included
    scale_dim: tl.constexpr,  # 16
    quant_block: tl.constexpr,  # 32
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_max: tl.constexpr,
):
    # One token per program with 1-D loads/stores.
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride
    # Per-token data and scales are interleaved: 512 B data + 16 B scales.
    token_data_ptr = cache_block_ptr + pos_in_block * (head_dim + scale_dim)
    token_scale_ptr = token_data_ptr + head_dim

    d = tl.arange(0, head_dim)
    x = tl.load(k_ptr + pid.to(tl.int64) * head_dim + d).to(tl.float32)

    tiles = tl.reshape(x, (scale_dim, quant_block))
    # Floor the absmax so an all-zero tile still lands on a finite scale.
    amax = tl.maximum(tl.max(tl.abs(tiles), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / fp8_max)))
    # Exact power of two from the exponent bits (see the V4 kernel).
    e_int = exponent.to(tl.int32)
    inv_scale = ((127 - e_int) << 23).to(tl.float32, bitcast=True)
    scaled = tl.clamp(tiles * tl.reshape(inv_scale, (scale_dim, 1)), -fp8_max, fp8_max)
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        token_data_ptr + d,
        tl.reshape(fp8.to(tl.uint8, bitcast=True), (head_dim,)),
    )

    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
    tl.store(token_scale_ptr + tl.arange(0, scale_dim), encoded.to(tl.uint8))


def _quantize_and_insert_k_impl(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    use_fnuz: bool,
    bytes_per_token: int,
) -> None:
    if use_fnuz:
        raise ValueError("A5 FP8 is OCP E4M3; use_fnuz=True is not supported")

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    block_stride = k_cache.stride(0)

    if bytes_per_token == V41_BYTES_PER_TOKEN:
        _quantize_and_insert_k_v41_kernel[(num_tokens,)](
            k,
            slot_mapping,
            k_cache,
            num_tokens,
            head_dim=512,
            scale_dim=V41_NUM_SCALES,
            quant_block=V41_QUANT_BLOCK,
            cache_block_size=block_size,
            block_stride=block_stride,
            fp8_max=_FP8_MAX_OCP,
        )
        return

    _quantize_and_insert_k_v4_kernel[(num_tokens,)](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=512,
        fp8_dim=V4_FP8_DIM,
        bf16_dim=V4_BF16_DIM,
        scale_dim=V4_SCALE_DIM,
        quant_block=V4_QUANT_BLOCK,
        cache_block_size=block_size,
        token_data_size=V4_DATA_SIZE,
        block_stride=block_stride,
        fp8_max=_FP8_MAX_OCP,
        n_quant_blocks=V4_N_QUANT_BLOCKS,
    )


@torch.library.custom_op(
    "mojo_npu_triton_a5::quantize_and_insert_k_cache",
    mutates_args=("k_cache",),
)
def _quantize_and_insert_k_cache_leaf(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    use_fnuz: bool,
    bytes_per_token: int,
) -> None:
    _quantize_and_insert_k_impl(k, k_cache, slot_mapping, block_size, use_fnuz, bytes_per_token)


@_quantize_and_insert_k_cache_leaf.register_fake
def _quantize_and_insert_k_cache_fake(k, k_cache, slot_mapping, block_size, use_fnuz, bytes_per_token):
    return None


def quantize_and_insert_k_cache_fwd(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
    is_ue8m0: bool = True,
    use_fnuz: bool = False,
    bytes_per_token: int = V4_BYTES_PER_TOKEN,
) -> torch.Tensor:
    assert is_ue8m0, "Only UE8M0 quantization is supported"
    _quantize_and_insert_k_cache_leaf(
        k, k_cache, slot_mapping, block_size, use_fnuz, bytes_per_token
    )
    return k_cache
