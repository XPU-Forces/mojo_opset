from typing import Tuple

import torch

from mojo_opset.modeling import torch_qwen3_dense
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def run_single_pass(config, device: str, dtype: torch.dtype, model_data: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    decoder_layer, rotary_emb, prefill_data, decode_data, batch_size = model_data

    hidden_states_prefill, attention_mask_prefill, position_ids_prefill = prefill_data
    hidden_states_decode, position_ids_decode_template = decode_data

    past_key_values = torch_qwen3_dense.PagedDummyCache(
        config=config,
        batch_size=batch_size,
        device=device,
        block_size=128,
    )

    position_embeddings_prefill = rotary_emb(hidden_states_prefill, position_ids_prefill)
    with torch.no_grad():
        output_prefill = decoder_layer(
            hidden_states=hidden_states_prefill,
            attention_mask=attention_mask_prefill,
            past_key_values=past_key_values,
            use_cache=True,
            position_ids=position_ids_prefill,
            position_embeddings=position_embeddings_prefill,
        )

    past_lens = past_key_values.get_seq_length()

    position_ids_decode = past_lens.unsqueeze(-1)

    position_embeddings_decode = rotary_emb(hidden_states_decode, position_ids_decode)
    with torch.no_grad():
        output_decode = decoder_layer(
            hidden_states=hidden_states_decode,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            position_ids=position_ids_decode,
            position_embeddings=position_embeddings_decode,
        )

    return output_prefill[0], output_decode[0]


def test_qwen3_dense_patch():
    device = "npu"

    dtype = torch.bfloat16
    torch.manual_seed(42)

    config = torch_qwen3_dense.Qwen3Config()
    config.num_key_value_heads = 2

    batch_size, prefill_len, decode_len = 8, 128, 1
    prefill_data = (
        torch.randn(batch_size, prefill_len, config.hidden_size, device=device, dtype=dtype),
        None,
        torch.arange(0, prefill_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1),
    )
    decode_data = (
        torch.randn(batch_size, decode_len, config.hidden_size, device=device, dtype=dtype),
        None,
    )

    native_decoder_layer = torch_qwen3_dense.Qwen3DecoderLayer(config, 0).to(device).to(dtype).eval()
    native_rotary_emb = torch_qwen3_dense.Qwen3RotaryEmbedding(config, device=device)

    native_model_data = (native_decoder_layer, native_rotary_emb, prefill_data, decode_data, batch_size)

    original_rmsnorm_class = torch_qwen3_dense.Qwen3RMSNorm
    original_mlp_class = torch_qwen3_dense.Qwen3MLP
    original_apply_rotary_pos_emb = torch_qwen3_dense.apply_rotary_pos_emb
    original_attn_prefill = torch_qwen3_dense.paged_attention_prefill
    original_attn_decode = torch_qwen3_dense.paged_attention_decode

    native_prefill_out, native_decode_out = run_single_pass(config, device, dtype, native_model_data)

    from mojo_opset.utils.patching import apply_mojo_to_qwen3

    apply_mojo_to_qwen3()

    patched_decoder_layer = torch_qwen3_dense.Qwen3DecoderLayer(config, 0).to(device).to(dtype).eval()
    patched_decoder_layer.load_state_dict(native_decoder_layer.state_dict())
    patched_rotary_emb = torch_qwen3_dense.Qwen3RotaryEmbedding(config, device=device)

    patched_model_data = (patched_decoder_layer, patched_rotary_emb, prefill_data, decode_data, batch_size)

    patched_rmsnorm_class = torch_qwen3_dense.Qwen3RMSNorm
    patched_mlp_class = torch_qwen3_dense.Qwen3MLP
    patched_apply_rotary_pos_emb = torch_qwen3_dense.apply_rotary_pos_emb
    patched_attn_prefill = torch_qwen3_dense.paged_attention_prefill
    patched_attn_decode = torch_qwen3_dense.paged_attention_decode

    patched_prefill_out, patched_decode_out = run_single_pass(config, device, dtype, patched_model_data)

    prefill_match = torch.allclose(native_prefill_out, patched_prefill_out, atol=1e-2, rtol=1e-2)
    decode_match = torch.allclose(native_decode_out, patched_decode_out, atol=1e-2, rtol=1e-2)

    if not prefill_match:
        logger.warning("Prefill outputs differ!")
        logger.warning("Max diff:", (native_prefill_out - patched_prefill_out).abs().max())
    if not decode_match:
        logger.warning("Decode outputs differ!")
        logger.warning("Max diff:", (native_decode_out - patched_decode_out).abs().max())

    assert prefill_match, "Paged prefill outputs do not match post-patching!"
    assert decode_match, "Paged decode outputs do not match post-patching!"

    assert patched_rmsnorm_class is not original_rmsnorm_class, "Qwen3RMSNorm class not patched!"
    assert patched_mlp_class is not original_mlp_class, "Qwen3MLP class not patched!"
    assert patched_apply_rotary_pos_emb is not original_apply_rotary_pos_emb, "apply_rotary_pos_emb func not patched!"
    assert patched_attn_prefill is not original_attn_prefill, "paged_attention_prefill func not patched!"
    assert patched_attn_decode is not original_attn_decode, "paged_attention_decode func not patched!"
