from typing import Tuple

import torch

from mojo_opset.modeling.qwen3 import torch_qwen3_dense
from mojo_opset.modeling.qwen3 import mojo_qwen3_dense
from mojo_opset.modeling.qwen3.mojo_qwen3_dense import Qwen3AttentionBackend
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def run_single_pass(config, device: str, dtype: torch.dtype, model_data: tuple, cache_class) -> Tuple[torch.Tensor, torch.Tensor]:
    decoder_layer, rotary_emb, prefill_data, decode_data, batch_size = model_data

    hidden_states_prefill, attention_mask_prefill, position_ids_prefill = prefill_data
    hidden_states_decode, position_ids_decode_template = decode_data

    past_key_values = cache_class(
        config=config,
        batch_size=batch_size,
        device=device,
        block_size=128,
    )
    Qwen3AttentionBackend.init_attention_info(
        num_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        block_size=128,
    )
    Qwen3AttentionBackend.prepare_prefill_attn(
        hidden_states_prefill,
        past_key_values,
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


def _transfer_weights(torch_layer, mojo_layer):
    """Transfer weights from torch_qwen3_dense decoder layer to mojo_qwen3_dense decoder layer.

    Handles structural differences:
    - torch has shared `gamma` parameters; mojo has separate norm weights
    - torch may have attention bias; mojo does not
    """
    mojo_layer.self_attn.q_proj.weight.data.copy_(torch_layer.self_attn.q_proj.weight.data)
    mojo_layer.self_attn.k_proj.weight.data.copy_(torch_layer.self_attn.k_proj.weight.data)
    mojo_layer.self_attn.v_proj.weight.data.copy_(torch_layer.self_attn.v_proj.weight.data)
    mojo_layer.self_attn.o_proj.weight.data.copy_(torch_layer.self_attn.o_proj.weight.data)

    mojo_layer.mlp.gate_proj.weight.data.copy_(torch_layer.mlp.gate_proj.weight.data)
    mojo_layer.mlp.up_proj.weight.data.copy_(torch_layer.mlp.up_proj.weight.data)
    mojo_layer.mlp.down_proj.weight.data.copy_(torch_layer.mlp.down_proj.weight.data)

    mojo_layer.input_layernorm.weight.data.copy_(torch_layer.gamma.data)
    mojo_layer.post_attention_layernorm.weight.data.copy_(torch_layer.gamma.data)

    mojo_layer.self_attn.q_norm.weight.data.copy_(torch_layer.self_attn.gamma.data)
    mojo_layer.self_attn.k_norm.weight.data.copy_(torch_layer.self_attn.gamma.data)


def test_qwen3_dense():
    device = "npu"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    config = torch_qwen3_dense.Qwen3Config()
    config.num_key_value_heads = 2
    config.attention_bias = False
    config.q_norm = True
    config.k_norm = True

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

    mojo_decoder = mojo_qwen3_dense.Qwen3DecoderLayer(config, 0).to(device).to(dtype).eval()
    mojo_rotary = mojo_qwen3_dense.Qwen3RotaryEmbedding(config, device=device)

    _transfer_weights(native_decoder_layer, mojo_decoder)

    native_model_data = (native_decoder_layer, native_rotary_emb, prefill_data, decode_data, batch_size)
    mojo_model_data = (mojo_decoder, mojo_rotary, prefill_data, decode_data, batch_size)

    native_prefill_out, native_decode_out = run_single_pass(
        config, device, dtype, native_model_data, torch_qwen3_dense.PagedDummyCache
    )
    mojo_prefill_out, mojo_decode_out = run_single_pass(
        config, device, dtype, mojo_model_data, mojo_qwen3_dense.PagedDummyCache
    )

    prefill_match = torch.allclose(native_prefill_out, mojo_prefill_out, atol=1e-2, rtol=1e-2)
    decode_match = torch.allclose(native_decode_out, mojo_decode_out, atol=1e-2, rtol=1e-2)

    if not prefill_match:
        logger.warning("Prefill outputs differ!")
        logger.warning("Max diff: %s", (native_prefill_out - mojo_prefill_out).abs().max())
    if not decode_match:
        logger.warning("Decode outputs differ!")
        logger.warning("Max diff: %s", (native_decode_out - mojo_decode_out).abs().max())

    assert prefill_match, "Prefill outputs do not match between torch and mojo implementations!"
    assert decode_match, "Decode outputs do not match between torch and mojo implementations!"
