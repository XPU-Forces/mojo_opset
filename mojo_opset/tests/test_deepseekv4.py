import json

import pytest
import torch
import torch.nn.functional as F

from safetensors.torch import save_file

from mojo_opset import MojoGemm
from mojo_opset import MojoQuantGemm
from mojo_opset.modeling.deepseekv4 import DeepseekV4Config
from mojo_opset.modeling.deepseekv4 import DeepseekV4ForCausalLM
from mojo_opset.modeling.deepseekv4.loader import _build_name_mapping
from mojo_opset.modeling.deepseekv4.loader import _configure_modelslim_w8a8_modules
from mojo_opset.modeling.deepseekv4.loader import _load_expert_weights
from mojo_opset.modeling.deepseekv4.loader import _load_mapped_weights_from_dir
from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import compress_kv
from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import quant_lightning_indexer
from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import sparse_attn_shared_kv


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        moe_intermediate_size=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        head_dim=4,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        o_lora_rank=4,
        o_groups=1,
        sliding_window=4,
        compress_ratios=[0],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_hash_layers=1,
        max_position_embeddings=32,
        pa_max_length=16,
    )


def test_modelslim_attention_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJO_BACKEND", "torch")
    with torch.device("cpu"):
        model = DeepseekV4ForCausalLM(_tiny_config())
    attention = model.model.layers[0].self_attn
    assert isinstance(attention.wq_a, MojoGemm)
    assert isinstance(attention.wkv, MojoGemm)
    assert isinstance(attention.wo_b, MojoQuantGemm)

    description = {
        "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC",
        "layers.0.attn.wkv.weight": "W8A8_DYNAMIC",
        "layers.0.attn.wo_a.weight": "FLOAT",
        "layers.0.attn.wo_b.weight": "FLOAT",
    }
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")

    assert _configure_modelslim_w8a8_modules(model, str(tmp_path))
    assert isinstance(attention.wq_a, MojoQuantGemm)
    assert isinstance(attention.wkv, MojoQuantGemm)
    assert isinstance(attention.wo_b, MojoGemm)
    assert attention.wq_a.weight_scale.dtype == torch.float32


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_modelslim_loader_transposes_weight_and_rejects_asymmetric_offset(
    monkeypatch,
    tmp_path,
    offset,
):
    monkeypatch.setenv("MOJO_BACKEND", "torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = MojoQuantGemm(in_features=2, out_features=3)
            self.proj.weight_scale.data = self.proj.weight_scale.data.float()
            self.ep_size = 1
            self.ep_rank = 0
            self.config = type("Config", (), {"n_routed_experts": 0})()

    model = Model()
    checkpoint_weight = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int8)
    checkpoint_scale = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    shard = "quant_model_weights-00001-of-00001.safetensors"
    tensors = {
        "projection.weight": checkpoint_weight,
        "projection.weight_scale": checkpoint_scale,
        "projection.weight_offset": torch.full((3, 1), offset, dtype=torch.float32),
    }
    save_file(tensors, tmp_path / shard)
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )
    mapping = {
        "projection.weight": "proj.weight",
        "projection.weight_scale": "proj.weight_scale",
        "projection.weight_offset": "__validate_zero_offset__",
    }

    if offset:
        with pytest.raises(NotImplementedError, match="Asymmetric W8A8"):
            _load_mapped_weights_from_dir(model, str(tmp_path), mapping, [])
        return

    _load_mapped_weights_from_dir(model, str(tmp_path), mapping, [])
    assert torch.equal(model.proj.weight, checkpoint_weight.t())
    torch.testing.assert_close(model.proj.weight_scale.float(), checkpoint_scale.squeeze(1))


def test_rank_local_expert_loading(monkeypatch):
    monkeypatch.setenv("MOJO_BACKEND", "torch")
    with torch.device("cpu"):
        model = DeepseekV4ForCausalLM(_tiny_config(), ep_size=2, ep_rank=1)
    experts = model.model.layers[0].mlp.experts
    shared = model.model.layers[0].mlp.shared_experts

    tensors = {}
    prefixes = ["layers.0.ffn.experts.2", "layers.0.ffn.experts.3", "layers.0.ffn.shared_experts"]
    for value, prefix in enumerate(prefixes, start=3):
        tensors[f"{prefix}.w1.weight"] = torch.arange(32, dtype=torch.int8).reshape(4, 8) + value
        tensors[f"{prefix}.w3.weight"] = torch.arange(32, dtype=torch.int8).reshape(4, 8) + value + 4
        tensors[f"{prefix}.w2.weight"] = torch.arange(32, dtype=torch.int8).reshape(8, 4) + value + 8
        tensors[f"{prefix}.w1.weight_scale"] = torch.full((4, 1), value / 10)
        tensors[f"{prefix}.w3.weight_scale"] = torch.full((4, 1), value / 10 + 0.1)
        tensors[f"{prefix}.w2.weight_scale"] = torch.full((8, 1), value / 10 + 0.2)

    _load_expert_weights(model, tensors)

    for module, local_eid, prefix in ((experts, 0, prefixes[0]), (experts, 1, prefixes[1]), (shared, 0, prefixes[2])):
        assert torch.equal(module.up_proj.weight[local_eid, :, :4], tensors[f"{prefix}.w1.weight"].t())
        assert torch.equal(module.up_proj.weight[local_eid, :, 4:], tensors[f"{prefix}.w3.weight"].t())
        assert torch.equal(module.down_proj.weight[local_eid], tensors[f"{prefix}.w2.weight"].t())
        assert torch.equal(
            module.dequant_swiglu_quant.weight_scale[local_eid, :4], tensors[f"{prefix}.w1.weight_scale"].flatten()
        )
        assert torch.equal(module.down_proj.weight_scale[local_eid], tensors[f"{prefix}.w2.weight_scale"].flatten())


def test_modelslim_loader_restores_operator_owned_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJO_BACKEND", "torch")
    config = _tiny_config()
    config.num_hash_layers = 0
    config.o_groups = 2
    with torch.device("cpu"):
        model = DeepseekV4ForCausalLM(config)
    layer = model.model.layers[0]
    tensors = {
        "layers.0.attn.wo_a.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "layers.0.ffn.gate.bias": torch.randn(4),
    }
    for branch in ("hc_attn", "hc_ffn"):
        op = getattr(layer, branch)
        tensors[f"layers.0.{branch}_fn"] = torch.randn_like(op.weight)
        tensors[f"layers.0.{branch}_scale"] = torch.randn_like(op.scale)
        tensors[f"layers.0.{branch}_base"] = torch.randn_like(op.bias)
    shard = "model.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    mapping = {source: target for source, target in _build_name_mapping(model).items() if source in tensors}
    _load_mapped_weights_from_dir(model, str(tmp_path), mapping, [])

    expected_wo_a = tensors["layers.0.attn.wo_a.weight"].view(2, 4, 4).transpose(1, 2)
    assert torch.equal(layer.self_attn.wo_a.weight, expected_wo_a)
    assert torch.equal(layer.mlp.moe_gating_top_k.bias, tensors["layers.0.ffn.gate.bias"])
    for branch in ("hc_attn", "hc_ffn"):
        op = getattr(layer, branch)
        assert torch.equal(op.weight, tensors[f"layers.0.{branch}_fn"])
        assert torch.equal(op.scale, tensors[f"layers.0.{branch}_scale"])
        assert torch.equal(op.bias, tensors[f"layers.0.{branch}_base"])


@pytest.mark.parametrize("is_hash", [False, True])
def test_routes_match_checkpoint_gating(monkeypatch, is_hash):
    monkeypatch.setenv("MOJO_BACKEND", "torch")
    config = _tiny_config()
    config.num_hash_layers = int(is_hash)
    with torch.device("cpu"):
        moe = DeepseekV4ForCausalLM(config).model.layers[0].mlp
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0], [0.0, 3.0, 1.0, 2.0]])
    if is_hash:
        with pytest.raises(ValueError, match="input_ids is required"):
            moe._gate_topk(logits, input_ids=None)
        input_ids = torch.tensor([3, 5])
        routes = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
        moe.tid2eid.data[input_ids] = routes
    else:
        input_ids = None
        moe.moe_gating_top_k.bias.data.copy_(torch.tensor([0.0, 4.0, 0.0, 0.0]))
        routes = torch.tensor([[1, 0], [1, 3]], dtype=torch.int32)
    indices, weights = moe._gate_topk(logits, input_ids)
    scores = torch.sqrt(F.softplus(logits.float()))
    expected = torch.gather(scores, 1, routes.long())
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20)

    assert torch.equal(indices, routes)
    torch.testing.assert_close(weights, expected * moe.routed_scaling_factor)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"norm_topk_prob": False}, "norm_topk_prob=true"),
        ({"n_shared_experts": 0}, "n_shared_experts=1"),
        ({"scoring_func": "unsupported"}, "Unsupported scoring_func"),
        ({"topk_method": "unsupported"}, "topk_method='noaux_tc'"),
    ],
)
def test_rejects_unsupported_moe_contract(monkeypatch, override, message):
    monkeypatch.setenv("MOJO_BACKEND", "torch")
    config = _tiny_config()
    for name, value in override.items():
        setattr(config, name, value)

    with pytest.raises(NotImplementedError, match=message):
        DeepseekV4ForCausalLM(config)


def test_compress_kv_matches_independent_formula():
    torch.manual_seed(11)
    ratio = 2
    head_dim = 4
    rope_head_dim = 2
    x = torch.randn(4, 3)
    wkv = torch.randn(head_dim, 3)
    wgate = torch.randn(head_dim, 3)
    ape = torch.randn(ratio, head_dim)
    norm_weight = torch.randn(head_dim)

    kv = F.linear(x, wkv).reshape(2, ratio, head_dim)
    scores = F.linear(x, wgate).reshape(2, ratio, head_dim) + ape
    pooled = (kv * torch.softmax(scores, dim=1)).sum(dim=1)
    expected = pooled * torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + 1e-6) * norm_weight
    expected[:, -rope_head_dim:] = torch.stack((-expected[:, -1], expected[:, -2]), dim=-1)

    state_cache = torch.zeros(2, 8, 2 * head_dim)
    actual = compress_kv(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        torch.ones(2, rope_head_dim),
        torch.zeros(2, rope_head_dim),
        rope_head_dim,
        ratio,
        state_block_table=torch.ones(1, 1, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(state_cache) > 0


def test_sparse_attention_matches_independent_formula():
    torch.manual_seed(29)
    query = torch.randn(2, 2, 3)
    logical_original = torch.randn(5, 1, 3)
    logical_compressed = torch.randn(2, 1, 3)
    original_cache = torch.zeros(3, 4, 1, 3)
    compressed_cache = torch.zeros(2, 4, 1, 3)
    original_table = torch.tensor([[2, 1]], dtype=torch.int32)
    compressed_table = torch.tensor([[1]], dtype=torch.int32)
    for position, row in enumerate(logical_original):
        original_cache[original_table[0, position // 4], position % 4] = row
    for position, row in enumerate(logical_compressed):
        compressed_cache[compressed_table[0, position // 4], position % 4] = row

    sinks = torch.tensor([0.2, -0.4])
    compressed_indices = torch.tensor([[[1, -1]], [[0, 1]]], dtype=torch.int32)
    scale = 0.7
    actual = sparse_attn_shared_kv(
        query,
        ori_kv=original_cache,
        cmp_kv=compressed_cache,
        cmp_sparse_indices=compressed_indices,
        ori_block_table=original_table,
        cmp_block_table=compressed_table,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seqused_kv=torch.tensor([5], dtype=torch.int32),
        sinks=sinks,
        softmax_scale=scale,
        cmp_ratio=2,
        ori_win_left=2,
    )

    expected = torch.empty_like(actual)
    for query_index, absolute_position in enumerate((3, 4)):
        original = logical_original[max(0, absolute_position - 2) : absolute_position + 1, 0]
        selected = compressed_indices[query_index, 0]
        selected = selected[selected >= 0].long()
        candidates = torch.cat((original, logical_compressed[selected, 0]), dim=0)
        scores = torch.matmul(query[query_index].float(), candidates.float().T) * scale
        probabilities = torch.softmax(torch.cat((scores, sinks[:, None]), dim=-1), dim=-1)[..., :-1]
        expected[query_index] = torch.matmul(probabilities, candidates.float())

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_quant_lightning_indexer_matches_independent_formula():
    query = torch.tensor([[[2, 0], [0, 1]], [[1, 1], [2, -1]]], dtype=torch.int8)
    query_scale = torch.tensor([[0.5, 1.0], [1.0, 0.25]])
    weights = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    logical_key = torch.tensor([[[1, 0]], [[0, 2]]], dtype=torch.int8)
    logical_scale = torch.tensor([[0.5], [1.5]])
    key = torch.zeros(3, 2, 1, 2, dtype=torch.int8)
    key_scale = torch.ones(3, 2, 1)
    block_table = torch.tensor([[2]], dtype=torch.int32)
    key[2, :2] = logical_key
    key_scale[2, :2] = logical_scale

    indices = quant_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([8], dtype=torch.int32),
        block_table=block_table,
        sparse_count=3,
        cmp_ratio=4,
    )

    expected_indices = torch.tensor([[[0, -1, -1]], [[1, 0, -1]]], dtype=torch.int32)
    for token_index, valid_key_count in enumerate((1, 2)):
        dequantized_query = query[token_index].float() * query_scale[token_index, :, None]
        dequantized_key = logical_key[:valid_key_count, 0].float() * logical_scale[:valid_key_count]
        scores = torch.relu(dequantized_query @ dequantized_key.T)
        scores = (scores * weights[token_index, :, None]).sum(dim=0)
        order = torch.argsort(scores, descending=True, stable=True)
        assert torch.equal(indices[token_index, 0, :valid_key_count], order.to(torch.int32))

    assert torch.equal(indices, expected_indices)
