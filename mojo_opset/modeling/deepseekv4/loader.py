import json
import os

import torch

from mojo_opset import MojoBatchGemm
from mojo_opset import MojoGemm
from mojo_opset import MojoQuantGemm


def _align_weight(weight, target):
    if weight.shape != target.shape:
        if weight.dim() == 2 and target.dim() == 1 and weight.shape[1] == 1:
            weight = weight.squeeze(-1)
        elif weight.dim() == 1 and target.dim() == 2 and target.shape[1] == 1:
            weight = weight.unsqueeze(-1)
    if weight.dtype != target.dtype:
        if target.dtype == torch.bfloat16 and weight.dtype == torch.float32:
            weight = weight.to(torch.bfloat16)
        elif target.dtype == torch.float32 and weight.dtype == torch.bfloat16:
            weight = weight.to(torch.float32)
        elif target.dtype == torch.int32 and weight.dtype == torch.int64:
            weight = weight.to(torch.int32)
    return weight


def _align_projection_weight(weight, target, module):
    if isinstance(module, MojoBatchGemm):
        weight = weight.view(module.num_groups, module.out_features, module.in_features).transpose(1, 2).contiguous()
    elif isinstance(module, MojoQuantGemm) and weight.dim() == 2 and target.dim() == 2:
        if weight.shape != target.shape and weight.t().shape == target.shape:
            weight = weight.t().contiguous()
    return _align_weight(weight, target)


def _configure_modelslim_w8a8_modules(model, weight_dir: str) -> bool:
    """Align attention projections with the public ModelSlim checkpoint."""
    description_path = os.path.join(weight_dir, "quant_model_description.json")
    if not os.path.isfile(description_path):
        return False

    with open(description_path, encoding="utf-8") as file:
        description = json.load(file)
    prefix = "layers.0"
    actual = {
        name: description.get(f"{prefix}.attn.{name}.weight") for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
    }
    expected = {
        "wq_a": "W8A8_DYNAMIC",
        "wq_b": "W8A8_DYNAMIC",
        "wkv": "W8A8_DYNAMIC",
        "wo_a": "FLOAT",
        "wo_b": "FLOAT",
    }
    if actual != expected:
        raise NotImplementedError(
            f"Unsupported quant_model_description attention layout: expected {expected}, got {actual}."
        )

    for layer in model.model.layers:
        module = layer.self_attn
        if not isinstance(module.wq_a, MojoQuantGemm):
            old = module.wq_a
            module.wq_a = MojoQuantGemm(
                in_features=old.in_features,
                out_features=old.out_features,
                device=old.weight.device,
            ).train(old.training)
        if not isinstance(module.wkv, MojoQuantGemm):
            old = module.wkv
            module.wkv = MojoQuantGemm(
                in_features=old.in_features,
                out_features=old.out_features,
                device=old.weight.device,
            ).train(old.training)
        if isinstance(module.wo_b, MojoQuantGemm):
            old = module.wo_b
            module.wo_b = MojoGemm(
                in_features=old.in_features,
                out_features=old.out_features,
                bias=False,
                dtype=torch.bfloat16,
                device=old.weight.device,
            ).train(old.training)

    for module in model.modules():
        scale = getattr(module, "weight_scale", None)
        if isinstance(scale, torch.Tensor) and scale.dtype != torch.float32:
            scale.data = scale.data.float()
    return True


def _checkpoint_index_path(weight_dir: str) -> str:
    candidates = (
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    )
    for filename in candidates:
        path = os.path.join(weight_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No supported safetensors index found in {weight_dir!r}; expected one of {candidates}.")


def _init_default_weights(model):
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "RMSNorm" in cls_name:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
        elif "DynamicQuant" in cls_name:
            if hasattr(module, "inv_smooth_scale") and module.inv_smooth_scale is not None:
                module.inv_smooth_scale.data.fill_(1.0)


def _load_mapped_weights_from_dir(model, weight_dir, name_mapping, expert_key_prefixes):
    from safetensors.torch import load_file

    index_path = _checkpoint_index_path(weight_dir)
    with open(index_path) as file:
        index = json.load(file)
    weight_map = index["weight_map"]

    needed_checkpoint_keys = set(name_mapping.keys())
    ep_size = model.ep_size
    ep_rank = model.ep_rank
    experts_per_rank = model.config.n_routed_experts // ep_size
    ep_start = ep_rank * experts_per_rank

    for prefix in expert_key_prefixes:
        expert_prefixes = [f"{prefix}.ffn.experts.{eid}" for eid in range(ep_start, ep_start + experts_per_rank)]
        expert_prefixes.append(f"{prefix}.ffn.shared_experts")
        for expert_prefix in expert_prefixes:
            for weight_name in (
                "w1.weight",
                "w1.scale",
                "w1.weight_scale",
                "w1.weight_offset",
                "w2.weight",
                "w2.scale",
                "w2.weight_scale",
                "w2.weight_offset",
                "w3.weight",
                "w3.scale",
                "w3.weight_scale",
                "w3.weight_offset",
            ):
                needed_checkpoint_keys.add(f"{expert_prefix}.{weight_name}")

    file_to_keys = {}
    for checkpoint_key in needed_checkpoint_keys:
        if checkpoint_key in weight_map:
            shard = weight_map[checkpoint_key]
            file_to_keys.setdefault(shard, []).append(checkpoint_key)

    params_dict = dict(model.named_parameters())
    buffers_dict = dict(model.named_buffers())
    modules_dict = dict(model.named_modules())

    sources_by_target = {}
    for checkpoint_name, model_name in name_mapping.items():
        if model_name == "__validate_zero_offset__":
            continue
        sources_by_target.setdefault(model_name, []).append(checkpoint_name)
    missing_source_targets = sorted(
        model_name
        for model_name, checkpoint_names in sources_by_target.items()
        if not any(name in weight_map for name in checkpoint_names)
    )
    if missing_source_targets:
        preview = ", ".join(missing_source_targets[:8])
        raise KeyError(
            f"Checkpoint is missing source tensors for {len(missing_source_targets)} "
            f"model tensors: {preview}"
        )

    expert_weights = {}
    loaded_targets = set()
    for shard in sorted(file_to_keys):
        data = load_file(os.path.join(weight_dir, shard))
        for checkpoint_key in file_to_keys[shard]:
            if checkpoint_key not in data:
                continue
            weight = data[checkpoint_key]
            if checkpoint_key.endswith(".weight_offset"):
                if torch.count_nonzero(weight).item() != 0:
                    raise NotImplementedError(
                        f"Asymmetric W8A8 zero points are not supported, but {checkpoint_key!r} is non-zero."
                    )
                continue
            if "experts." in checkpoint_key:
                expert_weights[checkpoint_key] = weight
                continue
            model_name = name_mapping.get(checkpoint_key)
            if model_name is None:
                continue
            if model_name in params_dict:
                target = params_dict[model_name]
            elif model_name in buffers_dict:
                target = buffers_dict[model_name]
            else:
                raise KeyError(
                    f"Mapped model tensor {model_name!r} for checkpoint key "
                    f"{checkpoint_key!r} does not exist."
                )

            module_name = model_name.rsplit(".", 1)[0]
            weight = _align_projection_weight(weight, target, modules_dict.get(module_name))
            if target.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for {checkpoint_key!r} -> {model_name!r}: "
                    f"checkpoint={tuple(weight.shape)}, model={tuple(target.shape)}"
                )
            target.data.copy_(weight)
            loaded_targets.add(model_name)
        del data

    missing_loaded_targets = sorted(set(sources_by_target) - loaded_targets)
    if missing_loaded_targets:
        preview = ", ".join(missing_loaded_targets[:8])
        raise RuntimeError(
            f"Loader did not populate {len(missing_loaded_targets)} mapped model "
            f"tensors: {preview}"
        )

    return expert_weights


def _prepare_quant_weights(model):
    for layer in model.model.layers:
        self_attn = layer.self_attn
        self_attn.prepare_quant_proj_weights()
        if self_attn.indexer is not None:
            self_attn.indexer.prepare_quant_proj_weights()
        layer.mlp.shared_experts.prepare_for_inference()
        layer.mlp.experts.prepare_for_inference()


def _add_layer_name_mapping(
    mapping,
    prefix,
    model_prefix,
    *,
    include_compressor=False,
    include_indexer=False,
    include_gate_bias=True,
    include_tid2eid=False,
    include_wo_b_scale=True,
):
    checkpoint_attn = f"{prefix}.attn"
    model_attn = f"{model_prefix}.self_attn"
    checkpoint_ffn = f"{prefix}.ffn"
    model_mlp = f"{model_prefix}.mlp"

    mapping[f"{prefix}.hc_attn_fn"] = f"{model_prefix}.hc_attn.weight"
    mapping[f"{prefix}.hc_attn_base"] = f"{model_prefix}.hc_attn.bias"
    mapping[f"{prefix}.hc_attn_scale"] = f"{model_prefix}.hc_attn.scale"
    mapping[f"{prefix}.hc_ffn_fn"] = f"{model_prefix}.hc_ffn.weight"
    mapping[f"{prefix}.hc_ffn_base"] = f"{model_prefix}.hc_ffn.bias"
    mapping[f"{prefix}.hc_ffn_scale"] = f"{model_prefix}.hc_ffn.scale"
    mapping[f"{prefix}.attn_norm.weight"] = f"{model_prefix}.attn_norm.weight"
    mapping[f"{prefix}.ffn_norm.weight"] = f"{model_prefix}.ffn_norm.weight"

    mapping[f"{checkpoint_attn}.wq_a.weight"] = f"{model_attn}.wq_a.weight"
    mapping[f"{checkpoint_attn}.wq_a.weight_scale"] = f"{model_attn}.wq_a.weight_scale"
    mapping[f"{checkpoint_attn}.wq_a.weight_offset"] = "__validate_zero_offset__"
    mapping[f"{checkpoint_attn}.q_norm.weight"] = f"{model_attn}.q_norm.weight"
    mapping[f"{checkpoint_attn}.wq_b.weight"] = f"{model_attn}.wq_b.weight"
    mapping[f"{checkpoint_attn}.wq_b.scale"] = f"{model_attn}.wq_b.weight_scale"
    mapping[f"{checkpoint_attn}.wq_b.weight_scale"] = f"{model_attn}.wq_b.weight_scale"
    mapping[f"{checkpoint_attn}.wq_b.weight_offset"] = "__validate_zero_offset__"
    mapping[f"{checkpoint_attn}.wkv.weight"] = f"{model_attn}.wkv.weight"
    mapping[f"{checkpoint_attn}.wkv.weight_scale"] = f"{model_attn}.wkv.weight_scale"
    mapping[f"{checkpoint_attn}.wkv.weight_offset"] = "__validate_zero_offset__"
    mapping[f"{checkpoint_attn}.kv_norm.weight"] = f"{model_attn}.kv_norm.weight"
    mapping[f"{checkpoint_attn}.wo_a.weight"] = f"{model_attn}.wo_a.weight"
    mapping[f"{checkpoint_attn}.wo_b.weight"] = f"{model_attn}.wo_b.weight"
    if include_wo_b_scale:
        mapping[f"{checkpoint_attn}.wo_b.scale"] = f"{model_attn}.wo_b.weight_scale"
    mapping[f"{checkpoint_attn}.attn_sink"] = f"{model_attn}.attn_sink"

    if include_compressor:
        checkpoint_compressor = f"{checkpoint_attn}.compressor"
        model_compressor = f"{model_attn}.sfa_compressor"
        mapping[f"{checkpoint_compressor}.wkv.weight"] = f"{model_compressor}.wkv.weight"
        mapping[f"{checkpoint_compressor}.wgate.weight"] = f"{model_compressor}.wgate.weight"
        mapping[f"{checkpoint_compressor}.ape"] = f"{model_compressor}.ape"
        mapping[f"{checkpoint_compressor}.norm.weight"] = f"{model_compressor}.norm.weight"

    if include_indexer:
        checkpoint_indexer = f"{checkpoint_attn}.indexer"
        model_indexer = f"{model_attn}.indexer"
        mapping[f"{checkpoint_indexer}.wq_b.weight"] = f"{model_indexer}.wq_b.weight"
        mapping[f"{checkpoint_indexer}.wq_b.scale"] = f"{model_indexer}.wq_b.weight_scale"
        mapping[f"{checkpoint_indexer}.wq_b.weight_scale"] = f"{model_indexer}.wq_b.weight_scale"
        mapping[f"{checkpoint_indexer}.wq_b.weight_offset"] = "__validate_zero_offset__"
        mapping[f"{checkpoint_indexer}.weights_proj.weight"] = f"{model_indexer}.weights_proj.weight"
        mapping[f"{checkpoint_indexer}.compressor.wkv.weight"] = f"{model_indexer}.compressor.wkv.weight"
        mapping[f"{checkpoint_indexer}.compressor.wgate.weight"] = f"{model_indexer}.compressor.wgate.weight"
        mapping[f"{checkpoint_indexer}.compressor.ape"] = f"{model_indexer}.compressor.ape"
        mapping[f"{checkpoint_indexer}.compressor.norm.weight"] = f"{model_indexer}.compressor.norm.weight"

    mapping[f"{checkpoint_ffn}.gate.weight"] = f"{model_mlp}.gate"
    if include_gate_bias:
        mapping[f"{checkpoint_ffn}.gate.bias"] = f"{model_mlp}.moe_gating_top_k.bias"
    if include_tid2eid:
        mapping[f"{checkpoint_ffn}.gate.tid2eid"] = f"{model_mlp}.tid2eid"

def _build_name_mapping(model):
    mapping = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "hc_head_fn": "model.hc_head_fn",
        "hc_head_base": "model.hc_head_base",
        "hc_head_scale": "model.hc_head_scale",
    }
    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        model_prefix = f"model.layers.{layer_idx}"
        compress_ratio = model.config.compress_ratios[layer_idx]
        is_hash_layer = layer_idx < model.config.num_hash_layers
        _add_layer_name_mapping(
            mapping,
            prefix,
            model_prefix,
            include_compressor=compress_ratio > 1,
            include_indexer=compress_ratio == 4,
            include_gate_bias=not is_hash_layer,
            include_tid2eid=is_hash_layer,
            include_wo_b_scale=isinstance(model.model.layers[layer_idx].self_attn.wo_b, MojoQuantGemm),
        )
    return mapping


def _load_expert_weights(model, expert_weights):
    experts_per_rank = model.config.n_routed_experts // model.ep_size
    ep_start = model.ep_rank * experts_per_rank

    for layer_idx in range(model.config.num_hidden_layers):
        mlp = model.model.layers[layer_idx].mlp
        prefix = f"layers.{layer_idx}.ffn"
        routed_prefixes = [f"{prefix}.experts.{eid}" for eid in range(ep_start, ep_start + experts_per_rank)]
        for experts_mod, prefixes in (
            (mlp.experts, routed_prefixes),
            (mlp.shared_experts, [f"{prefix}.shared_experts"]),
        ):
            missing = []
            resolved_keys = []
            for expert_prefix in prefixes:
                weight_keys = [f"{expert_prefix}.{proj}.weight" for proj in ("w1", "w3", "w2")]
                scale_keys = []
                for projection in ("w1", "w3", "w2"):
                    candidates = (f"{expert_prefix}.{projection}.scale", f"{expert_prefix}.{projection}.weight_scale")
                    scale_key = next((key for key in candidates if key in expert_weights), None)
                    if scale_key is None:
                        missing.append(" or ".join(candidates))
                    scale_keys.append(scale_key)
                missing.extend(key for key in weight_keys if key not in expert_weights)
                resolved_keys.append((*weight_keys, *scale_keys))
            if missing:
                preview = ", ".join(missing[:8])
                raise KeyError(f"Missing {len(missing)} rank-local or shared expert tensors: {preview}")

            up_proj_weight = torch.empty_like(experts_mod.up_proj.weight, device="cpu")
            up_proj_scale = torch.empty_like(experts_mod.dequant_swiglu_quant.weight_scale, device="cpu")
            down_proj_weight = torch.empty_like(experts_mod.down_proj.weight, device="cpu")
            down_proj_scale = torch.empty_like(experts_mod.down_proj.weight_scale, device="cpu")

            for local_eid, keys in enumerate(resolved_keys):
                w1_key, w3_key, w2_key, s1_key, s3_key, s2_key = keys
                up_proj_weight[local_eid] = torch.cat((expert_weights[w1_key], expert_weights[w3_key])).t()
                down_proj_weight[local_eid] = expert_weights[w2_key].t()
                up_proj_scale[local_eid] = torch.cat((expert_weights[s1_key], expert_weights[s3_key])).reshape(-1)
                down_proj_scale[local_eid] = expert_weights[s2_key].reshape(-1)

            experts_mod.up_proj.weight.copy_(up_proj_weight)
            experts_mod.dequant_swiglu_quant.weight_scale.data.copy_(up_proj_scale)
            experts_mod.down_proj.weight.copy_(down_proj_weight)
            experts_mod.down_proj.weight_scale.copy_(down_proj_scale)


def load_weights(model, weight_dir: str):
    _configure_modelslim_w8a8_modules(model, weight_dir)
    _init_default_weights(model)

    name_mapping = _build_name_mapping(model)
    max_layer_idx = model.config.num_hidden_layers - 1
    name_mapping = {
        key: value
        for key, value in name_mapping.items()
        if not key.startswith("layers.") or int(key.split(".")[1]) <= max_layer_idx
    }
    expert_key_prefixes = [f"layers.{layer_idx}" for layer_idx in range(model.config.num_hidden_layers)]
    expert_weights = _load_mapped_weights_from_dir(model, weight_dir, name_mapping, expert_key_prefixes)
    _load_expert_weights(model, expert_weights)
    _prepare_quant_weights(model)
