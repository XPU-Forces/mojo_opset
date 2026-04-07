import glob
import os

from collections import OrderedDict
from typing import List
from typing import Optional

import torch

from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import AutoConfig

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def _env_flag_true(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _resolve_local_files_only(model_id_or_path: str) -> bool:
    if os.path.isdir(os.path.expanduser(model_id_or_path)):
        return True
    return any(
        _env_flag_true(k)
        for k in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_LOCAL_FILES_ONLY",
        )
    )


def load_weights_direct(model_path: str, torch_model: nn.Module) -> None:
    # 1. Collect weight files
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))

    files = safetensors_files if safetensors_files else bin_files
    if not files:
        raise ValueError(f"No checkpoint files found in {model_path}")

    # 2. Prepare model state dict (destination)
    model_state_dict = torch_model.state_dict()
    expected_keys = set(model_state_dict.keys())
    loaded_keys = set()
    unexpected_keys = set()

    print(f"Loading weights from {len(files)} files...")

    # 3. Load each file
    for f in files:
        print(f"  Processing {os.path.basename(f)} ...")
        if f.endswith(".safetensors"):
            if load_safetensors is None:
                raise ImportError("safetensors is not installed. Please install it to load .safetensors files.")
            state_dict = load_safetensors(f)
        else:
            state_dict = torch.load(f, map_location="cpu")

        for key, tensor in state_dict.items():
            # HF keys often start with "model." or are direct.
            # Our torch_model has "model." prefix for the transformer body, and "lm_head" outside.
            # If the checkpoint keys match exactly, we are good.
            # Check for potential prefix mismatches if necessary.

            if key in expected_keys:
                # Check shape
                target_shape = model_state_dict[key].shape
                if target_shape != tensor.shape:
                    print(
                        f"    WARNING: Shape mismatch for {key}. Expected {target_shape}, got {tensor.shape}. Skipping."
                    )
                    continue

                with torch.no_grad():
                    model_state_dict[key].copy_(tensor)
                loaded_keys.add(key)
            else:
                unexpected_keys.add(key)

        # Free memory
        del state_dict
        torch.npu.empty_cache() if torch.npu.is_available() else None

    # 4. Report
    missing_keys = expected_keys - loaded_keys

    print("\nWeight Loading Report:")
    print(f"  Total Expected Keys: {len(expected_keys)}")
    print(f"  Successfully Loaded: {len(loaded_keys)}")
    print(f"  Missing Keys: {len(missing_keys)}")
    print(f"  Unexpected Keys: {len(unexpected_keys)}")

    if missing_keys:
        print("\n  Missing Keys:")
        for k in sorted(list(missing_keys)):
            print(f"    - {k}")

    if unexpected_keys:
        print("\n  Unexpected Keys:")
        for k in sorted(list(unexpected_keys)):
            print(f"    - {k}")


def build_model_from_hf(
    model_class: type[nn.Module],
    model_id_or_path: str,
    device: str,
    num_layers: Optional[int] = None,
    trust_remote_code: bool = True,
) -> nn.Module:
    local_files_only = _resolve_local_files_only(model_id_or_path)

    hf_config = AutoConfig.from_pretrained(
        model_id_or_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    # Check if the model class supports from_pretrained (standard HF models)
    if hasattr(model_class, "from_pretrained"):
        torch_model = model_class.from_pretrained(
            model_id_or_path,
            config=hf_config,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        return torch_model.to(device)
    else:
        # Use no_init_weights to skip random initialization
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            torch_model = model_class(hf_config)

        # Move to device directly.
        # NOT using to_empty() because it destroys initialized buffers (like RoPE inv_freq),
        # causing garbage output. .to() preserves buffers while moving parameters.
        torch_model = torch_model.to(torch.bfloat16).to(device).eval()

        load_weights_direct(model_id_or_path, torch_model)

        return torch_model


"""
An example for name_mapping_dict, the values should be weight-keys of our own model.
        weight_renaming = {
            "linear_qkv_proj": "linear_attn_block.projs.qkv",
            "dt_bias": "linear_attn_block.dt_bias",
            "linear_o_proj": "linear_attn_block.projs.output",
            "a_proj": "linear_attn_block.projs.a",
            "b_proj": "linear_attn_block.projs.b",
            "context_groupnorm_linear": "linear_attn_block.context_rms_norm",
            "q_norm": "flash_attn_block.rms_norms.query",
            "k_norm": "flash_attn_block.rms_norms.key",
            "qkv_proj": "flash_attn_block.projs.qkv",
            "o_proj": "flash_attn_block.projs.output",
            "context_groupnorm": "flash_attn_block.context_rms_norm",
            "o_norm": "rms_norm",
            "mlp.moe.experts": "ffn",
            "gate_up_proj": "fc1",
            "down_proj": "fc2",
            "self_attention": "attention",
        }
"""


def create_renaming_by_dict(name_mapping_dict: dict, longest_match_first=True):
    from transformers.core_model_loading import WeightRenaming

    assert name_mapping_dict

    # NOTE(liuyuan): Longest-match-first should be the most common match.
    if longest_match_first:
        name_mapping_dict = sorted(name_mapping_dict.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)
    # NOTE(liuyuan): Although WeightRenaming supports multi-pattern matching, it still requires careful control over the mapping logic. Therefore, we recommend that users create complex WeightRenaming themselves.
    return list(map(lambda x: WeightRenaming(x[0], x[1]), name_mapping_dict))


def load_hf_weights(hf_dir):
    state_dict = {}
    import glob

    safetensors_files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    from safetensors.torch import load_file as load_safetensors

    for f in safetensors_files:
        state_dict.update(load_safetensors(f))
    return state_dict


"""
An example to create a WeightConverter with custom ConversionOps.
        from typing import Any
        class SimpleConverter(ConversionOps):
            @torch.no_grad
            def convert(
                self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs
            ) -> dict[str, list[torch.Tensor]]:
                result_tensor = None
                target_pattern = self.get_target_pattern(input_dict, source_patterns, target_patterns)
                for source_pattern in source_patterns:
                    source_tensor = input_dict[source_pattern].float()
                    if result_tensor is None:
                        result_tensor = torch.zeros_like(source_tensor)
                    result_tensor += source_tensor
                return {target_pattern : result_tensor * 0.5}

            def get_target_pattern(self, input_dict: dict, source_patterns: str, target_patterns: list[str]) -> str:
                assert len(source_pattern) == 2
                return target_aptterns[0]
        weight_converter = WeightConverter(["Hello","moto"], ["Nokia"], operations=[SimpleConverter()])
"""


def convert_hf_state_dict(
    hf_state_dict: dict | OrderedDict,
    target_keys: set[str] | dict[str, ...],
    renamings: List["WeightRenaming"] = [],
    converters: List["WeightConverter"] = [],
) -> dict[str, torch.Tensor]:
    """Rename and convert an HF state dict to model-native keys.

    Returns the converted state dict without loading it into any model.
    This is the shared core used by both ``load_weights_with_renaming_and_converter``
    (single-device) and distributed layer-by-layer loading.
    """
    from copy import deepcopy

    import transformers

    from transformers.core_model_loading import WeightRenaming
    from transformers.core_model_loading import rename_source_key

    pattern_to_converter = {k: c for c in converters for k in c.source_patterns}
    param_name_to_load: dict = {}

    for key in hf_state_dict.keys():
        renamed_key, src_pat = rename_source_key(key, renamings, converters)
        if renamed_key in target_keys:
            logger.debug("Renaming\033[1;31m %s \033[0m as\033[1;32m %s \033[0m", key, renamed_key)
            if src_pat is not None:
                mapping = param_name_to_load.setdefault(renamed_key, deepcopy(pattern_to_converter[src_pat]))
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(key, renamed_key))
                src_pat = key
            mapping.add_tensor(renamed_key, key, src_pat, hf_state_dict[key])

    new_state_dict = {}
    for k, mapping in param_name_to_load.items():
        if transformers.__version__ >= "5.1.0":
            converted = mapping.convert(k)
        else:
            converted, _ = mapping.convert(k)
        for ck, cv in converted.items():
            new_state_dict[ck] = cv[0] if isinstance(cv, list) else cv
    return new_state_dict


def load_weights_with_renaming_and_converter(
    model: torch.nn.Module,
    hf_dir_or_preload_state_dict: str | dict | OrderedDict,
    strict_loading=True,
    renamings: List["WeightRenaming"] = [],
    converters: List["WeightConverter"] = [],
):
    if isinstance(hf_dir_or_preload_state_dict, str):
        state_dict = load_hf_weights(hf_dir_or_preload_state_dict)
    elif isinstance(hf_dir_or_preload_state_dict, (OrderedDict, dict)):
        state_dict = hf_dir_or_preload_state_dict
    else:
        raise TypeError(
            f"hf_dir_or_preload_state_dict is supposed to be string or OrderedDict, "
            f"but found {type(hf_dir_or_preload_state_dict)}"
        )

    new_state_dict = convert_hf_state_dict(state_dict, model.state_dict(), renamings, converters)
    return model.load_state_dict(new_state_dict, strict=strict_loading)
