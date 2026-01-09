import glob
import os
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
from transformers.modeling_utils import no_init_weights


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
                    print(f"    WARNING: Shape mismatch for {key}. Expected {target_shape}, got {tensor.shape}. Skipping.")
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
        with no_init_weights():
            torch_model = model_class(hf_config)
            
        # Move to device directly. 
        # NOT using to_empty() because it destroys initialized buffers (like RoPE inv_freq),
        # causing garbage output. .to() preserves buffers while moving parameters.
        torch_model = torch_model.to(torch.bfloat16).to(device).eval()
        
        load_weights_direct(model_id_or_path, torch_model)
        
        return torch_model



