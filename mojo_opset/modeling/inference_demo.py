import argparse
import os
import torch
import json
import importlib
from transformers import AutoTokenizer
from mojo_opset.utils.hf_utils import build_model_from_hf, _resolve_local_files_only


ARCH_MAP = {
    "Qwen3ForCausalLM": ("mojo_qwen3_dense", "Qwen3ForCausalLM"),
    "SeedOssForCausalLM": ("mojo_seed_oss_base", "SeedOssForCausalLM"),
}


def resolve_model_class(model_path: str):
    cfg_path = os.path.join(model_path, "config.json")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found under {model_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arch_list = cfg.get("architectures") or []
    arch = arch_list[0] if isinstance(arch_list, list) and len(arch_list) > 0 else None
    if arch not in ARCH_MAP:
        raise ValueError(f"Unsupported architecture: {arch}")
    mod_name, cls_name = ARCH_MAP[arch]
    module = importlib.import_module(mod_name)

    return getattr(module, cls_name)


def generate(model, tokenizer, prompt, max_new_tokens, device):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", thinking_budget=-1
    ).to(device)

    # Prefill
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            logits, past_key_values = outputs

    # Greedy sampling for the first token
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    generated_ids = [next_token_id.item()]

    # Decode loop
    input_ids = next_token_id

    for i in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                logits, past_key_values = outputs

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        generated_ids.append(next_token_id.item())

        input_ids = next_token_id

        if next_token_id.item() == tokenizer.eos_token_id:
            print("EOS reached.")
            break

    print("-" * 40)
    full_output = tokenizer.decode(generated_ids)
    print(f"Generated text: {full_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("QWEN3_MODEL_PATH", ""))
    parser.add_argument("--device", type=str, default=os.getenv("QWEN3_DEVICE", "npu"))
    parser.add_argument("--num_layers", type=int, default=int(os.getenv("QWEN3_NUM_LAYERS", "36")))
    parser.add_argument("--prompt", type=str, default="今天天气怎么样？")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--transformers", action="store_true", help="Use Transformers model")
    args = parser.parse_args()

    if not args.model_path:
        print("Warning: No model_path provided. Using default from env or failing.")
        if not os.getenv("QWEN3_MODEL_PATH"):
            # For testing purposes, if no model path, we might fail or mock.
            # But the user asked to "realize demo", presumably they will run it with path.
            # I will raise error if not found.
            raise ValueError("Please pass --model_path or set QWEN3_MODEL_PATH")

    local_files_only = _resolve_local_files_only(args.model_path)

    if args.transformers:
        from transformers import AutoModelForCausalLM as model_class
    else:
        model_class = resolve_model_class(args.model_path)

    print(f"Loading model from {args.model_path}...")
    model = build_model_from_hf(
        model_class,
        args.model_path,
        device=args.device,
        num_layers=args.num_layers,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer is None:
        raise ValueError("Tokenizer not found")

    generate(model, tokenizer, args.prompt, args.max_new_tokens, args.device)
