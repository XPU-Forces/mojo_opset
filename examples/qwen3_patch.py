import argparse
import os

import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer


def parse_args():
    p = argparse.ArgumentParser(description="Run Qwen3-8B with Transformers (robust local/HF handling)")
    p.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Local model directory or HF model id, e.g. './Qwen3-8B'",
    )
    p.add_argument(
        "--prompt", type=str, default="请用中文简要介绍 Qwen3-8B 的主要能力。", help="User prompt or chat message"
    )
    p.add_argument("--max-new-tokens", type=int, default=1024, help="Number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling; if omitted, uses greedy")
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return p.parse_args()


def resolve_src(model_path: str):
    """
    Robust source resolution:
    - If model_path is an existing directory, force local loading (no Hub requests).
    - If model_path looks like a path but doesn't exist, raise a clear error with suggestions.
    - Otherwise, treat as a HF repo id.
    """
    if os.path.isdir(model_path):
        return model_path, dict(local_files_only=True)
    if model_path.startswith(("./", "../", "/")) and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local path '{model_path}' not found.\n"
            f"- Use a valid HF repo id (e.g. 'Qwen/Qwen3-8B' or 'Qwen/Qwen3-8B-Instruct'),\n"
            f"- Or download the model to that directory first, e.g.:\n"
            f"  huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B --local-dir-use-symlinks False"
        )
    return model_path, {}


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from mojo_opset.utils.patching import apply_mojo_to_qwen3

    apply_mojo_to_qwen3()

    # Resolve source
    src, extra = resolve_src(args.model_path)

    # Load tokenizer & model (Qwen requires trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True, **extra)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **extra,
    )

    from mojo_opset.utils.platform import get_platform

    platform = get_platform()

    model.to(platform)
    model.eval()

    # Build input; prefer chat template when available
    messages = [
        {"role": "system", "content": "You are a helpful assistant that replies in Chinese."},
        {"role": "user", "content": args.prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt")
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.stream else None
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            streamer=streamer,
        )

    if not args.stream:
        gen_ids = out_ids[0][inputs["input_ids"].shape[-1] :]
        print(tokenizer.decode(gen_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
