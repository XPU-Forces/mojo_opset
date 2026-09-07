import argparse
import os

from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from encoding_dsv4 import encode_messages
from transformers import AutoTokenizer

from mojo_opset import MojoDequantSwiGLUClampQuant
from mojo_opset.modeling.deepseekv4 import DeepseekV4Config
from mojo_opset.modeling.deepseekv4 import DeepseekV4ForCausalLM


def select_torch_clamp(model):
    # Keep the example independent of torch_npu's optional fused clamp API.
    implementation = MojoDequantSwiGLUClampQuant.get_backend_impl("torch", strict=True)
    for layer in model.model.layers:
        for expert in (layer.mlp.experts, layer.mlp.shared_experts):
            old = expert.dequant_swiglu_quant
            replacement = implementation(
                expert_num=old.expert_num,
                hidden_size=old.hidden_size,
                clamp_limit=old.clamp_limit,
                device=old.weight_scale.device,
            )
            replacement.load_state_dict(old.state_dict(), strict=True)
            expert.dequant_swiglu_quant = replacement


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Briefly introduce yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    text = encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or args.max_new_tokens < 1 or len(ids) + args.max_new_tokens > 512:
        raise ValueError("Use a nonempty prompt and 1+ new tokens, with a combined limit of 512.")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("npu:" + os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.set_default_dtype(torch.bfloat16)
    dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    try:
        config = DeepseekV4Config.from_json(os.path.join(args.model_dir, "config.json"))
        config.pa_max_length = 512
        config.compress_ratios = config.compress_ratios[:config.num_hidden_layers]
        with torch.device(device):
            model = DeepseekV4ForCausalLM(config, ep_size=world_size, ep_rank=rank)
        select_torch_clamp(model)
        model.eval()
        model.load_weights(model, args.model_dir)
        model.init_parallel_comm_group()
        model.set_ep_group()
        if rank == 0:
            print("Weights loaded. Running prefill and decode...", flush=True)

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, cache = model(
            input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True, is_prefill=True
        )
        generated = []
        for step in range(args.max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1)
            # Keep the generated sequence and stopping decision identical on all EP ranks.
            dist.broadcast(next_token, src=0)
            generated.append(next_token.item())
            if generated[-1] == tokenizer.eos_token_id or step + 1 == args.max_new_tokens:
                break
            logits, cache = model(
                next_token[:, None], past_key_values=cache, use_cache=True, is_prefill=False
            )
        if rank == 0:
            print(tokenizer.decode(generated, skip_special_tokens=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
