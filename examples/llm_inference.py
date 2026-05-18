import argparse
import importlib
import json
import os
import logging
import sys

import torch
import torch_npu
import torch.distributed as dist

from transformers import AutoTokenizer

from mojo_opset.utils.hf_utils import _resolve_local_files_only
from mojo_opset.utils.hf_utils import build_model_from_hf

ARCH_MAP = {
    "Qwen3ForCausalLM": ("mojo_opset.modeling.qwen3.mojo_qwen3_dense", "Qwen3ForCausalLM"),
    "SeedOssForCausalLM": ("mojo_opset.modeling.seed_oss.mojo_seed_oss_base", "SeedOssForCausalLM"),
    "DeepseekV4ForCausalLM": ("mojo_opset.modeling.deepseekv4.mojo_deepseek_v4", "DeepseekV4ForCausalLM"),
}

_MEM_PROFILING = os.getenv("MEM_PROFILING", "0") == "1"

def _mem_snapshot(tag, device=None):
    if not _MEM_PROFILING:
        return
    if device is None:
        device = torch.npu.current_device()
    alloc = torch.npu.memory_allocated(device) / (1024 ** 3)
    reserved = torch.npu.memory_reserved(device) / (1024 ** 3)
    max_alloc = torch.npu.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.npu.max_memory_reserved(device) / (1024 ** 3)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[MEM][rank{rank}] {tag:50s} alloc={alloc:7.3f}GiB reserved={reserved:7.3f}GiB max_alloc={max_alloc:7.3f}GiB max_reserved={max_reserved:7.3f}GiB", flush=True)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

GOLDEN_DEEPSEEK_V4_ROOT = os.getenv(
    "GOLDEN_DEEPSEEK_V4_ROOT",
    "/data01/tbw/mojo_opset_info/cann-recipes-infer/models/deepseek-v4",
)
if os.path.isdir(GOLDEN_DEEPSEEK_V4_ROOT) and GOLDEN_DEEPSEEK_V4_ROOT not in sys.path:
    sys.path.insert(0, GOLDEN_DEEPSEEK_V4_ROOT)


def build_prompt_input_ids(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if model.__class__.__name__ == "DeepseekV4ForCausalLM":
        from utils.encoding_dsv4 import encode_messages

        prompt_text = encode_messages(messages, thinking_mode="chat")
        return tokenizer.encode(prompt_text, return_tensors="pt"), prompt_text

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (TypeError, NotImplementedError, ValueError):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_text = prompt
    return input_ids, prompt_text


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
    module_path, cls_name = ARCH_MAP[arch]
    module = importlib.import_module(module_path)

    return getattr(module, cls_name)


def init_distributed(ep_size):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    npu_device_idx = int(os.getenv("NPU_DEVICE_IDX", local_rank))
    torch.npu.set_device(torch.device(f"npu:{npu_device_idx}"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                world_size=world_size,
                rank=global_rank,
            )
            logger.info(f"HCCL init done: global_rank={global_rank}, world_size={world_size}, local_rank={local_rank}")
    return local_rank, global_rank, world_size


def generate(model, tokenizer, prompt, max_new_tokens, device, ep_size=1):
    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    is_main = (global_rank == 0)
    moe_ep_group = getattr(model, 'moe_ep_group', None)

    input_ids, prompt_text = build_prompt_input_ids(model, tokenizer, prompt)
    input_ids = input_ids.to(device)
    _mem_snapshot("after input_ids.to(device)")

    if is_main:
        print(f"\nPrompt: {prompt}")
        print(f"Rendered prompt: {repr(prompt_text)}")
        print(f"Input token IDs: {input_ids.detach().cpu().tolist()}")
        print("-" * 40)

    torch.npu.reset_peak_memory_stats()
    _mem_snapshot("before prefill (peak reset)")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, is_prefill=True)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            logits, past_key_values = outputs
    _mem_snapshot("after prefill")

    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    if ep_size > 1 and dist.is_initialized():
        dist.broadcast(next_token_id, src=0, group=moe_ep_group)

    generated_ids = [next_token_id.item()]
    input_ids = next_token_id

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, is_prefill=False)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                logits, past_key_values = outputs

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        if ep_size > 1 and dist.is_initialized():
            dist.broadcast(next_token_id, src=0, group=moe_ep_group)

        tid = next_token_id.item()
        generated_ids.append(tid)
        input_ids = next_token_id

        if tid == tokenizer.eos_token_id:
            if is_main:
                print(f"EOS reached at step {step}.")
            break

    _mem_snapshot("after all decode steps")

    if is_main:
        print("-" * 40)
        print(f"Generated token IDs: {generated_ids}")
        for tid in generated_ids:
            raw = tokenizer.decode([tid])
            clean = tokenizer.decode([tid], skip_special_tokens=True)
            print(f"  token {tid}: raw={repr(raw)}, clean={repr(clean)}")
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated text: {full_output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("QWEN3_MODEL_PATH", ""))
    parser.add_argument("--device", type=str, default=os.getenv("QWEN3_DEVICE", "npu"))
    parser.add_argument("--num_layers", type=int, default=int(os.getenv("QWEN3_NUM_LAYERS", "36")))
    parser.add_argument("--prompt", type=str, default="今天天气怎么样？")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--pa_max_length", type=int, default=int(os.getenv("PA_MAX_LENGTH", "2048")))
    parser.add_argument("--transformers", action="store_true", help="Use Transformers model")
    parser.add_argument("--ep_size", type=int, default=int(os.getenv("EP_SIZE", "1")))
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path and not os.getenv("QWEN3_MODEL_PATH"):
        raise ValueError("Please pass --model_path or set QWEN3_MODEL_PATH")

    ep_size = args.ep_size
    local_rank, global_rank, world_size = init_distributed(ep_size)

    torch_npu.npu.config.allow_internal_format = True

    local_files_only = _resolve_local_files_only(args.model_path)

    if args.transformers:
        from transformers import AutoModelForCausalLM as model_class
        model = build_model_from_hf(
            model_class,
            args.model_path,
            device=args.device,
            num_layers=args.num_layers,
            trust_remote_code=True,
        )
    else:
        model_class = resolve_model_class(args.model_path)
        local_files_only = _resolve_local_files_only(args.model_path)

        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(
                args.model_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
        except (ValueError, KeyError, ImportError):
            cfg_path = os.path.join(args.model_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import DeepseekV4Config
            hf_config = DeepseekV4Config(**cfg_dict)

        hf_config.pa_max_length = args.pa_max_length

        ep_rank = global_rank % ep_size if ep_size > 1 else 0

        npu_device_idx = int(os.getenv("NPU_DEVICE_IDX", local_rank))
    _mem_snapshot("before model construct", npu_device_idx)

    if hasattr(model_class, "load_weights"):
        from transformers.modeling_utils import no_init_weights
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with no_init_weights():
            model = model_class(hf_config, num_layers=args.num_layers, ep_size=ep_size, ep_rank=ep_rank)
        torch.set_default_dtype(origin_dtype)
        _mem_snapshot("after model construct (CPU)", npu_device_idx)
        model = model.to(f"npu:{npu_device_idx}").eval()
        _mem_snapshot("after model.to(npu)", npu_device_idx)

        if ep_size > 1 and dist.is_initialized():
            model.init_parallel_comm_group()
            model.set_ep_group()
        _mem_snapshot("after init_parallel_comm_group", npu_device_idx)

        model_class.load_weights(model, args.model_path)
        _mem_snapshot("after load_weights", npu_device_idx)
    else:
        model = build_model_from_hf(
            model_class,
            args.model_path,
            device=args.device,
            num_layers=args.num_layers,
            trust_remote_code=True,
        )

    _mem_snapshot("before tokenizer", npu_device_idx)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer is None:
        raise ValueError("Tokenizer not found")

    generate(model, tokenizer, args.prompt, args.max_new_tokens, f"npu:{local_rank}", ep_size=ep_size)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()