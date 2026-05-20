import argparse
import importlib
import json
import os
import logging
from typing import List

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


def encode_deepseek_v4_chat(prompt: str) -> str:
    return "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜></think>"


def build_batch_prompts(prompt: str, batch_size: int, ep_size: int = 1) -> List[str]:
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return [prompt] * batch_size

    if isinstance(parsed, str):
        return [parsed] * batch_size
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--prompt must be a string or a JSON string list")

    prompts = [item for item in parsed if item]
    if len(prompts) == 0:
        raise ValueError("--prompt list is empty after filtering")
    if len(prompts) < batch_size:
        logger.warning(
            "Received %d prompts but batch_size=%d; repeating prompts cyclically to fill batch",
            len(prompts), batch_size,
        )
        prompts = (prompts * (batch_size // len(prompts) + 1))[:batch_size]
    return prompts


def build_prompt_input_ids(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if model.__class__.__name__ == "DeepseekV4ForCausalLM":
        prompt_text = encode_deepseek_v4_chat(prompt)
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


def init_distributed():
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


def _extract_outputs(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits, outputs.past_key_values
    return outputs


def pad_batch(encoded, pad_token_id, device):
    lengths = torch.tensor([item.shape[-1] for item in encoded], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for idx, ids in enumerate(encoded):
        flat = ids.squeeze(0).cpu()
        input_ids[idx, : flat.shape[0]] = flat
        attention_mask[idx, : flat.shape[0]] = True
    return input_ids.to(device), attention_mask.to(device), lengths.to(device)


def last_logits_from_output(logits, lengths):
    if logits.shape[1] == 1:
        return logits[:, 0, :]
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, lengths - 1]


def sync_next_token(next_token, model):
    if dist.is_initialized():
        group = getattr(model, "moe_ep_group", None)
        dist.broadcast(next_token, src=0, group=group)
    return next_token


def forward_prefill_mojo(model, input_ids, attention_mask, lengths):
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True, is_prefill=True)
    logits, past_key_values = _extract_outputs(outputs)
    return last_logits_from_output(logits, lengths), past_key_values


def forward_decode_mojo(model, token_ids, past_key_values):
    outputs = model(token_ids, past_key_values=past_key_values, use_cache=True, is_prefill=False)
    logits, past_key_values = _extract_outputs(outputs)
    return logits[:, -1, :], past_key_values


def generate(model, tokenizer, prompt, max_new_tokens, device, ep_size=1, batch_size=1):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    is_main = (global_rank == 0)
    moe_ep_group = getattr(model, 'moe_ep_group', None)
    prompts = build_batch_prompts(prompt, batch_size, ep_size)
    batch_size = len(prompts)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    encoded = []
    rendered = []
    for p in prompts:
        ids, text = build_prompt_input_ids(model, tokenizer, p)
        encoded.append(ids)
        rendered.append(text)
    input_ids, attention_mask, lengths = pad_batch(encoded, pad_token_id, device)

    _mem_snapshot("after input preparation")

    if is_main:
        for i in range(batch_size):
            print(f"\n[Batch {i}] Prompt: {prompts[i]}")
            print(f"  Rendered prefix: {repr(rendered[i][:200])}")
            print(f"  Input length: {lengths[i].item()}")
        print("-" * 40)

    torch.npu.reset_peak_memory_stats()
    _mem_snapshot("before prefill (peak reset)")
    with torch.no_grad():
        next_token_logits, past_key_values = forward_prefill_mojo(model, input_ids, attention_mask, lengths)
    _mem_snapshot("after prefill")

    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    if ep_size > 1 and dist.is_initialized():
        dist.broadcast(next_token_id, src=0, group=moe_ep_group)

    generated = [[int(x)] for x in next_token_id.squeeze(-1).detach().cpu().tolist()]

    input_ids = next_token_id

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            next_token_logits, past_key_values = forward_decode_mojo(model, input_ids, past_key_values)

        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        if ep_size > 1 and dist.is_initialized():
            dist.broadcast(next_token_id, src=0, group=moe_ep_group)

        token_list = next_token_id.squeeze(-1).detach().cpu().tolist()
        for row, token_id in enumerate(token_list):
            generated[row].append(int(token_id))

        input_ids = next_token_id

    _mem_snapshot("after all decode steps")

    if is_main:
        print("-" * 40)
        for i, output_ids in enumerate(generated):
            res = tokenizer.decode(output_ids, skip_special_tokens=False)
            if tokenizer.eos_token in res:
                res = res.split(tokenizer.eos_token)[0]
            print(f"[Batch {i}] Generated text: {res}")


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
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "1")),
                        help="Batch size for inference (default: 1, single-batch)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path and not os.getenv("QWEN3_MODEL_PATH"):
        raise ValueError("Please pass --model_path or set QWEN3_MODEL_PATH")

    ep_size = args.ep_size
    local_rank, global_rank, _ = init_distributed()
    npu_device_idx = int(os.getenv("NPU_DEVICE_IDX", local_rank))

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

    generate(
        model, tokenizer, args.prompt, args.max_new_tokens, f"npu:{npu_device_idx}",
        ep_size=ep_size, batch_size=args.batch_size,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()