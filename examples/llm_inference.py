import argparse
import contextlib
import importlib
import json
import os
import logging
import time
import traceback
from typing import List

import torch
import torch_npu
import torch.distributed as dist

from transformers import AutoTokenizer

from mojo_opset.utils.hf_utils import _resolve_local_files_only
from mojo_opset.utils.hf_utils import build_model_from_hf
from mojo_opset.runtime import DeepseekSparseAttentionRuntimeState
from mojo_opset.runtime import DeepseekMTPRuntimeState
from mojo_opset.distributed.parallel import apply_llm_parallelize_plan
from mojo_opset.distributed.parallel import copy_paged_dummy_cache_batch
from mojo_opset.distributed.parallel import forward_cp_prefill_minibatch_mojo
from mojo_opset.distributed.parallel import gather_decode_shard_tensor
from mojo_opset.distributed.parallel import get_attn_dp_shard_range
from mojo_opset.distributed.parallel import LLMParallelConfig
from mojo_opset.distributed.parallel import pad_deepseek_v4_cp_prefill
from mojo_opset.distributed.parallel import shard_batch_for_attn_dp

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


def _device_snapshot(tag, *, local_rank, global_rank, world_size, target_device, ep_rank=None):
    current_device = torch.npu.current_device() if hasattr(torch, "npu") else "unknown"
    logger.info(
        "[DEVICE] %s | rank=%s local_rank=%s world_size=%s ep_rank=%s target_device=%s current_device=%s pid=%s",
        tag,
        global_rank,
        local_rank,
        world_size,
        ep_rank,
        target_device,
        current_device,
        os.getpid(),
    )

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
    if isinstance(outputs, tuple) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    return outputs


def pad_batch(encoded, pad_token_id, device):
    lengths = torch.tensor([item.shape[-1] for item in encoded], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for idx, ids in enumerate(encoded):
        flat = ids.squeeze(0)
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


class DeepseekV4DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values, position_ids, context_lens, attn_inputs, attn_metadata=None):
        logits, past_key_values, hidden_states = self.model(
            input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            context_lens=context_lens,
            attn_inputs=attn_inputs,
            attn_metadata=attn_metadata,
            use_cache=True,
            is_prefill=False,
        )
        return logits, past_key_values, hidden_states


class DeepseekV4MTPDecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, past_key_values, position_ids, context_lens, q_lens,
                input_ids, attn_inputs, attn_metadata=None):
        logits, past_key_values, mtp_hidden = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_prefill=False,
            attn_inputs=attn_inputs,
            attn_metadata=attn_metadata,
            context_lens=context_lens,
            q_lens=q_lens,
            input_ids=input_ids,
            return_hidden=True,
        )
        return logits, past_key_values, mtp_hidden



def sample_last_token(logits, *, sync_tokens=False, group=None):
    token = torch.argmax(logits, dim=-1)[:, -1:]
    if sync_tokens:
        token = token.contiguous()
        dist.broadcast(token, src=0, group=group)
    return token


def verify_mtp_spec_tokens(mtp_spec_tokens, all_logits, next_n, *, sync_tokens=False, group=None):
    main_greedy = torch.argmax(all_logits, dim=-1)
    if sync_tokens:
        main_greedy = main_greedy.contiguous()
        dist.broadcast(main_greedy, src=0, group=group)

    token_mask = mtp_spec_tokens == main_greedy[:, :next_n]
    invalid_mask = ~token_mask
    has_invalid = invalid_mask.any(dim=-1)
    invalid_pos = invalid_mask.int().argmax(dim=-1)
    accepted_num = torch.where(has_invalid, invalid_pos, token_mask.shape[-1])
    if sync_tokens:
        accepted_num = accepted_num.contiguous()
        dist.broadcast(accepted_num, src=0, group=group)
    return main_greedy, accepted_num


def mtp_post_process(
    mtp_model,
    mtp_runtime_state,
    mtp_prev_hidden,
    mtp_generate_ids,
    *,
    next_n,
    spec_len,
    batch_size,
    mtp_decode_fn=None,
    mtp_decode_graph_warmed=False,
    enable_mtp_cache_compile=False,
    sync_tokens=False,
    moe_ep_group=None,
    allow_decode_graph=True,
    initial_decode_input_ids=None,
):
    mtp_spec_tokens_list = []
    if initial_decode_input_ids is None:
        mtp_decode_input_ids = mtp_generate_ids[:, -spec_len:].contiguous().reshape(batch_size, spec_len)
    else:
        mtp_decode_input_ids = initial_decode_input_ids.contiguous().reshape(batch_size, spec_len)
    for _ in range(next_n):
        with torch.no_grad():
            if sync_tokens:
                mtp_decode_input_ids = mtp_decode_input_ids.contiguous().reshape(batch_size, spec_len)
                dist.broadcast(mtp_decode_input_ids, src=0, group=moe_ep_group)
            current_mtp_decode_fn = mtp_decode_fn if allow_decode_graph else None
            mtp_logits, _, mtp_hidden = forward_mtp_decode_mojo(
                mtp_model,
                mtp_prev_hidden,
                mtp_decode_input_ids,
                mtp_runtime_state,
                decode_fn=current_mtp_decode_fn,
                skip_guard_eval=(
                    allow_decode_graph and
                    enable_mtp_cache_compile and
                    current_mtp_decode_fn is not None and
                    mtp_decode_graph_warmed
                ),
            )
            if allow_decode_graph and enable_mtp_cache_compile and current_mtp_decode_fn is not None:
                mtp_decode_graph_warmed = True

        mtp_spec_token = sample_last_token(mtp_logits)
        mtp_spec_tokens_list.append(mtp_spec_token)
        mtp_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous().reshape(batch_size, spec_len, -1)
        mtp_generate_ids = torch.cat([mtp_generate_ids, mtp_spec_token], dim=-1)
        mtp_decode_input_ids = torch.cat(
            [mtp_decode_input_ids[:, 1:], mtp_spec_token],
            dim=-1,
        ).contiguous().reshape(batch_size, spec_len)

    mtp_spec_tokens = torch.cat(mtp_spec_tokens_list, dim=-1)
    return mtp_spec_tokens, mtp_prev_hidden, mtp_generate_ids, mtp_decode_graph_warmed


def initialize_mtp_speculation_after_prefill(
    *,
    mtp_model,
    mtp_runtime_state,
    input_ids,
    attention_mask,
    full_input_ids,
    full_attention_mask,
    next_token_id,
    full_next_token_id,
    prev_hidden,
    prev_hidden_all_for_mtp,
    pad_token_id,
    batch_size,
    cp_size,
    shard_start,
    shard_end,
    spec_len,
    next_n,
    use_attn_metadata,
    sync_tokens,
    moe_ep_group,
    mtp_decode_fn=None,
):
    mtp_prefill_source_input_ids = full_input_ids if cp_size > 1 else input_ids
    mtp_prefill_source_attention_mask = full_attention_mask if cp_size > 1 else attention_mask
    mtp_prefill_next_token_id = full_next_token_id if cp_size > 1 else next_token_id
    mtp_input_ids, mtp_attention_mask, confirmed_generate_ids_full = build_mtp_prefill_inputs(
        mtp_prefill_source_input_ids,
        mtp_prefill_source_attention_mask,
        mtp_prefill_next_token_id,
        pad_token_id,
    )
    if cp_size > 1:
        confirmed_generate_ids = confirmed_generate_ids_full[shard_start:shard_end]
        mtp_prev_hidden_for_prefill = prev_hidden_all_for_mtp
    else:
        confirmed_generate_ids = confirmed_generate_ids_full
        mtp_prev_hidden_for_prefill = prev_hidden
    mtp_generate_ids = confirmed_generate_ids

    with torch.no_grad():
        if sync_tokens:
            mtp_input_ids = mtp_input_ids.contiguous()
            dist.broadcast(mtp_input_ids, src=0, group=moe_ep_group)
            mtp_attention_mask = mtp_attention_mask.contiguous()
            dist.broadcast(mtp_attention_mask, src=0, group=moe_ep_group)
        if cp_size > 1:
            mtp_out = forward_mtp_cp_prefill_minibatch_mojo(
                mtp_model,
                mtp_prev_hidden_for_prefill,
                mtp_input_ids,
                mtp_attention_mask,
                mtp_runtime_state,
                local_shard_start=shard_start,
                local_shard_end=shard_end,
                max_seq_len=max(input_ids.shape[1] * 4, 4096),
                next_n=next_n,
                use_attn_metadata=use_attn_metadata,
            )
        else:
            mtp_out = forward_mtp_prefill_mojo(
                mtp_model,
                mtp_prev_hidden_for_prefill,
                mtp_input_ids,
                mtp_attention_mask,
                mtp_runtime_state,
                max_seq_len=max(input_ids.shape[1] * 4, 4096),
                next_n=next_n,
                use_attn_metadata=use_attn_metadata,
            )
        mtp_logits, _, mtp_hidden = mtp_out

    mtp_spec_token = sample_last_token(mtp_logits, sync_tokens=sync_tokens, group=moe_ep_group)
    mtp_spec_tokens = mtp_spec_token
    confirmed_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous() if cp_size > 1 else prev_hidden[:, -spec_len:, :].contiguous()
    mtp_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous().reshape(batch_size, spec_len, -1)
    mtp_generate_ids = torch.cat([mtp_generate_ids, mtp_spec_token], dim=-1)

    if next_n > 1:
        extra_spec_tokens, mtp_prev_hidden, mtp_generate_ids, _ = mtp_post_process(
            mtp_model,
            mtp_runtime_state,
            mtp_prev_hidden,
            mtp_generate_ids,
            next_n=next_n - 1,
            spec_len=spec_len,
            batch_size=batch_size,
            mtp_decode_fn=mtp_decode_fn,
            enable_mtp_cache_compile=False,
            sync_tokens=False,
            moe_ep_group=moe_ep_group,
            allow_decode_graph=False,
            initial_decode_input_ids=mtp_generate_ids[:, -spec_len:],
        )
        mtp_spec_tokens = torch.cat([mtp_spec_tokens, extra_spec_tokens], dim=-1)

    mtp_seq_lens_cached = mtp_runtime_state.paged_cache.seq_lens.clone()
    verify_input_ids = torch.cat([next_token_id, mtp_spec_tokens], dim=-1)
    if sync_tokens:
        dist.broadcast(verify_input_ids, src=0, group=moe_ep_group)

    return {
        'input_ids': verify_input_ids,
        'mtp_spec_tokens': mtp_spec_tokens,
        'confirmed_prev_hidden': confirmed_prev_hidden,
        'mtp_prev_hidden': mtp_prev_hidden,
        'confirmed_generate_ids': confirmed_generate_ids,
        'mtp_generate_ids': mtp_generate_ids,
        'mtp_seq_lens_cached': mtp_seq_lens_cached,
    }


def update_mtp_after_verify(
    *,
    main_greedy,
    main_hidden,
    accepted_num,
    generated,
    runtime_state,
    mtp_runtime_state,
    main_seq_lens_before_verify,
    mtp_seq_lens_cached,
    confirmed_prev_hidden,
    confirmed_generate_ids,
    pad_token_id,
    next_n,
    batch_size,
    spec_len,
    sync_tokens=False,
    moe_ep_group=None,
):
    accepted_num_cpu = accepted_num.detach().cpu().tolist()
    main_greedy_cpu = main_greedy[:, :spec_len].detach().cpu().tolist()
    for row, accepted in enumerate(accepted_num_cpu):
        generated[row].extend(int(token) for token in main_greedy_cpu[row][:accepted + 1])

    accepted_step = accepted_num + 1
    if next_n == 1:
        runtime_state.paged_cache.seq_lens += accepted_step
    elif main_seq_lens_before_verify is not None:
        runtime_state.paged_cache.seq_lens.copy_(main_seq_lens_before_verify + accepted_step)
    else:
        runtime_state.paged_cache.seq_lens += accepted_step

    accepted_idx = accepted_num.to(dtype=torch.long).view(batch_size, 1)
    next_token_id = main_greedy.gather(1, accepted_idx)
    if sync_tokens:
        next_token_id = next_token_id.contiguous()
        dist.broadcast(next_token_id, src=0, group=moe_ep_group)

    hidden_dim = main_hidden.shape[-1]
    hidden_positions = (
        accepted_step.to(dtype=torch.long).view(batch_size, 1, 1) +
        torch.arange(spec_len, device=main_hidden.device, dtype=torch.long).view(1, spec_len, 1)
    ).expand(batch_size, spec_len, hidden_dim)
    mtp_prev_hidden = torch.gather(
        torch.cat([confirmed_prev_hidden, main_hidden[:, :spec_len, :]], dim=1),
        dim=1,
        index=hidden_positions,
    ).contiguous().reshape(batch_size, spec_len, hidden_dim)
    confirmed_prev_hidden = mtp_prev_hidden

    old_history_len = confirmed_generate_ids.shape[1]
    new_history_len = old_history_len + spec_len
    history_pad = torch.full(
        (batch_size, spec_len),
        pad_token_id,
        dtype=confirmed_generate_ids.dtype,
        device=confirmed_generate_ids.device,
    )
    history_source = torch.cat([history_pad, confirmed_generate_ids, history_pad], dim=1)
    new_col_idx = torch.arange(new_history_len, device=confirmed_generate_ids.device, dtype=torch.long).view(1, new_history_len)
    old_shift_idx = accepted_step.to(dtype=torch.long).view(batch_size, 1) + new_col_idx
    shifted_history = history_source.gather(1, old_shift_idx.expand(batch_size, new_history_len))
    append_start = new_history_len - accepted_step.to(dtype=torch.long).view(batch_size, 1)
    append_idx = new_col_idx - append_start
    valid_append = append_idx >= 0
    append_idx = append_idx.clamp(min=0, max=spec_len - 1).expand(batch_size, new_history_len)
    append_tokens = main_greedy[:, :spec_len].gather(1, append_idx)
    confirmed_generate_ids = torch.where(valid_append, append_tokens, shifted_history).contiguous()

    mtp_runtime_state.paged_cache.seq_lens.copy_(mtp_seq_lens_cached + 1 + accepted_num)
    mtp_seq_lens_cached = mtp_runtime_state.paged_cache.seq_lens.clone()

    return {
        'next_token_id': next_token_id,
        'confirmed_prev_hidden': confirmed_prev_hidden,
        'mtp_prev_hidden': mtp_prev_hidden,
        'confirmed_generate_ids': confirmed_generate_ids,
        'mtp_generate_ids': confirmed_generate_ids,
        'mtp_seq_lens_cached': mtp_seq_lens_cached,
    }


def _npugraph_cache_dir(cache_name):
    base_dir = os.getenv("MOJO_GRAPH_CACHE_DIR")
    if not base_dir:
        case_name = os.getenv("CASE_NAME", "mojo_deepseekv4")
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile_cache", case_name)
    cache_dir = os.path.join(base_dir, cache_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _skip_guard_eval_context(enabled):
    if enabled and hasattr(torch.compiler, "set_stance"):
        return torch.compiler.set_stance(skip_guard_eval_unsafe=True)
    return contextlib.nullcontext()


def _compile_deepseek_v4_wrapper(wrapper, graph_mode, enable_cache_compile=False, cache_name=None):
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.cache_size_limit = 128

    if graph_mode == "npugraph_ex":
        compile_options = {
            "frozen_parameter": True,
            "static_kernel_compile": True,
        }
        if enable_cache_compile:
            if cache_name is None:
                raise ValueError("cache_name must be provided when enable_cache_compile=True")
            return torch.npu.npugraph_ex.inference.cache_compile(
                wrapper.forward,
                cache_dir=_npugraph_cache_dir(cache_name),
                dynamic=False,
                options=compile_options,
            )

        return torch.compile(
            wrapper,
            dynamic=False,
            fullgraph=True,
            backend="npugraph_ex",
            options=compile_options,
        )

    if graph_mode == "ge_graph":
        import torchair as tng
        import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
        from torchair.configs.compiler_config import CompilerConfig

        tng.patch_for_hcom()

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        return torch.compile(
            wrapper,
            dynamic=False,
            fullgraph=True,
            backend=npu_backend,
        )

    raise ValueError(f"Unsupported graph_mode: {graph_mode}")


def compile_deepseek_v4_decode(
    model,
    graph_mode,
    enable_cache_compile=False,
    cache_name=None,
):
    model_name = model.__class__.__name__
    if graph_mode == "eager" or model_name not in {"DeepseekV4ForCausalLM", "DeepseekV4ForMTP"}:
        return None
    if model_name == "DeepseekV4ForCausalLM":
        wrapper = DeepseekV4DecodeWrapper(model)
    else:
        wrapper = DeepseekV4MTPDecodeWrapper(model)
    return _compile_deepseek_v4_wrapper(
        wrapper,
        graph_mode,
        enable_cache_compile=enable_cache_compile,
        cache_name=cache_name,
    )


def forward_prefill_mojo(model, input_ids, attention_mask, lengths, runtime_state=None, use_attn_metadata=True):
    if runtime_state is not None:
        prefill_meta = runtime_state.prepare_prefill_inputs(input_ids, attention_mask=attention_mask, q_lens=None)
        logits, past_key_values, hidden_states = model(
            input_ids=prefill_meta["input_ids"],
            attention_mask=attention_mask,
            position_ids=prefill_meta["position_ids"],
            past_key_values=runtime_state.paged_cache,
            use_cache=True,
            is_prefill=True,
            attn_inputs=prefill_meta["attn_inputs"],
            attn_metadata=prefill_meta.get("attn_metadata") if use_attn_metadata else None,
            context_lens=prefill_meta["context_lens"],
            q_lens=prefill_meta["q_lens"],
        )
        return last_logits_from_output(logits, lengths), past_key_values, hidden_states
    else:
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True, is_prefill=True)
    logits, past_key_values = _extract_outputs(outputs)
    return last_logits_from_output(logits, lengths), past_key_values, None


def build_mtp_prefill_inputs(input_ids, attention_mask, next_token_id, pad_token_id):
    batch_size, seq_len = input_ids.shape
    mtp_input_ids = torch.full_like(input_ids, pad_token_id)
    mtp_attention_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    valid_lens = attention_mask.to(dtype=torch.int32).sum(dim=-1).to(dtype=torch.long)

    max_history_len = int((valid_lens.max() + 1).item()) if valid_lens.numel() > 0 else 0
    confirmed_generate_ids = torch.full(
        (batch_size, max_history_len),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    for row in range(batch_size):
        valid_len = int(valid_lens[row].item())
        if valid_len <= 0:
            raise ValueError("MTP prefill expects every request to have at least one valid input token.")
        shifted = torch.cat([input_ids[row, 1:valid_len], next_token_id[row]], dim=0)
        mtp_input_ids[row, : shifted.numel()] = shifted
        mtp_attention_mask[row, : shifted.numel()] = True

        history = torch.cat([input_ids[row, :valid_len], next_token_id[row]], dim=0)
        confirmed_generate_ids[row, max_history_len - history.numel():] = history

    return mtp_input_ids, mtp_attention_mask, confirmed_generate_ids


def forward_mtp_prefill_mojo(
    mtp_model,
    hidden_states,
    input_ids,
    attention_mask,
    runtime_state,
    *,
    max_seq_len,
    next_n,
    use_attn_metadata=True,
):
    cp_size = int(getattr(mtp_model, "cp_size", 1))
    if cp_size <= 1:
        mtp_prefill_meta = runtime_state.prepare_prefill_inputs(
            input_ids,
            attention_mask=attention_mask,
            q_lens=None,
        )
        mtp_out = mtp_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=mtp_prefill_meta["position_ids"],
            past_key_values=runtime_state.paged_cache,
            use_cache=True,
            is_prefill=True,
            attn_inputs=mtp_prefill_meta["attn_inputs"],
            attn_metadata=mtp_prefill_meta.get("attn_metadata") if use_attn_metadata else None,
            context_lens=mtp_prefill_meta["context_lens"],
            q_lens=mtp_prefill_meta["q_lens"],
            input_ids=input_ids,
            return_hidden=True,
        )
        return mtp_out

    local_batch = input_ids.shape[0]
    runtime_batch = int(runtime_state.paged_cache.batch_size)
    if runtime_batch != local_batch:
        raise ValueError(
            "MTP CP prefill runtime batch size mismatch: "
            f"runtime_batch={runtime_batch}, local_batch={local_batch}"
        )

    logits_chunks = []
    hidden_chunks = []
    past_key_values = runtime_state.paged_cache
    for local_slot in range(local_batch):
        minibatch_runtime = DeepseekMTPRuntimeState.from_model(
            mtp_model,
            batch_size=1,
            device=input_ids.device,
            max_seq_len=max_seq_len,
            next_n=next_n,
        )
        mtp_prefill_meta = minibatch_runtime.prepare_prefill_inputs(
            input_ids[local_slot: local_slot + 1],
            attention_mask=attention_mask[local_slot: local_slot + 1],
            q_lens=None,
        )
        mtp_logits, past_key_values, mtp_hidden = mtp_model(
            hidden_states=hidden_states[local_slot: local_slot + 1],
            attention_mask=attention_mask[local_slot: local_slot + 1],
            position_ids=mtp_prefill_meta["position_ids"],
            past_key_values=minibatch_runtime.paged_cache,
            use_cache=True,
            is_prefill=True,
            attn_inputs=mtp_prefill_meta["attn_inputs"],
            attn_metadata=mtp_prefill_meta.get("attn_metadata") if use_attn_metadata else None,
            context_lens=mtp_prefill_meta["context_lens"],
            q_lens=mtp_prefill_meta["q_lens"],
            input_ids=input_ids[local_slot: local_slot + 1],
            return_hidden=True,
        )
        copy_paged_dummy_cache_batch(runtime_state.paged_cache, minibatch_runtime.paged_cache, local_slot)
        logits_chunks.append(mtp_logits)
        hidden_chunks.append(mtp_hidden)
        del minibatch_runtime

    return torch.cat(logits_chunks, dim=0), past_key_values, torch.cat(hidden_chunks, dim=0)


def forward_mtp_cp_prefill_minibatch_mojo(
    mtp_model,
    hidden_states,
    input_ids,
    attention_mask,
    runtime_state,
    *,
    local_shard_start,
    local_shard_end,
    max_seq_len,
    next_n,
    use_attn_metadata=True,
):
    total_batch = input_ids.shape[0]
    local_batch = local_shard_end - local_shard_start
    runtime_batch = int(runtime_state.paged_cache.batch_size)
    if runtime_batch != local_batch:
        raise ValueError(
            "MTP Golden CP prefill runtime batch size mismatch: "
            f"runtime_batch={runtime_batch}, local_batch={local_batch}, "
            f"owner_range=[{local_shard_start}, {local_shard_end})"
        )
    if hidden_states is None or hidden_states.shape[0] != total_batch:
        raise ValueError(
            "MTP Golden CP prefill needs CP-local main hidden states for every global sample: "
            f"hidden_batch={None if hidden_states is None else hidden_states.shape[0]}, total_batch={total_batch}"
        )

    logits_chunks = []
    hidden_chunks = []
    past_key_values = runtime_state.paged_cache
    for global_batch_idx in range(total_batch):
        owns_sample = local_shard_start <= global_batch_idx < local_shard_end
        minibatch_runtime = DeepseekMTPRuntimeState.from_model(
            mtp_model,
            batch_size=1,
            device=input_ids.device,
            max_seq_len=max_seq_len,
            next_n=next_n,
        )
        mtp_prefill_meta = minibatch_runtime.prepare_prefill_inputs(
            input_ids[global_batch_idx: global_batch_idx + 1],
            attention_mask=attention_mask[global_batch_idx: global_batch_idx + 1],
            q_lens=None,
        )
        mtp_logits, past_key_values, mtp_hidden = mtp_model(
            hidden_states=hidden_states[global_batch_idx: global_batch_idx + 1],
            attention_mask=attention_mask[global_batch_idx: global_batch_idx + 1],
            position_ids=mtp_prefill_meta["position_ids"],
            past_key_values=minibatch_runtime.paged_cache,
            use_cache=True,
            is_prefill=True,
            attn_inputs=mtp_prefill_meta["attn_inputs"],
            attn_metadata=mtp_prefill_meta.get("attn_metadata") if use_attn_metadata else None,
            context_lens=mtp_prefill_meta["context_lens"],
            q_lens=mtp_prefill_meta["q_lens"],
            input_ids=input_ids[global_batch_idx: global_batch_idx + 1],
            return_hidden=True,
        )
        if owns_sample:
            local_slot = global_batch_idx - local_shard_start
            copy_paged_dummy_cache_batch(runtime_state.paged_cache, minibatch_runtime.paged_cache, local_slot)
            logits_chunks.append(mtp_logits)
            hidden_chunks.append(mtp_hidden)
        del minibatch_runtime

    if not logits_chunks:
        raise RuntimeError(
            "MTP Golden CP prefill did not collect owner logits. "
            f"owner_range=[{local_shard_start}, {local_shard_end}), total_batch={total_batch}"
        )
    return torch.cat(logits_chunks, dim=0), past_key_values, torch.cat(hidden_chunks, dim=0)


def forward_main_decode_mojo(
    model,
    token_ids,
    past_key_values,
    *,
    runtime_state=None,
    decode_fn=None,
    expected_seq_len=None,
    use_attn_metadata=True,
    advance_cache=True,
    return_last_logits=True,
    skip_guard_eval=False,
):
    """Run the main model decode path.

    The same main-model decode call is used for normal decode and MTP verify.
    MTP verify passes a wider token window and delays cache advancement until
    accepted tokens are known.
    """
    if runtime_state is None:
        outputs = model(token_ids, past_key_values=past_key_values, use_cache=True, is_prefill=False)
        logits, past_key_values = _extract_outputs(outputs)
        hidden_states = None
    else:
        prepare_kwargs = {}
        if expected_seq_len is not None:
            prepare_kwargs["expected_seq_len"] = expected_seq_len
        decode_meta = runtime_state.prepare_decode_inputs(
            token_ids,
            allow_decode_fast_path=(
                token_ids.shape[1] == 1 or
                expected_seq_len is not None
            ),
            **prepare_kwargs,
        )
        attn_metadata = decode_meta.get("attn_metadata") if use_attn_metadata else None

        if decode_fn is not None:
            with _skip_guard_eval_context(skip_guard_eval):
                logits, past_key_values, hidden_states = decode_fn(
                    token_ids,
                    runtime_state.paged_cache,
                    decode_meta["position_ids"],
                    decode_meta["context_lens"],
                    decode_meta["attn_inputs"],
                    attn_metadata,
                )
        else:
            logits, past_key_values, hidden_states = model(
                token_ids,
                past_key_values=runtime_state.paged_cache,
                position_ids=decode_meta["position_ids"],
                context_lens=decode_meta["context_lens"],
                attn_inputs=decode_meta["attn_inputs"],
                attn_metadata=attn_metadata,
                use_cache=True,
                is_prefill=False,
            )
        if advance_cache:
            runtime_state.post_decode_step(seq_len=token_ids.shape[1])

    if return_last_logits:
        logits = logits[:, -1, :]
    return logits, past_key_values, hidden_states


def forward_mtp_decode_mojo(
    mtp_model,
    hidden_states,
    input_ids,
    runtime_state,
    decode_fn=None,
    skip_guard_eval=False,
):
    """Run the MTP draft model decode path.

    This is intentionally separate from main decode: the MTP model consumes the
    previous main/MTP hidden states and advances its own draft cache.
    """
    decode_meta = runtime_state.prepare_decode_inputs(input_ids, allow_decode_fast_path=True)
    attn_metadata = decode_meta.get("attn_metadata")

    if decode_fn is not None:
        with _skip_guard_eval_context(skip_guard_eval):
            logits, past_key_values, mtp_hidden = decode_fn(
                hidden_states,
                runtime_state.paged_cache,
                decode_meta["position_ids"],
                decode_meta["context_lens"],
                decode_meta["q_lens"],
                input_ids,
                decode_meta["attn_inputs"],
                attn_metadata,
            )
    else:
        logits, past_key_values, mtp_hidden = mtp_model(
            hidden_states=hidden_states,
            position_ids=decode_meta["position_ids"],
            past_key_values=runtime_state.paged_cache,
            use_cache=True,
            is_prefill=False,
            attn_inputs=decode_meta["attn_inputs"],
            attn_metadata=attn_metadata,
            context_lens=decode_meta["context_lens"],
            q_lens=decode_meta["q_lens"],
            input_ids=input_ids,
            return_hidden=True,
        )

    if getattr(runtime_state.paged_cache, "next_n", 1) == 1:
        runtime_state.post_decode_step(seq_len=input_ids.shape[1])
    else:
        runtime_state.post_decode_step(seq_len=1)
    return logits, past_key_values, mtp_hidden

def generate(model, tokenizer, prompt, max_new_tokens, device, ep_size=1, cp_size=1, batch_size=1, graph_mode="eager", prof=False,
             use_attn_metadata=True, mtp_model=None, mtp_runtime_state=None, next_n=0, prof_prefill=False):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    os.environ["MOJO_BUILD_LEGACY_ATTN_INPUTS"] = "0" if use_attn_metadata else "1"

    use_mtp = next_n > 0 and mtp_model is not None
    spec_len = next_n + 1  # speculative length = next_n + 1

    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    is_main = (global_rank == 0)
    moe_ep_group = getattr(model, 'moe_ep_group', None)
    sync_mtp_tokens_across_ep = (
        ep_size > 1
        and dist.is_initialized()
        and model.__class__.__name__ != "DeepseekV4ForCausalLM"
    )
    lmhead_tp_size = getattr(model, "lmhead_tp_size", 1)
    attn_tp_size = getattr(model, "attn_tp_size", 1)
    shard_start = 0
    shard_end = 0
    attn_dp_size = 1
    dp_rank = 0
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
    full_input_ids, full_attention_mask, full_lengths = pad_batch(encoded, pad_token_id, device)
    if model.__class__.__name__ == "DeepseekV4ForCausalLM":
        shard_start, shard_end, attn_dp_size, dp_rank = get_attn_dp_shard_range(
            batch_size,
            global_rank=global_rank,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            attn_tp_size=attn_tp_size,
            cp_size=cp_size,
        )
        full_input_ids, full_attention_mask = pad_deepseek_v4_cp_prefill(
            full_input_ids, full_attention_mask, pad_token_id, cp_size
        )
        input_ids, attention_mask, lengths, prompts, rendered = shard_batch_for_attn_dp(
            full_input_ids,
            full_attention_mask,
            full_lengths,
            prompts,
            rendered,
            global_rank=global_rank,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            attn_tp_size=attn_tp_size,
            cp_size=cp_size,
        )
        batch_size = input_ids.shape[0]
    else:
        input_ids, attention_mask, lengths = full_input_ids, full_attention_mask, full_lengths

    if use_mtp:
        runtime_batch_size = getattr(getattr(mtp_runtime_state, "paged_cache", None), "batch_size", None)
        if runtime_batch_size != batch_size:
            mtp_runtime_state = DeepseekMTPRuntimeState.from_model(
                mtp_model,
                batch_size=batch_size,
                device=device,
                max_seq_len=4096,
                next_n=next_n,
            )

    _mem_snapshot("after input preparation")

    should_print = is_main
    if dist.is_initialized() and model.__class__.__name__ == "DeepseekV4ForCausalLM":
        should_print = (global_rank % attn_tp_size) == 0
    enable_cache_compile = (
        graph_mode == "npugraph_ex"
        and os.getenv("MOJO_ENABLE_CACHE_COMPILE", "0") == "1"
    )

    if should_print:
        print(
            f"[ATTN_DP_SHARD] rank={global_rank} dp_rank={dp_rank}/{attn_dp_size} "
            f"attn_tp_size={attn_tp_size} lmhead_tp_size={lmhead_tp_size} "
            f"shard=[{shard_start}, {shard_end}) decode_local_batch={batch_size} "
            f"prefill_batch={full_input_ids.shape[0] if model.__class__.__name__ == 'DeepseekV4ForCausalLM' and cp_size > 1 else batch_size}",
            flush=True,
        )
        for i in range(batch_size):
            print(f"\n[Batch {shard_start + i}] Prompt: {prompts[i]}")
        print(f"Graph mode: {graph_mode}")
        print(f"Enable cache_compile: {enable_cache_compile}")
        print(f"Use attn_metadata: {use_attn_metadata}")
        if use_mtp:
            print(f"MTP enabled: next_n={next_n}, spec_len={spec_len}")
            print(f"[MTP] Runtime local batch_size={batch_size}")
        print("-" * 40)

    torch.npu.reset_peak_memory_stats()
    _mem_snapshot("before prefill (peak reset)")

    runtime_state = None
    decode_fn = None
    mtp_decode_fn = None
    enable_main_cache_compile = enable_cache_compile
    enable_mtp_cache_compile = enable_cache_compile and use_mtp
    main_decode_graph_warmed = False
    mtp_decode_graph_warmed = False
    if model.__class__.__name__ == "DeepseekV4ForCausalLM":
        runtime_state = DeepseekSparseAttentionRuntimeState.from_model(
            model, batch_size=batch_size,
            max_seq_len=max(input_ids.shape[1] * 4, 4096),
        )
        if should_print:
            print("[RUNTIME] DeepseekSparseAttentionRuntimeState created before prefill")
        if graph_mode != "eager":
            if should_print:
                if use_mtp:
                    print(f"[GRAPH] Compiling main decode and MTP decode with {graph_mode} backend...")
                else:
                    print(f"[GRAPH] Compiling decode with {graph_mode} backend...")
            compile_t0 = time.time()
            decode_fn = compile_deepseek_v4_decode(
                model,
                graph_mode,
                enable_cache_compile=enable_main_cache_compile,
                cache_name=f"main_decode_hidden_next{next_n}" if use_mtp else "main_decode",
            )
            if use_mtp:
                mtp_decode_fn = compile_deepseek_v4_decode(
                    mtp_model,
                    graph_mode,
                    enable_cache_compile=enable_mtp_cache_compile,
                    cache_name=f"mtp_decode_next{next_n}",
                )
            if should_print:
                print(f"[GRAPH] Decode graph callable prepared in {time.time() - compile_t0:.2f}s (actual compilation happens on first call)")

    # ==================== Prefill ====================
    prefill_prof_ctx = None
    if prof_prefill and is_main:
        import torch_npu.profiler as prof_mod
        prefill_prof_dir = os.getenv(
            "MOJO_PROF_PREFILL_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling_prefill"),
        )
        prefill_prof_ctx = prof_mod.profile(
            activities=[prof_mod.ProfilerActivity.CPU, prof_mod.ProfilerActivity.NPU],
            # schedule=prof_mod.schedule(wait=0, warmup=0, active=prof_steps, repeat=1),
            on_trace_ready=prof_mod.tensorboard_trace_handler(prefill_prof_dir),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )
        prefill_prof_ctx.__enter__()
        print(f"[PROF] Start profiling prefill, output -> {prefill_prof_dir}", flush=True)

    with torch.no_grad():
        prev_hidden_all_for_mtp = None
        use_cp_minibatch_prefill = (
            model.__class__.__name__ == "DeepseekV4ForCausalLM"
            and runtime_state is not None
            and cp_size > 1
        )
        if use_cp_minibatch_prefill:
            cp_prefill_out = forward_cp_prefill_minibatch_mojo(
                model,
                full_input_ids,
                full_attention_mask,
                full_lengths,
                runtime_state,
                local_shard_start=shard_start,
                local_shard_end=shard_end,
                max_seq_len=max(input_ids.shape[1] * 4, 4096),
                runtime_state_factory=DeepseekSparseAttentionRuntimeState.from_model,
                forward_prefill_fn=forward_prefill_mojo,
                use_attn_metadata=use_attn_metadata,
                return_all_hidden=use_mtp,
            )
            if use_mtp:
                next_token_logits, past_key_values, prev_hidden, prev_hidden_all_for_mtp = cp_prefill_out
            else:
                next_token_logits, past_key_values, prev_hidden = cp_prefill_out
        else:
            next_token_logits, past_key_values, prev_hidden = forward_prefill_mojo(
                model, input_ids, attention_mask, lengths, runtime_state=runtime_state,
                use_attn_metadata=use_attn_metadata,
            )
        if not use_mtp:
            prev_hidden = None

    if prefill_prof_ctx is not None:
        prefill_prof_ctx.__exit__(None, None, None)
        prefill_prof_ctx = None
        print(f"[PROF] Prefill profiling done, output -> {prefill_prof_dir}", flush=True)

    _mem_snapshot("after prefill")

    # Sample first token (greedy)
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [B, 1]
    if use_mtp and cp_size > 1 and model.__class__.__name__ == "DeepseekV4ForCausalLM":
        full_next_token_id = gather_decode_shard_tensor(
            next_token_id,
            full_input_ids.shape[0],
            shard_start,
            attn_dp_size,
        )
    else:
        full_next_token_id = next_token_id

    if dist.is_initialized() and model.__class__.__name__ != "DeepseekV4ForCausalLM":
        if lmhead_tp_size > 1:
            lmhead_tp_group = getattr(model, "hccl_comm_dict", {}).get("lmhead_tp_group")
            src_rank = (global_rank // lmhead_tp_size) * lmhead_tp_size
            dist.broadcast(next_token_id, src=src_rank, group=lmhead_tp_group)
        elif ep_size > 1:
            dist.broadcast(next_token_id, src=0, group=moe_ep_group)

    generated = [[int(x)] for x in next_token_id.squeeze(-1).detach().cpu().tolist()]

    # ==================== MTP State ====================
    mtp_spec_tokens = None  # [B, next_n] spec tokens from MTP model
    confirmed_prev_hidden = None  # [B, spec_len, H] confirmed hidden history
    mtp_prev_hidden = None  # [B, spec_len, H] working hidden history for the current MTP round
    confirmed_generate_ids = None  # [B, S] confirmed history
    mtp_generate_ids = None  # [B, S] working draft history for the current MTP round
    mtp_total_accepted = 0
    mtp_total_spec = 0
    mtp_step_count = 0
    # Save MTP cache seq_lens before each round of speculation for rollback
    mtp_seq_lens_cached = None

    if use_mtp:
        mtp_state = initialize_mtp_speculation_after_prefill(
            mtp_model=mtp_model,
            mtp_runtime_state=mtp_runtime_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            full_input_ids=full_input_ids,
            full_attention_mask=full_attention_mask,
            next_token_id=next_token_id,
            full_next_token_id=full_next_token_id,
            prev_hidden=prev_hidden,
            prev_hidden_all_for_mtp=prev_hidden_all_for_mtp,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            cp_size=cp_size,
            shard_start=shard_start,
            shard_end=shard_end,
            spec_len=spec_len,
            next_n=next_n,
            use_attn_metadata=use_attn_metadata,
            sync_tokens=sync_mtp_tokens_across_ep,
            moe_ep_group=moe_ep_group,
            mtp_decode_fn=mtp_decode_fn,
        )
        input_ids = mtp_state["input_ids"]
        mtp_spec_tokens = mtp_state["mtp_spec_tokens"]
        confirmed_prev_hidden = mtp_state["confirmed_prev_hidden"]
        mtp_prev_hidden = mtp_state["mtp_prev_hidden"]
        confirmed_generate_ids = mtp_state["confirmed_generate_ids"]
        mtp_generate_ids = mtp_state["mtp_generate_ids"]
        mtp_seq_lens_cached = mtp_state["mtp_seq_lens_cached"]
        if is_main:
            print(f"[MTP] Prefill end: generated {mtp_spec_tokens.shape[-1]} spec token(s)")
    else:
        input_ids = next_token_id

    warmup_steps = int(os.getenv("MOJO_PROF_WARMUP_STEPS", "3"))
    prof_steps = int(os.getenv("MOJO_PROF_ACTIVE_STEPS", "10"))
    prof_active = False
    prof_ctx = None
    try:
        for step in range(max_new_tokens - 1):
            if prof and step == warmup_steps and is_main:
                import torch_npu.profiler as prof_mod
                prof_dir = os.getenv(
                    "MOJO_PROF_DIR",
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling_decode"),
                )
                prof_ctx = prof_mod.profile(
                    activities=[prof_mod.ProfilerActivity.CPU, prof_mod.ProfilerActivity.NPU],
                    schedule=prof_mod.schedule(wait=0, warmup=0, active=prof_steps, repeat=1),
                    on_trace_ready=prof_mod.tensorboard_trace_handler(prof_dir),
                    record_shapes=True,
                    with_stack=False,
                    with_flops=True,
                )
                prof_ctx.__enter__()
                prof_active = True
                print(f"[PROF] Start profiling decode at step {step}, output -> {prof_dir}", flush=True)

            # ==================== Main Model Decode ====================
            with torch.no_grad():
                if use_mtp:
                    main_seq_lens_before_verify = runtime_state.paged_cache.seq_lens.clone()
                    verify_input_ids = (
                        input_ids.contiguous().reshape(batch_size, spec_len)
                        if next_n > 1 else input_ids
                    )
                    all_logits, past_key_values, main_hidden = forward_main_decode_mojo(
                        model,
                        verify_input_ids,
                        past_key_values,
                        runtime_state=runtime_state,
                        decode_fn=decode_fn,
                        expected_seq_len=spec_len if next_n > 1 else None,
                        use_attn_metadata=use_attn_metadata,
                        advance_cache=False,
                        return_last_logits=False,
                        skip_guard_eval=(
                            enable_main_cache_compile and
                            decode_fn is not None and
                            main_decode_graph_warmed
                        ),
                    )
                    if enable_main_cache_compile and decode_fn is not None:
                        main_decode_graph_warmed = True
                else:
                    next_token_logits, past_key_values, _ = forward_main_decode_mojo(
                        model,
                        input_ids,
                        past_key_values,
                        runtime_state=runtime_state,
                        decode_fn=decode_fn,
                        use_attn_metadata=use_attn_metadata,
                    )

            if step == 0 and decode_fn is not None and should_print:
                print("[GRAPH] First decode step done (compilation triggered on this step)")
                print(
                    f"[MEM] After graph compile: allocated={torch.npu.memory_allocated()/1e9:.2f} GB, "
                    f"peak={torch.npu.max_memory_allocated()/1e9:.2f} GB, "
                    f"reserved={torch.npu.memory_reserved()/1e9:.2f} GB",
                    flush=True,
                )

            if prof_active:
                prof_ctx.step()

            if prof_active and (step - warmup_steps + 1) >= prof_steps:
                prof_ctx.__exit__(None, None, None)
                prof_ctx = None
                prof_active = False
                print(f"[PROF] Stop profiling at step {step}, collected {prof_steps} decode steps", flush=True)

            # ==================== MTP Verify & Speculate ====================
            if use_mtp and mtp_spec_tokens is not None:
                main_greedy, accepted_num = verify_mtp_spec_tokens(
                    mtp_spec_tokens,
                    all_logits,
                    next_n,
                    sync_tokens=sync_mtp_tokens_across_ep,
                    group=moe_ep_group,
                )

                if is_main and step < 3:
                    print(
                        f"[MTP DEBUG step={step}] spec_tokens={mtp_spec_tokens[0].tolist()}, "
                        f"main_greedy={main_greedy[0, :next_n].tolist()}, accepted={accepted_num[0].item()}",
                        flush=True,
                    )

                mtp_total_accepted += accepted_num.sum().item()
                mtp_total_spec += next_n * batch_size
                mtp_step_count += 1

                mtp_state = update_mtp_after_verify(
                    main_greedy=main_greedy,
                    main_hidden=main_hidden,
                    accepted_num=accepted_num,
                    generated=generated,
                    runtime_state=runtime_state,
                    mtp_runtime_state=mtp_runtime_state,
                    main_seq_lens_before_verify=main_seq_lens_before_verify,
                    mtp_seq_lens_cached=mtp_seq_lens_cached,
                    confirmed_prev_hidden=confirmed_prev_hidden,
                    confirmed_generate_ids=confirmed_generate_ids,
                    pad_token_id=pad_token_id,
                    next_n=next_n,
                    batch_size=batch_size,
                    spec_len=spec_len,
                    sync_tokens=sync_mtp_tokens_across_ep,
                    moe_ep_group=moe_ep_group,
                )
                next_token_id = mtp_state["next_token_id"]
                confirmed_prev_hidden = mtp_state["confirmed_prev_hidden"]
                mtp_prev_hidden = mtp_state["mtp_prev_hidden"]
                confirmed_generate_ids = mtp_state["confirmed_generate_ids"]
                mtp_generate_ids = mtp_state["mtp_generate_ids"]
                mtp_seq_lens_cached = mtp_state["mtp_seq_lens_cached"]

                mtp_spec_tokens, mtp_prev_hidden, mtp_generate_ids, mtp_decode_graph_warmed = mtp_post_process(
                    mtp_model,
                    mtp_runtime_state,
                    mtp_prev_hidden,
                    mtp_generate_ids,
                    next_n=next_n,
                    spec_len=spec_len,
                    batch_size=batch_size,
                    mtp_decode_fn=mtp_decode_fn,
                    mtp_decode_graph_warmed=mtp_decode_graph_warmed,
                    enable_mtp_cache_compile=enable_mtp_cache_compile,
                    sync_tokens=sync_mtp_tokens_across_ep,
                    moe_ep_group=moe_ep_group,
                )

                input_ids = torch.cat([next_token_id, mtp_spec_tokens], dim=-1)
                if sync_mtp_tokens_across_ep:
                    input_ids = input_ids.contiguous()
                    dist.broadcast(input_ids, src=0, group=moe_ep_group)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                if dist.is_initialized() and model.__class__.__name__ != "DeepseekV4ForCausalLM":
                    if lmhead_tp_size > 1:
                        lmhead_tp_group = getattr(model, "hccl_comm_dict", {}).get("lmhead_tp_group")
                        src_rank = (global_rank // lmhead_tp_size) * lmhead_tp_size
                        dist.broadcast(next_token_id, src=src_rank, group=lmhead_tp_group)
                    elif ep_size > 1:
                        dist.broadcast(next_token_id, src=0, group=moe_ep_group)

                token_list = next_token_id.squeeze(-1).detach().cpu().tolist()
                for row, token_id in enumerate(token_list):
                    generated[row].append(int(token_id))

                input_ids = next_token_id
    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

    _mem_snapshot("after all decode steps")

    if should_print:
        print("-" * 40)
        for i, output_ids in enumerate(generated):
            res = tokenizer.decode(output_ids, skip_special_tokens=False)
            if tokenizer.eos_token in res:
                res = res.split(tokenizer.eos_token)[0]
            print(f"[Batch {shard_start + i}] Generated text: {res}")
        if use_mtp and mtp_step_count > 0:
            avg_accept = mtp_total_accepted / mtp_total_spec if mtp_total_spec > 0 else 0
            print(f"[MTP Stats] avg acceptance rate: {avg_accept:.4f} ({mtp_total_accepted}/{mtp_total_spec}) over {mtp_step_count} steps")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("QWEN3_MODEL_PATH", ""))
    parser.add_argument("--device", type=str, default=os.getenv("QWEN3_DEVICE", "npu"))
    parser.add_argument("--num_layers", type=int, default=int(os.getenv("QWEN3_NUM_LAYERS", "36")))
    parser.add_argument("--prompt", type=str, default="今天天气怎么样？")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--pa_max_length", type=int, default=int(os.getenv("PA_MAX_LENGTH", "2048")))
    parser.add_argument("--transformers", action="store_true", help="Use Transformers model")
    parser.add_argument("--ep_size", type=int, default=int(os.getenv("EP_SIZE", "8")))
    parser.add_argument("--cp_size", type=int, default=int(os.getenv("CP_SIZE", "8")))
    parser.add_argument("--attn_tp_size", type=int, default=int(os.getenv("ATTN_TP_SIZE", "1")))
    parser.add_argument("--lmhead_tp_size", type=int, default=int(os.getenv("LMHEAD_TP_SIZE", "1")))
    parser.add_argument("--o_proj_tp_size", type=int, default=int(os.getenv("O_PROJ_TP_SIZE", "1")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "1")),
                        help="Batch size for inference (default: 1, single-batch)")
    parser.add_argument("--prof", action="store_true", default=bool(int(os.getenv("MOJO_PROF", "0"))),
                        help="Enable NPU profiling for decode")
    parser.add_argument("--prof_prefill", action="store_true", default=bool(int(os.getenv("MOJO_PROF_PREFILL", "0"))),
                        help="Enable NPU profiling for prefill")
    parser.add_argument("--graph_mode", type=str, default=os.getenv("MOJO_GRAPH_MODE", "eager"),
                        choices=["eager", "npugraph_ex", "ge_graph"], help="Graph compilation mode for decode")
    parser.add_argument("--use_attn_metadata", type=int, choices=[0, 1],
                        default=int(os.getenv("MOJO_USE_ATTN_METADATA", "1")),
                        help="Use runtime attn_metadata in DeepSeek-V4 attention path (default: 1)")
    parser.add_argument("--next_n", type=int, default=int(os.getenv("NEXT_N", "0")),
                        help="Number of MTP speculative tokens (0=disabled, 1-3 enabled)")
    parser.add_argument("--use_parallelize_module_tp", type=int, choices=[0, 1],
                        default=int(os.getenv("MOJO_USE_PARALLELIZE_MODULE_TP", "0")),
                        help="Use distributed.parallel mojo_parallelize_module for TP plan")
    parser.add_argument("--use_parallelize_module_ep", type=int, choices=[0, 1],
                        default=int(os.getenv("MOJO_USE_PARALLELIZE_MODULE_EP", "0")),
                        help="Use distributed.parallel mojo_parallelize_module for DeepSeekV4 MoE EP binding")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path and not os.getenv("QWEN3_MODEL_PATH"):
        raise ValueError("Please pass --model_path or set QWEN3_MODEL_PATH")

    local_rank, global_rank, world_size = init_distributed()
    ep_size = args.ep_size
    cp_size = args.cp_size
    attn_tp_size = args.attn_tp_size
    lmhead_tp_size = args.lmhead_tp_size
    o_proj_tp_size = args.o_proj_tp_size
    parallel_layout_config = LLMParallelConfig(
        ep_size=ep_size,
        cp_size=cp_size,
        attn_tp_size=attn_tp_size,
        lmhead_tp_size=lmhead_tp_size,
        o_proj_tp_size=o_proj_tp_size,
    )
    parallel_layout_config.validate(world_size, require_pure_ep=True)
    ep_rank = parallel_layout_config.ep_rank(global_rank)
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
        hf_config.next_n = args.next_n
        logger.info(
            "[CONFIG] DeepSeek-V4 runtime config: world_size=%s ep_size=%s cp_size=%s "
            "attn_tp_size=%s lmhead_tp_size=%s o_proj_tp_size=%s pa_max_length=%s next_n=%s",
            world_size,
            ep_size,
            cp_size,
            attn_tp_size,
            lmhead_tp_size,
            o_proj_tp_size,
            args.pa_max_length,
            args.next_n,
        )

        parallel_config = parallel_layout_config.as_model_parallel_config(world_size)
        parallel_config["use_parallelize_module_tp"] = int(args.use_parallelize_module_tp)
        parallel_config["use_parallelize_module_ep"] = int(args.use_parallelize_module_ep)

    _mem_snapshot("before model construct", npu_device_idx)
    _device_snapshot(
        "before model construct",
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        ep_rank=ep_rank if not args.transformers else None,
        target_device=f"npu:{npu_device_idx}",
    )

    if hasattr(model_class, "load_weights"):
        from transformers.modeling_utils import no_init_weights
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with no_init_weights():
                model = model_class(
                    hf_config,
                    num_layers=args.num_layers,
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    global_rank=global_rank,
                    parallel_config=parallel_config,
                )
        finally:
            torch.set_default_dtype(origin_dtype)
        _mem_snapshot("after model construct (CPU)", npu_device_idx)
        _device_snapshot(
            "before model.to(npu)",
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            ep_rank=ep_rank,
            target_device=f"npu:{npu_device_idx}",
        )
        model_to_start = time.perf_counter()
        model = model.to(f"npu:{npu_device_idx}").eval()
        logger.info(
            "[DEVICE] after model.to(npu) | rank=%s local_rank=%s elapsed=%.3fs current_device=%s",
            global_rank,
            local_rank,
            time.perf_counter() - model_to_start,
            torch.npu.current_device(),
        )
        _mem_snapshot("after model.to(npu)", npu_device_idx)

        if dist.is_initialized():
            _device_snapshot(
                "before init_parallel_comm_group",
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                ep_rank=ep_rank,
                target_device=f"npu:{npu_device_idx}",
            )
            init_group_start = time.perf_counter()
            model.init_parallel_comm_group()
            model.set_ep_group(bind_moe=not args.use_parallelize_module_ep)
            logger.info(
                "[DEVICE] after init_parallel_comm_group | rank=%s local_rank=%s elapsed=%.3fs current_device=%s",
                global_rank,
                local_rank,
                time.perf_counter() - init_group_start,
                torch.npu.current_device(),
            )
        _mem_snapshot("after init_parallel_comm_group", npu_device_idx)

        _device_snapshot(
            "before load_weights",
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            ep_rank=ep_rank,
            target_device=f"npu:{npu_device_idx}",
        )
        load_weights_start = time.perf_counter()
        try:
            model_class.load_weights(model, args.model_path)
        except Exception:
            logger.exception(
                "[DEVICE] load_weights failed | rank=%s local_rank=%s elapsed=%.3fs traceback=%s",
                global_rank,
                local_rank,
                time.perf_counter() - load_weights_start,
                traceback.format_exc(),
            )
            raise
        logger.info(
            "[DEVICE] after load_weights | rank=%s local_rank=%s elapsed=%.3fs current_device=%s",
            global_rank,
            local_rank,
            time.perf_counter() - load_weights_start,
            torch.npu.current_device(),
        )
        _mem_snapshot("after load_weights", npu_device_idx)
        if args.use_parallelize_module_tp or args.use_parallelize_module_ep:
            model = apply_llm_parallelize_plan(
                model,
                parallel_layout_config,
                enable_lmhead_tp=True,
                enable_attention_tp=True,
                enable_moe_ep=bool(args.use_parallelize_module_ep),
                device_type="npu",
            )
            _mem_snapshot("after apply_llm_parallelize_plan", npu_device_idx)
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

    # ==================== MTP Model Setup ====================
    mtp_model = None
    mtp_runtime_state = None
    next_n = args.next_n

    if next_n > 0 and model.__class__.__name__ == "DeepseekV4ForCausalLM":
        from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import DeepseekV4ForMTP

        is_main = (global_rank == 0) if dist.is_initialized() else True
        if is_main:
            print(f"[MTP] Creating MTP model with next_n={next_n}...")

        model.config.num_nextn_predict_layers = next_n
        model.config.next_n = next_n

        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        from transformers.modeling_utils import no_init_weights
        with no_init_weights():
            mtp_model = DeepseekV4ForMTP(model.config, ep_size=ep_size, ep_rank=ep_rank)
        torch.set_default_dtype(origin_dtype)
        mtp_model = mtp_model.to(f"npu:{npu_device_idx}").eval()
        _mem_snapshot("after MTP model construct", npu_device_idx)

        # Share embed_tokens and lm_head from main model to MTP model
        mtp_model.model.embed_tokens = model.model.embed_tokens
        mtp_model.lm_head = model.lm_head
        mtp_model.attn_dp_size = getattr(model, "attn_dp_size", 1)
        mtp_model.prefill_dp_size = getattr(model, "prefill_dp_size", 1)
        mtp_model.lmhead_tp_size = getattr(model, "lmhead_tp_size", 1)
        mtp_model.local_vocab_size = getattr(model, "local_vocab_size", model.config.vocab_size)
        mtp_model.cp_size = getattr(model, "cp_size", cp_size)
        mtp_model.global_rank = getattr(model, "global_rank", global_rank)
        mtp_model.hccl_comm_dict = model.hccl_comm_dict.copy()
        if args.use_parallelize_module_ep:
            mtp_model.set_ep_group(bind_moe=False)

        DeepseekV4ForMTP.load_weights(mtp_model, args.model_path, main_model=model)
        _mem_snapshot("after MTP load_weights", npu_device_idx)

        if ep_size > 1 and dist.is_initialized():
            # CRITICAL: MTP model must reuse the main model's hccl_comm_dict (EP communication groups).
            # Creating independent comm groups via dist.new_group() causes HCCL communication mismatch
            # between MTP and main model, leading to npu_moe_distribute_dispatch_v2 aicore timeout (507014).
            # This matches note: behavior where comm groups are implicitly shared via caching.
            if not args.use_parallelize_module_ep:
                mtp_model.set_ep_group(bind_moe=True)
            # Only smooth_scale_1 needs all_gather (used by moe_init_routing_v2 before all_to_all,
            # indexed by global expert_num=256). smooth_scale_2 must stay local [experts_per_rank, ...]
            # (used by npu_dequant_swiglu_quant after all_to_all, indexed by local group_index len=32).
            # This matches the main model's load_weights behavior (L2759-2771).
            moe_ep_group = mtp_model.hccl_comm_dict.get("moe_ep_group")
            if moe_ep_group is not None:
                for layer_key, layer in mtp_model.model.layers.items():
                    mlp = layer.mlp
                    if hasattr(mlp, 'smooth_scale_1') and mlp.smooth_scale_1 is not None:
                        all_smooth_scale_1 = mlp.smooth_scale_1.data.new_empty(
                            mlp.smooth_scale_1.data.shape[0] * ep_size,
                            mlp.smooth_scale_1.data.shape[1])
                        dist.all_gather_into_tensor(
                            all_smooth_scale_1, mlp.smooth_scale_1.data,
                            group=moe_ep_group)
                        mlp.smooth_scale_1.data = all_smooth_scale_1

        if args.use_parallelize_module_ep:
            mtp_model = apply_llm_parallelize_plan(
                mtp_model,
                parallel_layout_config,
                enable_lmhead_tp=False,
                enable_attention_tp=False,
                enable_moe_ep=True,
                device_type="npu",
            )

        if is_main:
            print("[MTP] MTP model created; runtime state will be created after local batch sharding")

    generate(
        model, tokenizer, args.prompt, args.max_new_tokens, f"npu:{npu_device_idx}",
        ep_size=ep_size, cp_size=cp_size, batch_size=args.batch_size, graph_mode=args.graph_mode, prof=args.prof,
        use_attn_metadata=bool(args.use_attn_metadata),
        mtp_model=mtp_model, mtp_runtime_state=mtp_runtime_state, next_n=next_n,
        prof_prefill=args.prof_prefill,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
