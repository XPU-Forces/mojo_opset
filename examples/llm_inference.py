import argparse
import contextlib
import importlib
import json
import os
import logging
import time
from typing import List

import torch
import torch_npu
import torch.distributed as dist

from transformers import AutoTokenizer

from mojo_opset.utils.hf_utils import _resolve_local_files_only
from mojo_opset.utils.hf_utils import build_model_from_hf
from mojo_opset.runtime import DeepseekSparseAttentionRuntimeState
from mojo_opset.runtime import DeepseekMTPRuntimeState

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
    if isinstance(outputs, tuple) and len(outputs) >= 2:
        return outputs[0], outputs[1]
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

    def forward(self, hidden_states, past_key_values, position_ids, context_lens, q_lens, input_ids, attn_inputs):
        logits, past_key_values, mtp_hidden = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_prefill=False,
            attn_inputs=attn_inputs,
            context_lens=context_lens,
            q_lens=q_lens,
            input_ids=input_ids,
            return_hidden=True,
        )
        return logits, past_key_values, mtp_hidden


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
        if enable_cache_compile:
            if cache_name is None:
                raise ValueError("cache_name must be provided when enable_cache_compile=True")
            cache_compile_options = {
                "frozen_parameter": True,
                "static_kernel_compile": False,
            }
            return torch.npu.npugraph_ex.inference.cache_compile(
                wrapper.forward,
                cache_dir=_npugraph_cache_dir(cache_name),
                dynamic=False,
                options=cache_compile_options,
            )

        graph_pool = torch.npu.graph_pool_handle()
        compile_options = {
           "frozen_parameter": True,
            "static_kernel_compile": True,
            "clone_input": False,
            "use_graph_pool": graph_pool,
        }
        return torch.compile(
            wrapper,
            dynamic=True,
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


def forward_decode_mojo(
    model,
    token_ids,
    past_key_values,
    decode_fn=None,
    runtime_state=None,
    use_attn_metadata=True,
    update_runtime_state=True,
    skip_guard_eval=False,
):
    if runtime_state is not None:
        decode_meta = runtime_state.prepare_decode_inputs(token_ids)
        attn_metadata = decode_meta.get("attn_metadata") if use_attn_metadata else None
        if decode_fn is not None:
            with _skip_guard_eval_context(skip_guard_eval):
                logits, past_key_values, hidden_states = decode_fn(
                    token_ids,
                    past_key_values,
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
        if update_runtime_state:
            runtime_state.post_decode_step(seq_len=token_ids.shape[1])
            return logits[:, -1, :], past_key_values, hidden_states
        return logits, past_key_values, hidden_states
    else:
        outputs = model(token_ids, past_key_values=past_key_values, use_cache=True, is_prefill=False)
    logits, past_key_values = _extract_outputs(outputs)
    return logits[:, -1, :], past_key_values, None


def forward_mtp_verify_mojo(
    model,
    input_ids,
    past_key_values,
    runtime_state,
    next_n,
    decode_fn=None,
    use_attn_metadata=True,
    skip_guard_eval=False,
):
    verify_meta = runtime_state.prepare_mtp_verify_inputs(input_ids, next_n)
    attn_metadata = verify_meta.get("attn_metadata") if use_attn_metadata else None
    if decode_fn is not None:
        with _skip_guard_eval_context(skip_guard_eval):
            logits, past_key_values, hidden_states = decode_fn(
                input_ids,
                past_key_values,
                verify_meta["position_ids"],
                verify_meta["context_lens"],
                verify_meta["attn_inputs"],
                attn_metadata,
            )
    else:
        logits, past_key_values, hidden_states = model(
            input_ids,
            past_key_values=runtime_state.paged_cache,
            position_ids=verify_meta["position_ids"],
            context_lens=verify_meta["context_lens"],
            attn_inputs=verify_meta["attn_inputs"],
            attn_metadata=attn_metadata,
            use_cache=True,
            is_prefill=False,
        )
    return logits, past_key_values, hidden_states


def forward_mtp_decode_mojo(
    mtp_model,
    hidden_states,
    input_ids,
    runtime_state,
    decode_fn=None,
    skip_guard_eval=False,
):
    decode_meta = runtime_state.prepare_decode_inputs(input_ids)
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
            )
    else:
        logits, past_key_values, mtp_hidden = mtp_model(
            hidden_states=hidden_states,
            position_ids=decode_meta["position_ids"],
            past_key_values=runtime_state.paged_cache,
            use_cache=True,
            is_prefill=False,
            attn_inputs=decode_meta["attn_inputs"],
            context_lens=decode_meta["context_lens"],
            q_lens=decode_meta["q_lens"],
            input_ids=input_ids,
            return_hidden=True,
        )
    # NEXT_N=1 follows the stable cache_compile path from mojo_opset_AI:
    # the MTP cache advances by the full input window. For NEXT_N>=2, keep the
    # note: draft semantics where each speculative step commits one token.
    if getattr(runtime_state.paged_cache, "next_n", 1) == 1:
        runtime_state.post_decode_step(seq_len=input_ids.shape[1])
    else:
        runtime_state.post_decode_step(seq_len=1)
    return logits, past_key_values, mtp_hidden


def generate(model, tokenizer, prompt, max_new_tokens, device, ep_size=1, batch_size=1, graph_mode="eager", prof=False,
             use_attn_metadata=True, mtp_model=None, mtp_runtime_state=None, next_n=0):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    os.environ["MOJO_BUILD_LEGACY_ATTN_INPUTS"] = "0" if use_attn_metadata else "1"

    use_mtp = next_n > 0 and mtp_model is not None and mtp_runtime_state is not None
    spec_len = next_n + 1  # speculative length = next_n + 1

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
        print(f"Graph mode: {graph_mode}")
        print(f"Use attn_metadata: {use_attn_metadata}")
        if use_mtp:
            print(f"MTP enabled: next_n={next_n}, spec_len={spec_len}")
        print("-" * 40)

    torch.npu.reset_peak_memory_stats()
    _mem_snapshot("before prefill (peak reset)")

    runtime_state = None
    decode_fn = None
    mtp_decode_fn = None
    mtp_decode_fns = None
    enable_mtp_cache_compile = graph_mode == "npugraph_ex" and use_mtp and next_n == 1
    main_decode_graph_warmed = False
    mtp_decode_graph_warmed = False
    mtp_decode_graph_warmed_steps = None
    if model.__class__.__name__ == "DeepseekV4ForCausalLM":
        runtime_state = DeepseekSparseAttentionRuntimeState.from_model(
            model, batch_size=batch_size,
            max_seq_len=max(input_ids.shape[1] * 4, 4096),
        )
        if is_main:
            print("[RUNTIME] DeepseekSparseAttentionRuntimeState created before prefill")
        if graph_mode != "eager":
            if is_main:
                if use_mtp:
                    print(f"[GRAPH] Compiling main decode and MTP decode with {graph_mode} backend...")
                else:
                    print(f"[GRAPH] Compiling decode with {graph_mode} backend...")
            compile_t0 = time.time()
            decode_fn = compile_deepseek_v4_decode(
                model,
                graph_mode,
                enable_cache_compile=enable_mtp_cache_compile,
                cache_name=f"main_decode_hidden_next{next_n}" if use_mtp else "main_decode",
            )
            if use_mtp:
                if next_n > 1:
                    mtp_decode_fns = [
                        compile_deepseek_v4_decode(
                            mtp_model,
                            graph_mode,
                            enable_cache_compile=False,
                            cache_name=f"mtp_decode_next{next_n}_step{step_idx}",
                        )
                        for step_idx in range(next_n)
                    ]
                    mtp_decode_graph_warmed_steps = [False] * next_n
                else:
                    mtp_decode_fn = compile_deepseek_v4_decode(
                        mtp_model,
                        graph_mode,
                        enable_cache_compile=enable_mtp_cache_compile,
                        cache_name=f"mtp_decode_next{next_n}",
                    )
            if is_main:
                print(f"[GRAPH] Decode graph callable prepared in {time.time() - compile_t0:.2f}s (actual compilation happens on first call)")

    # ==================== Prefill ====================
    with torch.no_grad():
        if use_mtp:
            next_token_logits, past_key_values, prev_hidden = forward_prefill_mojo(
                model, input_ids, attention_mask, lengths, runtime_state=runtime_state,
                use_attn_metadata=use_attn_metadata,
            )
        else:
            next_token_logits, past_key_values, _ = forward_prefill_mojo(
                model, input_ids, attention_mask, lengths, runtime_state=runtime_state,
                use_attn_metadata=use_attn_metadata,
            )
    _mem_snapshot("after prefill")

    # Sample first token (greedy)
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [B, 1]

    if ep_size > 1 and dist.is_initialized():
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
        # ---- MTP Prefill: generate initial spec tokens ----
        # Following note:: after main model prefill, append next_token to generate_ids,
        # then MTP prefill uses generate_ids[:, 1:] as input_ids and main_hidden as prev_hidden_states.
        # This means MTP sees the next_token that main model just sampled.
        confirmed_generate_ids = torch.cat([input_ids, next_token_id], dim=-1).contiguous()  # [B, S+1]
        mtp_generate_ids = confirmed_generate_ids
        mtp_input_ids = mtp_generate_ids[:, 1:]  # [B, S] (skip BOS, includes next_token)
        mtp_prev_hidden_for_prefill = prev_hidden  # [B, S, H] (all positions from main model)

        with torch.no_grad():
            if ep_size > 1 and dist.is_initialized():
                mtp_input_ids = mtp_input_ids.contiguous()
                dist.broadcast(mtp_input_ids, src=0, group=moe_ep_group)
            mtp_prefill_meta = mtp_runtime_state.prepare_prefill_inputs(
                mtp_input_ids, attention_mask=None,
                q_lens=torch.full((batch_size,), mtp_input_ids.shape[1], dtype=torch.int32, device=device),
            )
            mtp_out = mtp_model(
                hidden_states=mtp_prev_hidden_for_prefill,
                position_ids=mtp_prefill_meta["position_ids"],
                past_key_values=mtp_runtime_state.paged_cache,
                use_cache=True,
                is_prefill=True,
                attn_inputs=mtp_prefill_meta["attn_inputs"],
                context_lens=mtp_prefill_meta["context_lens"],
                q_lens=mtp_prefill_meta["q_lens"],
                input_ids=mtp_input_ids,
                return_hidden=True,
            )
            mtp_logits, _, mtp_hidden = mtp_out

        # Sample first spec token
        mtp_spec_token = torch.argmax(mtp_logits, dim=-1)[:, -1:]  # [B, 1]
        if ep_size > 1 and dist.is_initialized():
            mtp_spec_token = mtp_spec_token.contiguous()
            dist.broadcast(mtp_spec_token, src=0, group=moe_ep_group)
        mtp_spec_tokens = mtp_spec_token
        confirmed_prev_hidden = prev_hidden[:, -spec_len:, :].contiguous()
        # Update prev_hidden for next MTP step: keep last spec_len positions
        mtp_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous().reshape(batch_size, spec_len, -1)  # [B, spec_len, H]
        # Note: mtp_generate_ids already contains next_token_id,
        # so we don't need to append it again here.
        # Following note:: append spec_token to generate_ids for sliding window
        mtp_generate_ids = torch.cat([mtp_generate_ids, mtp_spec_token], dim=-1)

        # For next_n > 1, run additional MTP decode steps
        mtp_decode_input_ids = torch.cat(
            [mtp_input_ids[:, -(spec_len - 1):], mtp_spec_token],
            dim=-1,
        ).contiguous()
        for _mtp_step in range(1, next_n):
            with torch.no_grad():
                # MTP decode: input_ids = [B, spec_len], prev_hidden = [B, spec_len, H]
                # This prefill-side supplement path only exists when next_n > 1.
                # Keep it eager: its cache/sliding-window semantics differ from
                # the steady speculate loop and can crash when captured.
                current_mtp_decode_fn = None if next_n > 1 else mtp_decode_fn
                mtp_logits, _, mtp_hidden = forward_mtp_decode_mojo(
                    mtp_model,
                    mtp_prev_hidden,
                    mtp_decode_input_ids,
                    mtp_runtime_state,
                    decode_fn=current_mtp_decode_fn,
                    # Prefill-side supplement is a one-off graph pattern; do not
                    # use it to warm steady speculate decode steps.
                    skip_guard_eval=False,
                )

            mtp_spec_token = torch.argmax(mtp_logits, dim=-1)[:, -1:]  # [B, 1]
            mtp_spec_tokens = torch.cat([mtp_spec_tokens, mtp_spec_token], dim=-1)
            mtp_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous().reshape(batch_size, spec_len, -1)  # [B, spec_len, H]
            mtp_generate_ids = torch.cat([mtp_generate_ids, mtp_spec_token], dim=-1)
            mtp_decode_input_ids = torch.cat(
                [mtp_decode_input_ids[:, 1:], mtp_spec_token],
                dim=-1,
            ).contiguous().reshape(batch_size, spec_len)

        # Save MTP cache seq_lens for rollback after verification
        mtp_seq_lens_cached = mtp_runtime_state.paged_cache.seq_lens.clone()

        # Feed [next_token + spec_tokens] to main model in next iteration
        input_ids = torch.cat([next_token_id, mtp_spec_tokens], dim=-1)  # [B, spec_len]

        # CRITICAL: Broadcast mtp_spec_tokens to all ranks to ensure consistency.
        # MTP model uses EP, so different ranks may produce different spec tokens.
        # If input_ids are inconsistent across ranks, npu_moe_distribute_dispatch_v2
        # will deadlock (manifests as aicore timeout 507014).
        if ep_size > 1 and dist.is_initialized():
            dist.broadcast(input_ids, src=0, group=moe_ep_group)
        if is_main:
            print(f"[MTP] Prefill: generated {mtp_spec_tokens.shape[-1]} spec token(s)")
    else:
        input_ids = next_token_id

    warmup_steps = int(os.getenv("MOJO_PROF_WARMUP_STEPS", "3"))
    prof_steps = int(os.getenv("MOJO_PROF_ACTIVE_STEPS", "3"))
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
                    with_stack=True,
                    with_flops=True,
                )
                prof_ctx.__enter__()
                prof_active = True
                print(f"[PROF] Start profiling decode at step {step}, output -> {prof_dir}", flush=True)

            # ==================== Main Model Decode ====================
            with torch.no_grad():
                if use_mtp:
                    main_seq_lens_before_verify = runtime_state.paged_cache.seq_lens.clone()
                    if next_n > 1:
                        all_logits, past_key_values, main_hidden = forward_mtp_verify_mojo(
                            model,
                            input_ids.contiguous().reshape(batch_size, spec_len),
                            past_key_values,
                            runtime_state,
                            next_n,
                            decode_fn=decode_fn,
                            use_attn_metadata=use_attn_metadata,
                            skip_guard_eval=False,
                        )
                    else:
                        # MTP mode: decode spec_len tokens, return all logits + hidden states
                        all_logits, past_key_values, main_hidden = forward_decode_mojo(
                            model, input_ids, past_key_values, decode_fn=decode_fn, runtime_state=runtime_state,
                            use_attn_metadata=use_attn_metadata,
                            update_runtime_state=False,
                            skip_guard_eval=(
                                enable_mtp_cache_compile and
                                decode_fn is not None and
                                main_decode_graph_warmed
                            ),
                        )
                        if enable_mtp_cache_compile and decode_fn is not None:
                            main_decode_graph_warmed = True
                else:
                    next_token_logits, past_key_values, _ = forward_decode_mojo(
                        model, input_ids, past_key_values, decode_fn=decode_fn, runtime_state=runtime_state,
                        use_attn_metadata=use_attn_metadata,
                    )

            if step == 0 and decode_fn is not None and is_main:
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
                # all_logits: [B, spec_len, V], main_hidden: [B, spec_len, H]
                main_greedy = torch.argmax(all_logits, dim=-1)  # [B, spec_len]
                if ep_size > 1 and dist.is_initialized():
                    main_greedy = main_greedy.contiguous()
                    dist.broadcast(main_greedy, src=0, group=moe_ep_group)

                # Verify spec tokens: compare MTP spec tokens with main model's greedy output
                # main_greedy[:, 0] predicts token after real_token -> should match spec_tokens[:, 0]
                # main_greedy[:, i] predicts token after spec_tokens[:, i-1] -> should match spec_tokens[:, i]
                token_mask = mtp_spec_tokens == main_greedy[:, :next_n]  # [B, next_n]
                has_invalid = (token_mask == False).any(dim=-1)
                invalid_pos = (token_mask == False).int().argmax(dim=-1)
                accepted_num = torch.where(has_invalid, invalid_pos, token_mask.shape[-1])  # [B]

                # Debug: print MTP vs main model comparison
                if is_main and step < 3:
                    print(f"[MTP DEBUG step={step}] spec_tokens={mtp_spec_tokens[0].tolist()}, main_greedy={main_greedy[0, :next_n].tolist()}, accepted={accepted_num[0].item()}", flush=True)

                # CRITICAL: Broadcast accepted_num to ensure all ranks have consistent values
                # Different ranks may compute different accepted_num due to MTP output differences
                if ep_size > 1 and dist.is_initialized():
                    accepted_num = accepted_num.contiguous()
                    dist.broadcast(accepted_num, src=0, group=moe_ep_group)

                mtp_total_accepted += accepted_num.sum().item()
                mtp_total_spec += next_n * batch_size
                mtp_step_count += 1

                # Collect accepted tokens: accepted_num spec tokens + 1 next token
                for row in range(batch_size):
                    cur_accepted = accepted_num[row].item() + 1  # +1 for the next token (first rejected or bonus)
                    for j in range(cur_accepted):
                        generated[row].append(int(main_greedy[row, j].item()))

                # ---- Update main model cache seq_lens (per-batch) ----
                accepted_step = (1 + accepted_num.unsqueeze(0)).squeeze(0)
                if next_n == 1:
                    runtime_state.paged_cache.seq_lens += accepted_step
                elif main_seq_lens_before_verify is not None:
                    runtime_state.paged_cache.seq_lens.copy_(main_seq_lens_before_verify + accepted_step)
                else:
                    runtime_state.paged_cache.seq_lens += accepted_step

                # ---- Determine next_token_id: the token after accepted positions ----
                next_token_id = torch.stack(
                    [main_greedy[row, accepted_num[row].item()] for row in range(batch_size)], dim=0
                ).unsqueeze(-1)  # [B, 1]

                if ep_size > 1 and dist.is_initialized():
                    next_token_id = next_token_id.contiguous()
                    dist.broadcast(next_token_id, src=0, group=moe_ep_group)

                # ---- Update MTP prev_hidden and generate_ids ----
                # Following note:: concat last_step_hidden with accepted hidden, take last spec_len
                mtp_prev_hidden_list = []
                confirmed_generate_ids_list = []
                for row in range(batch_size):
                    cur_len = accepted_num[row].item() + 1
                    cur_accepted_hidden = main_hidden[row, :cur_len, :].unsqueeze(0)  # [1, cur_len, H]
                    mtp_prev_hid = torch.cat([confirmed_prev_hidden[row:row+1], cur_accepted_hidden], dim=1)[:, -spec_len:, :]
                    mtp_prev_hidden_list.append(mtp_prev_hid)
                    # Match note:: only confirmed tokens are kept in generate_ids.
                    cur_accepted_ids = main_greedy[row, :cur_len]
                    confirmed_generate_ids_list.append(torch.cat([confirmed_generate_ids[row], cur_accepted_ids]))
                mtp_prev_hidden = torch.cat(mtp_prev_hidden_list, dim=0).reshape(batch_size, spec_len, -1)  # [B, spec_len, H]
                confirmed_prev_hidden = mtp_prev_hidden
                # Pad generate_ids to same length (different batches may have different accepted_num)
                from torch.nn.utils.rnn import pad_sequence
                # Match note:: left-pad variable-length histories so the
                # newest suffix always stays aligned in the last `spec_len` positions.
                confirmed_generate_ids_rev = [ids.flip(0) for ids in confirmed_generate_ids_list]
                confirmed_generate_ids = pad_sequence(
                    confirmed_generate_ids_rev,
                    batch_first=True,
                    padding_value=pad_token_id,
                ).flip(1).contiguous()
                mtp_generate_ids = confirmed_generate_ids

                # ---- Update MTP cache seq_lens (per-batch) ----
                mtp_runtime_state.paged_cache.seq_lens.copy_(mtp_seq_lens_cached + 1 + accepted_num)
                # Match note: `kv_len_cached`: cache the confirmed
                # sequence length before this round's new draft tokens are appended.
                mtp_seq_lens_cached = mtp_runtime_state.paged_cache.seq_lens.clone()

                # ---- MTP Post-Process: generate new spec tokens ----
                mtp_spec_tokens = None
                mtp_decode_input_ids = mtp_generate_ids[:, -spec_len:].contiguous().reshape(batch_size, spec_len)
                for mtp_step_idx in range(next_n):
                    with torch.no_grad():
                        # MTP decode: input_ids = [B, spec_len], prev_hidden = [B, spec_len, H]
                        # CRITICAL: Broadcast mtp_input_ids and mtp_prev_hidden to all ranks
                        # to ensure EP communication consistency in MoE dispatch/combine.
                        # Different ranks may have different mtp_prev_hidden due to EP mode.
                        if ep_size > 1 and dist.is_initialized():
                            mtp_decode_input_ids = mtp_decode_input_ids.contiguous().reshape(batch_size, spec_len)
                            dist.broadcast(mtp_decode_input_ids, src=0, group=moe_ep_group)
                        current_mtp_decode_fn = (
                            mtp_decode_fns[mtp_step_idx]
                            if mtp_decode_fns is not None else mtp_decode_fn
                        )
                        current_mtp_warmed = (
                            mtp_decode_graph_warmed_steps[mtp_step_idx]
                            if mtp_decode_graph_warmed_steps is not None else mtp_decode_graph_warmed
                        )
                        mtp_logits, _, mtp_hidden = forward_mtp_decode_mojo(
                            mtp_model,
                            mtp_prev_hidden,
                            mtp_decode_input_ids,
                            mtp_runtime_state,
                            decode_fn=current_mtp_decode_fn,
                            skip_guard_eval=(
                                enable_mtp_cache_compile and
                                current_mtp_decode_fn is not None and
                                current_mtp_warmed
                            ),
                        )
                        if enable_mtp_cache_compile and current_mtp_decode_fn is not None:
                            if mtp_decode_graph_warmed_steps is not None:
                                mtp_decode_graph_warmed_steps[mtp_step_idx] = True
                            else:
                                mtp_decode_graph_warmed = True
                    mtp_spec_token = torch.argmax(mtp_logits, dim=-1)[:, -1:]  # [B, 1]
                    if mtp_spec_tokens is None:
                        mtp_spec_tokens = mtp_spec_token
                    else:
                        mtp_spec_tokens = torch.cat([mtp_spec_tokens, mtp_spec_token], dim=-1)

                    # Update mtp_prev_hidden for next MTP step
                    mtp_prev_hidden = mtp_hidden[:, -spec_len:, :].contiguous().reshape(batch_size, spec_len, -1)  # [B, spec_len, H]
                    # Update generate_ids
                    mtp_generate_ids = torch.cat([mtp_generate_ids, mtp_spec_token], dim=-1)
                    mtp_decode_input_ids = torch.cat(
                        [mtp_decode_input_ids[:, 1:], mtp_spec_token],
                        dim=-1,
                    ).contiguous().reshape(batch_size, spec_len)

                # Feed [next_token + spec_tokens] to main model
                input_ids = torch.cat([next_token_id, mtp_spec_tokens], dim=-1)  # [B, spec_len]

                # CRITICAL: Broadcast input_ids to all ranks (same reason as prefill broadcast)
                if ep_size > 1 and dist.is_initialized():
                    input_ids = input_ids.contiguous()
                    dist.broadcast(input_ids, src=0, group=moe_ep_group)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                if ep_size > 1 and dist.is_initialized():
                    dist.broadcast(next_token_id, src=0, group=moe_ep_group)

                token_list = next_token_id.squeeze(-1).detach().cpu().tolist()
                for row, token_id in enumerate(token_list):
                    generated[row].append(int(token_id))

                input_ids = next_token_id
    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

    _mem_snapshot("after all decode steps")

    if is_main:
        print("-" * 40)
        for i, output_ids in enumerate(generated):
            res = tokenizer.decode(output_ids, skip_special_tokens=False)
            if tokenizer.eos_token in res:
                res = res.split(tokenizer.eos_token)[0]
            print(f"[Batch {i}] Generated text: {res}")
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
    parser.add_argument("--ep_size", type=int, default=int(os.getenv("EP_SIZE", "1")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "1")),
                        help="Batch size for inference (default: 1, single-batch)")
    parser.add_argument("--prof", action="store_true", default=bool(int(os.getenv("MOJO_PROF", "0"))),
                        help="Enable NPU profiling for decode")
    parser.add_argument("--graph_mode", type=str, default=os.getenv("MOJO_GRAPH_MODE", "eager"),
                        choices=["eager", "npugraph_ex", "ge_graph"], help="Graph compilation mode for decode")
    parser.add_argument("--use_attn_metadata", type=int, choices=[0, 1],
                        default=int(os.getenv("MOJO_USE_ATTN_METADATA", "1")),
                        help="Use runtime attn_metadata in DeepSeek-V4 attention path (default: 1)")
    parser.add_argument("--next_n", type=int, default=int(os.getenv("NEXT_N", "0")),
                        help="Number of MTP speculative tokens (0=disabled, 1-3 enabled)")
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

        DeepseekV4ForMTP.load_weights(mtp_model, args.model_path, main_model=model)
        _mem_snapshot("after MTP load_weights", npu_device_idx)

        if ep_size > 1 and dist.is_initialized():
            # CRITICAL: MTP model must reuse the main model's hccl_comm_dict (EP communication groups).
            # Creating independent comm groups via dist.new_group() causes HCCL communication mismatch
            # between MTP and main model, leading to npu_moe_distribute_dispatch_v2 aicore timeout (507014).
            # This matches note: behavior where comm groups are implicitly shared via caching.
            mtp_model.hccl_comm_dict = model.hccl_comm_dict.copy()
            mtp_model.set_ep_group()
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

        mtp_runtime_state = DeepseekMTPRuntimeState.from_model(
            mtp_model,
            batch_size=args.batch_size,
            device=f"npu:{npu_device_idx}",
            max_seq_len=4096,
            next_n=next_n,
        )
        if is_main:
            print("[MTP] MTP model and runtime state created")

    generate(
        model, tokenizer, args.prompt, args.max_new_tokens, f"npu:{npu_device_idx}",
        ep_size=ep_size, batch_size=args.batch_size, graph_mode=args.graph_mode, prof=args.prof,
        use_attn_metadata=bool(args.use_attn_metadata),
        mtp_model=mtp_model, mtp_runtime_state=mtp_runtime_state, next_n=next_n,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
