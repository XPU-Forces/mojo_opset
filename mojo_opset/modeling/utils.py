from abc import abstractmethod
from itertools import accumulate
from multiprocessing import Pipe
from multiprocessing import Process
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def merge_group_and_share_ffn(
    config,
    group_ffn_output: torch.Tensor,
    share_ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    if config.dp_size == 1:
        return group_ffn_output + share_ffn_output
    dp_rank = config.dp_rank
    if use_padding:
        raise NotImplementedError("merge_group_ffn not implemented.")
        # TODO: Implement the NPU Triton version.
        global_max_batch_size = config.max_batch_size * config.dp_size
        assert group_ffn_output.shape[0] == global_max_batch_size
        merge_group_ffn(group_ffn_output, share_ffn_output, dp_rank_input_len, global_max_batch_size, dp_rank)
    else:
        rank_start = sum(host_dp_rank_input_len[:dp_rank])
        group_ffn_output[rank_start : rank_start + share_ffn_output.shape[0], :] += share_ffn_output
    return group_ffn_output


def dp_allreduce(
    config,
    hidden_states: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    if config.dp_size == 1:
        return hidden_states
    dp_rank = config.dp_rank
    if use_padding:
        raise NotImplementedError("dp_pad not implemented.")
        # TODO: Implement the NPU Triton version.
        global_max_batch_size = config.max_batch_size * config.dp_size
        hidden_states = dp_pad(hidden_states, dp_rank_input_len, global_max_batch_size, dp_rank)
    else:
        left_len = sum(host_dp_rank_input_len[:dp_rank])
        right_len = sum(host_dp_rank_input_len[dp_rank + 1 :])
        hidden_states = F.pad(hidden_states, (0, 0, left_len, right_len))
    if config.is_deterministic:
        raise NotImplementedError("all_reduce_with_all_to_all not implemented.")
        # TODO: Implement the NPU Triton version.
    else:
        dist.all_reduce(hidden_states, group=config.dp_group)
    return hidden_states


def dp_scatter(
    config,
    ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    local_token_num: int,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    dp_rank = config.dp_rank
    if config.dp_size == 1:
        return ffn_output
    if use_padding:
        raise NotImplementedError("dp_unpad not implemented.")
        # TODO: Implement the NPU Triton version.
        return dp_unpad(ffn_output, dp_rank_input_len, local_token_num, dp_rank)
    else:
        cu_lens = list(accumulate([0] + host_dp_rank_input_len))
        return ffn_output[cu_lens[dp_rank] : cu_lens[dp_rank + 1]]


class MojoSession:
    @property
    @abstractmethod
    def kv_cache(self): ...


class PagedKVCache:
    def __init__(
        self,
        config: MojoConfig,
        batch_size: int,
        num_layers: int,
        device,
        dtype,
        block_size: int = 16,
    ):
        from mojo_opset import MojoStorePagedKVCache

        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = config.model_config.num_key_value_heads
        self.head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads
        self.batch_size = batch_size

        max_blocks_per_seq = (config.model_config.max_position_embeddings + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        self.k_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=dtype,
            device=device,
        )

        self.block_tables = torch.zeros(
            (self.num_layers, self.batch_size, max_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        self.seq_lens = torch.zeros((self.num_layers, self.batch_size), dtype=torch.int64, device=device)

        self.free_blocks = torch.arange(total_blocks, device=device, dtype=torch.int32)
        self.num_free_blocks = total_blocks
        self.store_paged_kv = MojoStorePagedKVCache()

    def _allocate_blocks(self, num_blocks: int):
        if num_blocks > self.num_free_blocks:
            raise ValueError("PagedKVCache: Out of memory!")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks : self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_len: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
    ):
        if input_len is None:
            input_len = torch.ones(self.batch_size, device=key_states.device, dtype=torch.int64)

        current_seq_lens = self.seq_lens[layer_idx]
        for i in range(self.batch_size):
            context_len = current_seq_lens[i].item()

            old_num_blocks = (context_len + self.block_size - 1) // self.block_size
            new_total_len = context_len + input_len[i]
            new_num_blocks = (new_total_len + self.block_size - 1) // self.block_size

            if new_num_blocks > old_num_blocks:
                num_to_allocate = new_num_blocks - old_num_blocks
                newly_allocated = self._allocate_blocks(num_to_allocate)
                self.block_tables[layer_idx, i, old_num_blocks:new_num_blocks] = newly_allocated

        self.store_paged_kv(
            key_states,
            value_states,
            self.k_cache,
            self.v_cache,
            self.block_tables[layer_idx],
            cu_seqlens,
            current_seq_lens,
        )
        self.seq_lens[layer_idx] += input_len

    def get_block_tables_for_decode(self, layer_idx: int):
        max_blocks = (self.seq_lens[layer_idx].max().item() + self.block_size - 1) // self.block_size
        return self.block_tables[layer_idx, :, :max_blocks]


class MojoSampler(torch.nn.Module):
    @abstractmethod
    def forward(self, logits, session: MojoSession = None): ...


class MojoSimpleSampler(MojoSampler):
    def __init__(self, temperature: float = 1.0, top_p: float = 0.9):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p

    def forward(self, logits, session: MojoSession = None):
        if self.temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs - sorted_probs > self.top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token_indices = torch.multinomial(sorted_probs, num_samples=1)
        next_tokens = torch.gather(sorted_indices, -1, next_token_indices)
        return next_tokens


class MojoGenerator(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        sampler: MojoSampler,
        device: torch.device,
        max_new_tokens=128,
        enable_typewriter=False,
    ):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sampler = sampler
        self._enable_typewriter = enable_typewriter
        if self._enable_typewriter:
            self._producer_conn, self._consumer_conn = Pipe()
            self._daemon_process = Process(target=self.typewriter, args=(self.tokenizer, self._consumer_conn))
            self._daemon_process.start()
            # NOTE(liuyuan): close the unnecessary connection for parent process.
            self._consumer_conn.close()

    def __del__(self):
        if self._enable_typewriter:
            self._consumer_conn.close()
            self._producer_conn.close()
            if self._daemon_process.is_alive():
                self._daemon_process.join()
                self._daemon_process.close()

    @staticmethod
    def typewriter(tokenizer, conn):
        print("-" * 40)
        print("Generated text: ")
        try:
            full_output = None
            while generated_ids := conn.recv():
                output = tokenizer.decode(torch.cat(generated_ids, dim=1))
                if full_output is None:
                    full_output = output
                else:
                    for idx in range(len(full_output)):
                        full_output[idx] = "".join((full_output[idx], output[idx]))

                str2print = "\n".join(full_output)
                print(
                    "\033[H\033[0J" + str2print,
                    end="",
                    flush=True,
                )
        except EOFError:
            print("\nGeneration is done.")

    def forward(self, prompts):
        input_ids = self.tokenizer(prompts, return_tensors=None).input_ids
        context_input_len = torch.tensor([len(seq) for seq in input_ids], dtype=torch.int64, device=self.device)
        input_ids = (
            torch.cat(
                list(
                    map(
                        lambda x: torch.tensor(x, dtype=torch.int64),
                        input_ids,
                    )
                )
            )
            .squeeze()
            .to(self.device)
        )

        # Prefill
        print(f"\nPrompt: {prompts}")
        print("-" * 40)
        print(f"Tokens: {input_ids}")
        print(f"{context_input_len=}")

        with torch.inference_mode():
            logits, session = self.model(
                input_ids,
                context_input_len=context_input_len,
            )

        next_token_id = self.sampler(logits, session)

        generated_ids = [next_token_id.cpu()]

        # Decode loop
        input_ids = next_token_id

        for _ in range(1, self.max_new_tokens):
            with torch.inference_mode():
                logits, session = self.model(
                    input_ids,
                    session=session,
                )

            # next_token_id = simple_sample(logits, temperature=0.7)
            next_token_id = self.sampler(logits, session)

            generated_ids.append(next_token_id.cpu())

            input_ids = next_token_id

            if all(next_token_id == self.tokenizer.eos_token_id):
                break
            if self._enable_typewriter and len(generated_ids) >= 4:
                self._producer_conn.send(generated_ids)
                generated_ids.clear()

        if self._enable_typewriter:
            generated_ids and self._producer_conn.send(generated_ids)
            self._producer_conn.close()
