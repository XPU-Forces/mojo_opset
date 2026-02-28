from abc import abstractmethod
import torch
from mojo_opset.modeling.config import MojoConfig
class MojoSession:
    @property
    @abstractmethod
    def kv_cache(self):
        ...

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

        self.seq_lens = torch.zeros(
            (self.num_layers, self.batch_size), dtype=torch.int64, device=device
        )

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
        self.top_p= top_p

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
        typewriter_buffer=4,
    ):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sampler = sampler
        self._enable_typewriter = enable_typewriter
        self._typewriter_buffer = typewriter_buffer
        if self._enable_typewriter:
            from multiprocessing import Process, Pipe
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
        print(f"Generated text: ")
        try:
            full_output = None
            while (generated_ids := conn.recv()):
                output = tokenizer.decode(torch.cat(generated_ids, dim=1))
                if full_output is None:
                    full_output = [f"[{idx}] " + msg for idx, msg in enumerate(output)]
                else:
                    for idx in range(len(full_output)):
                        full_output[idx] = ''.join((full_output[idx], output[idx]))

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
        context_input_len = torch.tensor(
            [len(seq) for seq in input_ids], dtype=torch.int64, device=self.device
        )
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
        print(f"Prompt: {prompts}")
        print("-" * 40)

        with torch.inference_mode():
            logits, session = self.model(
                input_ids,
                context_input_len=context_input_len,
            )

        next_token_id = self.sampler(logits, session)

        generated_ids = [next_token_id.cpu()]

        # Decode loop
        input_ids = next_token_id
        should_end = next_token_id == self.tokenizer.eos_token_id

        for _ in range(1, self.max_new_tokens):
            with torch.inference_mode():
                logits, session = self.model(
                    input_ids,
                    session=session,
                )

            next_token_id = self.sampler(logits, session)

            should_end = should_end | (next_token_id == self.tokenizer.eos_token_id)
            if all(should_end):
                break

            next_token_id[should_end] = self.tokenizer.eos_token_id
            generated_ids.append(next_token_id.cpu())
            input_ids = next_token_id

            if self._enable_typewriter and len(generated_ids) >= self._typewriter_buffer:
                self._producer_conn.send(generated_ids)
                generated_ids.clear()

        if self._enable_typewriter:
            generated_ids and self._producer_conn.send(generated_ids)
            self._producer_conn.close()
        else:
            print(generated_ids)
