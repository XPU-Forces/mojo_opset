from abc import abstractmethod
import torch
from mojo_opset.modeling.config import MojoConfig
class MojoSession:
    @property
    @abstractmethod
    def kv_cache(self):
        ...

class MojoSampler(torch.nn.Module):
    @abstractmethod
    def forward(self, logits, session: MojoSession = None): ...

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