import time

from abc import abstractmethod
from pathlib import Path

import torch

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoSession:
    @property
    @abstractmethod
    def kv_cache(self): ...


class MojoSampler(torch.nn.Module):
    @abstractmethod
    def forward(self, logits, session: MojoSession = None): ...


class GeneratorHook:
    def before_prefill(self, *, input_ids, context_input_len): ...
    def after_prefill(self, *, logits, session): ...
    def before_decode(self): ...
    def after_decode_step(self, *, step, logits, next_token_id): ...
    def after_decode(self, *, decode_steps, generated_ids): ...


class PerfHook(GeneratorHook):
    def __init__(self, device):
        self._device = device
        self._prefill_start = 0.0
        self._prefill_ms = 0.0
        self._decode_start = 0.0
        self._batch_size = 0
        self._total_input_tokens = 0

    def _sync(self):
        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "mlu":
            torch.mlu.synchronize()
        else:
            raise ValueError(f"Unsupported device: {self._device}")

    def before_prefill(self, *, input_ids, context_input_len):
        self._batch_size = context_input_len.shape[0]
        self._total_input_tokens = int(context_input_len.sum().item())
        self._sync()
        self._prefill_start = time.perf_counter()

    def after_prefill(self, *, logits, session):
        self._sync()
        self._prefill_ms = (time.perf_counter() - self._prefill_start) * 1000

    def before_decode(self):
        self._sync()
        self._decode_start = time.perf_counter()

    def after_decode(self, *, decode_steps, generated_ids):
        self._sync()
        decode_total_ms = (time.perf_counter() - self._decode_start) * 1000
        decode_avg_ms = decode_total_ms / decode_steps if decode_steps > 0 else 0
        throughput = self._batch_size / (decode_avg_ms / 1000) if decode_avg_ms > 0 else 0
        logger.info(
            f"[Perf] bs={self._batch_size} in_tok={self._total_input_tokens} | "
            f"prefill={self._prefill_ms:.1f}ms | "
            f"decode={decode_steps}steps {decode_total_ms:.1f}ms avg={decode_avg_ms:.1f}ms/step {throughput:.1f}tok/s"
        )


class DumpHook(GeneratorHook):
    def __init__(self, dump_dir: str, max_decode_steps: int = 20):
        self._dump_dir = Path(dump_dir)
        self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._max_decode_steps = max_decode_steps

    def after_prefill(self, *, logits, session):
        path = self._dump_dir / "prefill_logits.pt"
        torch.save(logits.cpu(), path)

    def after_decode_step(self, *, step, logits, next_token_id):
        if step <= self._max_decode_steps:
            path = self._dump_dir / f"decode_step_{step:03d}_logits.pt"
            torch.save(logits.cpu(), path)


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
        hooks: list[GeneratorHook] | None = None,
    ):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sampler = sampler
        self._enable_typewriter = enable_typewriter
        self._typewriter_buffer = typewriter_buffer
        self._hooks = hooks or []
        if self._enable_typewriter:
            from multiprocessing import Pipe
            from multiprocessing import Process

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

    def _run_hooks(self, method: str, **kwargs):
        for hook in self._hooks:
            getattr(hook, method)(**kwargs)

    @staticmethod
    def typewriter(tokenizer, conn):
        print("-" * 40)
        print("Generated text: ")
        try:
            full_output = None
            while generated_ids := conn.recv():
                output = tokenizer.decode(torch.cat(generated_ids, dim=1))
                if full_output is None:
                    full_output = [f"[{idx}] " + msg for idx, msg in enumerate(output)]
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
        print(f"Prompt: {prompts}")
        print("-" * 40)

        self._run_hooks("before_prefill", input_ids=input_ids, context_input_len=context_input_len)

        with torch.inference_mode():
            logits, session = self.model(
                input_ids,
                context_input_len=context_input_len,
            )

        self._run_hooks("after_prefill", logits=logits, session=session)

        next_token_id = self.sampler(logits, session)

        generated_ids = [next_token_id.cpu()]

        # Decode loop
        input_ids = next_token_id
        should_end = next_token_id == self.tokenizer.eos_token_id
        decode_steps = 0

        self._run_hooks("before_decode")

        for step in range(1, self.max_new_tokens):
            with torch.inference_mode():
                logits, session = self.model(
                    input_ids,
                    session=session,
                )

            next_token_id = self.sampler(logits, session)
            decode_steps += 1

            self._run_hooks("after_decode_step", step=step, logits=logits, next_token_id=next_token_id)

            should_end = should_end | (next_token_id == self.tokenizer.eos_token_id)
            if all(should_end):
                break

            next_token_id[should_end] = self.tokenizer.eos_token_id
            generated_ids.append(next_token_id.cpu())
            input_ids = next_token_id

            if self._enable_typewriter and len(generated_ids) >= self._typewriter_buffer:
                self._producer_conn.send(generated_ids)
                generated_ids.clear()

        self._run_hooks("after_decode", decode_steps=decode_steps, generated_ids=generated_ids)

        if self._enable_typewriter:
            generated_ids and self._producer_conn.send(generated_ids)
            self._producer_conn.close()
        else:
            print(generated_ids)
