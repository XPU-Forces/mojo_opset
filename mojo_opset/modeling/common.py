import time

from abc import abstractmethod
from pathlib import Path

import torch

from mojo_opset.compile.device_graph import DeviceGraphPool
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
    def __init__(self, device, silent=False):
        self._device = device
        self._silent = silent
        self._prefill_start = 0.0
        self._prefill_ms = 0.0
        self._decode_start = 0.0
        self._batch_size = 0
        self._total_input_tokens = 0
        self.records = []

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

        self.records.append(
            {
                "batch_size": self._batch_size,
                "in_tok": self._total_input_tokens,
                "prefill_ms": self._prefill_ms,
                "decode_steps": decode_steps,
                "decode_total_ms": decode_total_ms,
                "decode_avg_ms": decode_avg_ms,
                "throughput": throughput,
            }
        )

        if not self._silent:
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
        self.use_device_graph = False
        if hasattr(model, "config") and hasattr(model.config, "runtime_config"):
            if getattr(model.config.runtime_config, "use_device_graph", False):
                self.use_device_graph = True
        self._graph_pool = DeviceGraphPool(model, device=device) if self.use_device_graph else None
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

        self.generate_from_ids(input_ids, context_input_len)

    def generate_from_ids(
        self,
        input_ids,
        context_input_len,
        max_decode_steps=None,
        ignore_eos=False,
        silent=False,
    ):
        if max_decode_steps is None:
            max_decode_steps = self.max_new_tokens

        self._run_hooks("before_prefill", input_ids=input_ids, context_input_len=context_input_len)

        with torch.inference_mode():
            logits, session = self.model(
                input_ids,
                context_input_len=context_input_len,
            )

            if hasattr(session, "pre_allocate"):
                session.pre_allocate(max_decode_steps)

        self._run_hooks("after_prefill", logits=logits, session=session)

        next_token_id = self.sampler(logits, session)

        generated_ids = [next_token_id.cpu()]

        # Decode loop
        input_ids = next_token_id
        should_end = next_token_id == self.tokenizer.eos_token_id
        decode_steps = 0

        self._run_hooks("before_decode")

        graph_runner = None
        if self._graph_pool is not None:
            graph_runner = self._graph_pool.get_runner(input_ids, session)

        for step in range(1, max_decode_steps):
            with torch.inference_mode():
                if graph_runner is not None:
                    logits, session = graph_runner.replay(input_ids, session)
                else:
                    logits, session = self.model(
                        input_ids,
                        session=session,
                    )

            next_token_id = self.sampler(logits, session)
            decode_steps += 1

            self._run_hooks(
                "after_decode_step",
                step=step,
                logits=logits,
                next_token_id=next_token_id,
            )

            should_end = should_end | (next_token_id == self.tokenizer.eos_token_id)
            if not ignore_eos and all(should_end):
                break

            if not ignore_eos:
                next_token_id[should_end] = self.tokenizer.eos_token_id
            generated_ids.append(next_token_id.cpu())
            input_ids = next_token_id

            if not silent and self._enable_typewriter and len(generated_ids) >= self._typewriter_buffer:
                self._producer_conn.send(generated_ids)
                generated_ids.clear()

        self._run_hooks("after_decode", decode_steps=decode_steps, generated_ids=generated_ids)

        if not silent:
            if self._enable_typewriter:
                generated_ids and self._producer_conn.send(generated_ids)
                self._producer_conn.close()
            else:
                print(generated_ids)


class PerfMojoGenerator(MojoGenerator):
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
        from mojo_opset.utils.platform import get_platform

        hooks = hooks or []
        self.perf_hook = PerfHook(get_platform(), silent=True)
        hooks.append(self.perf_hook)
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            device=device,
            max_new_tokens=max_new_tokens,
            enable_typewriter=False,
            typewriter_buffer=typewriter_buffer,
            hooks=hooks,
        )

    def _run_perf_case(self, batch_size, seqlen, max_decode_steps):
        vocab_size = getattr(self.model.config, "vocab_size", 32000) if hasattr(self.model, "config") else 32000
        input_ids = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.int64, device=self.device)
        context_input_len = torch.full((batch_size,), seqlen, dtype=torch.int64, device=self.device)

        self.generate_from_ids(
            input_ids=input_ids,
            context_input_len=context_input_len,
            max_decode_steps=max_decode_steps,
            ignore_eos=True,
            silent=True,
        )

    def forward(self, prompts=None):
        logger.info("Starting Prefill Latency Tests...")
        prefill_seqlens = [512, 1024, 2048, 4096, 8192]
        self.perf_hook.records.clear()
        for seqlen in prefill_seqlens:
            self._run_perf_case(batch_size=1, seqlen=seqlen, max_decode_steps=1)

        logger.info("\n" + "=" * 60, extra={"clean": True})
        logger.info(f"{'Prefill Latency Tests':^60}", extra={"clean": True})
        logger.info("=" * 60, extra={"clean": True})
        logger.info(
            f"{'SeqLen':<15} | {'Batch Size':<15} | {'Prefill Latency (ms)':<20}",
            extra={"clean": True},
        )
        logger.info("-" * 60, extra={"clean": True})
        for r in self.perf_hook.records:
            logger.info(
                f"{r['in_tok']:<15} | {r['batch_size']:<15} | {r['prefill_ms']:<20.2f}",
                extra={"clean": True},
            )
        logger.info("=" * 60 + "\n", extra={"clean": True})

        logger.info("Starting Decode Throughput Tests...")
        decode_batch_sizes = [1, 2, 4, 8, 16, 24]
        decode_seqlen = 4000
        self.perf_hook.records.clear()
        for bs in decode_batch_sizes:
            self._run_perf_case(
                batch_size=bs,
                seqlen=decode_seqlen,
                max_decode_steps=self.max_new_tokens,
            )

        logger.info("\n" + "=" * 80, extra={"clean": True})
        logger.info(
            f"{'Decode Throughput Tests (Context Len = 4000)':^80}",
            extra={"clean": True},
        )
        logger.info("=" * 80, extra={"clean": True})
        logger.info(
            f"{'Batch Size':<12} | {'Decode Steps':<15} | {'Avg Latency (ms/step)':<22} | {'Throughput (tok/s)':<20}",
            extra={"clean": True},
        )
        logger.info("-" * 80, extra={"clean": True})
        for r in self.perf_hook.records:
            logger.info(
                f"{r['batch_size']:<12} | {r['decode_steps']:<15} | {r['decode_avg_ms']:<22.2f} | {r['throughput']:<20.2f}",
                extra={"clean": True},
            )
        logger.info("=" * 80 + "\n", extra={"clean": True})
