import datetime
import gzip
import os

from mojo_opset.modeling.common import GeneratorHook
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def create_npu_profiler(
    start_step: int,
    end_step: int,
    trace_dir: str,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
):
    import torch_npu

    def handler_fn(p):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file_extention = "pt.trace.json"
        memory_timeline_file_extention = "html"

        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, f"npu_profile_{time_str}.{trace_file_extention}")
        memory_timeline_file = os.path.join(trace_dir, f"npu_profile_{time_str}.{memory_timeline_file_extention}")

        p.export_chrome_trace(trace_file)
        logger.info(f"Profiling result saved at {trace_file}.")

        p.export_memory_timeline(memory_timeline_file)
        logger.info(f"Profiling memory timeline saved at {memory_timeline_file}.")

        # In NPU, compress the trace file to .gz format ourselves.
        gz_path = trace_file + ".gz"
        with open(trace_file, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(trace_file)
        trace_file = gz_path
        logger.info(f"Profiling result compressed to {trace_file}.")

    profiler_module = torch_npu.profiler
    activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.NPU]

    warmup = 0 if start_step <= 1 else 1
    wait = max(0, start_step - warmup - 1)
    active = max(1, end_step - start_step)
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")

    schedule = profiler_module.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=1,
    )

    return profiler_module.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=with_stack,
        with_stack=False,
    )


class NPUProfilerHook(GeneratorHook):
    def __init__(self, start_step: int = 2, end_step: int = 10, trace_dir: str = "./profiling_logs"):
        self.profiler = create_npu_profiler(start_step, end_step, trace_dir)
        self.current_step = 0
        self.profiler.start()

    def _step(self):
        self.current_step += 1
        self.profiler.step()

    def after_prefill(self, *, logits, session):
        self._step()

    def after_decode_step(self, *, step, logits, next_token_id):
        self._step()

    def after_decode(self, *, decode_steps, generated_ids):
        # We don't stop here because PerfMojoGenerator runs multiple decodes
        pass

    def __del__(self):
        if hasattr(self, "profiler") and self.profiler is not None:
            self.profiler.stop()
