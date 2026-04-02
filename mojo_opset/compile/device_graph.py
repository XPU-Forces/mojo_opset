import torch

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class DeviceGraphRunner:
    """
    Graph capture/replay runner backed by xpu_graph's GraphRunner.
    Supports NPU, MLU, and any target registered in xpu_graph.
    """

    def __init__(self, model, device=None):
        from xpu_graph.config import Target
        from xpu_graph.device_graph_runner import GraphRunner

        if device is None:
            from mojo_opset.utils.platform import get_platform

            device = get_platform()

        self.model = model
        self._device = device
        self._device_mod = getattr(torch, device)

        def logits_only_forward(input_ids, **kwargs):
            logits, _ = model(input_ids, **kwargs)
            return logits

        RunnerClass = GraphRunner[Target(device)]
        if RunnerClass is None:
            raise RuntimeError(
                f"xpu_graph has no GraphRunner for device '{device}'. "
                f"Ensure the corresponding torch extension (e.g. torch_npu/torch_mlu) is installed."
            )

        self._runner = RunnerClass(
            callable_target=logits_only_forward,
            init_param_callback=lambda input_ids, **kw: input_ids,
            copy_param_callback=lambda buf, input_ids, **kw: (
                buf.copy_(input_ids),
                True,
            ),
        )
        self.session = None

    def capture(self, input_ids, session, memory_pool=None):
        self.session = session

        if hasattr(session, "backup_state"):
            session.backup_state()

        with torch.inference_mode():
            s = self._device_mod.Stream()
            s.wait_stream(self._device_mod.current_stream())
            with self._device_mod.stream(s):
                self.model(input_ids.clone(), session=session)
            self._device_mod.current_stream().wait_stream(s)

            if hasattr(session, "restore_state"):
                session.restore_state()

            self._runner.capture(input_ids, memory_pool=memory_pool, session=session)

            if hasattr(session, "restore_state"):
                session.restore_state()

    def replay(self, input_ids, session):
        logits = self._runner(input_ids, session=session)
        return logits, session


class DeviceGraphPool:
    """Batch-size-keyed cache of DeviceGraphRunners with a shared memory pool.

    All runners share a single device memory pool (via graph_pool_handle()),
    so intermediate tensors from different batch-size graphs can reuse the
    same memory — only one graph runs at a time during decode.

    Bucket range: bs ∈ [1, 16].
    """

    BATCH_SIZES = list(range(1, 17))

    def __init__(self, model, device):
        self._model = model
        self._device = device
        self._device_mod = getattr(torch, device)
        self._pool = self._device_mod.graph_pool_handle()
        self._runners: dict[int, DeviceGraphRunner] = {}
        self._bound_session_id: int | None = None

    def get_runner(self, input_ids, session) -> DeviceGraphRunner:
        """Return cached runner, or lazily capture one for this batch size."""
        if id(session) != self._bound_session_id:
            self._runners.clear()
            self._bound_session_id = id(session)
        bs = input_ids.shape[0]
        if bs not in self._runners:
            runner = DeviceGraphRunner(self._model, device=self._device)
            runner.capture(input_ids, session, memory_pool=self._pool)
            self._runners[bs] = runner
            logger.debug(f"[DeviceGraphPool] Captured graph for bs={bs}")
        return self._runners[bs]

    @property
    def captured_batch_sizes(self) -> list[int]:
        return sorted(self._runners.keys())
