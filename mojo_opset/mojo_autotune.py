import time
import torch
from functools import lru_cache

from mojo_opset.core.mojo_operator import MojoOperator
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


_tuning_cache = {}
# _autotuned_instances = {}


# NOTE: just as example.
def benchmark_op(func_to_run, *args, **kwargs) -> float:
    warmup = 3
    repeat = 5

    device = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            device = arg.device
            break

    sync_func = lambda: None
    assert device and device.type == "npu", "Only NPU is supported for auto-tuning now."
    sync_func = torch.npu.synchronize

    for _ in range(warmup):
        func_to_run(*args, **kwargs)
    sync_func()

    start_time = time.time()
    for _ in range(repeat):
        func_to_run(*args, **kwargs)
    sync_func()
    end_time = time.time()

    return (end_time - start_time) / repeat * 1000


@lru_cache(maxsize=128)
def create_autotuned_op(cls_to_tune, default_key_func: callable, *init_args, **init_kwargs):
    # instance_key = (cls_to_tune, str(init_args), str(init_kwargs.items()))
    # if instance_key in _autotuned_instances:
    #     return _autotuned_instances[instance_key]

    class AutoTunerWrapper(cls_to_tune):
        _is_autotuner_wrapper = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._family_head = self._get_family_head()
            if not hasattr(self._family_head, "_tuning_cache"):
                self._family_head._tuning_cache = {}

        def _get_family_head(self):
            for base in self.__class__.mro():
                if MojoOperator in getattr(base, "__bases__", []):
                    return base
            return None

        def _tune_for_key(self, current_key, *args, **kwargs):
            logger.info(f"Running Auto-tuner for {self._family_head.__name__} with key={current_key}")

            best_time = float("inf")
            best_performer_forward_std = None

            for priority, impl_class in self._family_head._registry:
                try:
                    temp_instance = impl_class(*self._init_args, **self._init_kwargs)

                    cloned_args = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
                    cloned_kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

                    execution_time = benchmark_op(temp_instance.forward_std, *cloned_args, **cloned_kwargs)
                    logger.info(
                        f"  - Benchmarking {impl_class.__name__} for key={current_key}: {execution_time:.4f} ms"
                    )

                    if execution_time < best_time:
                        best_time = execution_time
                        best_performer_forward_std = temp_instance.forward_std

                except Exception as e:
                    logger.warning(f"Failed to benchmark {impl_class.__name__} for key={current_key}: {e}")

            if best_performer_forward_std:
                logger.info(
                    f"--- Auto-tuner selected {best_performer_forward_std.__self__.__class__.__name__} for key={current_key} ---"
                )
                return best_performer_forward_std
            else:
                logger.warning(
                    f"Auto-tuner for {self._family_head.__name__} failed to find a valid performer for key={current_key}."
                )
                return None

        def forward(self, *args, **kwargs):
            key_func = kwargs.pop("key_func", default_key_func)

            if key_func is None:
                return super().forward(*args, **kwargs)

            current_key = key_func(*args, **kwargs)

            cache = self._family_head._tuning_cache
            if current_key in cache:
                best_func = cache[current_key]
                if best_func:
                    return best_func(*args, **kwargs)
                else:
                    return super().forward(*args, **kwargs)

            best_func = self._tune_for_key(current_key, *args, **kwargs)

            cache[current_key] = best_func

            if best_func:
                return best_func(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    instance = AutoTunerWrapper(*init_args, **init_kwargs)
    instance._init_args = init_args
    instance._init_kwargs = init_kwargs

    # _autotuned_instances[instance_key] = instance
    return instance
