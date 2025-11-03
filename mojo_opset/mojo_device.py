from abc import ABC, abstractmethod
from typing import Any

import torch


class MojoDevice(ABC):
    """
    MojoDevice is a class that provides a wrapper for the meta device. Simply dispatch
    API calling to the backend device lib.
    """

    """
    NOTICE: a certain backend need to replace device_lib with the corresponding device lib.
    """
    device_lib = torch.cuda

    """
    NOTICE: a certian backend need to implemetn this API.
    XXX: whether it's proper to put this classmethod here?
    """

    @abstractmethod
    def Compile(self):
        raise NotImplementedError

    @classmethod
    def __getattr__(cls, name: str) -> Any:
        if hasattr(cls.device_lib, name):
            return getattr(cls.device_lib, name)
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    """
    Hint for backend: following APIs are not necessary to be implemented.
    """

    """
    device related api
    """

    @classmethod
    def get_device_name(cls, device):
        return cls.device_lib.get_device_name(device)

    @classmethod
    def current_device(cls) -> int:
        return cls.device_lib.current_device()

    """
    stream and sync related api
    """

    @classmethod
    def Stream(cls, device=None, priority=0, **kwargs):
        return cls.device_lib.Stream(device=device, priority=priority, **kwargs)

    @classmethod
    def default_stream(cls, device=None):
        return cls.device_lib.default_stream(device)

    @classmethod
    def set_stream(cls, stream):
        return cls.device_lib.set_stream(stream)

    @classmethod
    def stream(cls, device_stream=None):
        return cls.device_lib.stream(device_stream)

    @classmethod
    def current_stream(cls, device=None):
        return cls.device_lib.current_stream(device)

    @classmethod
    def Event(cls, enable_timing=False, blocking=False, interprocess=False):
        return cls.device_lib.Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)

    @classmethod
    def synchronize(cls, device):
        return cls.device_lib.synchronize(device)

    """
    memory related api
    """

    @classmethod
    def empty_cache(cls):
        return cls.device_lib.empty_cache()

    @classmethod
    def memory_usage(cls, device=None):
        return cls.device_lib.memory_usage(device)

    """
    rng related api
    """

    @classmethod
    def manual_seed(cls, seed: int):
        return cls.device_lib.manual_seed(seed)

    @classmethod
    def manual_seed_all(cls, seed: int):
        return cls.device_lib.manual_seed(seed)
