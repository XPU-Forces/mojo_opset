def get_bool_env(key, default=True):
    import os

    value = os.environ.get(key, None)
    if value is None:
        return default
    value = value.lower()
    if value in ("1", "yes", "true"):
        return True
    elif value in ("0", "no", "false"):
        return False
    else:
        return default


def configure_torch_deterministic(seed=0):
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_tensor_factory_kwargs(**kwargs):
    factory_kwargs = {}
    for k, v in kwargs.items():
        if v is not None and k in ("device", "dtype", "layout", "requires_grad", "pin_memory", "memory_format"):
            factory_kwargs[k] = v
    return factory_kwargs
