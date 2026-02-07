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

def get_tensor_factory_kwargs(**kwargs):
    factory_kwargs = {}
    for k,v in kwargs.items():
        if v is not None and k in ('device', 'dtype', 'layout', 'requires_grad', 'pin_memory', 'memory_format'):
            factory_kwargs[k] = v
    return factory_kwargs
    