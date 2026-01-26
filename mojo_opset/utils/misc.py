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