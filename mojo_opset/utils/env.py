import os


def get_bool_env(key: str, default: bool = True) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "yes", "true"}:
        return True
    if normalized in {"0", "no", "false"}:
        return False
    return default
