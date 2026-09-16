from importlib import import_module


def load_platform(platform):
    if platform not in ("npu", "ilu", "mlu"):
        raise ValueError(f"No performance adapter for platform={platform!r}")
    return import_module(f"{__name__}.{platform}")
