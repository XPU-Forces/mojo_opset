import torch

runtime = torch.cuda


def collect_trace(run, directory):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as profile:
        run()
    path = directory / "trace.json"
    profile.export_chrome_trace(str(path))
    return path


def is_kernel(event):
    return event.get("cat") == "kernel"


def l2_bytes(properties):
    return getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", None))
