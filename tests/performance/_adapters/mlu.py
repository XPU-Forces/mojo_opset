import torch
import torch_mlu  # noqa: F401 -- registers the MLU runtime and profiler support

runtime = torch.mlu


def collect_trace(run, directory):
    activity = getattr(torch.profiler.ProfilerActivity, "MLU", None)
    if activity is None:
        raise RuntimeError("This torch_mlu build does not expose ProfilerActivity.MLU")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, activity]) as profile:
        run()
    path = directory / "trace.json"
    profile.export_chrome_trace(str(path))
    return path


def is_kernel(event):
    return event.get("cat") == "kernel"


def l2_bytes(properties):
    return getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", None))
