# Performance testing

```sh
pytest tests/performance
pytest tests/performance -k rms_norm --mojo-implementation triton
pytest tests/performance --perf-timers e2e --perf-output after.json
```

The suite measures public functions/modules, not every implementation automatically.
Select cases with pytest paths or `-k`; list them with `--collect-only -q`.
Use an idle device and run revisions serially, without pytest-xdist.

Cases include the original Mojo/Ext performance shapes, including long attention
sequences and large VWN/GDN states. The full suite is expensive; select files or
cases for development. Forward, backward, and combined forward/backward are
reported separately. Input construction and saved backward graphs are outside
the timed region; legacy hardware-specific MFU estimates are not reported.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--mojo-target` | Detected | Target configuration, e.g. `npu.a2`; does not select a physical card |
| `--mojo-implementation` | Config | Exact implementation for all selected cases |
| `--perf-device` | `0` | Logical device index within the visible devices |
| `--perf-timers` | `profiler,e2e` | Comma-separated `profiler`, `e2e`, or `event` |
| `--perf-warmup` | `10` | Warmup calls, at least 1 |
| `--perf-repeats` | `20` | Samples per timer, at least 3 |
| `--perf-instances` | `2` | Independent input/state instances to rotate, at least 2 |
| `--perf-output` | `tests/performance/results/current.json` | JSON output, overwritten each run |

`MOJO_ACCURACY_IMPLEMENTATION` is no longer used. Select implementations through
config (including `MOJO_OPS_CONFIG_PATH`) or `--mojo-implementation`.

## Measurements

- **Profiler:** time from the first device kernel's start to the last kernel's
  end for one call, including gaps. Reports include individual kernel details;
  NPU backward kernels launched by autograd worker threads are included.
- **E2E:** host wall time for one call followed by device synchronization.
- **Event:** optional current-stream event time for one call, not all-stream time.
- **Memory:** allocator peak/incremental memory, measured separately after timing.
  Allocations outside PyTorch's allocator may not be visible.

Each input instance is initialized before measurement, and warmup is excluded.
Rotate alternates independent inputs/state; increasing instances uses more memory
and does not guarantee cold L2. It is not an explicit cache eviction policy.
Stateful workloads must reset or otherwise bound state across repeated calls.

## Comparing revisions

Measure both revisions on the same device and software stack, with matching cases
and measurement settings. No baseline is committed to the repository.

```sh
# Run in the respective revision checkouts, serially.
pytest tests/performance --perf-output /tmp/before.json
pytest tests/performance --perf-output /tmp/after.json
python scripts/compare_perf.py /tmp/before.json /tmp/after.json --output /tmp/comparison.md
```

The script prints and optionally saves a Markdown comparison of latency and
memory. The threshold defaults to 5% (`--threshold-percent`). Regressions and
noisy results are report-only; invalid, incompatible or missing results fail.
