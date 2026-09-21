# Mojo Opset

PyTorch operators with config-based implementation selection. Public APIs are
exposed through `functions` and `modules`; `kernels` contains their implementations.

```text
modules/    torch.nn.Module wrappers
    ↓
functions/  ordinary functions, autograd orchestration and dispatch
    ↓
kernels/    kernel implementations and Python/custom-op wrappers
```

## Quick start

Install in the matching accelerator PyTorch environment:

```sh
python -m pip install -e .
```

```python
import torch
from mojo_opset import functions, modules

x = torch.randn(8, 1024, device="npu", dtype=torch.bfloat16)
y = functions.silu(x)  # Uses configuration for the detected target.
y_ref = functions.silu(x, implementation="torch_reference")
norm = modules.RMSNorm(1024, implementation="triton").npu()
y = norm(x)
```

Mojo Opset is a pure-Python distribution. Native implementations require
optional, hardware-specific extensions under `mojo_opset_lib`; users
still call `mojo_opset.functions` and `mojo_opset.modules`. Python/Triton-only
development needs no lib package.

## Operator support

See [operator support](docs/operator_support.md) for available implementations.

## Targets and configuration

Target identity is `platform.arch[.sku]`, for example `npu.a2`,
`npu.a2.910b4`, `npu.a5.950pr` or `npu.a5.950dt`.

Defaults live in [mojo_opset/config](mojo_opset/config): `npu_a2.yaml`,
`npu_a5.yaml`, `ilu.yaml` and `mlu.yaml`. Selection precedence is:

```text
explicit implementation= → exact SKU config → architecture config
```

This is configuration inheritance, not execution fallback. Use exact
implementation names from the support table, or `torch_reference`.

An external YAML file can override just a few entries:

```yaml
npu.a2:
  rms_norm_infer: torch_npu
```

Load it explicitly before calling operators or compiling a model:

```python
from mojo_opset import config

config.reload_config("/path/to/overrides.yaml")
# Also accepts a directory of YAML files.
```

Alternatively, set the environment variable before the first configuration
read (normally the first operator call that needs configured dispatch):

```sh
export MOJO_OPS_CONFIG_PATH=/path/to/overrides.yaml
# A directory of YAML files is accepted here too.
```

Both methods overlay only the specified entries on packaged defaults.
Each reload starts fresh; earlier overrides do not accumulate.

`reload_config(path)` uses the explicit path; `reload_config()` reads
`MOJO_OPS_CONFIG_PATH`, or restores defaults if unset. Call it after changing
the environment variable to refresh cached configuration.

Configuration and kernel implementations load on first use and are cached;
importing `mojo_opset` does not load them.

Before a cold fullgraph capture, preload the selected implementation or perform
an eager call:

```python
from mojo_opset import preload

preload("silu", implementation="triton")
compiled = torch.compile(
    lambda x: functions.silu(x, implementation="triton"), fullgraph=True
)
```

`preload` imports wrappers without executing kernels. Config/API changes can
trigger recompilation; a backward keeps the implementation selected by its
forward. Reload configuration between calls, not during graph capture.

## Testing

Run in the matching accelerator environment. Target and implementation default
to device detection and config; override with `--mojo-target npu.a2` and
`--mojo-implementation triton` when needed.

```sh
# Accuracy and training bitwise checks (inference runs accuracy only).
pytest tests/functions tests/modules

# Run either check separately.
pytest tests/functions tests/modules --check accuracy
pytest tests/functions tests/modules --check bitwise

# Performance: device time, end-to-end time and allocator memory.
pytest tests/performance
```

Performance tests use rotating inputs and exclude warmup. Run them on an idle
device; ordinary `pytest` does not collect them. Results are saved to
`tests/performance/results/current.json` (git-ignored).

See [performance testing](docs/performance_test.md) for options and report comparison.
