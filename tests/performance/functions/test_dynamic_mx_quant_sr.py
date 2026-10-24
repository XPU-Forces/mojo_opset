from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from mojo_opset import functions

# Sizes select the actual optimization categories on dav-3510, alongside
# each public layout. Full-tile admission also depends on the current quota.
_MXQ_CASES = [
    ("001_tail_16k", (128, 128), -1),
    ("002_tail_1m", (1024, 1024), -1),
    ("003_stride2_fused", (2, 1024, 2), 1),
    ("004_stride2", (2, 1152, 2), 1),
    ("005_width8", (4096, 8), 0),
    ("006_width16", (2048, 16), 0),
    ("007_width32", (512, 128, 32), -2),
    ("008_width64", (256, 128, 64), -2),
    ("009_width128", (128, 128, 128), -2),
    ("010_transpose", (64, 48, 3), 1),
    ("011_eight_below", (8191, 256), -1),
    ("012_eight_at", (8192, 256), -1),
    ("013_eight_tail", (8193, 256), -1),
    ("014_full_wave56", (56 * 128, 256), -1),
    ("015_partial_wave56", (56 * 128 + 1, 256), -1),
    ("016_compact", (2048, 16384), -1),
    ("017_crossdraw", (32768, 64), -1),
    ("018_width16_five_waves56", (56 * 5, 256, 16), 1),
]


def _execution_metadata(op, prepared):
    """Read the actual cache key rather than recomputing policy thresholds."""
    with op._PROGRAM_CACHE._lock:
        key = next((key for key, program in op._PROGRAM_CACHE._programs.items() if program is prepared.program), None)
    if key is None:
        raise RuntimeError("SR performance requires enabled Program caching and disabled diagnostic compilation")
    identity = asdict(key)
    identity["layout"] = key.layout.value
    identity["codegen_options"] = key.codegen_options.hex()
    return {"layout": prepared.plan.layout.value, "blocks": prepared.plan.blocks, "compile_key": identity}


@pytest.fixture
def dynamic_mx_quant_sr_perf_backend(perf_environment):
    from mojo_opset import Target
    from mojo_opset.functions._dispatch import resolve_implementation

    _, device, target, implementation = perf_environment
    selected = implementation or resolve_implementation("dynamic_mx_quant_sr", Target.parse(target))
    if selected == "torch":
        return selected, device
    if not target.startswith("npu.a5") or selected != "cannbotdsl":
        pytest.skip("Dynamic MX quant SR performance currently has an A5 cannbotdsl provider only")
    return selected, device


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("case_id,shape,axis", _MXQ_CASES, ids=[case_id for case_id, *_ in _MXQ_CASES])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("dst_type", [24, 23], ids=["e4", "e5"])
@pytest.mark.parametrize("scale_alg,bound", [(0, 0.0), (1, 0.0), (1, 32.0)], ids=["floor", "ceil", "bound"])
@pytest.mark.parametrize("entry", ["public", "replay"])
def test_dynamic_mx_quant_sr(
    benchmark, dynamic_mx_quant_sr_perf_backend, case_id, shape, axis, dtype, dst_type, scale_alg, bound, entry
):
    implementation, device = dynamic_mx_quant_sr_perf_backend
    if entry == "replay" and implementation == "torch":
        pytest.skip("preallocated replay measures the DSL provider")
    execution = {}

    def factory():
        generator = torch.Generator().manual_seed(43)
        cpu = (torch.randn(shape, generator=generator) * 7).to(dtype)
        x = cpu.to(device)
        attributes = dict(axis=axis, dst_type=dst_type, scale_alg=scale_alg, max_low_bound=bound)
        reference = functions.dynamic_mx_quant_sr(cpu, implementation="torch", **attributes)
        if implementation != "torch":
            from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as op

            target = torch.float8_e5m2 if dst_type == 23 else torch.float8_e4m3fn
            prepared = op._prepare_launch(x, target, axis % x.ndim, scale_alg, op._encode_bound(bound))
            execution.update(_execution_metadata(op, prepared))
        else:
            execution.update(layout="reference", provider="torch")
        if entry == "replay":
            state = op._framework_rng._make_rng_state(0, 0, device=x.device)
            prepared.launch(rng_state=state)
            actual = prepared.outputs()

            def run():
                prepared.launch(rng_state=state)

        else:
            # Validate fixed-state bytes outside timing, then restore the caller.
            # Timed public calls reserve fresh framework state on every invocation.
            if implementation != "torch":
                saved = torch.npu.get_rng_state(x.device)
                try:
                    torch.npu.default_generators[x.device.index].manual_seed(0)
                    actual = functions.dynamic_mx_quant_sr(x, implementation=implementation, **attributes)
                finally:
                    torch.npu.set_rng_state(saved, x.device)
            else:
                actual = functions.dynamic_mx_quant_sr(x, implementation=implementation, **attributes)

            def run():
                return functions.dynamic_mx_quant_sr(x, implementation=implementation, **attributes)

        for value, expected in zip(actual, reference):
            assert torch.equal(value.view(torch.uint8).cpu(), expected.view(torch.uint8).cpu())
        return run

    benchmark(
        factory=factory,
        op="dynamic_mx_quant_sr",
        case_id=case_id,
        shape=str(shape),
        axis=axis,
        dtype=str(dtype).removeprefix("torch."),
        dst_type=dst_type,
        scale_alg=scale_alg,
        max_low_bound=bound,
        entry=entry,
        rng_abi=60,
        execution=execution,
    )
