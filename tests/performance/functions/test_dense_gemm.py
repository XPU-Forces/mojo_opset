import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions as F
from mojo_opset.functions._dispatch import resolve_implementation

_CASES = [
    pytest.param(op, packed_sf, marks=pytest.mark.api(f"functions.{op}"))
    for op in ("gemm", "mxfp8_dequant_gemm", "mxfp4_gemm")
    for packed_sf in ((False,) if op == "gemm" else (False, True))
]


@pytest.mark.parametrize("op,packed_sf", _CASES)
@pytest.mark.parametrize("m,n,k", [(256, 256, 256), (1024, 1024, 1024), (32, 4096, 4096), (4096, 4096, 4096)])
def test_dense_gemm(benchmark, perf_environment, op, packed_sf, m, n, k):
    _, device, target, implementation = perf_environment
    selected = implementation or resolve_implementation(op, Target.parse(target))
    if selected == "native":
        try:
            from mojo_opset_lib.npu_a5 import _native

            getattr(_native, "dense_gemm")
        except (ImportError, AttributeError):
            pytest.skip("dense_gemm native build requires CANN 9.2+")

    def factory():
        if op == "gemm":
            a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            b = torch.randn(n, k, dtype=torch.bfloat16, device=device)
            return lambda: F.gemm(a, b, trans_weight=True, output_dtype=torch.bfloat16, implementation=implementation)
        a_dtype = torch.int8 if op == "mxfp4_gemm" else torch.float8_e4m3fn
        a = (
            torch.randint(-128, 128, (m, k // 2), dtype=torch.int8, device="cpu").to(device)
            if op == "mxfp4_gemm"
            else torch.randn(m, k, device="cpu").to(a_dtype).to(device)
        )
        b = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, device="cpu").to(device)
        if packed_sf:
            sa = torch.full(((k + 63) // 64, m), 0x7F7F, dtype=torch.int16, device=device).t()
            sb = torch.full(((k + 63) // 64, n), 0x7F7F, dtype=torch.int16, device=device).t()
        else:
            sa = torch.ones((m, (k + 31) // 32), dtype=torch.float32, device=device)
            sb = torch.ones((n, (k + 31) // 32), dtype=torch.float32, device=device)
        fn = getattr(F, op)
        return lambda: fn(
            a, b, sa, sb, trans_weight=True, output_dtype=torch.bfloat16, implementation=implementation
        )

    benchmark(
        factory=factory,
        op=op,
        m=m,
        n=n,
        k=k,
        trans_weight=True,
        packed_sf=packed_sf,
        dtype="bfloat16",
        phase="forward",
    )
