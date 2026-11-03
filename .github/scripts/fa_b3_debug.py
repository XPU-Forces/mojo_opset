"""Temporary, isolated B3 diagnosis; production sources are patched only in this job."""

import sys
from pathlib import Path


def probe():
    import ctypes
    import torch
    import torch_npu

    torch.npu.set_device(0)
    print("device", torch.npu.get_device_name(0), flush=True)
    print("torch", torch.__version__, "torch_npu", torch_npu.__version__, flush=True)
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetDeviceInfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    for name, attribute in (("cube", 102), ("vector", 201)):
        value = ctypes.c_int64()
        result = acl.aclrtGetDeviceInfo(0, attribute, ctypes.byref(value))
        assert result == 0, result
        print(name, value.value, flush=True)


def select():
    path = Path("native/CMakeLists.txt")
    source = path.read_text()
    anchor = "foreach(recipe IN LISTS _recipes)"
    assert source.count(anchor) == 1
    path.write_text(source.replace(anchor, 'list(FILTER _recipes INCLUDE REGEX "/fa_(fwd|bwd)/CMakeLists.txt$")\n' + anchor))


def patch():
    path = Path("native/src/npu_a2/fa_bwd/vec_op_det.h")
    source = path.read_text()
    function = source.index("void VecOpDet<FAGT>::DetVector2(")
    start = source.index("  if (vecCoreIdx < 16) {", function)
    end = source.index("  wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdAPing);", start)
    body = source[start:end]
    replacement = (
        "  const uint32_t physicalVecCoreIdx = vecCoreIdx;\n"
        "  // Preserve 48 logical reduction workers and their 16 KiB buffer bounds.\n"
        "  for (uint32_t logicalVecCoreIdx = physicalVecCoreIdx; logicalVecCoreIdx < 48;\n"
        "       logicalVecCoreIdx += tilingData->coreNum) {\n"
        "    vecCoreIdx = logicalVecCoreIdx;\n"
        + "".join("  " + line if line.strip() else line for line in body.splitlines(keepends=True))
        + "  }\n  vecCoreIdx = physicalVecCoreIdx;\n\n"
    )
    path.write_text(source[:start] + replacement + source[end:])


def gradients():
    import torch
    from mojo_opset import functions as functions

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    inputs = [torch.randn(256, heads, 128, device="npu", dtype=torch.float16, requires_grad=True) for heads in (4, 2, 2)]
    candidate = [x.detach().clone().requires_grad_() for x in inputs]
    cu = torch.tensor([0, 256], dtype=torch.int32, device="npu")
    options = dict(max_q_len=256, max_k_len=256, causal=False)
    expected = functions.flash_attention(*inputs, cu, cu, implementation="torch", **options)
    actual = functions.flash_attention(*candidate, cu, cu, implementation="native", **options)
    dy = torch.randn_like(expected)
    reference_grads = torch.autograd.grad(expected, inputs, dy)
    actual_grads = torch.autograd.grad(actual, candidate, dy)
    for name, value, reference in zip(("output", "dq", "dk", "dv"), (actual, *actual_grads), (expected, *reference_grads)):
        bad = ~torch.isclose(value, reference, rtol=1e-3, atol=1e-3)
        print(name, "mismatches", bad.sum().item(), "of", bad.numel(), "max_error", (value-reference).abs().max().item(), flush=True)
        print(name, "zero_rows", (value == 0).all(dim=-1).all(dim=-1).nonzero().flatten().cpu().tolist(), flush=True)


{"probe": probe, "select": select, "patch": patch, "gradients": gradients}[sys.argv[1]]()
