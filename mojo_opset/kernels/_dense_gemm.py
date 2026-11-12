"""Shared metadata contract for dense BF16 and MX GEMMs."""
import torch


def logical_shape(x):
    if x.ndim != 2:
        raise ValueError("dense GEMM operands must be matrices")
    rows, cols = x.shape
    if x.stride(0) != 1 and x.stride(1) != 1:
        raise ValueError("dense GEMM requires one contiguous matrix axis")
    if x.dtype == torch.int8:
        if x.stride(0) == 1 and x.stride(1) == 1:
            raise ValueError("ambiguous FP4 packing axis")
        if x.stride(1) == 1:
            cols *= 2
        else:
            rows *= 2
    return rows, cols


def check(op, a, b, sa, sb, trans_input, trans_weight, output_dtype, c, alpha, recipe, compiled_dims):
    pairs = {"gemm": (torch.bfloat16, torch.bfloat16),
             "mxfp8_dequant_gemm": (torch.float8_e4m3fn, torch.int8),
             "mxfp4_gemm": (torch.int8, torch.int8)}
    if (a.dtype, b.dtype) != pairs[op]:
        raise ValueError(f"{op} requires operand dtypes {pairs[op]}; FP4 uses packed int8 storage")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("output_dtype must be bfloat16 or float32")
    if (len(recipe) != 3
            or any(not isinstance(g, int) or isinstance(g, bool) or g <= 0 for g in recipe)
            or recipe[2] != 32):
        raise ValueError("recipe must be positive (gran_m, gran_n, 32)")
    if any(g > 128 or g & (g - 1) for g in recipe[:2]):
        raise ValueError("recipe MN granularities must be powers of two in [1,128]")
    if any(dim not in "mnk" for dim in compiled_dims):
        raise ValueError("compiled_dims may only contain m, n, k")
    a = a.t() if trans_input else a
    b = b if trans_weight else b.t()
    m, k = logical_shape(a)
    n, bk = logical_shape(b)
    if (a.dtype == torch.int8 or b.dtype == torch.int8) and k % 2:
        raise ValueError("FP4 GEMM requires an even logical K")
    if k != bk:
        raise ValueError("operand K dimensions differ")
    for x in (b, sa, sb, c):
        if x is not None and x.device != a.device:
            raise ValueError("all GEMM tensors must share a device")
    if op == "gemm":
        if sa is not None or sb is not None:
            raise ValueError("BF16 GEMM does not take scales")
    else:
        for sf, mn, gran in ((sa, m, recipe[0]), (sb, n, recipe[1])):
            if sf is None or sf.dtype not in (torch.float32, torch.int16):
                raise ValueError("scales must be float32 or packed UE8M0 int16")
            div = 32 if sf.dtype == torch.float32 else 64
            if sf.shape != ((mn + gran - 1) // gran, (k + div - 1) // div):
                raise ValueError("scale shape must match logical MN/K and recipe")
            if sf.stride(0) != 1 and sf.stride(1) != 1:
                raise ValueError("scales require one contiguous matrix axis")
    if c is not None:
        if c.dtype != output_dtype or c.shape not in ((m, n), (1, n)):
            raise ValueError("C must have output dtype and shape [M,N] or [1,N]")
        if c.shape[0] == 1:
            if alpha is not None or not c.is_contiguous():
                raise ValueError("row bias must be contiguous and cannot be combined with alpha")
    return m, n, k
