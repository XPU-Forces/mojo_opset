"""Independent decoding and FP32 accumulation for dense GEMM validation."""
import torch


def _decode(x):
    if x.dtype != torch.int8:
        return x.float()
    transposed = x.stride(1) != 1
    packed = x.t() if transposed else x
    bits = packed.to(torch.int32) & 255
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=x.device)
    values = torch.stack((lut[(bits & 15).long()], lut[(bits >> 4).long()]), dim=-1)
    values = values.reshape(packed.shape[0], packed.shape[1] * 2)
    return values.t() if transposed else values


def _scale(sf, rows, k, gran):
    if sf.dtype == torch.int16:
        bits = sf.to(torch.int32) & 65535
        exponents = torch.stack((bits & 255, (bits >> 8) & 255), dim=-1).flatten(-2)
        sf = torch.pow(2.0, exponents.float() - 127)
    return sf.float().repeat_interleave(gran, 0).repeat_interleave(32, 1)[:rows, :k]


def dense_gemm_fwd(input, weight, input_scale, weight_scale, trans_input, trans_weight,
                   output_dtype, c, alpha, gran_m, gran_n, compiled_dims):
    a = _decode(input.t() if trans_input else input)
    b = _decode(weight if trans_weight else weight.t())
    if input_scale is not None:
        a = a * _scale(input_scale, *a.shape, gran_m)
        b = b * _scale(weight_scale, *b.shape, gran_n)
    out = a @ b.t()
    if alpha is not None:
        out = out * alpha
    if c is not None:
        out = out + c.float()
    return out.to(output_dtype)


gemm_fwd = dense_gemm_fwd
mxfp8_dequant_gemm_fwd = dense_gemm_fwd
mxfp4_gemm_fwd = dense_gemm_fwd
