"""Independent CPU oracle for the Triton expression's BF16 round points."""

import pytest
import torch

from mojo_opset import functions as F


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("h,w,d,t", [(32, 48, 128, 1), (32, 48, 1152, 1), (32, 48, 1152, 3), (60, 80, 1000, 3)])
def test_bf16_expression_rounding(accuracy_backend, h, w, d, t):
    _, _, device = accuracy_backend
    torch.manual_seed(42)
    weight = torch.randn(48 * 48, d, dtype=torch.bfloat16) * 0.25
    pixel = torch.arange(h * w)
    row = pixel // 4 // (w // 2) * 2 + pixel % 4 // 2
    col = pixel // 4 % (w // 2) * 2 + pixel % 2
    hr = row.float() * (47 / (h - 1))
    wr = col.float() * (47 / (w - 1))
    hf, wf = hr.int(), wr.int()
    hc, wc = (hf + 1).clamp(max=47), (wf + 1).clamp(max=47)
    dh, dw = hr - hf, wr - wf
    w11 = dh * dw
    w10, w01 = dh - w11, dw - w11
    w00 = 1.0 - dh - w01
    weights = [value[:, None].to(torch.bfloat16) for value in (w00, w01, w10, w11)]
    samples = [weight[hf * 48 + wf], weight[hf * 48 + wc], weight[hc * 48 + wf], weight[hc * 48 + wc]]
    expected = weights[0] * samples[0] + weights[1] * samples[1]
    expected = expected + weights[2] * samples[2]
    expected = expected + weights[3] * samples[3]
    actual = F.vision_pos_embed_interp(weight.to(device), t, h, w, 48, 2, weight.dtype, implementation="triton")
    torch.testing.assert_close(actual.cpu(), expected.repeat(t, 1), atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("input_dtype,output_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_mixed_dtype_rounding(accuracy_backend, input_dtype, output_dtype):
    _, _, device = accuracy_backend
    h, w, d = 32, 48, 128
    torch.manual_seed(42)
    weight = torch.randn(48 * 48, d, dtype=input_dtype) * 0.25
    pixel = torch.arange(h * w)
    row = pixel // 4 // (w // 2) * 2 + pixel % 4 // 2
    col = pixel // 4 % (w // 2) * 2 + pixel % 2
    hr = row.float() * (47 / (h - 1))
    wr = col.float() * (47 / (w - 1))
    hf, wf = hr.int(), wr.int()
    hc, wc = (hf + 1).clamp(max=47), (wf + 1).clamp(max=47)
    dh, dw = hr - hf, wr - wf
    w11 = dh * dw
    w10, w01 = dh - w11, dw - w11
    w00 = 1.0 - dh - w01
    weights = [value[:, None].to(output_dtype) for value in (w00, w01, w10, w11)]
    samples = [weight[hf * 48 + wf], weight[hf * 48 + wc], weight[hc * 48 + wf], weight[hc * 48 + wc]]
    expected = weights[0] * samples[0] + weights[1] * samples[1]
    expected = expected + weights[2] * samples[2]
    expected = expected + weights[3] * samples[3]
    actual = F.vision_pos_embed_interp(weight.to(device), 1, h, w, 48, 2, output_dtype, implementation="triton")
    torch.testing.assert_close(actual.cpu(), expected.to(output_dtype), atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("g,m,h,d", [(14, 2, 10, 33), (15, 3, 12, 17), (14, 2, 10, 32), (15, 3, 12, 32), (8, 1, 1, 64)])
def test_bf16_integral_width_generalization(accuracy_backend, g, m, h, d):
    _, _, device = accuracy_backend
    torch.manual_seed(42)
    weight = torch.randn(g * g, d, dtype=torch.bfloat16) * 0.25
    weight[:, 0] = -0.0
    weight[:, 1] = 0.0
    pixel = torch.arange(h * g)
    row = pixel // (m * m) // (g // m) * m + pixel % (m * m) // m
    col = pixel // (m * m) % (g // m) * m + pixel % m
    hr = row.float() * ((g - 1) / (h - 1) if h > 1 else 0.0)
    hf = hr.int()
    hc, wc = (hf + 1).clamp(max=g - 1), (col + 1).clamp(max=g - 1)
    dh = hr - hf
    q0, q2 = (1.0 - dh)[:, None].to(weight.dtype), dh[:, None].to(weight.dtype)
    zero = torch.zeros_like(q0)
    expected = q0 * weight[hf * g + col] + zero * weight[hf * g + wc]
    expected = expected + q2 * weight[hc * g + col]
    expected = expected + zero * weight[hc * g + wc]
    actual = F.vision_pos_embed_interp(weight.to(device), 1, h, g, g, m, weight.dtype, implementation="triton")
    repeated = F.vision_pos_embed_interp(weight.to(device), 3, h, g, g, m, weight.dtype, implementation="triton")
    # Integer views also check the sign of zero and the exact repeated bits.
    torch.testing.assert_close(actual.cpu().view(torch.int16), expected.view(torch.int16), atol=0, rtol=0)
    torch.testing.assert_close(repeated.cpu().view(torch.int16), expected.repeat(3, 1).view(torch.int16), atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("g,m,h,w,d", [(14, 2, 58, 66, 384), (15, 3, 63, 63, 128), (8, 1, 64, 64, 256), (14, 2, 42, 14, 384), (15, 1, 36, 15, 384)])
def test_bf16_large_grid_generalization(accuracy_backend, g, m, h, w, d):
    """Check partial spatial tiles and exceptional BF16 values independently."""
    _, _, device = accuracy_backend
    bits = (torch.arange(g * g * d, dtype=torch.int32) % 65536).to(torch.int16)
    weight = bits.view(torch.bfloat16).reshape(g * g, d)
    pixel = torch.arange(h * w)
    row = pixel // (m * m) // (w // m) * m + pixel % (m * m) // m
    col = pixel // (m * m) % (w // m) * m + pixel % m
    hr, wr = row.float() * ((g - 1) / (h - 1)), col.float() * ((g - 1) / (w - 1))
    hf, wf = hr.int(), wr.int()
    hc, wc = (hf + 1).clamp(max=g - 1), (wf + 1).clamp(max=g - 1)
    dh, dw = hr - hf, wr - wf
    q3 = dh * dw
    q2, q1 = dh - q3, dw - q3
    q0 = 1.0 - dh - q1
    q = [v[:, None].to(weight.dtype) for v in (q0, q1, q2, q3)]
    expected = q[0] * weight[hf * g + wf] + q[1] * weight[hf * g + wc]
    expected = expected + q[2] * weight[hc * g + wf]
    expected = expected + q[3] * weight[hc * g + wc]
    actual = F.vision_pos_embed_interp(weight.to(device), 1, h, w, g, m, weight.dtype, implementation="triton")
    actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu.isnan(), expected.isnan(), atol=0, rtol=0)
    # NaN payloads are not part of the numeric contract; all other bits,
    # including the sign of zero and infinities, must agree with the oracle.
    torch.testing.assert_close(
        torch.where(actual_cpu.isnan(), 0, actual_cpu).view(torch.int16),
        torch.where(expected.isnan(), 0, expected).view(torch.int16), atol=0, rtol=0,
    )
    repeated = F.vision_pos_embed_interp(weight.to(device), 3, h, w, g, m, weight.dtype, implementation="triton")
    torch.testing.assert_close(repeated.cpu().view(torch.int16), actual_cpu.repeat(3, 1).view(torch.int16), atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("g,h,w,d,kind", [
    (48, 60, 80, 1152, "random"),
    (48, 60, 80, 1152, "all_encodings"),
    (48, 60, 80, 1152, "max_finite"),
    (48, 60, 80, 1152, "negative_max_finite"),
    (48, 60, 80, 1152, "mixed_zero"),
    (17, 48, 60, 1008, "all_encodings"),
    (8, 16, 60, 256, "single_qnan"),
])
def test_bf16_stream_rounding_and_overflow(accuracy_backend, g, h, w, d, kind):
    """Exercise the streaming domain with an independent CPU BF16 expression."""
    _, _, device = accuracy_backend
    torch.manual_seed(42)
    if kind == "all_encodings":
        weight = (torch.arange(g * g * d, dtype=torch.int32) % 65536).to(torch.int16)
        weight = weight.view(torch.bfloat16).reshape(g * g, d)
    elif kind in ("max_finite", "negative_max_finite"):
        value = torch.finfo(torch.bfloat16).max
        weight = torch.full((g * g, d), -value if kind.startswith("negative") else value,
                            dtype=torch.bfloat16)
    elif kind == "mixed_zero":
        weight = (torch.randint(0, 2, (g * g, d), dtype=torch.int32) * 32768)
        weight = weight.to(torch.int16).view(torch.bfloat16)
    elif kind == "single_qnan":
        weight = torch.zeros(g * g, d, dtype=torch.bfloat16)
        weight.view(torch.int16)[g * g // 2, d // 2] = 0x7FC1
    else:
        weight = torch.randn(g * g, d, dtype=torch.bfloat16) * 0.25
    pixel = torch.arange(h * w)
    row = pixel // 4 // (w // 2) * 2 + pixel % 4 // 2
    col = pixel // 4 % (w // 2) * 2 + pixel % 2
    hr, wr = row.float() * ((g - 1) / (h - 1)), col.float() * ((g - 1) / (w - 1))
    hf, wf = hr.int(), wr.int()
    hc, wc = (hf + 1).clamp(max=g - 1), (wf + 1).clamp(max=g - 1)
    dh, dw = hr - hf, wr - wf
    q3 = dh * dw
    q2, q1 = dh - q3, dw - q3
    q0 = 1.0 - dh - q1
    qs = [q[:, None].to(torch.bfloat16) for q in (q0, q1, q2, q3)]
    indices = [hf * g + wf, hf * g + wc, hc * g + wf, hc * g + wc]
    products = [qs[k] * weight[indices[k]] for k in range(4)]
    expected = products[0] + products[1]
    expected = expected + products[2]
    expected = expected + products[3]
    actual = F.vision_pos_embed_interp(weight.to(device), 1, h, w, g, 2, weight.dtype,
                                     implementation="triton").cpu()
    torch.testing.assert_close(actual.isnan(), expected.isnan(), atol=0, rtol=0)
    # NaN payload is unspecified; every other BF16 encoding, including signed
    # zero, infinity and subnormals, must agree exactly.
    comparable = ~expected.isnan()
    torch.testing.assert_close(actual.view(torch.int16)[comparable],
                               expected.view(torch.int16)[comparable], atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.accuracy
def test_bf16_stream_time_and_batch_bits(accuracy_backend):
    _, _, device = accuracy_backend
    g, h, w, d = 48, 60, 80, 128
    bits = (torch.arange(g * g * d, dtype=torch.int32) % 65536).to(torch.int16)
    weight = bits.view(torch.bfloat16).reshape(g * g, d).to(device)
    single = F.vision_pos_embed_interp(weight, 1, h, w, g, 2, weight.dtype,
                                      implementation="triton").cpu()
    video = F.vision_pos_embed_interp(weight, 3, h, w, g, 2, weight.dtype,
                                     implementation="triton").cpu()
    batch = F.vision_pos_embed_interp_batch(weight, [(1, h, w), (1, 8, 16)], g, 2,
                                           weight.dtype, implementation="triton").cpu()
    # These API routes must preserve even the same NaN payload and zero sign.
    torch.testing.assert_close(video.view(torch.int16), single.repeat(3, 1).view(torch.int16),
                               atol=0, rtol=0)
    torch.testing.assert_close(batch[:h * w].view(torch.int16), single.view(torch.int16),
                               atol=0, rtol=0)
