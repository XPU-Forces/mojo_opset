# SPDX-License-Identifier: Apache-2.0
"""A2/A3 SIMD span primitive using Triton-Ascend's public custom-op API.

The caller keeps allocation, GM transfers, spatial merge and frame broadcasting
in Triton. The bitcode primitive preserves all BF16 round points explicitly;
each contiguous source span is converted once and reused by neighboring pixels.
CANN's compiler builds the device bitcode once in Triton's persistent cache.
"""

import functools
import hashlib
import math
import os
import shutil
import subprocess
import tempfile

from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from triton.runtime.cache import get_cache_manager

from ._npu_triton_utils import get_num_cores
from ._npu_vision_pos_embed_interp import _coefficient_fp32


_DEVICE_SOURCE = r"""

using i64=long long;
template<typename T,unsigned D>struct MemRef {T*allocated;T*aligned;i64 offset;i64 sizes[D];i64 strides[D];};
__aicore__ __attribute__((always_inline)) float rb(float v){union{float f;unsigned u;}a;a.f=v;a.u=(a.u+32767+((a.u>>16)&1))&0xffff0000;return a.f;}
extern "C" __aicore__ __attribute__((always_inline)) void _mlir_ciface_vision_pos_embed_bf16_span4(
 MemRef<__ubuf__ __bf16,1>*a,MemRef<__ubuf__ __bf16,1>*b,float dh,int col0,int base,float ws,int pad,int maxcol,
 MemRef<__ubuf__ __bf16,1>*y,MemRef<__ubuf__ float,1>*f,MemRef<__ubuf__ __bf16,1>*h) {
 const i64 n=y->sizes[0]/4;const i64 z=n*4;const i64 zp=z+pad;auto fp=f->aligned+f->offset;auto hp=h->aligned+h->offset;auto yp=y->aligned+y->offset;
 auto ap=a->aligned+a->offset;auto bp=b->aligned+b->offset;
 auto sp=(__ubuf__ float*)hp;const i64 sn=a->sizes[0];
 set_mask_count();if(pad){set_vector_mask(0,pad);for(int k=0;k<4;++k)vector_dup(fp+k*zp+z,0.0f,1,1,1,8,0);pipe_barrier(PIPE_V);}
 set_vector_mask(0,sn);
 vconv_bf162f32(sp,ap,1,1,1,8,4);
 vconv_bf162f32(sp+sn,bp,1,1,1,8,4);pipe_barrier(PIPE_V);
 set_vector_mask(0,n);
 for(int i=0;i<4;++i){
 float wr=float(col0+i)*ws;int wf=int(wr);float dw=wr-float(wf);float q3=dh*dw,q2=dh-q3,q1=dw-q3,q0=(1.f-dh)-q1;
 int lo=wf-base,hi=(wf<maxcol?wf+1:maxcol)-base;
 vmuls(fp+i*n,sp+lo*n,rb(q0),1,1,1,8,8);
 vmuls(fp+zp+i*n,sp+hi*n,rb(q1),1,1,1,8,8);
 vmuls(fp+2*zp+i*n,sp+sn+lo*n,rb(q2),1,1,1,8,8);
 vmuls(fp+3*zp+i*n,sp+sn+hi*n,rb(q3),1,1,1,8,8);
 }
 pipe_barrier(PIPE_V);set_vector_mask(0,4*zp);
 vconv_f322bf16r(hp,fp,1,1,1,4,8);pipe_barrier(PIPE_V);
 vconv_bf162f32(fp,hp,1,1,1,8,4);pipe_barrier(PIPE_V);set_vector_mask(0,z);
vadd(fp,fp,fp+1*zp,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
 vconv_f322bf16r(hp,fp,1,1,1,4,8);pipe_barrier(PIPE_V);
 vconv_bf162f32(fp,hp,1,1,1,8,4);pipe_barrier(PIPE_V);
vadd(fp,fp,fp+2*zp,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
 vconv_f322bf16r(hp,fp,1,1,1,4,8);pipe_barrier(PIPE_V);
 vconv_bf162f32(fp,hp,1,1,1,8,4);pipe_barrier(PIPE_V);
vadd(fp,fp,fp+3*zp,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);vconv_f322bf16r(yp,fp,1,1,1,4,8);set_mask_norm();}
// AXPY is used only for exact zero products; all nonzero products retain
// their separate BF16 round points and additions remain left-associated.
extern "C" __aicore__ __attribute__((always_inline)) void _mlir_ciface_vision_pos_embed_bf16_integral8(
MemRef<__ubuf__ __bf16,1>*top,MemRef<__ubuf__ __bf16,1>*bottom,float q0,float q2,int width,int edge,int last,
MemRef<__ubuf__ __bf16,1>*y,MemRef<__ubuf__ float,1>*f){
i64 n=y->sizes[0];const int pad=32;auto a=top->aligned+top->offset;auto c=bottom->aligned+bottom->offset;auto x=f->aligned+f->offset;auto b=x+n+pad;auto z=b+n+pad;auto out=y->aligned+y->offset;
set_mask_count();set_vector_mask(0,n+width);vconv_bf162f32(z,a,1,1,1,8,4);pipe_barrier(PIPE_V);
set_vector_mask(0,n);vmuls(x,z,q0,1,1,1,8,8);pipe_barrier(PIPE_V);vconv_f322bf16r(out,x,1,1,1,4,8);pipe_barrier(PIPE_V);vconv_bf162f32(x,out,1,1,1,8,4);pipe_barrier(PIPE_V);
if(edge){set_vector_mask(0,width);vconv_bf162f32(z+(last+1)*width,a+last*width,1,1,1,8,4);set_vector_mask(0,n);pipe_barrier(PIPE_V);}
vaxpy(x,z+width,0.0f,1,1,1,8,8);pipe_barrier(PIPE_V);
set_vector_mask(0,n+width);vconv_bf162f32(z,c,1,1,1,8,4);pipe_barrier(PIPE_V);set_vector_mask(0,n);vmuls(b,z,q2,1,1,1,8,8);pipe_barrier(PIPE_V);vconv_f322bf16r(out,b,1,1,1,4,8);pipe_barrier(PIPE_V);vconv_bf162f32(b,out,1,1,1,8,4);pipe_barrier(PIPE_V);
vadd(x,x,b,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
if(edge){set_vector_mask(0,width);vconv_bf162f32(z+(last+1)*width,c+last*width,1,1,1,8,4);set_vector_mask(0,n);pipe_barrier(PIPE_V);}
vaxpy(x,z+width,0.0f,1,1,1,8,8);pipe_barrier(PIPE_V);vconv_f322bf16r(out,x,1,1,1,4,8);set_mask_norm();}
"""


@functools.lru_cache(maxsize=1)
def _device_bitcode():
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("CANN bisheng is required to compile the BF16 SIMD span primitive")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True).stdout
    key = hashlib.sha256(_DEVICE_SOURCE.encode() + os.fsencode(compiler) + version).hexdigest()
    manager = get_cache_manager(key)
    cached = manager.get_file("vision_pos_embed_bf16_span4.bc")
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory(prefix="vision-pos-embed-") as directory:
        source = Path(directory) / "span.cpp"
        bitcode = Path(directory) / "span.bc"
        source.write_text(_DEVICE_SOURCE)
        subprocess.run(
            [compiler, "-x", "cce", "--cce-aicore-arch=dav-c220-vec", "--cce-aicore-only",
             "-O2", "-ffp-contract=off", "-c", "-emit-llvm", str(source), "-o", str(bitcode)],
            check=True, capture_output=True,
        )
        return manager.put(bitcode.read_bytes(), "vision_pos_embed_bf16_span4.bc", binary=True)


def _scratch_elements(elements):
    padding = (64 - elements % 128) % 128
    return (elements + padding) * 4


@al.register_custom_op
class vision_pos_embed_bf16_span4:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "vision_pos_embed_bf16_span4"
    bitcode = ""

    def __init__(self, top, bottom, dh, col0, base, ws, pad, maxcol, out=None):
        # Each neighbor plane may need 64 FP32 elements of bank padding.
        elements = _scratch_elements(out.numel)
        source_elements = max(elements, 4 * top.numel)
        self.extra_buffers = [(tl.float32, elements), (tl.bfloat16, source_elements)]


@triton.jit
def _span_kernel(
    embed_ptr, output_ptr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
    HS: tl.constexpr, WS: tl.constexpr, CORES: tl.constexpr, COMPILE_KEY: tl.constexpr,
):
    for task in range(tl.program_id(0), H * triton.cdiv(W, 4), CORES):
        row = task // triton.cdiv(W, 4)
        col0 = task % triton.cdiv(W, 4) * 4
        hr = row.to(tl.float32) * HS
        hf = hr.to(tl.int32)
        hc = tl.minimum(hf + 1, G - 1)
        dh = hr - hf.to(tl.float32)
        base = (col0.to(tl.float32) * WS).to(tl.int32)
        source_cols = tl.arange(0, S)
        features = tl.arange(0, C)
        offsets = (base + source_cols[:, None]) * D + features[None, :]
        mask = (base + source_cols[:, None] < G) & (features[None, :] < D)
        top = tl.load(embed_ptr + hf * G * D + offsets, mask, 0).reshape(S * C)
        bottom = tl.load(embed_ptr + hc * G * D + offsets, mask, 0).reshape(S * C)
        PAD: tl.constexpr = (64 - (4 * C) % 128) % 128
        value = al.custom(
            "vision_pos_embed_bf16_span4", top, bottom, dh, col0, base,
            tl.full((), WS, tl.float32), tl.full((), PAD, tl.int32), tl.full((), G - 1, tl.int32),
            out=tl.full((4 * C,), 0, tl.bfloat16),
        )
        frames = tl.arange(0, triton.next_power_of_2(T))
        for j in tl.static_range(4):
            col = col0 + j
            pixel = ((row // M) * (W // M) + col // M) * M * M + row % M * M + col % M
            part = al.extract_slice(value, (j * C,), (C,), (1,))
            repeated = tl.broadcast_to(part[None, :], (triton.next_power_of_2(T), C))
            tl.store(
                output_ptr + (frames[:, None] * H * W + pixel) * D + features[None, :], repeated,
                (frames[:, None] < T) & (col < W) & (features[None, :] < D),
            )


def launch_span(embed_weight, t, h, w, g, m):
    d = embed_weight.shape[1]
    c = triton.cdiv(d, 16) * 16
    hs = float(g - 1) / (h - 1) if h > 1 else 0.0
    ws = float(g - 1) / (w - 1) if w > 1 else 0.0
    span = math.ceil(3 * ws) + 2
    vision_pos_embed_bf16_span4.bitcode = _device_bitcode()
    # Custom-op registry mutations are not tracked by Triton's dependency
    # finder. Include the content-addressed bitcode path and scratch geometry
    # in the specialization even though they do not affect device arithmetic.
    compile_key = (vision_pos_embed_bf16_span4.bitcode, _scratch_elements(4 * c),
                   max(_scratch_elements(4 * c), 4 * span * c))
    output = torch.empty((t * h * w, d), device=embed_weight.device, dtype=embed_weight.dtype)
    cores = get_num_cores()
    with torch.npu.device(embed_weight.device):
        _span_kernel[(cores,)](
            embed_weight, output, t, h, w, g, m, d, c, span, hs, ws, cores, compile_key,
            enable_fp_fusion=False, multibuffer=True,
        )
    return output


@al.register_custom_op
class vision_pos_embed_bf16_integral8:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "vision_pos_embed_bf16_integral8"
    bitcode = ""

    def __init__(self, top, bottom, q0, q2, width, edge, last, out=None):
        self.extra_buffers = [(tl.float32, out.numel * 3 + top.numel // 9 + 64)]


@triton.jit
def _integral_kernel(
    embed_ptr, output_ptr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    HS: tl.constexpr, CORES: tl.constexpr, COMPILE_KEY: tl.constexpr,
):
    R: tl.constexpr = 8
    for task in range(tl.program_id(0), H * triton.cdiv(W, R), CORES):
        row = task // triton.cdiv(W, R)
        col0 = task % triton.cdiv(W, R) * R
        hr = row.to(tl.float32) * HS
        hf = hr.to(tl.int32)
        hc = tl.minimum(hf + 1, G - 1)
        dh = hr - hf.to(tl.float32)
        source = tl.arange(0, (R + 1) * D)
        top = tl.load(embed_ptr + (hf * G + col0) * D + source, source < (G - col0) * D, 0)
        bottom = tl.load(embed_ptr + (hc * G + col0) * D + source, source < (G - col0) * D, 0)
        value = al.custom(
            "vision_pos_embed_bf16_integral8", top, bottom,
            _coefficient_fp32(1.0 - dh), _coefficient_fp32(dh), tl.full((), D, tl.int32),
            (col0 + R >= W).to(tl.int32), tl.minimum(R - 1, W - 1 - col0),
            out=tl.full((R * D,), 0, tl.bfloat16),
        )
        group = col0 // M + tl.arange(0, R // M)
        features = tl.arange(0, M * D)
        pixel_base = ((row // M) * (W // M) + group[:, None]) * M * M
        address = (pixel_base + row % M * M) * D + features[None, :]
        for frame in tl.static_range(T):
            if W % R == 0:
                tl.store(output_ptr + frame * H * W * D + address, value.reshape(R // M, M * D))
            else:
                tl.store(output_ptr + frame * H * W * D + address,
                         value.reshape(R // M, M * D), group[:, None] < W // M)


def launch_integral(embed_weight, t, h, w, g, m):
    d = embed_weight.shape[1]
    hs = float(g - 1) / (h - 1) if h > 1 else 0.0
    bitcode = _device_bitcode()
    vision_pos_embed_bf16_integral8.bitcode = bitcode
    output = torch.empty((t * h * w, d), device=embed_weight.device, dtype=embed_weight.dtype)
    cores = get_num_cores()
    with torch.npu.device(embed_weight.device):
        _integral_kernel[(cores,)](
            embed_weight, output, t, h, w, g, m, d, hs, cores, (bitcode, 25 * d + 64),
            enable_fp_fusion=False, multibuffer=False,
        )
    return output
