# SPDX-License-Identifier: Apache-2.0
"""A5 native BF16 vector arithmetic for contiguous interpolation spans.

Triton owns GM loads, row scheduling and spatial merge. Three independent
128-element chunks interleave the four BF16 multiplies and three left-associated
BF16 adds without changing the upstream rounding points.
"""
import functools
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime.cache import get_cache_manager

from ._npu_triton_utils import get_num_cores

_DEVICE_SOURCE = r"""
#include <stdint.h>
using i64=long long;
template<typename T,unsigned D>struct MemRef {T*allocated;T*aligned;i64 offset;i64 sizes[D];i64 strides[D];};
__aicore__ __attribute__((always_inline)) unsigned rb(float x){union{float f;unsigned u;}b;b.f=x;return (b.u+32767+((b.u>>16)&1))>>16;}
extern "C" __aicore__ __attribute__((always_inline)) void _mlir_ciface_vision_pos_embed_bf16_vf_span16(
 MemRef<__ubuf__ __bf16,1>*top,MemRef<__ubuf__ __bf16,1>*bottom,float dh,int col0,int base,float ws,int rows,int maxcol,
 MemRef<__ubuf__ __bf16,1>*y) {
 auto ap=top->aligned+top->offset;auto bp=bottom->aligned+bottom->offset;auto yp=y->aligned+y->offset;unsigned n=y->sizes[0]/rows;
 for(int j=0;j<rows;j++){
  float wr=float(col0+j)*ws;int wf=int(wr);float dw=wr-float(wf);
  float q3=dh*dw,q2=dh-q3,q1=dw-q3,q0=(1.f-dh)-q1;
  unsigned b0=rb(q0)*65537u,b1=rb(q1)*65537u,b2=rb(q2)*65537u,b3=rb(q3)*65537u;
  int lo=wf-base,hi=(wf<maxcol?wf+1:maxcol)-base;
  auto a=ap+lo*n;auto b=ap+hi*n;auto c=bp+lo*n;auto d=bp+hi*n;auto out=yp+j*n;
  __VEC_SCOPE__ {
   vector_bool all=pset_b16(PAT_ALL);vector_u32 t0,t1,t2,t3;
   vbr(t0,b0);vbr(t1,b1);vbr(t2,b2);vbr(t3,b3);
   uint16_t main_repeats=n/384;
   for(uint16_t i=0;i<main_repeats;i++){
    vector_bf16 v0a,v0b,v0c,v0d;
    vlds(v0a,a,i*384+0,NORM);vlds(v0b,b,i*384+0,NORM);vlds(v0c,c,i*384+0,NORM);vlds(v0d,d,i*384+0,NORM);
    vector_bf16 v1a,v1b,v1c,v1d;
    vlds(v1a,a,i*384+128,NORM);vlds(v1b,b,i*384+128,NORM);vlds(v1c,c,i*384+128,NORM);vlds(v1d,d,i*384+128,NORM);
    vector_bf16 v2a,v2b,v2c,v2d;
    vlds(v2a,a,i*384+256,NORM);vlds(v2b,b,i*384+256,NORM);vlds(v2c,c,i*384+256,NORM);vlds(v2d,d,i*384+256,NORM);
    vmul(v0a,v0a,(vector_bf16&)t0,all,MODE_ZEROING);
    vmul(v0b,v0b,(vector_bf16&)t1,all,MODE_ZEROING);
    vmul(v0c,v0c,(vector_bf16&)t2,all,MODE_ZEROING);
    vmul(v0d,v0d,(vector_bf16&)t3,all,MODE_ZEROING);
    vmul(v1a,v1a,(vector_bf16&)t0,all,MODE_ZEROING);
    vmul(v1b,v1b,(vector_bf16&)t1,all,MODE_ZEROING);
    vmul(v1c,v1c,(vector_bf16&)t2,all,MODE_ZEROING);
    vmul(v1d,v1d,(vector_bf16&)t3,all,MODE_ZEROING);
    vmul(v2a,v2a,(vector_bf16&)t0,all,MODE_ZEROING);
    vmul(v2b,v2b,(vector_bf16&)t1,all,MODE_ZEROING);
    vmul(v2c,v2c,(vector_bf16&)t2,all,MODE_ZEROING);
    vmul(v2d,v2d,(vector_bf16&)t3,all,MODE_ZEROING);
    vadd(v0a,v0a,v0b,all,MODE_ZEROING);
    vadd(v1a,v1a,v1b,all,MODE_ZEROING);
    vadd(v2a,v2a,v2b,all,MODE_ZEROING);
    vadd(v0a,v0a,v0c,all,MODE_ZEROING);
    vadd(v1a,v1a,v1c,all,MODE_ZEROING);
    vadd(v2a,v2a,v2c,all,MODE_ZEROING);
    vadd(v0a,v0a,v0d,all,MODE_ZEROING);
    vadd(v1a,v1a,v1d,all,MODE_ZEROING);
    vadd(v2a,v2a,v2d,all,MODE_ZEROING);
    vsts(v0a,out,i*384+0,NORM_B16,all);
    vsts(v1a,out,i*384+128,NORM_B16,all);
    vsts(v2a,out,i*384+256,NORM_B16,all);
   }
   uint16_t tail_repeats=(n%384)/128;
   for(uint16_t i=0;i<tail_repeats;i++){
    vector_bf16 ra,rb,rc,rd;
    vlds(ra,a,main_repeats*384+i*128,NORM);vlds(rb,b,main_repeats*384+i*128,NORM);vlds(rc,c,main_repeats*384+i*128,NORM);vlds(rd,d,main_repeats*384+i*128,NORM);
    vmul(ra,ra,(vector_bf16&)t0,all,MODE_ZEROING);
    vmul(rb,rb,(vector_bf16&)t1,all,MODE_ZEROING);
    vmul(rc,rc,(vector_bf16&)t2,all,MODE_ZEROING);
    vmul(rd,rd,(vector_bf16&)t3,all,MODE_ZEROING);
    vadd(ra,ra,rb,all,MODE_ZEROING);
    vadd(ra,ra,rc,all,MODE_ZEROING);
    vadd(ra,ra,rd,all,MODE_ZEROING);
    vsts(ra,out,main_repeats*384+i*128,NORM_B16,all);
   }
  }
 }
}
"""

@functools.lru_cache(maxsize=1)
def _device_bitcode():
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("CANN bisheng is required to compile the BF16 SIMD span primitive")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True).stdout
    key = hashlib.sha256(_DEVICE_SOURCE.encode() + os.fsencode(compiler) + version).hexdigest()
    manager = get_cache_manager(key)
    cached = manager.get_file("vision_pos_embed_bf16_vf_span16.bc")
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory(prefix="vision-pos-embed-") as directory:
        source = Path(directory) / "span.cpp"
        bitcode = Path(directory) / "span.bc"
        source.write_text(_DEVICE_SOURCE)
        subprocess.run(
            [compiler, "-x", "cce", "--cce-aicore-arch=dav-c310-vec", "--cce-aicore-only",
             "-O2", "-ffp-contract=off", "-c", "-emit-llvm", str(source), "-o", str(bitcode)],
            check=True, capture_output=True,
        )
        return manager.put(bitcode.read_bytes(), "vision_pos_embed_bf16_vf_span16.bc", binary=True)



@al.register_custom_op
class vision_pos_embed_bf16_vf_span16:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "vision_pos_embed_bf16_vf_span16"
    bitcode = ""

    def __init__(self, top, bottom, dh, col0, base, ws, rows, maxcol, out=None):
        pass


@triton.jit
def _vf_span_kernel(
    embed_ptr, output_ptr, H: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    HS: tl.constexpr, WS: tl.constexpr, CORES: tl.constexpr, COMPILE_KEY: tl.constexpr,
):
    R: tl.constexpr = 16
    for task in range(tl.program_id(0), H * triton.cdiv(W, R), CORES):
        row = task // triton.cdiv(W, R)
        col0 = task % triton.cdiv(W, R) * R
        hr = row.to(tl.float32) * HS
        hf = hr.to(tl.int32)
        hc = tl.minimum(hf + 1, G - 1)
        dh = hr - hf.to(tl.float32)
        base = (col0.to(tl.float32) * WS).to(tl.int32)
        source = tl.arange(0, S * D)
        mask = source < (G - base) * D
        top = tl.load(embed_ptr + (hf * G + base) * D + source, mask, 0)
        bottom = tl.load(embed_ptr + (hc * G + base) * D + source, mask, 0)
        value = al.custom(
            "vision_pos_embed_bf16_vf_span16", top, bottom, dh, col0, base,
            tl.full((), WS, tl.float32), tl.full((), R, tl.int32),
            tl.full((), G - 1, tl.int32), out=tl.full((R * D,), 0, tl.bfloat16),
        )
        # M divides R, so each output slice contains whole merge groups.
        group = col0 // M + tl.arange(0, R // M)
        feature = tl.arange(0, M * D)
        address = ((row // M * (W // M) + group[:, None]) * M * M * D
                   + row % M * M * D + feature[None, :])
        tl.store(output_ptr + address, value.reshape(R // M, M * D), group[:, None] < W // M)


def launch_vf_span(embed_weight, output, h, w, g, m):
    d = embed_weight.shape[1]
    hs = float(g - 1) / (h - 1) if h > 1 else 0.0
    ws = float(g - 1) / (w - 1) if w > 1 else 0.0
    span = math.ceil(15 * ws) + 2
    bitcode = _device_bitcode()
    vision_pos_embed_bf16_vf_span16.bitcode = bitcode
    cores = get_num_cores()
    _vf_span_kernel[(cores,)](
        embed_weight, output, h, w, g, m, d, span, hs, ws, cores, bitcode,
        enable_fp_fusion=False, multibuffer=False, compile_mode="simd",
    )
