# SPDX-License-Identifier: Apache-2.0
"""C220 BF16 interpolation with ordered atomic sums and exact exceptional fallback.

Only complete two-row, ten-column tiles with balanced core work use this path.
Geometry is compiled and cached independently of input data. NaN/Inf and numeric
exceptions trigger a vector recomputation in the same kernel. A shape-only bound
also detects finite BF16 atomic overflow, which the atomic engine does not flag.
"""

import functools
import hashlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime.cache import get_cache_manager

from mojo_opset.kernels._npu_triton_utils import get_num_cores

_COLS = 10
_PAD = 64
_compile_lock = threading.RLock()

_SOURCE_TEMPLATE = r"""
constexpr int G=@G@,H=@H@,W=@W@,D=@D@,CORES=@CORES@,SPAN=@SPAN@,CTRL_SAT_BIT=48;
constexpr float HS=@HS@,WS=@WS@;
using i64=long long;template<typename T,unsigned D>struct MemRef{T*allocated;T*aligned;i64 offset;i64 sizes[D];i64 strides[D];};
__aicore__ __attribute__((always_inline)) float rb(float v){union{float f;unsigned u;}a;a.f=v;a.u=(a.u+32767+((a.u>>16)&1))&0xffff0000;return a.f;}
// Exact fallback: four BF16 products and left-associated BF16 sums.
__aicore__ __attribute__((always_inline)) void fallback(int pid,__gm__ __bf16*x,__gm__ __bf16*y,__ubuf__ float*f,__ubuf__ __bf16*h){
 for(int task=pid;task<(H/2)*(W/10);task+=CORES){
  int row0=task/(W/10)*2,col0=task%(W/10)*10;
  for(int lr=0;lr<2;lr++)for(int cc=0;cc<10;cc++){
   int row=row0+lr,col=col0+cc;float hr=float(row)*(HS),wr=float(col)*(WS);int hf=int(hr),wf=int(wr),hc=hf<(G-1)?hf+1:(G-1),wc=wf<(G-1)?wf+1:(G-1);float dh=hr-float(hf),dw=wr-float(wf),q3=dh*dw,q2=dh-q3,q1=dw-q3,q0=(1.f-dh)-q1;
   #pragma unroll
   for(int k=0;k<4;k++){int index=(k<2?hf:hc)*G+(k%2?wc:wf);copy_gm_to_ubuf((__ubuf__ __bf16*)(f+k*D)+D,x+index*D,0,1,D/16,0,0);}
   set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);set_mask_count();set_vector_mask(0,D);
   #pragma unroll
   for(int k=0;k<4;k++)vconv_bf162f32(f+k*D,(__ubuf__ __bf16*)(f+k*D)+D,1,1,1,8,4);
   pipe_barrier(PIPE_V);
   #pragma unroll
   for(int k=0;k<4;k++)vmuls(f+k*D,f+k*D,rb(k==0?q0:(k==1?q1:(k==2?q2:q3))),1,1,1,8,8);
   pipe_barrier(PIPE_V);set_vector_mask(0,4*D);vconv_f322bf16r(h,f,1,1,1,4,8);pipe_barrier(PIPE_V);vconv_bf162f32(f,h,1,1,1,8,4);pipe_barrier(PIPE_V);set_vector_mask(0,D);
   #pragma unroll
   for(int k=1;k<4;k++){vadd(f,f,f+k*D,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);vconv_f322bf16r(h,f,1,1,1,4,8);pipe_barrier(PIPE_V);if(k<3){vconv_bf162f32(f,h,1,1,1,8,4);pipe_barrier(PIPE_V);}}
   set_mask_norm();set_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);int pixel=(row/2*(W/2)+col/2)*4+lr*2+col%2;copy_ubuf_to_gm(y+pixel*D,h,0,1,D/16,0,0);
   set_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);wait_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);set_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
  }
 }
}

@RANGE_MASKS@
extern "C" __aicore__ void _mlir_ciface_@SYMBOL@(unsigned long long xaddr,unsigned long long yaddr,int pid,MemRef<__ubuf__ float,1>*ff,MemRef<__ubuf__ __bf16,1>*oo){
 constexpr int cols=10,span=SPAN,base_nf=D,cores=CORES,pad=64,prefetch=1,mode=1;
 auto saved_ctrl=get_ctrl();set_ctrl(saved_ctrl|(1ULL<<CTRL_SAT_BIT));
 auto x=(__gm__ __bf16*)xaddr;auto y=(__gm__ __bf16*)yaddr;auto f=ff->aligned+ff->offset;auto o=oo->aligned+oo->offset;int groups=(W+cols-1)/cols,fs=D/base_nf,total=(H/2)*groups*fs,rem=total%cores,main=total-rem,split=rem?cores/rem:1;
 for(int work=pid;work<main+(rem?cores:0);work+=cores){
  int task=work<main?work:main+(work-main)/split,nf=work<main?base_nf:base_nf/split,tail=work<main?0:(work-main)%split;
  int sn=span*nf,slots=sn+pad;auto tmp=f+2*slots;auto q=tmp+6*nf;auto broad=q+8*cols;int full=2*cols;
  int row=task/(groups*fs)*2,col0=task/fs%groups*cols,f0=task%fs*base_nf+tail*nf,ncols=W-col0<cols?W-col0:cols,np=2*ncols;float hr0=float(row)*(HS),hr1=float(row+1)*(HS);int hf0=int(hr0),hf1=int(hr1);float dh0=hr0-float(hf0),dh1=hr1-float(hf1);int base=int(float(col0)*(WS)),ns=G-base<span?G-base:span;

  unsigned long long lo0,hi0,lo1,hi1;range_masks(row,lo0,hi0);range_masks(row+1,lo1,hi1);
  unsigned long long risk0=col0>=64?hi0>>(col0-64):(col0?(lo0>>col0)|(hi0<<(64-col0)):lo0);
  unsigned long long risk1=col0>=64?hi1>>(col0-64):(col0?(lo1>>col0)|(hi1<<(64-col0)):lo1);
  #pragma unroll
  for(int stage=0;stage<4;stage++){
   if(stage==0||stage==2){
    if(stage==0&&(!prefetch||work==pid||work>=main)){
     for(int lr=0;lr<2;lr++){int hf=lr?hf1:hf0;auto sp=f+lr*slots;if(mode&&nf==D)copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+base)*D,0,1,ns*nf/16,0,0);else copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+base)*D+f0,0,ns,nf/16,(D-nf)/16,0);}
    }
    set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);set_mask_count();set_vector_mask(0,ns*nf);
    for(int lr=0;lr<2;lr++){auto sp=f+lr*slots;vconv_bf162f32(sp,(__ubuf__ __bf16*)sp+sn,1,1,1,8,4);}pipe_barrier(PIPE_V);
   }
   auto out=o+(stage%2)*(2*cols*nf);
   #pragma unroll
   for(int lr=0;lr<2;lr++){
    set_mask_count();set_vector_mask(0,nf);
    #pragma unroll
    for(int cb=0;cb<cols;cb+=10){
    set_vector_mask(0,nf);
    #pragma unroll
    for(int j=0;j<10;j++){constexpr int dummy=0;int cc=cb+j;if(cc<cols){int pp=cc/2*4+lr*2+cc%2;float wr=float(col0+cc)*(WS);int wf=int(wr);float dw=wr-float(wf),dh=lr?dh1:dh0,q3=dh*dw;float weight=stage==0?(1.f-dh)-(dw-q3):(stage==1?dw-q3:(stage==2?dh-q3:q3));int idx=((stage%2&&wf<(G-1))?wf+1:wf)-base;auto target=j<6?tmp+j*nf:(__ubuf__ float*)(out+((j-6)*4+2)*nf);// The probe and product are identical VMUL writes, with no intervening read.
    // Their batch V barrier below completes both before any BF16 conversion.
    if(((lr?risk1:risk0)>>cc)&1){vmuls(target,f+lr*slots+idx*nf,2.f,1,1,1,8,8);}
vmuls(target,f+lr*slots+idx*nf,rb(weight),1,1,1,8,8);}}
    pipe_barrier(PIPE_V);
    if(cb+10>=cols){
    if(stage==1){
     int hf=(lr?hf1:hf0)+1;hf=hf<G?hf:(G-1);auto sp=f+lr*slots;set_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);if(mode&&nf==D)copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+base)*D,0,1,ns*nf/16,0,0);else copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+base)*D+f0,0,ns,nf/16,(D-nf)/16,0);
    }
    if(stage==3&&prefetch&&work+cores<main){
     int nt=work+cores,nr=nt/(groups*fs)*2,nc=nt/fs%groups*cols,nf0=nt%fs*base_nf,nb=int(float(nc)*(WS)),nspan=G-nb<span?G-nb:span;int hf=int(float(nr+lr)*(HS));auto sp=f+lr*slots;set_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);if(mode&&nf==D)copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+nb)*D,0,1,nspan*nf/16,0,0);else copy_gm_to_ubuf((__ubuf__ __bf16*)sp+sn,x+(hf*G+nb)*D+nf0,0,nspan,nf/16,(D-nf)/16,0);
    }
    }
    set_vector_mask(0,nf);
    #pragma unroll
    for(int cc=9;cc>=6;cc--)vconv_f322bf16r(out+(cc/2*4+lr*2+cc%2)*nf,(__ubuf__ float*)(out+((cc-6)*4+2)*nf),1,1,1,4,8);
    set_vector_mask(0,2*nf);
    #pragma unroll
    for(int cc=0;cc<3;cc++)vconv_f322bf16r(out+(cc*4+lr*2)*nf,tmp+cc*2*nf,1,1,1,4,8);
    pipe_barrier(PIPE_V);
    }

   }
   set_mask_norm();set_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);
   // Drain the preceding transfer after computing this stage in the other buffer.
   set_flag(PIPE_MTE3,PIPE_S,EVENT_ID0);wait_flag(PIPE_MTE3,PIPE_S,EVENT_ID0);
   if(stage==0)set_atomic_none();else if(stage==1){set_atomic_add();set_atomic_bf16();}
   if(mode&&nf==D)copy_ubuf_to_gm(y+(row/2*(W/2)+col0/2)*4*D,out,0,1,np*nf/16,0,0);else copy_ubuf_to_gm(y+(row/2*(W/2)+col0/2)*4*D+f0,out,0,np,nf/16,0,(D-nf)/16);
  }
 }
 set_flag(PIPE_MTE3,PIPE_S,EVENT_ID0);wait_flag(PIPE_MTE3,PIPE_S,EVENT_ID0);set_atomic_none();if(get_status()!=0)fallback(pid,x,y,f,o);set_ctrl(saved_ctrl);
}

"""


def _f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return math.nan
    bits = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _range_masks(g, h, w):
    # Monotonic rounding bounds every finite signed input by using MAX * abs(q).
    # If this bound can overflow BF16, inspect those four source values with an
    # exact 2x multiply. Its FP32 overflow flag catches large finite inputs too.
    hs, ws = _f32((g - 1) / (h - 1)), _f32((g - 1) / (w - 1))
    maximum = struct.unpack("<f", struct.pack("<I", 0x7F7F0000))[0]
    masks = []
    for row in range(h):
        hr = _f32(row * hs)
        dh = _f32(hr - int(hr))
        bits = 0
        for col in range(w):
            wr = _f32(col * ws)
            dw = _f32(wr - int(wr))
            q3 = _f32(dh * dw)
            q2, q1 = _f32(dh - q3), _f32(dw - q3)
            q0 = _f32(_f32(1.0 - dh) - q1)
            products = [_bf16(_f32(abs(_bf16(q)) * maximum)) for q in (q0, q1, q2, q3)]
            value = products[0]
            for product in products[1:]:
                value = _bf16(_f32(value + product))
            if math.isinf(value):
                bits |= 1 << col
        masks.append((bits & ((1 << 64) - 1), bits >> 64))

    def tree(start, stop):
        if stop - start == 1:
            lo, hi = masks[start]
            return f"lo={lo}ULL;hi={hi}ULL;"
        mid = (start + stop) // 2
        return f"if(row<{mid}){{{tree(start, mid)}}}else{{{tree(mid, stop)}}}"

    return ("__aicore__ __attribute__((always_inline)) void range_masks("
            "int row,unsigned long long &lo,unsigned long long &hi){" + tree(0, h) + "}\n")


def supports_stream(weight, t, h, w, g, merge, dtype):
    if not (t == 1 and merge == 2 and g > 1 and 2 <= h <= 128 and 2 <= w <= 128
            and h % 2 == 0 and w % _COLS == 0 and w > g and h * w >= 512
            and weight.dtype == dtype == torch.bfloat16 and weight.is_contiguous()):
        return False
    d = weight.shape[1]
    if not (128 <= d <= 1152 and d % 16 == 0):
        return False
    cores = get_num_cores()
    tiles = (h // 2) * (w // _COLS)
    span = math.ceil((_COLS - 1) * (g - 1) / (w - 1)) + 2
    ub_bytes = 4 * (2 * (span * d + _PAD) + 6 * d) + 2 * (4 * _COLS * d)
    return tiles >= cores and tiles % cores == 0 and ub_bytes <= 192 * 1024


@functools.lru_cache(maxsize=1)
def _compiler_identity():
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("CANN bisheng is required for the C220 BF16 interpolation primitive")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True).stdout
    return compiler, version


@functools.lru_cache(maxsize=None)
def _specialization(g, h, w, d, cores):
    span = math.ceil((_COLS - 1) * (g - 1) / (w - 1)) + 2
    source = _SOURCE_TEMPLATE
    replacements = {"G": g, "H": h, "W": w, "D": d, "CORES": cores, "SPAN": span,
                    "HS": f"{_f32((g - 1) / (h - 1)):.20e}f",
                    "WS": f"{_f32((g - 1) / (w - 1)):.20e}f",
                    "RANGE_MASKS": _range_masks(g, h, w)}
    for name, value in replacements.items():
        source = source.replace("@" + name + "@", str(value))
    symbol = "vision_pos_embed_stream_" + hashlib.sha256(source.encode()).hexdigest()[:20]
    source = source.replace("@SYMBOL@", symbol)
    compiler, version = _compiler_identity()
    key = hashlib.sha256(source.encode() + os.fsencode(compiler) + version).hexdigest()
    manager = get_cache_manager(key)
    bitcode = manager.get_file("stream.bc")
    if bitcode is None:
        with tempfile.TemporaryDirectory(prefix="vision-pos-stream-") as directory:
            cpp, bc = Path(directory) / "stream.cpp", Path(directory) / "stream.bc"
            cpp.write_text(source)
            subprocess.run([compiler, "-x", "cce", "--cce-aicore-arch=dav-c220-vec",
                            "--cce-aicore-only", "-O2", "-fno-jump-tables", "-ffp-contract=off",
                            "-c", "-emit-llvm", str(cpp), "-o", str(bc)], check=True, capture_output=True)
            bitcode = manager.put(bc.read_bytes(), "stream.bc", binary=True)

    def init(self, x, y, pid):
        self.extra_buffers = [(tl.float32, 2 * (span * d + _PAD) + 6 * d),
                              (tl.bfloat16, 4 * _COLS * d)]

    op = type(symbol, (), {"__init__": init, "core": al.CORE.VECTOR,
                          "pipe": (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3),
                          "mode": al.MODE.SIMD, "symbol": symbol, "bitcode": bitcode})
    al.register_custom_op(op)
    return symbol, bitcode


@triton.jit
def _stream_kernel(X, Y, OP: tl.constexpr, BITCODE: tl.constexpr):
    al.custom(OP, X.to(tl.uint64), Y.to(tl.uint64), tl.program_id(0))


def launch_stream(weight, h, w, g):
    cores = get_num_cores()
    with _compile_lock:
        symbol, bitcode = _specialization(g, h, w, weight.shape[1], cores)
    output = weight.new_empty((h * w, weight.shape[1]))
    _stream_kernel[(cores,)](weight, output, symbol, bitcode,
                            enable_fp_fusion=False, multibuffer=False)
    return output
