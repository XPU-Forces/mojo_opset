"""Compare warm dense GEMM calls using NPU graph replay; run from the repository root."""
import argparse
import json, runpy, time
from pathlib import Path
import torch, torch_npu
from mojo_opset import functions as F
h = runpy.run_path('tests/functions/test_dense_gemm.py')
parser=argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
options=parser.parse_args()
rows = []
for op in h['OPS']:
    for m,n,k in [(32,4096,4096),(256,512,1024),(1024,1024,1024)]:
        torch.manual_seed(123)
        args = h['inputs'](op,m,n,k,False,True,'npu',packed_sf=True)
        kwargs = dict(trans_weight=True, output_dtype=torch.bfloat16, implementation='native', compiled_dims='nk')
        call = lambda: getattr(F,op)(*args,**kwargs)
        t=time.perf_counter(); out=call(); torch.npu.synchronize(); first=time.perf_counter()-t
        h['compare'](out,h['cpu_reference'](op,args,{k:v for k,v in kwargs.items() if k!='implementation'}))
        for _ in range(5): call()
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g):
            for _ in range(20): captured=call()
        for _ in range(3): g.replay()
        torch.npu.synchronize()
        h['compare'](captured,h['cpu_reference'](op,args,{k:v for k,v in kwargs.items() if k!='implementation'}))
        samples=[]
        for _ in range(3):
            start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
            start.record()
            g.replay()
            end.record(); end.synchronize(); samples.append(start.elapsed_time(end)*1000/20)
        row=dict(op=op,shape=[m,n,k],us=sorted(samples)[1],first_s=first, timing='NPUGraph replay, 20 calls per graph')
        print(json.dumps(row),flush=True); rows.append(row)
options.output.write_text(json.dumps(rows,indent=2))
