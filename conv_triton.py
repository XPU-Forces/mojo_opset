import argparse
import os
import sys
import time
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _sync(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu", choices=["npu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--b", type=int, default=16)
    p.add_argument("--t", type=int, default=1)
    p.add_argument("--d", type=int, default=2048)
    p.add_argument("--w", type=int, default=4)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--mode", default="update_states", choices=["update_states", "fwd_with_state"])
    p.add_argument("--sleep-ms", type=int, default=0)
    args = p.parse_args()

    if not (hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available()):
        raise RuntimeError("NPU is not available")

    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

    device = torch.device("npu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    b, t, d, w = int(args.b), int(args.t), int(args.d), int(args.w)

    x = torch.randn((b, t, d), device=device, dtype=dtype)
    init_state = torch.randn((b, d, w), device=device, dtype=dtype)
    weight = torch.randn((d, w), device=device, dtype=dtype)

    print(f"device={device} dtype={dtype} x={tuple(x.shape)} init_state={tuple(init_state.shape)} weight={tuple(weight.shape)}")
    _sync(device)

    if args.mode == "update_states":
        from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_update_states

        for i in range(int(args.iters)):
            st = causal_conv1d_update_states(x=x, state_len=w, initial_state=init_state, cu_seqlens=None)
            if not isinstance(st, torch.Tensor):
                raise RuntimeError(f"unexpected state type: {type(st)}")
            init_state = st
            if (i + 1) % 10 == 0:
                _sync(device)
                print(f"iter={i+1} ok state={tuple(init_state.shape)}")
            if args.sleep_ms > 0:
                time.sleep(float(args.sleep_ms) / 1000.0)

    else:
        from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_fwd

        for i in range(int(args.iters)):
            y, st = causal_conv1d_fwd(
                x=x,
                weight=weight,
                bias=None,
                residual=None,
                initial_state=init_state,
                output_final_state=True,
                activation=None,
                cu_seqlens=None,
            )
            if not isinstance(y, torch.Tensor):
                raise RuntimeError(f"unexpected y type: {type(y)}")
            if not isinstance(st, torch.Tensor):
                raise RuntimeError(f"unexpected st type: {type(st)}")
            init_state = st
            if (i + 1) % 10 == 0:
                _sync(device)
                print(f"iter={i+1} ok y={tuple(y.shape)} state={tuple(init_state.shape)}")
            if args.sleep_ms > 0:
                time.sleep(float(args.sleep_ms) / 1000.0)


if __name__ == "__main__":
    main()