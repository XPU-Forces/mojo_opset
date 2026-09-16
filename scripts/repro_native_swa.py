"""Recreate the two intermittent 950PR SWA failures without pytest.

Run from the repo root: python -m scripts.repro_native_swa --repeats 20
The saved input is regenerated from the original test recipe, not a dump from
the original failing process. Same inputs do not guarantee the intermittent
failure; allocator/device-buffer state is not captured by an input file.
"""

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from mojo_opset import functions as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--input", type=Path, help="Replay tensors saved by --save-input")
    parser.add_argument("--save-input", type=Path, help="Save regenerated inputs; never overwrites")
    parser.add_argument("--failure", type=Path, help="Save first failed outputs and inputs; never overwrites")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for destination in (args.save_input, args.failure):
        if destination is not None and destination.exists():
            parser.error(f"refusing to overwrite {destination}")

    if args.input:
        saved = torch.load(args.input, map_location="cpu", weights_only=True)
        q, k, v, cq, ck = (saved[name].to("npu") for name in ("q", "k", "v", "cu_q", "cu_k"))
    else:
        # Same generation order, device RNG, dtype and seed as test_swa_infer.
        torch.manual_seed(42)
        cq = torch.tensor([0, 129, 257], device="npu", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        ck = torch.tensor([0, 129, 257], device="npu", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        q = torch.randn(386, 16, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(386, 4, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)

    options = dict(is_causal=True, global_window_size=4, softmax_scale=128**-0.5)
    inputs = None
    references = {}
    print(f"device={torch.npu.get_device_name()}, torch={torch.__version__}", flush=True)
    for repeat in range(args.repeats):
        for interleave, window in ((False, 1023), (True, 255)):
            kwargs = dict(options, gqa_interleave=interleave, local_window_size=window)
            actual = F.swa_infer(q, k, v, cq, ck, implementation="native", **kwargs)
            if interleave not in references:
                references[interleave] = F.swa_infer(
                    q, k, v, cq, ck, implementation="torch_reference", **kwargs
                ).cpu()
            actual = actual.cpu()  # Wait for completion before checking/saving.
            expected = references[interleave]
            if inputs is None:
                inputs = dict(zip(("q", "k", "v", "cu_q", "cu_k"),
                                  (t.cpu() for t in (q, k, v, cq, ck))))
                assert all(torch.isfinite(inputs[name]).all() for name in ("q", "k", "v"))
                if args.save_input:
                    with args.save_input.open("xb") as output:
                        torch.save(inputs, output)
                    print(f"Saved inputs: {args.save_input}", flush=True)
            bad = ~torch.isclose(actual, expected, atol=0.02, rtol=0.02, equal_nan=False)
            count = bad.sum().item()
            if count:
                indices = bad.nonzero()[:8].tolist()
                print(f"FAIL repeat={repeat} interleave={interleave} window={window} "
                      f"mismatches={count} nan={actual.isnan().sum().item()} first={indices}", flush=True)
                if args.failure:
                    with args.failure.open("xb") as output:
                        torch.save(dict(inputs=inputs, actual=actual, reference=expected,
                                        options=kwargs, repeat=repeat), output)
                raise SystemExit(1)
    print(f"PASS: {args.repeats * 2} calls; this does not establish stability across processes.")


if __name__ == "__main__":
    main()
