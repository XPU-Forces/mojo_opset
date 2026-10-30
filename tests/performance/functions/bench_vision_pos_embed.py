"""MBU benchmark for vision_pos_embed_interp (task spec).

Measurement protocol (per the task book):
  * ``do_bench_npu_profiler`` with 5 warmup + 30 active iterations, no
    ``target_kernel_name`` -> mean device kernel time per call (ms);
  * bandwidth calibrated with a ``fill_`` pure-write probe (write-dominated op),
    5 rounds of (5 warmup + 30 active) reduced by median;
  * ideal bytes = 2 x (G^2 x hidden + rows x hidden) bytes for BF16,
    accumulating the position table once per grid.
"""

import statistics

import torch
import torch_npu  # noqa: F401
from triton.backends.ascend.testing import do_bench_npu_profiler

from mojo_opset import functions as F

G = 48
M = 2
HID = 1152
DT = torch.bfloat16
SIZE_MB = 512
ROUNDS = 5


def _gbps(nbytes, ms):
    return nbytes / (ms / 1e3) / 1e9


def calibrate(probe="fill", size_mb=SIZE_MB, rounds=ROUNDS):
    n = size_mb * 1024 * 1024 // 2
    a = torch.empty(n, dtype=DT, device="npu")
    if probe == "fill":
        fn, nbytes = (lambda: a.fill_(0.0)), n * 2
    else:
        b = torch.empty(n, dtype=DT, device="npu")
        fn, nbytes = (lambda: b.copy_(a)), n * 2 * 2
    samples = [_gbps(nbytes, do_bench_npu_profiler(fn, warmup=5, active=30)) for _ in range(rounds)]
    return statistics.median(samples), samples


def bench_single(t, h, w, hidden=HID, impl="triton"):
    ew = torch.randn(G * G, hidden, dtype=DT, device="npu") * 0.25
    fn = lambda: F.vision_pos_embed_interp(ew, t, h, w, G, M, DT, implementation=impl)
    fn()
    torch.npu.synchronize()
    ms = do_bench_npu_profiler(fn, warmup=5, active=30)
    rows = t * h * w
    return ms, (G * G + rows) * hidden * 2


def bench_multi(grids, hidden=HID):
    ew = torch.randn(G * G, hidden, dtype=DT, device="npu") * 0.25
    fn = lambda: F.vision_pos_embed_interp_multi(ew, grids, G, M, DT, implementation="triton")
    fn()
    torch.npu.synchronize()
    ms = do_bench_npu_profiler(fn, warmup=5, active=30)
    rows = sum(t * h * w for t, h, w in grids)
    return ms, (len(grids) * G * G + rows) * hidden * 2


def main():
    bw, bw_samples = calibrate("fill")
    bw_copy, bw_copy_samples = calibrate("copy")
    print(f"[calib] fill_ = {bw:.1f} GB/s  samples={[round(x, 1) for x in bw_samples]}")
    print(f"[calib] copy_ = {bw_copy:.1f} GB/s  samples={[round(x, 1) for x in bw_copy_samples]}")
    print(f"{'case':<34}{'rows':>8}{'MB':>9}{'ms':>9}{'MBU%':>8}{'MBU%copy':>10}")

    for t, h, w in [(1, 8, 16), (1, 32, 48), (1, 60, 80), (3, 16, 16), (16, 100, 168)]:
        ms, nb = bench_single(t, h, w)
        print(
            f"{f'grid {t}x{h}x{w}':<34}{t*h*w:>8}{nb/1e6:>9.1f}{ms:>9.3f}"
            f"{_gbps(nb, ms) / bw * 100:>8.1f}{_gbps(nb, ms) / bw_copy * 100:>10.1f}"
        )

    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    ms, nb = bench_multi(grids)
    print(
        f"{'multi 3 grids':<34}{sum(t*h*w for t, h, w in grids):>8}{nb/1e6:>9.1f}{ms:>9.3f}"
        f"{_gbps(nb, ms) / bw * 100:>8.1f}{_gbps(nb, ms) / bw_copy * 100:>10.1f}"
    )

    ms, nb = bench_single(3, 60, 80, hidden=1000)
    print(
        f"{'tail 3x60x80 hidden=1000':<34}{3*60*80:>8}{nb/1e6:>9.1f}{ms:>9.3f}"
        f"{_gbps(nb, ms) / bw * 100:>8.1f}{_gbps(nb, ms) / bw_copy * 100:>10.1f}"
    )

    ms_t, _ = bench_single(1, 8, 16, impl="triton")
    ms_n, _ = bench_single(1, 8, 16, impl="torch")
    print(f"[small] triton {ms_t*1e3:.1f} us vs native {ms_n*1e3:.1f} us (ratio {ms_t/ms_n:.3f})")


if __name__ == "__main__":
    main()
