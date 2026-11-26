# vllm-ascend integration: `fused_qk_norm_rope` dispatch

`qk_norm_rope_infer` is the Ascend Triton port of vLLM-Omni's `fused_qk_norm_rope`
(`vllm_omni/diffusion/layers/fused_qk_norm_rope.py`), the MiniMax-H3 contract:
`head_dim=128`, `rotary_dim=96`, BF16, `[tokens, heads, head_dim]`, half-split pairing.

## Design (source-level)

Default production path for H3 half-split (`kernels/npu_a2_triton/qk_norm_rope.py`):

1. **Persist schedule** — `grid ≈ num_vectorcore`, each core strides tokens with `tl.range`
   (triton-ascend HighPerfGuide / profiling Method 2).
2. **extract/insert half-split** — Ascend `tl.extra.cann.extension.extract_slice` /
   `insert_slice` on a `[HPP, head_dim]` tile for the two rotary halves (`rotary_half=48`),
   so partner halves are taken from the already-normalized UB tile instead of a second
   partner HBM load.
3. One launch per Q/K stream.

Interleaved (GPT-NeoX) pairing and non-power-of-two `head_dim` keep the general multi-head
tile kernel for coverage.

Numerical contract (must match upstream bit-for-bit cases):

1. RMSNorm statistics once per head; rotary pairs share `inv_rms`.
2. Normalize → **round to BF16** → rotate in FP32 → store (BF16 round even for FP32 inputs).
3. Scale left-to-right: `x * inv_rms * weight` (not `x * (inv_rms * weight)`).
4. Dims `>= rotary_dim` pass through as the BF16-rounded norm value.

## HiAscend acceptance evidence (taskbook MBU)

Calibration uses on-device `copy_` bandwidth; RW bytes follow the taskbook formula
`2*(4*T*H*128 + T*96 + 2*128)`. Warmup 5 + active 30 profiler samples.

### A5 / Ascend950PR — **MBU gate PASS**

| tokens×heads | gate | latency | MBU | result |
| --- | --- | --- | --- | --- |
| 32×8 | relative_A | 0.003 ms | 6.7% | PASS |
| 320×8 | relative_A | 0.009 ms | 20.7% | PASS |
| 320×56 | mbu≥60% | 0.021 ms | **64.6%** | PASS |
| 1024×56 | mbu≥60% | 0.057 ms | **74.3%** | PASS |
| 4096×56 | mbu≥60% | 0.212 ms | **80.0%** | PASS |
| 16384×56 | mbu≥60% | 0.828 ms | **81.9%** | PASS |
| 69888×56 | mbu≥60% | 3.513 ms | **82.3%** | PASS |
| 4096×57 | mbu≥50% | 0.217 ms | **79.5%** | PASS |

`copy_` ≈ 1390 GB/s. Stack: triton-ascend 3.6.0 + CANN 9.2 + torch_npu 2.10.

### A3 / 910_9382 — precision + relative_A PASS; large-shape MBU below 60%

Local persist best: 4096×56 **~56.9%** (`copy_` ≈ 1163 GB/s). Tail 4096×57 ≥50% PASS on
local runs. Official review harness may report a lower band (~40–43%); the qualitative
gap vs 60% is the same.

### A2 / 910B3 — precision + relative_A PASS; large-shape MBU below 60%

Local persist: ~**47%** on ≥16MB shapes (`copy_` ≈ 1143 GB/s, 40 vector cores); +1.4pt vs
classic. Tail 4096×57 still under 50% on local runs.

## Why A2/A3 miss the 60%/50% lines (hardware affinity)

This is the clarification requested for acceptance when ≥16MB MBU stays under the gate.

### 1. Operator geometry is vector/descriptor heavy, not a pure `copy_` stream

For MiniMax-H3, one head is only **256 B** (`128 × bf16`). Even when a program covers all
56 heads of a token (~14 KB), the kernel still has to:

- reduce RMS over `head_dim` (vector reduce);
- apply weight with the **left-to-right** scale order required by the contract;
- **round to BF16 before RoPE** (mandatory BF16↔FP32 round-trips);
- split rotary halves (`rotary_half=48`, non-power-of-two) via Ascend
  `extract_slice` / `insert_slice` (or an equivalent partner gather).

So the working set is small relative to HBM peak, and a large fraction of cycles is
**vector compute + cast + slice**, not bulk DMA. Against a `copy_`-calibrated MBU
denominator this structurally caps utilization on A2/A3.

### 2. On-device profile agrees (A3 4096×56 representative)

| Signal | Observation | Interpretation |
| --- | --- | --- |
| kernels / call | 1–2 (one launch per stream) | not multi-kernel inflation |
| `aiv_vec_ratio` | ≈0.87–0.90 | **vector-compute dominated** |
| `aiv_mte2_ratio` | ≈0.25–0.27 | **not** a pure HBM bandwidth wall |
| Block / vector cores | saturates die vector cores | little unused parallel slack |

Effective payload throughput sits well below the `copy_` probe (e.g. A3 ~625 GB/s
effective vs ~1170 GB/s `copy_`). Raising MBU further would require removing contract
steps (BF16 pre-round / slice), which would break the numerical suite.

### 3. Same source reaches the gate on A5 — affinity, not a wrong algorithm

Identical Triton path + measurement recipe is **all_pass on Ascend950PR (74–82% MBU)**.
That anchors correctness of the implementation and the MBU formula. The residual gap on
A2/A3 is therefore **platform / BiShengIR efficiency on the slice+RMS+cast path**, not a
missing fusion or a mis-counted RW byte formula.

### 4. Exhaustive scheduling / compiler sweeps did not close the gate

Already tried on A2/A3 (all precision-OK where applicable): persist vs classic vs
token-block; HPP sweeps; pot/rearr/split/hyb/vllmzeros/direct/tok2 graphs; AscendC fused
rewrite (~12–14% MBU, not competitive); two rounds of Triton-Ascend vector
`compile_meta` / compiler options (`multibuffer`, `num_stages`, `tile_mix_vector_loop`,
preload/av2/flatten/bind/cast-align, `compile_mode`, …). **None** pushed ≥16MB MBU over
60%. Production default remains persist + extract/insert as the best measured point.

### 5. Acceptance mapping (taskbook)

| Item | Status |
| --- | --- |
| Accuracy / stability on A2·A3·A5 | PASS |
| Small-shape relative_A (&lt;16MB) | PASS |
| A5 ≥16MB MBU ≥60% + tail ≥50% | PASS |
| A2/A3 ≥16MB MBU ≥60% (and A2 tail ≥50%) | **below line** — profile + affinity rationale above |
| Taskbook clause for unmet configs | attach bottleneck analysis → **acceptance judgment** |

## Dispatch note

`mojo_opset` config selects `triton` for this op. Consumers that also ship
`npu_rms_norm` + `npu_rotary_mul` may keep that composition as default on A2/A3 for
bandwidth-bound shapes; on A5/950PR the Triton persist path meets the taskbook MBU line.

## Escape hatches

| env | effect |
| --- | --- |
| `MOJO_T1_TILE=1` | force the general tile kernel (disable persist extract) |
| `MOJO_T1_HEADS` / `MOJO_T1_ROWS` / `MOJO_T1_WAVES` | tile-path tuning only |
