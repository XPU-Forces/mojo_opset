# vllm-ascend integration: `fast_pos_embed_interpolate` dispatch

What NPU users currently get, and what changes.

## Current state

`vllm_ascend/patch/worker/patch_qwen3vl.py` replaces
`Qwen3_VisionTransformer.fast_pos_embed_interpolate` with the eager native path, so the upstream
Triton kernel is unreachable on Ascend:

```python
def _fast_pos_embed_interpolate(self, grid_thw):
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(pos_embed_interpolate_native(...))
    return torch.cat(outputs, dim=0)

Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate
```

That is `len(grid_thw)` launches of a ~38-kernel eager chain, plus a `torch.cat` copy.

## Change

Replace that tail with the code below. The Triton route is taken when it is usable, the native
route stays as the fallback, and the multi-image case becomes a single launch instead of
`len(grid_thw)` launches plus a copy.

Dispatch is two-level on purpose: `HAS_TRITON` is a fact about the vllm build, not about the
device (a CUDA or CPU build has Triton installed and still has no working Ascend kernel), so it
only gates the cheap import attempt. The real decision is the runtime probe, and it is cached
because this runs on every forward. The `try/except` around the call is deliberate: this is a
monkey-patch inside a serving stack, and a shape the Triton path does not cover must degrade to
native rather than raise.

## Verification

`harness/verify_vllm_ascend_dispatch.py` stubs `vllm`, `torch_npu` and `mojo_opset` and checks
the decisions without needing a serving stack or an NPU:

```
OK  triton disabled (HAS_TRITON=False)     multi=0 native=3 out=(2432, 1152)
OK  npu unavailable (HAS_TRITON=True)      multi=0 native=3
OK  both available                         multi=1 native=0
OK  triton raises -> native fallback       multi=0 native=3
OK  grid list coerced to int tuples        [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
     impl=triton G=48 m_size=2 dtype=torch.bfloat16
RESULT: ALL PASS
```

The batched operator's output is bit-identical to per-image calls concatenated; that is asserted
on hardware by `tests/functions/test_pos_embed_interp.py::test_pos_embed_interp_multi_matches_cat`.

Device-side effect, measured on A2 with triton-ascend 3.6.0: single image 3.4x to 26x faster than
the native chain, and the 3-image case becomes one launch (43.0 us) instead of 116 kernels.

## Code

```python
# SPDX-License-Identifier: Apache-2.0
"""Replacement for the ``fast_pos_embed_interpolate`` patch in vllm-ascend.

Delivered alongside the mojo_opset operator (task requirement 5: "provide the code that
removes the ``patch_qwen3vl.py`` native forcing and dispatches on ``HAS_TRITON`` + NPU
capability, with a before/after verification").

Where this goes
---------------
``vllm_ascend/patch/worker/patch_qwen3vl.py`` currently ends with::

    def _fast_pos_embed_interpolate(self, grid_thw):
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(pos_embed_interpolate_native(...))
        return torch.cat(outputs, dim=0)

    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate

i.e. the Triton kernel is unconditionally replaced by the eager native path. Replace that
tail with the code below. What changes:

* the Triton route is taken when it is actually usable, and the native route stays as the
  fallback -- never the other way round, so a missing operator or an unexpected shape can
  only cost performance, never correctness;
* the multi-image case becomes one launch instead of ``len(grid_thw)`` launches plus a
  ``torch.cat``.

Dispatch is two-level on purpose
--------------------------------
``HAS_TRITON`` is a compile-time-ish fact about vllm, not about the device: a TPU or a
CPU build can have Triton installed and still have no working Ascend kernel. So it gates
only the cheap import attempt, and the real decision is the runtime probe (NPU present and
the mojo operator importable). The probe result is cached, because it runs on every forward.

The device check is ``torch.npu.is_available()``, not ``torch_npu.is_available()``: importing
``torch_npu`` only registers the ``npu`` backend on ``torch``, it does not add a module-level
``is_available``. Calling the latter raises ``AttributeError``, which a single surrounding
``except Exception`` would swallow and turn into "Triton unusable" -- the operator would then
never run on NPU, silently. The two probes are therefore separate ``try`` blocks: a broken
device probe and a missing operator are different failures.

The ``try/except`` around the call is deliberate: this is a monkey-patch inside a serving
stack, and a shape the Triton path does not cover must degrade to native rather than raise.

The authoritative copy of this patch is ``docs/vllm_ascend_pos_embed_dispatch.py``; it is what
``tests/test_vllm_ascend_dispatch.py`` executes, so the snippet below and that file must not
drift apart.
"""

import torch

from vllm.triton_utils import HAS_TRITON
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.qwen3_vl import pos_embed_interpolate_native

# Cached result of the runtime probe: None = not probed yet.
_TRITON_POS_EMBED_OK = None


def _triton_pos_embed_available() -> bool:
    """True when the NPU-backed Triton operator can actually be used right now.

    ``HAS_TRITON`` alone is not enough: it only says the Triton Python package is
    importable, which is also true on CUDA and CPU builds where this operator has no
    implementation. The three conditions that matter are Triton present, an Ascend device
    usable, and the operator itself reachable from mojo_opset.

    The device check is ``torch.npu.is_available()`` (the ``torch_npu`` module only registers
    the ``npu`` backend; it has no module-level ``is_available``). The two probes are
    separate ``try`` blocks on purpose: a broken device probe and a missing operator are
    different failures, and folding them together would let an ``AttributeError`` from the
    operator import masquerade as "no NPU".
    """
    global _TRITON_POS_EMBED_OK
    if _TRITON_POS_EMBED_OK is None:
        ok = False
        if HAS_TRITON:
            try:
                import torch_npu  # noqa: F401  registers the "npu" device on torch

                npu_ready = torch.npu.is_available()
            except Exception:
                npu_ready = False
            if npu_ready:
                try:
                    from mojo_opset import functions as _F

                    ok = hasattr(_F, "pos_embed_interp_multi_infer")
                except Exception:
                    ok = False
        _TRITON_POS_EMBED_OK = ok
    return _TRITON_POS_EMBED_OK


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    """Interpolate the position table for every image, one launch when Triton is usable.

    Output is identical to the per-image native path concatenated: the batched kernel
    addresses each image through its own scalar set, so the per-row arithmetic is the same
    instruction sequence (verified bit-exact by
    ``tests/functions/test_pos_embed_interp.py::test_pos_embed_interp_multi_matches_cat``).
    """
    weight = self.pos_embed.weight
    if _triton_pos_embed_available():
        try:
            from mojo_opset import functions as _F

            return _F.pos_embed_interp_multi_infer(
                weight,
                [tuple(int(v) for v in grid) for grid in grid_thw],
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
                implementation="triton",
            )
        except Exception:
            # Fall through to the native path: a serving stack must not fail here.
            pass

    outputs = [
        pos_embed_interpolate_native(
            weight,
            t,
            h,
            w,
            self.num_grid_per_side,
            self.spatial_merge_size,
            self.dtype,
        )
        for t, h, w in grid_thw
    ]
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate
```
