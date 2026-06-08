"""UC backend for MojoApplyPenaltiesTempurate.

Mojo core op (``mojo_opset/core/operators/sampling.py``) is a dense
elementwise pipeline with per-batch Python branches for skipping each
penalty. The UC kernel
(``uc-kernel/kernels/mojo_apply_penalties_tempurate_bf16.py``) evaluates
the formula unconditionally on dense ``(B, V)`` tensors; this wrapper
materialises those tensors, picking neutral values for batches that the
mojo wrapper would otherwise skip:

  - ``token_freqs[i] is None``  -> dense_freq row=0, mask row=0, fp=0, pp=0, rp=1
  - ``frequency_penalties[i] == 0`` -> fp[i]=0 (kernel applies ``l - 0*f``)
  - ``presence_penalties[i] == 0``  -> pp[i]=0
  - ``repetition_penalties[i] == 1`` -> rp[i]=1 (kernel writes pos/1+neg*1 = l)
  - ``temps`` None / empty  -> tp[i]=1 (kernel applies ``l / 1``)
  - ``temps[i] is None``    -> tp[i]=1

For any unsupported configuration (dtype != bf16, dim != 2, list length
mismatch) we delegate to the parent ``MojoApplyPenaltiesTempurate.forward``
torch native implementation.
"""

from typing import List, Optional, Union

import torch

from mojo_opset.core import MojoApplyPenaltiesTempurate

from ._utils import _uc_kernels


_KERNEL_API = "mojo_apply_penalties_tempurate_bf16"
_SUPPORTED_DTYPES = (torch.bfloat16,)


class UCApplyPenaltiesTempurate(MojoApplyPenaltiesTempurate):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        if logits.dtype not in _SUPPORTED_DTYPES or logits.dim() != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        batch_size, vocab = logits.shape
        if batch_size == 0 or vocab == 0:
            return logits.clone()

        if (
            len(token_freqs) != batch_size
            or len(presence_penalties) != batch_size
            or len(frequency_penalties) != batch_size
            or len(repetition_penalties) != batch_size
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ``temps`` may be None, empty list, or per-batch list with optional Nones.
        if temps is not None and len(temps) > 0 and len(temps) != batch_size:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernels = _uc_kernels()
        if _KERNEL_API not in kernels:
            raise NotImplementedError(
                f"UC kernel {_KERNEL_API!r} is not in the loaded uc-kernel wheel. "
                "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )
        kernel = kernels[_KERNEL_API]

        device = logits.device

        # Dense (B, V) freq plane + per-element mask. Rows for None entries
        # stay zero so the corresponding penalty terms degenerate to no-ops.
        dense_freq = torch.zeros((batch_size, vocab), dtype=torch.float32, device=device)
        for i, freq_tensor in enumerate(token_freqs):
            if freq_tensor is None:
                continue
            row = freq_tensor.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
            if row.numel() != vocab:
                # Shape mismatch on this row -- bail out to parent for safety.
                raise NotImplementedError(
                    "UC backend cannot service this call (shape/dtype/contract not "
                    "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                    "(2026-06-08), this wrapper does not silently fall back to torch — "
                    "use TTX / torch_npu / torch_native backend for unsupported inputs."
                )
            dense_freq[i].copy_(row)

        has_freq_mask = (dense_freq > 0).to(torch.float32)

        # Per-batch scalars: neutral values (0 / 0 / 1) for batches that the
        # mojo wrapper would have skipped because ``token_freqs[i] is None``.
        fp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        pp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        rp = torch.ones(batch_size, dtype=torch.float32, device=device)
        for i, freq_tensor in enumerate(token_freqs):
            if freq_tensor is None:
                continue
            fp[i] = float(frequency_penalties[i])
            pp[i] = float(presence_penalties[i])
            rp[i] = float(repetition_penalties[i])

        tp = torch.ones(batch_size, dtype=torch.float32, device=device)
        if temps is not None and len(temps) > 0:
            for i, t in enumerate(temps):
                if t is not None:
                    tp[i] = float(t)

        logits_c = logits.contiguous()
        out = torch.empty_like(logits_c)

        kernel(
            logits_c,
            dense_freq.contiguous(),
            has_freq_mask.contiguous(),
            fp,
            pp,
            rp,
            tp,
            out,
            batch_size,
            vocab,
        )
        return out
