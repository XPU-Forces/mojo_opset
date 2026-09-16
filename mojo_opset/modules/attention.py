from typing import Optional

import torch

from mojo_opset import functions


class SWAInfer(torch.nn.Module):
    """Original MojoSWA options, using the forward-only packed SWA kernel."""

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
        *,
        implementation: Optional[str] = None,
    ):
        super().__init__()
        if gqa_layout not in ("AABB", "ABAB"):
            raise ValueError("gqa_layout must be AABB or ABAB")
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size
        self.implementation = implementation

    def forward(self, q, k, v, cu_q_lens, cu_total_seq_lens, softmax_scale=None):
        return functions.swa_infer(
            q,
            k,
            v,
            cu_q_lens,
            cu_total_seq_lens,
            is_causal=self.is_causal,
            local_window_size=self.local_window_size,
            global_window_size=self.global_window_size,
            softmax_scale=softmax_scale,
            gqa_interleave=self.gqa_interleave,
            implementation=self.implementation,
        )

    def extra_repr(self):
        return (
            f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, "
            f"global_window_size={self.global_window_size}, local_window_size={self.local_window_size}"
        )


class VarlenFAInfer(torch.nn.Module):
    """Original VarlenPrefillGQA options; calls the temporary varlen_fa_infer API."""

    def __init__(self, is_causal: bool = True, gqa_interleave: bool = False, *, implementation: Optional[str] = None):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_interleave = gqa_interleave
        self.implementation = implementation

    def forward(self, q, k, v, cu_q_lens, cu_k_lens=None, softmax_scale=None):
        return functions.varlen_fa_infer(
            q,
            k,
            v,
            cu_q_lens,
            cu_k_lens,
            is_causal=self.is_causal,
            softmax_scale=softmax_scale,
            gqa_interleave=self.gqa_interleave,
            implementation=self.implementation,
        )
