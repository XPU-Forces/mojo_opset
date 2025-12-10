import os

from typing import Optional
from typing import Tuple

import torch

# Note: now we only support ascend backend
from .ascend.flash_attention import paged_attention_decode_impl
from .ascend.flash_attention import paged_attention_prefill_impl
from .ascend.gelu import gelu_bwd_impl
from .ascend.gelu import gelu_fwd_impl
from .ascend.rmsnorm import rmsnorm_bwd_impl
from .ascend.rmsnorm import rmsnorm_fwd_impl
from .ascend.rmsnorm import rmsnorm_infer_impl
from .ascend.rope import rope_bwd_impl
from .ascend.rope import rope_fwd_impl
from .ascend.silu import silu_bwd_impl
from .ascend.silu import silu_fwd_impl
from .ascend.swiglu import swiglu_bwd_impl
from .ascend.swiglu import swiglu_fwd_impl

if os.getenv("MOJO_RUN_MODE", "compile") == "compile":
    assert torch.version.__version__ >= "2.7.0", "Work with torch.compile request your torch version >= 2.7.0"

    # =====================================
    # Register GELU
    # =====================================
    @torch.library.custom_op("ttx::gelu", mutates_args={})
    def gelu_fwd(x: torch.Tensor) -> torch.Tensor:
        return gelu_fwd_impl(x)

    @gelu_fwd.register_fake
    def gelu_fwd_fake(x: torch.tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("ttx::gelu_bwd", mutates_args={})
    def gelu_bwd(
        dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return gelu_bwd_impl(dy, x)

    @gelu_bwd.register_fake
    def gelu_bwd_fake(
        dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(dy)

    # =====================================
    # Register SiLU
    # =====================================

    @torch.library.custom_op("ttx::silu", mutates_args={})
    def silu_fwd(x: torch.Tensor) -> torch.Tensor:
        return silu_fwd_impl(x)

    @silu_fwd.register_fake
    def silu_fwd_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("ttx::silu_bwd", mutates_args={})
    def silu_bwd(
        dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return silu_bwd_impl(dy, x)

    @silu_bwd.register_fake
    def silu_bwd_fake(
        dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(dy)

    # ====================================
    # Register SwiGLU
    # ====================================

    @torch.library.custom_op("ttx::swiglu", mutates_args={})
    def swiglu_fwd(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return swiglu_fwd_impl(a, b)

    @swiglu_fwd.register_fake
    def swiglu_fwd_fake(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(a)

    @torch.library.custom_op("ttx::swiglu_bwd", mutates_args={})
    def swiglu_bwd(
        dc: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return swiglu_bwd_impl(dc, a, b)

    @swiglu_bwd.register_fake
    def swiglu_bwd_fake(
        dc: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(dc), torch.empty_like(dc)

    # ====================================
    # Register Attention
    # ====================================

    @torch.library.custom_op("ttx::paged_attention_prefill", mutates_args={})
    def paged_attention_prefill(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        return paged_attention_prefill_impl(q, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale)

    @paged_attention_prefill.register_fake
    def paged_attention_prefill_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    @torch.library.custom_op("ttx::paged_attention_decode", mutates_args={})
    def paged_attention_decode(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        return paged_attention_decode_impl(q, k_cache, v_cache, seqlens, block_tables, sm_scale)

    @paged_attention_decode.register_fake
    def paged_attention_decode_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    # ====================================
    # Register Rope
    # ====================================

    @torch.library.custom_op("ttx::rope", mutates_args={})
    def rope_fwd(
        q: torch.Tensor,  # [BNSD]
        k: torch.Tensor,  # [BNSD]
        cos: torch.Tensor,  # [BSD]
        sin: torch.Tensor,  # [BSD]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [BNSD]
        return rope_fwd_impl(q, k, cos, sin)

    @rope_fwd.register_fake
    def rope_fwd_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), torch.empty_like(k)

    @torch.library.custom_op("ttx::rope_bwd", mutates_args={})
    def rope_bwd(
        dq: torch.Tensor,
        dk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_bwd_impl(dq, dk, cos, sin)

    @rope_bwd.register_fake
    def rope_bwd_fake(
        dq: torch.Tensor,
        dk: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(dq), torch.empty_like(dk)

    # ====================================
    # Register rmsnorm
    # ====================================

    @torch.library.custom_op("ttx::rmsnorm_infer", mutates_args={})
    def rmsnorm_infer(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rmsnorm_infer_impl(x, w, eps)

    @rmsnorm_infer.register_fake
    def rmsnorm_infer_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("ttx::rmsnorm_fwd", mutates_args={})
    def rmsnorm_fwd(
        X: torch.Tensor,
        W: torch.Tensor,
        eps: float,
        offset: float,
        casting_mode_int: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_fwd_impl(X, W, eps, offset, casting_mode_int)

    @rmsnorm_fwd.register_fake
    def rmsnorm_fwd_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        eps: float,
        offset: float,
        casting_mode_int: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Y = torch.empty_like(X)
        X_2d = X.reshape(-1, X.shape[-1])

        rstd_dtype = torch.float32 if casting_mode_int in (0, 1) else X.dtype  # fp32 @llama or @gemma
        RSTD = torch.empty(X_2d.shape[0], dtype=rstd_dtype, device=X.device)
        return Y, RSTD

    @torch.library.custom_op("ttx::rmsnorm_bwd", mutates_args={})
    def rmsnorm_bwd(
        dY: torch.Tensor,
        X: torch.Tensor,
        W: torch.Tensor,
        RSTD: torch.Tensor,
        offset: float,
        casting_mode_int: int,
        X_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_bwd_impl(dY, X, W, RSTD, offset, casting_mode_int, X_dtype)

    @rmsnorm_bwd.register_fake
    def rmsnorm_bwd_fake(
        dY: torch.Tensor,
        X: torch.Tensor,
        W: torch.Tensor,
        RSTD: torch.Tensor,
        offset: float,
        casting_mode_int: int,
        X_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dX = torch.empty_like(X)
        dW = torch.empty(dY.shape[-1], dtype=W.dtype, device=W.device)
        return dX, dW

else:
    gelu_fwd = gelu_fwd_impl
    gelu_bwd = gelu_bwd_impl
    silu_fwd = silu_fwd_impl
    silu_bwd = silu_bwd_impl
    swiglu_fwd = swiglu_fwd_impl
    swiglu_bwd = swiglu_bwd_impl
    paged_attention_prefill = paged_attention_prefill_impl
    paged_attention_decode = paged_attention_decode_impl
    rope_fwd = rope_fwd_impl
    rope_bwd = rope_bwd_impl
    rmsnorm_fwd = rmsnorm_fwd_impl
    rmsnorm_bwd = rmsnorm_bwd_impl
    rmsnorm_infer = rmsnorm_infer_impl
