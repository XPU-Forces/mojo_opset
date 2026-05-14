import math

from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


def assert_paged_prefill_contract(
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
) -> None:
    assert isinstance(cu_q_lens, torch.Tensor)
    assert isinstance(block_tables, torch.Tensor)
    assert cu_q_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    if cu_total_seq_lens is not None:
        assert isinstance(cu_total_seq_lens, torch.Tensor)
        assert cu_total_seq_lens.dtype == torch.int32
        assert cu_total_seq_lens.dim() == 1
        assert cu_total_seq_lens.shape[0] == q_lens.shape[0] + 1
    assert block_tables.shape[0] == q_lens.shape[0]
    assert block_tables.dim() == 2


def assert_paged_decode_contract(block_tables: torch.Tensor, total_seq_lens: torch.Tensor) -> None:
    assert isinstance(block_tables, torch.Tensor)
    assert isinstance(total_seq_lens, torch.Tensor)
    assert total_seq_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32
    assert block_tables.shape[0] == total_seq_lens.shape[0]
    assert block_tables.dim() == 2


def _seq_lens_from_cu(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


class MojoDecodeGQA(MojoOperator):
    """Non-paged GQA decode attention (single query token per batch)."""

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
    ):
        super().__init__()
        if gqa_layout not in ("ABAB", "AABB"):
            raise ValueError(f"gqa_layout must be 'ABAB' or 'AABB', got {gqa_layout}")
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        total_seq_lens: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, Hq, D)`` — one token per batch.
            key:   ``(B, Hkv, S, D)`` — full key cache.
            value: ``(B, Hkv, S, D)`` — full value cache.
            total_seq_lens: ``(B,)`` total visible sequence lengths. ``None`` → use full S.
            softmax_scale: scaling factor; defaults to ``1/sqrt(D)``.

        Returns:
            ``(B, Hq, D)``
        """
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        B, Hq, D = query.shape
        _, Hkv, S, _ = key.shape
        group = Hq // Hkv
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        outputs = torch.zeros(B, Hq, D, dtype=query.dtype, device=query.device)
        for i in range(B):
            sl = total_seq_lens[i].item() if total_seq_lens is not None else S
            if sl <= 0:
                continue
            q_i = query[i]                         # (Hq, D)
            k_i = key[i, :, :sl, :]                # (Hkv, sl, D)
            v_i = value[i, :, :sl, :]              # (Hkv, sl, D)

            if group > 1:
                if self.gqa_layout == "AABB":
                    k_i = k_i.repeat_interleave(group, dim=0)
                    v_i = v_i.repeat_interleave(group, dim=0)
                else:
                    k_i = k_i.repeat(group, 1, 1)
                    v_i = v_i.repeat(group, 1, 1)

            scores = torch.einsum("hd,hsd->hs", q_i, k_i) * softmax_scale

            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            o_i = torch.einsum("hs,hsd->hd", probs, v_i)
            outputs[i] = o_i
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}".replace("self.", "")


class MojoPagedDecodeGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
    ):
        """
        Initialize the Paged Decode GQA attention operator.

        Args:
            is_causal (bool, default=True): Enable causal masking (lower-triangular) if True.
            gqa_layout (str, default="ABAB"): GQA head grouping layout; one of {"ABAB", "AABB"}.

        Raises:
            ValueError: If `gqa_layout` is not in {"ABAB", "AABB"} 

        Notes:
            This initializer stores configuration only. Actual causal masking and window enforcement
            are applied in the forward path according to these settings.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ):
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Causal mask uses per-batch total visible KV lengths `total_seq_lens`.
            - Softmax is computed in float32 and cast back to the input dtype.
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)

        batch_size, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, head_dim = key_cache.shape

        num_share_q_heads = num_q_heads // num_kv_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        outputs = torch.zeros(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        for i in range(batch_size):
            seq_len = total_seq_lens[i].item()
            if seq_len <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged decode requires a valid block table for rows with kv lens > 0.")

            q = query[i]

            k_ref = torch.zeros(seq_len, num_kv_heads, head_dim, device=query.device, dtype=query.dtype)
            v_ref = torch.zeros(seq_len, num_kv_heads, head_dim, device=query.device, dtype=query.dtype)
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()
                if physical_block_id < 0:
                    break

                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]
                v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]

                k_ref[start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
                v_ref[start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

            if num_share_q_heads > 1:
                if self.gqa_layout == "AABB":
                    k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=1)
                    v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=1)
                else:
                    k_ref = k_ref.repeat((1, num_share_q_heads, 1))
                    v_ref = v_ref.repeat((1, num_share_q_heads, 1))

            attn_scores = torch.einsum("hd,khd->hk", q, k_ref) * softmax_scale
            # Note: if is_causal=True, we just do full attention over 1 query to seq_len key/value
            if not self.is_causal and mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[seq_len, :seq_len]
                attn_scores.masked_fill_(attn_mask.unsqueeze(0), -torch.inf)

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            outputs[i] = torch.einsum("hk,khd->hd", attn_probs, v_ref)
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}".replace("self.", "")


class MojoPrefillGQA(MojoOperator):
    """
    GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        gqa_layout (str): Layout for GQA attention.
    """

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__()

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout

    """
    Forward pass of the Mojo GQA attention operator, reference for backend.
    Args:
        query (torch.Tensor): Query tensor, in shape [B, Q_H, S, D].
        key (torch.Tensor): Key tensor, in shape [B, K_H, S, D].
        value (torch.Tensor): Value tensor, inshape [B, V_H, S, D].
        softmax_scale (float): Scaling factor for the softmax operation.

    Returns:
        torch.Tensor: Output tensor.
    """

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:

        assert cu_q_lens.dtype == torch.int32
        batch_size, num_attn_heads, seq_len, head_dim = query.size()

        num_kv_heads = k_cache.shape[1]

        group = num_attn_heads // num_kv_heads

        query = query.reshape(-1, seq_len, head_dim)
        k_cache = torch.transpose(k_cache, -2, -1)

        if self.gqa_layout == "ABAB":
            k_cache = torch.cat([k_cache] * group, axis=1).reshape(-1, head_dim, seq_len)
            v_cache = torch.cat([v_cache] * group, axis=1).reshape(-1, seq_len, head_dim)
        elif self.gqa_layout == "AABB":
            k_cache = k_cache.repeat_interleave(group, dim=1).reshape(-1, head_dim, seq_len)
            v_cache = v_cache.repeat_interleave(group, dim=1).reshape(-1, seq_len, head_dim)
        else:
            raise NotImplementedError

        score = torch.bmm(query, k_cache).float()

        if softmax_scale is None:
            score *= 1 / (head_dim**0.5)
        else:
            score *= softmax_scale

        if self.is_causal:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=query.device))
            score.masked_fill_(~mask, float("-inf"))
        else:
            raise NotImplementedError

        score = torch.softmax(score, -1).to(query.dtype)

        attn_output = torch.bmm(score, v_cache)
        attn_output = attn_output.view(batch_size, num_attn_heads, seq_len, head_dim).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, num_attn_heads, head_dim)

        return attn_output


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i; `cu_q_lens[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            cu_total_seq_lens (Optional[torch.Tensor]): Cumulative total KV lengths, shape (B+1,);
                `cu_total_seq_lens[i+1] - cu_total_seq_lens[i]` is the total visible KV length for batch i.
                If None, defaults to `cu_q_lens`.
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                If mask is None, it means a full mask or causal mask based on `is_causal`.
                If mask is not None, and is_causal=False, applies the mask to the attention scores.
                Currently we do not constrain the shape of mask, it is recommended be of shape (B, T, T) or (T, T),
                where B is the block size, and T >= max(max(total_seq_lens), max(q_lens)).

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
            - Despite the type annotation Tuple[Any], this implementation returns a single tensor.
        """
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        total_q_tokens, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        outputs = torch.zeros(total_q_tokens, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        q_lens = _seq_lens_from_cu(cu_q_lens)
        total_seq_lens = q_lens if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)
        batch_size = len(q_lens)

        for i in range(batch_size):
            q_seq_len = q_lens[i].item()
            start_loc = cu_q_lens[i].item()
            end_loc = cu_q_lens[i + 1].item()
            q = query[start_loc:end_loc]
            kv_seq_len = total_seq_lens[i].item()
            if q_seq_len == 0 or kv_seq_len <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged prefill requires a valid block table for rows with kv lens > 0.")

            num_blocks_for_seq = (kv_seq_len + block_size - 1) // block_size
            k_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)
            v_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()
                if physical_block_id < 0:
                    break

                start_pos_in_seq = j * block_size
                end_pos_in_seq = min(start_pos_in_seq + block_size, kv_seq_len)
                tokens_in_block = end_pos_in_seq - start_pos_in_seq

                k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]

                k_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_slice.permute(1, 0, 2)

                v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
                v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

            if num_q_heads != num_kv_heads:
                if self.gqa_layout == "AABB":
                    k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                    v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                else:
                    k_expanded = k_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
                    v_expanded = v_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
            else:
                k_expanded = k_unpadded
                v_expanded = v_unpadded

            attn_scores = torch.einsum("thd,khd->thk", q, k_expanded).float() * softmax_scale
            if self.is_causal:
                attn_mask = torch.ones(q_seq_len, kv_seq_len, device=query.device, dtype=torch.bool).tril(
                    kv_seq_len - q_seq_len
                )
                attn_scores.masked_fill_(~attn_mask.unsqueeze(1), -torch.inf)
            elif mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[kv_seq_len - q_seq_len : kv_seq_len, :kv_seq_len]
                attn_scores.masked_fill_(~attn_mask.unsqueeze(1), -torch.inf)

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            outputs[start_loc:end_loc] = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}".replace("self.", "")


def _make_attn_sink(num_heads: int, tensor_factory_kwargs: dict) -> torch.nn.Parameter:
    factory_kwargs = dict(tensor_factory_kwargs)
    factory_kwargs["dtype"] = torch.float32
    return torch.nn.Parameter(torch.empty(num_heads, **factory_kwargs))


def _attention_probs_with_optional_sink(
    scores: torch.Tensor,
    output_dtype: torch.dtype,
    attn_sink: Optional[torch.Tensor],
) -> torch.Tensor:
    if attn_sink is None:
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.nan_to_num(probs, nan=0.0).to(output_dtype)
    if scores.dim() < 2:
        raise ValueError(f"scores must have at least 2 dimensions, but got {scores.dim()}")
    if attn_sink.dim() != 1 or attn_sink.numel() != scores.shape[-2]:
        raise ValueError(
            f"attn_sink must be 1D with length equal to num_heads {scores.shape[-2]}, "
            f"but got shape {tuple(attn_sink.shape)}"
        )

    sink_shape = [1] * scores.dim()
    sink_shape[-2] = attn_sink.numel()
    sink_shape[-1] = 1
    sink_scores = attn_sink.float().view(sink_shape).expand(*scores.shape[:-1], 1)
    scores_with_sink = torch.cat([scores.float(), sink_scores], dim=-1)
    probs = torch.softmax(scores_with_sink, dim=-1, dtype=torch.float32)[..., :-1]
    return torch.nan_to_num(probs, nan=0.0).to(output_dtype)


class MojoDecodeMLA(MojoOperator):
    """Non-paged MLA (Multi-head Latent Attention) decode.

    KV cache stores compressed latent ``c_kv`` (shape ``kv_lora_rank``) and
    positional key ``k_pe`` (shape ``qk_rope_head_dim``).  During attention the
    latent is decompressed via ``kv_b_proj`` to recover ``k_nope`` and ``v``.
    """

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        use_attn_sink: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.use_attn_sink = use_attn_sink

        self.kv_b_proj = torch.nn.Parameter(
            torch.empty(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        )
        if use_attn_sink:
            self.attn_sink = _make_attn_sink(num_heads, self.tensor_factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        total_seq_lens: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, H, qk_nope_head_dim + qk_rope_head_dim)``
            compressed_kv: ``(B, S, kv_lora_rank)``
            k_pe: ``(B, S, 1, qk_rope_head_dim)``
            total_seq_lens: ``(B,)`` total visible KV lengths.
            softmax_scale: defaults to ``1/sqrt(qk_head_dim)``.

        Returns:
            ``(B, H, v_head_dim)``
        """
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        B, H, _ = query.shape
        S = compressed_kv.shape[1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        # Decompress → (B, S, H, qk_nope + v)
        kv = (compressed_kv @ self.kv_b_proj.T).view(
            B, S, H, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv[..., : self.qk_nope_head_dim]           # (B, S, H, d_nope)
        v = kv[..., self.qk_nope_head_dim :]                 # (B, S, H, d_v)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, H, -1)], dim=-1)  # (B, S, H, d_qk)

        scores = torch.einsum("bhd,bshd->bhs", query, k) * softmax_scale

        if total_seq_lens is not None:
            for i in range(B):
                scores[i, :, total_seq_lens[i].item() :] = float("-inf")

        probs = _attention_probs_with_optional_sink(
            scores, query.dtype, getattr(self, "attn_sink", None)
        )
        return torch.einsum("bhs,bshd->bhd", probs, v)

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, v_head_dim={self.v_head_dim}, "
            f"kv_lora_rank={self.kv_lora_rank}, use_attn_sink={self.use_attn_sink}"
        )


class MojoPagedDecodeMLA(MojoOperator):
    """Paged MLA decode — compressed KV and k_pe are stored in block caches."""

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        use_attn_sink: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.use_attn_sink = use_attn_sink

        self.kv_b_proj = torch.nn.Parameter(
            torch.empty(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        )
        if use_attn_sink:
            self.attn_sink = _make_attn_sink(num_heads, self.tensor_factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, H, qk_nope + qk_rope)``
            compressed_kv_cache: ``(N_blocks, 1, block_size, kv_lora_rank)``
            k_pe_cache: ``(N_blocks, 1, block_size, qk_rope_head_dim)``
            total_seq_lens: ``(B,)``
            block_tables: ``(B, max_num_blocks)``
            softmax_scale: defaults to ``1/sqrt(qk_head_dim)``.

        Returns:
            ``(B, H, v_head_dim)``
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)
        B, H, _ = query.shape
        block_size = compressed_kv_cache.shape[2]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        outputs = torch.zeros(B, H, self.v_head_dim, dtype=query.dtype, device=query.device)
        for i in range(B):
            sl = total_seq_lens[i].item()
            if sl <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged decode requires a valid block table for rows with kv lens > 0.")
            num_blocks = (sl + block_size - 1) // block_size

            # Reconstruct compressed_kv and k_pe from paged cache
            c_parts, pe_parts = [], []
            for j in range(num_blocks):
                bid = block_tables[i, j].item()
                if bid < 0:
                    break
                tokens = min(block_size, sl - j * block_size)
                c_parts.append(compressed_kv_cache[bid, 0, :tokens])
                pe_parts.append(k_pe_cache[bid, 0, :tokens])
            if not c_parts:
                continue
            c_kv = torch.cat(c_parts, dim=0)       # (sl, kv_lora_rank)
            k_pe = torch.cat(pe_parts, dim=0)       # (sl, qk_rope)

            # Decompress
            kv = (c_kv @ self.kv_b_proj.T).view(sl, H, self.qk_nope_head_dim + self.v_head_dim)
            k_nope = kv[..., : self.qk_nope_head_dim]
            v = kv[..., self.qk_nope_head_dim :]
            k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, H, -1)], dim=-1)

            scores = torch.einsum("hd,shd->hs", query[i], k) * softmax_scale
            probs = _attention_probs_with_optional_sink(
                scores, query.dtype, getattr(self, "attn_sink", None)
            )
            outputs[i] = torch.einsum("hs,shd->hd", probs, v)
        return outputs

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, v_head_dim={self.v_head_dim}, "
            f"kv_lora_rank={self.kv_lora_rank}, use_attn_sink={self.use_attn_sink}"
        )


# ---------------------------------------------------------------------------
# NSA (Native Sparse Attention) helpers  — module-level to avoid triggering
# MojoOperator.__init_subclass__ registration.
# ---------------------------------------------------------------------------

def _nsa_compress_kv(k, v, compress_ratio):
    """Average-pool K/V in blocks of ``compress_ratio`` along sequence dim."""
    S, H, D = k.shape
    n = (S // compress_ratio) * compress_ratio
    k_t = k[:n].view(-1, compress_ratio, H, D).mean(dim=1)
    v_t = v[:n].view(-1, compress_ratio, H, D).mean(dim=1)
    return k_t, v_t


def _nsa_select_blocks(query, comp_k, sl, softmax_scale,
                        compress_ratio, block_size, num_selected_blocks):
    """Select top-k blocks by compressed attention score, returning a mask."""
    H, D = query.shape
    C = comp_k.shape[0]
    
    # 1. Compute softmax probabilities for compressed tokens
    qk = torch.einsum("hd,chd->hc", query, comp_k) * softmax_scale
    qk = qk.softmax(dim=-1, dtype=torch.float32) # [H, C]
    
    # 2. Aggregate into blocks of size `block_size`
    tokens_per_block = block_size // compress_ratio
    num_blocks = math.ceil(sl / block_size)
    
    block_score = torch.zeros(H, num_blocks, dtype=torch.float32, device=query.device)
    for b in range(num_blocks):
        start_c = b * tokens_per_block
        end_c = min((b + 1) * tokens_per_block, C)
        if start_c < C:
            block_score[:, b] = qk[:, start_c:end_c].sum(dim=-1)
            
    # 3. Select topk blocks per head
    num_sel = min(num_selected_blocks, num_blocks)
    topk_idx = block_score.topk(num_sel, dim=-1).indices # [H, num_sel]
    
    # 4. Create mask
    mask = torch.zeros(H, sl, dtype=torch.bool, device=query.device)
    for h in range(H):
        for b in topk_idx[h]:
            start = b.item() * block_size
            end = min(start + block_size, sl)
            mask[h, start:end] = True
            
    return mask


def _nsa_window_kv(k, v, sl, window_size):
    start = max(0, sl - window_size)
    return k[start:sl], v[start:sl]


def _nsa_attend(q, k, v, softmax_scale, mask=None):
    """q: (Tq, H, D), k/v: (Tk, H, D) → (Tq, H, D)"""
    scores = torch.einsum("thd,shd->ths", q, k).float() * softmax_scale
    if mask is not None:
        # mask shape: [H, Tk] -> [1, H, Tk]
        scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    if mask is not None:
        probs = torch.nan_to_num(probs, nan=0.0)
    return torch.einsum("ths,shd->thd", probs, v)


def _nsa_gate(query, gate_proj):
    return torch.sigmoid(torch.einsum("...hd,hdc->...hc", query, gate_proj))


def _nsa_init(self, num_heads, head_dim, compress_ratio, num_selected_blocks,
              block_size, window_size, is_causal, **kwargs):
    MojoOperator.__init__(self, **kwargs)
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.compress_ratio = compress_ratio
    self.num_selected_blocks = num_selected_blocks
    self.block_size = block_size
    self.window_size = window_size
    self.is_causal = is_causal
    self.gate_proj = torch.nn.Parameter(torch.empty(num_heads, head_dim, 3))


def _nsa_extra_repr(self):
    return (
        f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
        f"compress_ratio={self.compress_ratio}, "
        f"num_selected_blocks={self.num_selected_blocks}, "
        f"block_size={self.block_size}, window_size={self.window_size}, "
        f"is_causal={self.is_causal}"
    )


def _nsa_decode_core(self, q_i, k_i, v_i, sl, softmax_scale):
    """Shared per-sample decode logic for NSA."""
    if sl <= 0:
        return torch.zeros_like(q_i)
    H, D = q_i.shape
    comp_k, comp_v = _nsa_compress_kv(k_i, v_i, self.compress_ratio)
    sel_mask = _nsa_select_blocks(
        q_i, comp_k, sl, softmax_scale,
        self.compress_ratio, self.block_size, self.num_selected_blocks,
    )
    win_k, win_v = _nsa_window_kv(k_i, v_i, sl, self.window_size)

    q_u = q_i.unsqueeze(0)
    out_comp = _nsa_attend(q_u, comp_k, comp_v, softmax_scale).squeeze(0)
    out_sel = _nsa_attend(q_u, k_i, v_i, softmax_scale, mask=sel_mask).squeeze(0)
    out_win = _nsa_attend(q_u, win_k, win_v, softmax_scale).squeeze(0)

    g = _nsa_gate(q_i, self.gate_proj)  # (H, 3)
    return g[..., 0:1] * out_comp + g[..., 1:2] * out_sel + g[..., 2:3] * out_win


class MojoDecodeNSA(MojoOperator):
    """Non-paged NSA decode (single query token per batch).

    Three attention branches — compressed (global), selected (top-k blocks),
    and sliding window (local) — blended by a per-head sigmoid gate.
    """

    def __init__(self, num_heads, head_dim, compress_ratio=4,
                 num_selected_blocks=16, block_size=64, window_size=512,
                 is_causal=True, **kwargs):
        _nsa_init(self, num_heads, head_dim, compress_ratio,
                  num_selected_blocks, block_size, window_size, is_causal, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        total_seq_lens: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, H, D)``
            key / value: ``(B, S, H, D)``
            total_seq_lens: ``(B,)``

        Returns:
            ``(B, H, D)``
        """
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        B, H, D = query.shape
        S = key.shape[1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        outputs = torch.zeros_like(query)
        for i in range(B):
            sl = total_seq_lens[i].item() if total_seq_lens is not None else S
            if sl <= 0:
                continue
            outputs[i] = _nsa_decode_core(self, query[i], key[i, :sl], value[i, :sl], sl, softmax_scale)
        return outputs

    extra_repr = _nsa_extra_repr


class MojoPagedDecodeNSA(MojoOperator):
    """Paged NSA decode with blocked KV caches."""

    def __init__(self, num_heads, head_dim, compress_ratio=4,
                 num_selected_blocks=16, block_size=64, window_size=512,
                 is_causal=True, **kwargs):
        _nsa_init(self, num_heads, head_dim, compress_ratio,
                  num_selected_blocks, block_size, window_size, is_causal, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, H, D)``
            key_cache / value_cache: ``(N_blocks, H, block_size, D)``
            total_seq_lens: ``(B,)``
            block_tables: ``(B, max_num_blocks)``

        Returns:
            ``(B, H, D)``
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)
        B, H, D = query.shape
        blk = key_cache.shape[2]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        outputs = torch.zeros_like(query)
        for i in range(B):
            sl = total_seq_lens[i].item()
            if sl <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged decode requires a valid block table for rows with kv lens > 0.")
            nb = (sl + blk - 1) // blk
            k_parts, v_parts = [], []
            for j in range(nb):
                bid = block_tables[i, j].item()
                if bid < 0:
                    break
                t = min(blk, sl - j * blk)
                k_parts.append(key_cache[bid, :, :t].permute(1, 0, 2))
                v_parts.append(value_cache[bid, :, :t].permute(1, 0, 2))
            if not k_parts:
                continue
            k_i = torch.cat(k_parts, dim=0)
            v_i = torch.cat(v_parts, dim=0)
            outputs[i] = _nsa_decode_core(self, query[i], k_i, v_i, sl, softmax_scale)
        return outputs

    extra_repr = _nsa_extra_repr


class MojoPrefillMLA(MojoOperator):
    """Non-paged MLA prefill — variable-length sequences."""

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        is_causal: bool = True,
        use_attn_sink: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.is_causal = is_causal
        self.use_attn_sink = use_attn_sink

        self.kv_b_proj = torch.nn.Parameter(
            torch.empty(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        )
        if use_attn_sink:
            self.attn_sink = _make_attn_sink(num_heads, self.tensor_factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(T, H, qk_nope + qk_rope)`` packed queries.
            compressed_kv: ``(T, kv_lora_rank)`` packed compressed KV.
            k_pe: ``(T, 1, qk_rope_head_dim)`` packed positional keys.
            cu_q_lens: ``(B+1,)`` cumulative query lengths.
            softmax_scale: defaults to ``1/sqrt(qk_head_dim)``.

        Returns:
            ``(T, H, v_head_dim)``
        """
        assert cu_q_lens.dtype == torch.int32
        T, H, _ = query.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        # Decompress all at once → (T, H, d_nope + d_v)
        kv = (compressed_kv @ self.kv_b_proj.T).view(T, H, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v_all = kv[..., self.qk_nope_head_dim :]
        k_all = torch.cat([k_nope, k_pe.expand(-1, H, -1)], dim=-1)  # (T, H, d_qk)

        outputs = torch.zeros(T, H, self.v_head_dim, dtype=query.dtype, device=query.device)
        batch_size = cu_q_lens.shape[0] - 1

        for i in range(batch_size):
            s = cu_q_lens[i].item()
            e = cu_q_lens[i + 1].item()
            q_i = query[s:e]       # (L, H, d_qk)
            k_i = k_all[s:e]       # (L, H, d_qk)
            v_i = v_all[s:e]       # (L, H, d_v)

            scores = torch.einsum("thd,shd->ths", q_i, k_i) * softmax_scale

            if self.is_causal:
                L = e - s
                causal_mask = torch.tril(torch.ones(L, L, device=query.device, dtype=torch.bool))
                scores.masked_fill_(~causal_mask.unsqueeze(1), float("-inf"))

            probs = _attention_probs_with_optional_sink(
                scores, query.dtype, getattr(self, "attn_sink", None)
            )
            outputs[s:e] = torch.einsum("ths,shd->thd", probs, v_i)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, v_head_dim={self.v_head_dim}, "
            f"kv_lora_rank={self.kv_lora_rank}, is_causal={self.is_causal}, "
            f"use_attn_sink={self.use_attn_sink}"
        )


class MojoPagedPrefillMLA(MojoOperator):
    """Paged MLA prefill with blocked compressed KV and k_pe caches."""

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        is_causal: bool = True,
        use_attn_sink: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.is_causal = is_causal
        self.use_attn_sink = use_attn_sink

        self.kv_b_proj = torch.nn.Parameter(
            torch.empty(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        )
        if use_attn_sink:
            self.attn_sink = _make_attn_sink(num_heads, self.tensor_factory_kwargs)

    def _unpage(self, cache, block_tables_i, sl, block_size):
        """Reconstruct contiguous sequence from paged blocks for one batch."""
        if sl <= 0:
            return None
        num_blocks = (sl + block_size - 1) // block_size
        parts = []
        for j in range(num_blocks):
            bid = block_tables_i[j].item()
            if bid < 0:
                break
            tokens = min(block_size, sl - j * block_size)
            parts.append(cache[bid, 0, :tokens])
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(T, H, qk_nope + qk_rope)``
            compressed_kv_cache: ``(N_blocks, 1, block_size, kv_lora_rank)``
            k_pe_cache: ``(N_blocks, 1, block_size, qk_rope_head_dim)``
            cu_q_lens: ``(B+1,)``
            block_tables: ``(B, max_num_blocks)``
            softmax_scale: defaults to ``1/sqrt(qk_head_dim)``.
            cu_total_seq_lens: ``(B+1,)`` cumulative total KV lengths. ``None`` → same as ``cu_q_lens``.

        Returns:
            ``(T, H, v_head_dim)``
        """
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        T, H, _ = query.shape
        block_size = compressed_kv_cache.shape[2]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        q_lens = _seq_lens_from_cu(cu_q_lens)
        total_seq_lens = q_lens if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)
        batch_size = len(q_lens)
        outputs = torch.zeros(T, H, self.v_head_dim, dtype=query.dtype, device=query.device)

        for i in range(batch_size):
            qs = cu_q_lens[i].item()
            qe = cu_q_lens[i + 1].item()
            q_i = query[qs:qe]
            kv_len = total_seq_lens[i].item()
            if q_i.shape[0] == 0 or kv_len <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged prefill requires a valid block table for rows with kv lens > 0.")

            c_kv = self._unpage(compressed_kv_cache, block_tables[i], kv_len, block_size)
            kpe = self._unpage(k_pe_cache, block_tables[i], kv_len, block_size)
            if c_kv is None or kpe is None:
                continue

            kv = (c_kv @ self.kv_b_proj.T).view(kv_len, H, self.qk_nope_head_dim + self.v_head_dim)
            k_nope = kv[..., : self.qk_nope_head_dim]
            v = kv[..., self.qk_nope_head_dim :]
            k = torch.cat([k_nope, kpe.unsqueeze(1).expand(-1, H, -1)], dim=-1)

            scores = torch.einsum("thd,shd->ths", q_i, k).float() * softmax_scale

            if self.is_causal:
                q_len = qe - qs
                causal_mask = torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool).tril(
                    kv_len - q_len
                )
                scores.masked_fill_(~causal_mask.unsqueeze(1), float("-inf"))

            probs = _attention_probs_with_optional_sink(
                scores, query.dtype, getattr(self, "attn_sink", None)
            )
            outputs[qs:qe] = torch.einsum("ths,shd->thd", probs, v)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, v_head_dim={self.v_head_dim}, "
            f"kv_lora_rank={self.kv_lora_rank}, is_causal={self.is_causal}, "
            f"use_attn_sink={self.use_attn_sink}"
        )


class MojoPrefillNSA(MojoOperator):
    """Non-paged NSA prefill — variable-length packed sequences."""

    def __init__(self, num_heads, head_dim, compress_ratio=4,
                 num_selected_blocks=16, block_size=64, window_size=512,
                 is_causal=True, **kwargs):
        _nsa_init(self, num_heads, head_dim, compress_ratio,
                  num_selected_blocks, block_size, window_size, is_causal, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            query / key / value: ``(T, H, D)``
            cu_q_lens: ``(B+1,)``

        Returns:
            ``(T, H, D)``
        """
        assert cu_q_lens.dtype == torch.int32
        T, H, D = query.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        outputs = torch.zeros_like(query)
        batch_size = cu_q_lens.shape[0] - 1
        cr = self.compress_ratio

        for i in range(batch_size):
            s = cu_q_lens[i].item()
            e = cu_q_lens[i + 1].item()
            q_seq, k_seq, v_seq = query[s:e], key[s:e], value[s:e]

            for t in range(e - s):
                t_sl = t + 1 if self.is_causal else (e - s)
                k_ctx, v_ctx = k_seq[:t_sl], v_seq[:t_sl]

                ck, cv = _nsa_compress_kv(k_ctx, v_ctx, cr) if t_sl >= cr else (k_ctx, v_ctx)
                sel_mask = _nsa_select_blocks(
                    q_seq[t], ck, t_sl, softmax_scale,
                    cr, self.block_size, self.num_selected_blocks,
                )
                win_k, win_v = _nsa_window_kv(k_ctx, v_ctx, t_sl, self.window_size)

                q_t = q_seq[t:t + 1]
                out_comp = _nsa_attend(q_t, ck, cv, softmax_scale).squeeze(0)
                out_sel = _nsa_attend(q_t, k_ctx, v_ctx, softmax_scale, mask=sel_mask).squeeze(0)
                out_win = _nsa_attend(q_t, win_k, win_v, softmax_scale).squeeze(0)

                g = _nsa_gate(q_seq[t], self.gate_proj)
                outputs[s + t] = g[..., 0:1] * out_comp + g[..., 1:2] * out_sel + g[..., 2:3] * out_win

        return outputs

    extra_repr = _nsa_extra_repr


class MojoPagedPrefillNSA(MojoOperator):
    """Paged NSA prefill with blocked KV caches."""

    def __init__(self, num_heads, head_dim, compress_ratio=4,
                 num_selected_blocks=16, block_size=64, window_size=512,
                 is_causal=True, **kwargs):
        _nsa_init(self, num_heads, head_dim, compress_ratio,
                  num_selected_blocks, block_size, window_size, is_causal, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(T, H, D)``
            key_cache / value_cache: ``(N_blocks, H, block_size, D)``
            cu_q_lens: ``(B+1,)``
            block_tables: ``(B, max_num_blocks)``
            cu_total_seq_lens: ``(B+1,)``

        Returns:
            ``(T, H, D)``
        """
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        T, H, D = query.shape
        blk = key_cache.shape[2]
        cr = self.compress_ratio
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        q_lens = _seq_lens_from_cu(cu_q_lens)
        total_seq_lens = q_lens if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)
        batch_size = len(q_lens)
        outputs = torch.zeros_like(query)

        for i in range(batch_size):
            qs, qe = cu_q_lens[i].item(), cu_q_lens[i + 1].item()
            q_seq = query[qs:qe]
            kv_len = total_seq_lens[i].item()
            q_len = qe - qs
            if q_len == 0 or kv_len <= 0:
                continue
            if block_tables[i, 0].item() < 0:
                raise ValueError("Paged prefill requires a valid block table for rows with kv lens > 0.")

            nb = (kv_len + blk - 1) // blk
            k_parts, v_parts = [], []
            for j in range(nb):
                bid = block_tables[i, j].item()
                if bid < 0:
                    break
                t = min(blk, kv_len - j * blk)
                k_parts.append(key_cache[bid, :, :t].permute(1, 0, 2))
                v_parts.append(value_cache[bid, :, :t].permute(1, 0, 2))
            if not k_parts:
                continue
            k_seq = torch.cat(k_parts, dim=0)
            v_seq = torch.cat(v_parts, dim=0)

            for t_idx in range(q_len):
                t_kv = (kv_len - q_len + t_idx + 1) if self.is_causal else kv_len
                k_ctx, v_ctx = k_seq[:t_kv], v_seq[:t_kv]

                ck, cv = _nsa_compress_kv(k_ctx, v_ctx, cr) if t_kv >= cr else (k_ctx, v_ctx)
                sel_mask = _nsa_select_blocks(
                    q_seq[t_idx], ck, t_kv, softmax_scale,
                    cr, self.block_size, self.num_selected_blocks,
                )
                win_k, win_v = _nsa_window_kv(k_ctx, v_ctx, t_kv, self.window_size)

                q_t = q_seq[t_idx:t_idx + 1]
                out_comp = _nsa_attend(q_t, ck, cv, softmax_scale).squeeze(0)
                out_sel = _nsa_attend(q_t, k_ctx, v_ctx, softmax_scale, mask=sel_mask).squeeze(0)
                out_win = _nsa_attend(q_t, win_k, win_v, softmax_scale).squeeze(0)

                g = _nsa_gate(q_seq[t_idx], self.gate_proj)
                outputs[qs + t_idx] = g[..., 0:1] * out_comp + g[..., 1:2] * out_sel + g[..., 2:3] * out_win

        return outputs

    extra_repr = _nsa_extra_repr


class MojoSdpa(MojoOperator):
    def __init__(
        self,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.enable_gqa = enable_gqa

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention (SDPA) operator.

        Args:
            query (torch.Tensor): Query tensor; shape must be compatible with SDPA.
            key (torch.Tensor): Key tensor; same embedding dimension as query.
            value (torch.Tensor): Value tensor; same embedding dimension as key.
            attn_mask (Optional[torch.Tensor]): Attention mask tensor; shape must be broadcastable with SDPA.

        Returns:
            torch.Tensor: Attention output with the same batch/head layout as `query`.

        Notes:
            - Uses `attn_mask=attn_mask` (provided externally) and disables dropout.
            - `scale=self.scale` sets custom scaling; if None, SDPA uses default scaling.
            - `enable_gqa=self.enable_gqa` allows grouped query attention when supported.
        """
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
        )

        return output

    def extra_repr(self) -> str:
        return f"{self.scale=}, {self.enable_gqa=}".replace("self.", "")


def _generate_window_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_computed_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            )
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (torch.arange(0, kv_seq_len) < global_window_size)[None, :] if global_window_size is not None else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask

    return mask

class MojoPagedPrefillSWA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - global_window_size (Optional[int]): Global attention window length; None means no global window, default None. Only effective when is_causal=True.
        - local_window_size (Optional[int]): Local attention window length; None means no local window, default None. Only effective when is_causal=True.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size

    def forward(
        self,
        query: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        cu_q_lens: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,  # [bsz + 1]
        *,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.
        # max_q_lens / max_total_seq_lens match fused backends (e.g. ixformer); unused here.

        assert_paged_prefill_contract(cu_q_lens, block_table, cu_total_seq_lens)
        total_q_len, n_q_heads, head_dim = query.shape
        _, n_kv_heads, page_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        total_seq_lens = _seq_lens_from_cu(cu_q_lens) if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)

        o = torch.empty_like(query)
        bsz = cu_q_lens.shape[0] - 1
        for i in range(bsz):
            q_i = query[cu_q_lens[i] : cu_q_lens[i + 1]]
            q_seq_len = q_i.shape[0]
            if q_seq_len == 0:
                # skip padded query
                continue
            q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

            kv_seq_len = total_seq_lens[i].item()
            if kv_seq_len <= 0:
                continue
            if block_table[i, 0].item() < 0:
                raise ValueError("Paged prefill requires a valid block table for rows with kv lens > 0.")
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = key_cache[block_table[i, :kv_blocks]]  # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * softmax_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

            if self.is_causal:
                s_mask = _generate_window_mask(
                    q_seq_len,
                    kv_seq_len,
                    self.local_window_size,
                    self.global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
            s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]
            p_i = p_i.to(query.dtype)

            v_i = value_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
            o[cu_q_lens[i] : cu_q_lens[i + 1]] = o_i.to(o.dtype)
        return o
    
    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, global_window_size={self.global_window_size}, local_window_size={self.local_window_size}"


class MojoPagedDecodeSWA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - global_window_size (Optional[int]): Global attention window length; None means no global window, default None. Only effective when is_causal=True.
        - local_window_size (Optional[int]): Local attention window length; None means no local window, default None. Only effective when is_causal=True.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size

    def forward(
        self,
        query: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        total_seq_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        # Note: for decode kernel, is_causal = False should never happen

        assert_paged_decode_contract(block_table, total_seq_lens)
        bsz, n_q_heads, head_dim = query.shape
        _, n_kv_heads, page_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        o = torch.zeros_like(query)
        for i in range(bsz):
            q_i = query[i].unsqueeze(1) # -> [n_q_heads, 1, head_dim]

            kv_seq_len = total_seq_lens[i].item()
            if kv_seq_len <= 0:
                # skip padded tokens
                continue
            if block_table[i, 0].item() < 0:
                raise ValueError("Paged decode requires a valid block table for rows with kv lens > 0.")
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = key_cache[block_table[i, :kv_blocks]]  # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * softmax_scale  # -> [n_q_heads, 1, kv_seq_len]

            if self.is_causal:
                s_mask = _generate_window_mask(
                    1,
                    kv_seq_len,
                    self.local_window_size,
                    self.global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, 1, 1]
            s_i = s_i - m_i  # -> [n_q_heads, 1, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, 1, 1]
            p_i = p_i.to(query.dtype)

            v_i = value_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, 1, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, 1, head_dim]
            o_i = o_i.squeeze(1)  # -> [n_q_heads, head_dim]
            o[i] = o_i.to(o.dtype)
        return o

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, global_window_size={self.global_window_size}, local_window_size={self.local_window_size}"


class MojoSWA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - global_window_size (Optional[int]): Global attention window length; None means no global window, default None. Only effective when is_causal=True.
        - local_window_size (Optional[int]): Local attention window length; None means no local window, default None. Only effective when is_causal=True.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size

    def forward(
        self,
        query: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        key: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        value: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_q_lens: torch.Tensor,  # [bsz + 1]
        cu_total_seq_lens: torch.Tensor,  # [bsz + 1]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        assert cu_q_lens.dtype == torch.int32
        assert cu_total_seq_lens.dtype == torch.int32
        total_q_len, n_q_heads, head_dim = query.shape
        n_kv_heads = key.shape[1]
        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        o = torch.empty_like(query)
        bsz = cu_q_lens.shape[0] - 1
        for i in range(bsz):
            q_i = query[cu_q_lens[i] : cu_q_lens[i + 1]]
            q_seq_len = q_i.shape[0]
            q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

            k_i = key[cu_total_seq_lens[i] : cu_total_seq_lens[i + 1]]
            kv_seq_len = k_i.shape[0]
            k_i_T = k_i.permute(1, 2, 0)
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * softmax_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

            if self.is_causal:
                s_mask = _generate_window_mask(
                    q_seq_len,
                    kv_seq_len,
                    self.local_window_size,
                    self.global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
            s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]
            p_i = p_i.to(value.dtype)

            v_i = value[cu_total_seq_lens[i] : cu_total_seq_lens[i + 1]].permute(1, 0, 2)  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
            o[cu_q_lens[i] : cu_q_lens[i + 1]] = o_i.to(o.dtype)
        return o

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, global_window_size={self.global_window_size}, local_window_size={self.local_window_size}"


def _dynamic_quantize(tensor, qmax, qmin, quant_dtype):
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / qmax
    scale = torch.where(scale < 1e-6, 1.0, scale)
    
    tensor_scaled = tensor / scale
    tensor_quant = tensor_scaled.round().clamp(qmin, qmax).to(quant_dtype)
    
    scale = scale.view(*tensor.shape[:-1], 1)
    return tensor_quant, scale

class MojoPagedPrefillQuantGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - is_causal (bool): Whether to enable causal masking, default True.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - query_dtype (torch.dtype): the dtype for query, default torch.bfloat16 for non-quantized query.
        - context_dtype (torch.dtype): The context dtype for key_cache and value_cache, default torch.int8.
        - compute_dtype (torch.dtype): The quant matmul dtype for Q@K and P@V, default torch.bfloat16.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.query_dtype = query_dtype
        self.context_dtype = context_dtype
        self.compute_dtype = compute_dtype
        assert self.query_dtype in (torch.bfloat16, torch.int8), f"Unsupported query dtype {self.query_dtype}"
        if self.query_dtype == torch.int8:
            raise NotImplementedError("Quantized query is not implemented")
        assert self.context_dtype == torch.int8, f"Quant attention support int8 context only, but got {self.context_dtype}"
        assert self.compute_dtype in (torch.bfloat16, torch.int8), f"Unsupported compute dtype {self.compute_dtype}"
        if self.compute_dtype == torch.int8:
            bits = 8
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -(2 ** (bits - 1))

    def forward(
        self,
        query: torch.Tensor,
        query_scale: Optional[torch.Tensor],
        key_cache: torch.Tensor,
        key_scale: torch.Tensor,
        value_cache: torch.Tensor,
        value_scale: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D), it can be of dtype bf16 or int8
            query_scale (torch.Tensor): if query is quantized, it should be per-token scale of query shape (T, Hq, 1) and dtype bfloat16; if not, it should be None
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            key_scale (torch.Tensor): per-channel scale of key, shape (Hkv, D) and dtype bfloat16
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D) and dtype in8.
            value_scale (torch.Tensor): per-channel scale of value, shape (Hkv, D)
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i; `cu_q_lens[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            cu_total_seq_lens (Optional[torch.Tensor]): Cumulative total KV lengths, shape (B+1,);
                `cu_total_seq_lens[i+1] - cu_total_seq_lens[i]` is the total visible KV length for batch i.
                If None, defaults to `cu_q_lens`.
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                If mask is None, it means a full mask or causal mask based on `is_causal`.
                If mask is not None, and is_causal=False, applies the mask to the attention scores.
            max_q_lens (Optional[int]): Hint for the maximum query length (unused).
            max_total_seq_lens (Optional[int]): Hint for the maximum total KV length (unused).

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
        """
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        if self.query_dtype == torch.int8:
            assert query_scale is not None and query.dtype == self.query_dtype, "query_scale must be provided for quantized query"
        else:
            assert query_scale is None and query.dtype == self.query_dtype, "query_scale must be None for non-quantized query"
        
        total_q_tokens, num_q_heads, head_dim = query.shape
        _, num_kv_heads, page_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        total_seq_lens = (
            _seq_lens_from_cu(cu_q_lens) if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)
        )

        if num_q_heads != num_kv_heads:
            if self.gqa_layout == "AABB":
                key_scale = key_scale.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
                value_scale = value_scale.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
            else:
                key_scale = key_scale.repeat((num_q_heads // num_kv_heads, 1))
                value_scale = value_scale.repeat((num_q_heads // num_kv_heads, 1))

        outputs = torch.zeros(total_q_tokens, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        batch_size = len(q_lens)

        for i in range(batch_size):
            q_seq_len = q_lens[i].item()
            start_loc = cu_q_lens[i].item()
            end_loc = cu_q_lens[i + 1].item()
            q = query[start_loc:end_loc].permute(1, 0, 2) # [n_q_heads, q_seq_len, head_dim]
            kv_seq_len = total_seq_lens[i].item()

            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_unpadded = key_cache[block_tables[i, :kv_blocks]] # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_unpadded = k_unpadded.permute(1, 0, 2, 3).reshape(num_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            v_unpadded = value_cache[block_tables[i, :kv_blocks]] # [kv_blocks, n_kv_heads, page_size, head_dim]
            v_unpadded = v_unpadded.permute(1, 0, 2, 3).reshape(num_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]

            if num_q_heads != num_kv_heads:
                if self.gqa_layout == "AABB":
                    k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
                    v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
                else:
                    k_expanded = k_unpadded.repeat((num_q_heads // num_kv_heads, 1, 1))
                    v_expanded = v_unpadded.repeat((num_q_heads // num_kv_heads, 1, 1))
            else:
                k_expanded = k_unpadded
                v_expanded = v_unpadded

            if self.compute_dtype == torch.int8:
                q_quant, q_scale = _dynamic_quantize(q * key_scale.unsqueeze(1), self.qmax, self.qmin, self.compute_dtype)
                attn_scores = torch.matmul(q_quant.float(), k_expanded.mT.float()) * q_scale * softmax_scale
            else:
                k_expanded_scaled = k_expanded.float() * key_scale.unsqueeze(1).float()
                attn_scores = torch.matmul(q.float(), k_expanded_scaled.mT) * softmax_scale
            if self.is_causal:
                attn_mask = torch.ones(q_seq_len, kv_seq_len, device=query.device, dtype=torch.bool).tril(
                    kv_seq_len - q_seq_len
                )
                attn_scores = torch.where(attn_mask, attn_scores, float("-inf"))
            elif mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[kv_seq_len - q_seq_len : kv_seq_len, :kv_seq_len]
                attn_scores = torch.where(attn_mask, attn_scores, float("-inf"))

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            if self.compute_dtype == torch.int8:
                attn_probs_quant, attn_probs_scale = _dynamic_quantize(attn_probs, self.qmax, self.qmin, self.compute_dtype)
                o = torch.matmul(attn_probs_quant.float(), v_expanded.float()) * attn_probs_scale * value_scale.unsqueeze(1)
            else:
                v_expanded_scaled = v_expanded.float() * value_scale.unsqueeze(1).float()
                o = torch.matmul(attn_probs.float(), v_expanded_scaled)
            outputs[start_loc:end_loc] = o.permute(1, 0, 2)
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}, {self.query_dtype=}, {self.context_dtype=}, {self.compute_dtype=}".replace("self.", "")


class MojoPagedDecodeQuantGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Paged Decode GQA attention operator.

        Args:
            is_causal (bool, default=True): Enable causal masking (lower-triangular) if True.
            gqa_layout (str, default="ABAB"): GQA head grouping layout; one of {"ABAB", "AABB"}.
            query_dtype (torch.dtype): the dtype for query, default torch.bfloat16 for non-quantized query.
            context_dtype (torch.dtype): The context dtype for key_cache and value_cache, default torch.int8.
            compute_dtype (torch.dtype): The quant matmul dtype for Q@K and P@V, default torch.bfloat16.

        Raises:
            ValueError: If `gqa_layout` is not in {"ABAB", "AABB"}

        Notes:
            This initializer stores configuration only. Actual causal masking and window enforcement
            are applied in the forward path according to these settings.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.query_dtype = query_dtype
        self.context_dtype = context_dtype
        self.compute_dtype = compute_dtype
        assert self.query_dtype in (torch.bfloat16, torch.int8), f"Unsupported query dtype {self.query_dtype}"
        if self.query_dtype == torch.int8:
            raise NotImplementedError("Quantized query is not implemented")
        assert self.context_dtype == torch.int8, f"Quant attention support int8 context only, but got {self.context_dtype}"
        assert self.compute_dtype in (torch.bfloat16, torch.int8), f"Unsupported compute dtype {self.compute_dtype}"
        if self.compute_dtype == torch.int8:
            bits = 8
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -(2 ** (bits - 1))

    def forward(
        self,
        query: torch.Tensor,
        query_scale: Optional[torch.Tensor],
        key_cache: torch.Tensor,
        key_scale: torch.Tensor,
        value_cache: torch.Tensor,
        value_scale: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        $$\text{Attention}(Q, K, V) = \text{dequant} \left( \text{Quant} \left( \text{softmax} \left( \frac{\text{Quant}(Q) \cdot \text{Quant}(K)^T}{\text{scale}} \right) \right) \cdot V_{int8} \right)$$

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D), it can be of dtype bf16 or int8
            query_scale (torch.Tensor): if query is quantized, it should be per-token scale of query shape (B, Hq, 1) and dtype bfloat16; if not, it should be None
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            key_scale (torch.Tensor): per-channel scale of key, shape (Hkv, D) and dtype bfloat16.
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            value_scale (torch.Tensor): per-channel scale of value, shape (Hkv, D) and dtype bfloat16.
            total_seq_lens (torch.Tensor): Per-batch KV lengths, shape (B,).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
            max_total_seq_len (Optional[int]): Hint for the maximum total KV length (unused).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Causal mask uses per-batch sequence lengths `total_seq_lens`.
            - Softmax is computed in float32 and cast back to the input dtype.
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)
        if self.query_dtype == torch.int8:
            assert query_scale is not None and query.dtype == self.query_dtype, "query_scale must be provided for quantized query"
        else:
            assert query_scale is None and query.dtype == self.query_dtype, "query_scale must be None for non-quantized query"
        

        batch_size, num_q_heads, head_dim = query.shape
        _, num_kv_heads, page_size, head_dim = key_cache.shape

        num_share_q_heads = num_q_heads // num_kv_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if num_share_q_heads > 1:
            if self.gqa_layout == "AABB":
                key_scale = key_scale.repeat_interleave(num_share_q_heads, dim=0)
                value_scale = value_scale.repeat_interleave(num_share_q_heads, dim=0)
            else:
                key_scale = key_scale.repeat((num_share_q_heads, 1))
                value_scale = value_scale.repeat((num_share_q_heads, 1))

        outputs = torch.zeros(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        for i in range(batch_size):
            seq_len = total_seq_lens[i].item()
            if seq_len == 0:
                # skip padded batches
                continue

            q = query[i].unsqueeze(1) # [n_q_heads, 1, head_dim]

            kv_blocks = (seq_len + page_size - 1) // page_size
            k_unpadded = key_cache[block_tables[i, :kv_blocks]] # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_unpadded = k_unpadded.permute(1, 0, 2, 3).reshape(num_kv_heads, kv_blocks * page_size, head_dim)[:, :seq_len]
            v_unpadded = value_cache[block_tables[i, :kv_blocks]] # [kv_blocks, n_kv_heads, page_size, head_dim]
            v_unpadded = v_unpadded.permute(1, 0, 2, 3).reshape(num_kv_heads, kv_blocks * page_size, head_dim)[:, :seq_len]

            if num_q_heads != num_kv_heads:
                if self.gqa_layout == "AABB":
                    k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
                    v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
                else:
                    k_expanded = k_unpadded.repeat((num_q_heads // num_kv_heads, 1, 1))
                    v_expanded = v_unpadded.repeat((num_q_heads // num_kv_heads, 1, 1))
            else:
                k_expanded = k_unpadded
                v_expanded = v_unpadded

            if self.compute_dtype == torch.int8:
                q_quant, q_scale = _dynamic_quantize(q * key_scale.unsqueeze(1), self.qmax, self.qmin, self.compute_dtype)
                attn_scores = torch.matmul(q_quant.float(), k_expanded.mT.float()) * q_scale * softmax_scale
            else:
                k_expanded_scaled = k_expanded.float() * key_scale.unsqueeze(1).float()
                attn_scores = torch.matmul(q.float(), k_expanded_scaled.mT) * softmax_scale
            # Note: if is_causal=True, we just do full attention over 1 query to seq_len key/value
            if not self.is_causal and mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[seq_len, :seq_len]
                attn_scores = torch.where(attn_mask, attn_scores, float("-inf"))

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            if self.compute_dtype == torch.int8:
                attn_probs_quant, attn_probs_scale = _dynamic_quantize(attn_probs, self.qmax, self.qmin, self.compute_dtype)
                o = torch.matmul(attn_probs_quant.float(), v_expanded.float()) * attn_probs_scale * value_scale.unsqueeze(1)
            else:
                v_expanded_scaled = v_expanded.float() * value_scale.unsqueeze(1).float()
                o = torch.matmul(attn_probs.float(), v_expanded_scaled)
            outputs[i] = o.squeeze(1)
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}, {self.query_dtype=}, {self.context_dtype=}, {self.compute_dtype=}".replace("self.", "")


class MojoPagedPrefillQuantSWA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Paged Prefill SWA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - global_window_size (Optional[int]): Global attention window length; None means no global window, default None. Only effective when is_causal=True.
        - local_window_size (Optional[int]): Local attention window length; None means no local window, default None. Only effective when is_causal=True.
        - query_dtype (torch.dtype): the dtype for query, default torch.bfloat16 for non-quantized query.
        - context_dtype (torch.dtype): The context dtype for key_cache and value_cache, default torch.int8.
        - compute_dtype (torch.dtype): The quant matmul dtype for Q@K and P@V, default torch.bfloat16.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size
        self.query_dtype = query_dtype
        self.context_dtype = context_dtype
        self.compute_dtype = compute_dtype
        assert self.query_dtype in (torch.bfloat16, torch.int8), f"Unsupported query dtype {self.query_dtype}"
        if self.query_dtype == torch.int8:
            raise NotImplementedError("Quantized query is not implemented")
        assert self.context_dtype == torch.int8, f"Quant attention support int8 context only, but got {self.context_dtype}"
        assert self.compute_dtype in (torch.bfloat16, torch.int8), f"Unsupported compute dtype {self.compute_dtype}"
        if self.compute_dtype == torch.int8:
            bits = 8
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -(2 ** (bits - 1))

    def forward(
        self,
        query: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        query_scale: Optional[torch.Tensor],  # [total_q_len, n_q_heads, 1]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        key_scale: torch.Tensor, # [n_kv_heads, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_scale: torch.Tensor, # [n_kv_heads, head_dim]
        cu_q_lens: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,  # [bsz + 1]
    ) -> torch.Tensor:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D), it can be of dtype bf16 or int8
            query_scale (torch.Tensor): if query is quantized, it should be per-token scale of query shape (T, Hq, 1) and dtype bfloat16; if not, it should be None
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            key_scale (torch.Tensor): per-channel scale of key, shape (Hkv, D) and dtype bfloat16
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D) and dtype in8.
            value_scale (torch.Tensor): per-channel scale of value, shape (Hkv, D)
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i; `cu_q_lens[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            cu_total_seq_lens (Optional[torch.Tensor]): Cumulative total KV lengths, shape (B+1,);
                `cu_total_seq_lens[i+1] - cu_total_seq_lens[i]` is the total visible KV length for batch i.
                If None, defaults to `cu_q_lens`.

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
        """
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        assert_paged_prefill_contract(cu_q_lens, block_table, cu_total_seq_lens)
        if self.query_dtype == torch.int8:
            assert query_scale is not None and query.dtype == self.query_dtype, "query_scale must be provided for quantized query"
        else:
            assert query_scale is None and query.dtype == self.query_dtype, "query_scale must be None for non-quantized query"
        
        total_q_len, n_q_heads, head_dim = query.shape
        _, n_kv_heads, page_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        seqlens_kv = (
            _seq_lens_from_cu(cu_q_lens) if cu_total_seq_lens is None else _seq_lens_from_cu(cu_total_seq_lens)
        )

        if n_q_heads != n_kv_heads:
            if self.gqa_interleave:
                key_scale = key_scale.repeat((n_q_heads // n_kv_heads, 1)) # -> [n_q_heads, head_dim]
                value_scale = value_scale.repeat((n_q_heads // n_kv_heads, 1)) # -> [n_q_heads, head_dim]
            else:                
                key_scale = key_scale.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim]
                value_scale = value_scale.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim]
        
        o = torch.empty_like(query)
        bsz = cu_q_lens.shape[0] - 1
        for i in range(bsz):
            q_i = query[cu_q_lens[i] : cu_q_lens[i + 1]]
            q_seq_len = q_i.shape[0]
            if q_seq_len == 0:
                # skip padded query
                continue
            q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

            kv_seq_len = seqlens_kv[i].item()
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = key_cache[block_table[i, :kv_blocks]] # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            
            if self.compute_dtype == torch.int8:
                q_i_quant, q_i_scale = _dynamic_quantize(q_i * key_scale.unsqueeze(1), self.qmax, self.qmin, self.compute_dtype)
                s_i = torch.bmm(q_i_quant.float(), k_i_T.float()) * q_i_scale * softmax_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]
            else:
                k_i_T = k_i_T.float() * key_scale.unsqueeze(-1).float()
                s_i = torch.bmm(q_i.float(), k_i_T.float()) * softmax_scale

            if self.is_causal:
                s_mask = _generate_window_mask(
                    q_seq_len,
                    kv_seq_len,
                    self.local_window_size,
                    self.global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
            s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]

            v_i = value_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            if self.compute_dtype == torch.int8:
                p_i_quant, p_i_scale = _dynamic_quantize(p_i, self.qmax, self.qmin, self.compute_dtype)
                o_i = torch.bmm(p_i_quant.float(), v_i.float()) * p_i_scale * value_scale.unsqueeze(1)  # -> [n_q_heads, q_seq_len, head_dim]
            else:
                v_i = v_i.float() * value_scale.unsqueeze(1).float()
                o_i = torch.bmm(p_i.float(), v_i.float()) # -> [n_q_heads, q_seq_len, head_dim]
            
            o_i = o_i / l_i
            o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
            o[cu_q_lens[i] : cu_q_lens[i + 1]] = o_i.to(o.dtype)
        return o
    
    def extra_repr(self):
        return f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, global_window_size={self.global_window_size}, local_window_size={self.local_window_size}, query_dtype={self.query_dtype}, context_dtype={self.context_dtype}, compute_dtype={self.compute_dtype}"

class MojoPagedDecodeQuantSWA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Paged Prefill SWA attention operator with common parameters.
        Parameter descriptions:
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - global_window_size (Optional[int]): Global attention window length; None means no global window, default None. Only effective when is_causal=True.
        - local_window_size (Optional[int]): Local attention window length; None means no local window, default None. Only effective when is_causal=True.
        - query_dtype (torch.dtype): the dtype for query, default torch.bfloat16 for non-quantized query.
        - context_dtype (torch.dtype): The context dtype for key_cache and value_cache, default torch.int8.
        - compute_dtype (torch.dtype): The quant matmul dtype for Q@K and P@V, default torch.bfloat16.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.gqa_interleave = gqa_layout == "ABAB"
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size
        self.query_dtype = query_dtype
        self.context_dtype = context_dtype
        self.compute_dtype = compute_dtype
        assert self.query_dtype in (torch.bfloat16, torch.int8), f"Unsupported query dtype {self.query_dtype}"
        if self.query_dtype == torch.int8:
            raise NotImplementedError("Quantized query is not implemented")
        assert self.context_dtype == torch.int8, f"Quant attention support int8 context only, but got {self.context_dtype}"
        assert self.compute_dtype in (torch.bfloat16, torch.int8), f"Unsupported compute dtype {self.compute_dtype}"
        if self.compute_dtype == torch.int8:
            bits = 8
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -(2 ** (bits - 1))

    def forward(
        self,
        query: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        query_scale: Optional[torch.Tensor],  # [bsz, n_q_heads, 1]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        key_scale: torch.Tensor, # [n_kv_heads, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_scale: torch.Tensor, # [n_kv_heads, head_dim]
        total_seq_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Paged decode attention with Sliding-Window (SWA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D), it can be of dtype bf16 or int8
            query_scale (torch.Tensor): if query is quantized, it should be per-token scale of query shape (B, Hq, 1) and dtype bfloat16; if not, it should be None
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            key_scale (torch.Tensor): per-channel scale of key, shape (Hkv, D) and dtype bfloat16.
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D) and dtype int8.
            value_scale (torch.Tensor): per-channel scale of value, shape (Hkv, D) and dtype bfloat16.
            total_seq_lens (torch.Tensor): Per-batch KV lengths, shape (B,).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
        """
        # Note: for decode kernel, is_causal = False should never happen

        assert_paged_decode_contract(block_table, total_seq_lens)
        if self.query_dtype == torch.int8:
            assert query_scale is not None and query.dtype == self.query_dtype, "query_scale must be provided for quantized query"
        else:
            assert query_scale is None and query.dtype == self.query_dtype, "query_scale must be None for non-quantized query"
        
        bsz, n_q_heads, head_dim = query.shape
        _, n_kv_heads, page_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        if n_q_heads != n_kv_heads:
            if self.gqa_interleave:
                key_scale = key_scale.repeat((n_q_heads // n_kv_heads, 1)) # -> [n_q_heads, head_dim]
                value_scale = value_scale.repeat((n_q_heads // n_kv_heads, 1)) # -> [n_q_heads, head_dim]
            else:                
                key_scale = key_scale.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim]
                value_scale = value_scale.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim]
 

        o = torch.zeros_like(query)
        for i in range(bsz):
            q_i = query[i].unsqueeze(1) # -> [n_q_heads, 1, head_dim]
            kv_seq_len = total_seq_lens[i].item()
            if kv_seq_len == 0:
                # skip padded tokens
                continue
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = key_cache[block_table[i, :kv_blocks]]  # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]

            if self.compute_dtype == torch.int8:
                q_i_quant, q_i_scale = _dynamic_quantize(q_i * key_scale.unsqueeze(1), self.qmax, self.qmin, self.compute_dtype)
                s_i = torch.bmm(q_i_quant.float(), k_i_T.float()) * q_i_scale * softmax_scale  # -> [n_q_heads, 1, kv_seq_len]
            else:
                k_i_T = k_i_T.float() * key_scale.unsqueeze(-1).float()
                s_i = torch.bmm(q_i.float(), k_i_T.float()) * softmax_scale

            if self.is_causal:
                s_mask = _generate_window_mask(
                    1,
                    kv_seq_len,
                    self.local_window_size,
                    self.global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, 1, 1]
            s_i = s_i - m_i  # -> [n_q_heads, 1, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, 1, 1]

            v_i = value_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if self.gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            if self.compute_dtype == torch.int8:
                p_i_quant, p_i_scale = _dynamic_quantize(p_i, self.qmax, self.qmin, self.compute_dtype)
                o_i = torch.bmm(p_i_quant.float(), v_i.float()) * p_i_scale * value_scale.unsqueeze(1)  # -> [n_q_heads, 1, head_dim]
            else:
                v_i = v_i.float() * value_scale.unsqueeze(1).float()
                o_i = torch.bmm(p_i.float(), v_i.float()) # -> [n_q_heads, 1, head_dim]

            o_i = o_i / l_i
            o_i = o_i.squeeze(1)  # -> [n_q_heads, head_dim]
            o[i] = o_i.to(o.dtype)
        return o

    def extra_repr(self):
        return f"is_causal={self.is_causal}, gqa_layout={self.gqa_layout}, global_window_size={self.global_window_size}, local_window_size={self.local_window_size}, query_dtype={self.query_dtype}, context_dtype={self.context_dtype}, compute_dtype={self.compute_dtype}"

