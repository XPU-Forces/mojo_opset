import math

from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator
from .quantize import MojoDynamicQuant


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

def get_scale_and_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor],
    quant_dims,
    q_max: int = 127,
    q_min: int = -128,
    eps: float = 1e-6,
    quant_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integer quantization helper used by the sage-style ``per_*_int8`` wrappers.

    Computes (or reuses) a per-group scale by reducing ``x.abs()`` along
    ``quant_dims`` (with ``keepdim=True``), then quantizes ``x`` to
    ``quant_dtype`` via ``round(x / scale)`` and clamps into ``[q_min, q_max]``.

    Args:
        x (torch.Tensor): Input tensor in any floating-point dtype.
        scale (Optional[torch.Tensor]): Pre-computed scale.
            - ``None``: compute a dynamic scale from ``x``;
            - keepdim layout (``scale.ndim == x.ndim``): used as-is;
            - non-keepdim layout: missing reduced dims are restored via ``unsqueeze``
              so the result is broadcastable to ``x``.
        quant_dims (int | Tuple[int, ...] | List[int]): Axis (or axes) over which
            the absolute-max is reduced. Negative indices are accepted.
        q_max (int): Positive saturation bound, default ``127`` (int8).
        q_min (int): Negative saturation bound, default ``-128`` (int8).
        eps (float): Lower clamp on the dynamic scale to avoid division-by-zero.
        quant_dtype (torch.dtype): Target integer dtype, default ``torch.int8``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``q``: integer tensor with the same shape as ``x`` and dtype
              ``quant_dtype``.
            - ``scale``: float tensor broadcastable to ``x`` (keepdim layout).
    """
    assert scale is None or isinstance(scale, torch.Tensor), \
        f"scale must be None or Tensor, got {type(scale)}"

    if scale is None:
        scale = x.abs().amax(dim=quant_dims, keepdim=True).float() / q_max
        scale = scale.clamp(min=eps)
    elif scale.ndim != x.ndim:
        # Caller passed a non-keepdim scale; restore the reduced dims.
        dims = (quant_dims,) if isinstance(quant_dims, int) else tuple(quant_dims)
        dims = sorted({d % x.ndim for d in dims})
        for d in dims:
            scale = scale.unsqueeze(d)

    q = torch.clamp(torch.round(x.float() / scale), q_min, q_max).to(quant_dtype)
    return q, scale


def per_block_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    blk: int = 16,
    seq_dim: int = -2,
    q_max: int = 127,
    q_min: int = -128,
):
    """Per-block int8 quantization along the sequence axis.

    Splits ``x`` along ``seq_dim`` into contiguous chunks of size ``blk`` and
    computes one scale per (chunk, ...remaining dims) by reducing the abs-max
    inside each chunk. Typical usage in sage-style attention:

    - For the K cache of shape ``[T, H, D]`` and ``blk=128``, this yields one
      scale per (128-token block, H, D), i.e. scale shape ``[T//blk, 1, H, D]``
      after broadcasting back to the original layout.
    - For paged KV of shape ``[N_blocks, H, block_size, D]``, set
      ``seq_dim=-2`` (the default) and ``blk=block_size`` to obtain one scale
      per (logical block, H, D).

    Args:
        x (torch.Tensor): Input tensor whose dimension at ``seq_dim`` is
            divisible by ``blk``.
        xm (Optional[torch.Tensor]): Optional zero-point / mean tensor that is
            subtracted from ``x`` before quantization (broadcastable to ``x``).
        scale (Optional[torch.Tensor]): Pre-computed scale; same conventions
            as in :func:`get_scale_and_quant`. ``None`` triggers dynamic
            per-block scale computation.
        blk (int): Block size along ``seq_dim``. Default ``16``.
        seq_dim (int): Axis to split into blocks. Default ``-2``.
        q_max (int): Positive saturation bound, default ``127``.
        q_min (int): Negative saturation bound, default ``-128``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``x_q``: int8 tensor with the same shape as ``x``.
            - ``x_scale``: float scale tensor broadcastable to ``x`` (keepdim
              layout, with the block axis kept at length ``S // blk``).
    """
    if xm is not None:
        x = x - xm

    seq_dim_norm = seq_dim % x.ndim
    seq_len = x.shape[seq_dim_norm]
    assert seq_len % blk == 0, (
        f"per_block_int8: dim {seq_dim_norm} (size {seq_len}) must be divisible by blk={blk}"
    )

    # Reshape seq_dim from S into (S // blk, blk) so the inner ``blk`` axis
    # can be reduced independently.
    new_shape = (
        list(x.shape[:seq_dim_norm])
        + [seq_len // blk, blk]
        + list(x.shape[seq_dim_norm + 1:])
    )
    x_reshaped = x.reshape(new_shape)

    # The reduction axis is the inner ``blk`` axis we just inserted.
    q, scale_kd = get_scale_and_quant(
        x_reshaped,
        scale,
        quant_dims=seq_dim_norm + 1,
        q_max=q_max,
        q_min=q_min,
    )

    # Restore original layout for q; collapse the inner ``blk`` axis (which is
    # length 1 after reduction) so scale broadcasts back to the original shape.
    q = q.reshape(x.shape)
    scale_out = scale_kd.squeeze(seq_dim_norm + 1)
    return q, scale_out


def per_token_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    q_max: int = 127,
    q_min: int = -127,
):
    """Per-token int8 quantization (one scale per row of the last dim).

    Reduces ``abs(x)`` along the last dimension to produce one scale per
    "token row". Works for any tensor whose last axis is the feature/head-dim
    axis; e.g. ``x`` of shape ``[T, H, D]`` produces a scale of shape
    ``[T, H, 1]`` broadcastable back to ``x``.

    Args:
        x (torch.Tensor): Input tensor of any shape ending in ``D``.
        xm (Optional[torch.Tensor]): Optional zero-point / mean tensor that is
            subtracted from ``x`` before quantization (broadcastable to ``x``).
        scale (Optional[torch.Tensor]): Pre-computed scale; ``None`` triggers
            dynamic per-token scale computation. See
            :func:`get_scale_and_quant` for accepted layouts.
        q_max (int): Positive saturation bound, default ``127``.
        q_min (int): Negative saturation bound, default ``-127``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``x_q``: int8 tensor with the same shape as ``x``.
            - ``x_scale``: float scale tensor with the last dim collapsed to
              ``1`` (keepdim layout), easily to broadcastable. e.g. [T, H, 1].
    """
    if xm is not None:
        x = x - xm
    return get_scale_and_quant(x, scale, quant_dims=-1, q_max=q_max, q_min=q_min)


def per_channel_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    seq_dim: int = -2,
    q_max: int = 127,
    q_min: int = -127,
):
    """Per-channel int8 quantization (one scale per (head, dim) channel).

    Reduces ``abs(x)`` along ``seq_dim`` (the sequence/token axis) to produce
    one scale per non-seq position. For example, ``x`` of shape ``[T, H, D]``
    with the default ``seq_dim=-2`` (which is ``T`` here once contiguous use
    is mapped accordingly — see notes below) yields a scale broadcastable to
    ``[1, H, D]``.

    Args:
        x (torch.Tensor): Input tensor with an explicit sequence axis.
        xm (Optional[torch.Tensor]): Optional zero-point / mean tensor that is
            subtracted from ``x`` before quantization (broadcastable to ``x``).
        scale (Optional[torch.Tensor]): Pre-computed scale; ``None`` triggers
            dynamic per-channel scale computation. See
            :func:`get_scale_and_quant` for accepted layouts.
        seq_dim (int): Axis to reduce over (the sequence/token axis).
            Default ``-2``.
        q_max (int): Positive saturation bound, default ``127``.
        q_min (int): Negative saturation bound, default ``-127``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``x_q``: int8 tensor with the same shape as ``x``.
            - ``x_scale``: float scale tensor with ``seq_dim`` collapsed to
              ``1`` (keepdim layout), broadcastable to ``x``.

    Notes:
        Callers are expected to pass ``seq_dim`` matching their tensor layout;
        e.g. for ``[T, H, D]`` with the seq axis at position ``0`` use
        ``seq_dim=0``, while for ``[B, H, S, D]`` the default ``-2`` already
        points at ``S``.
    """
    if xm is not None:
        x = x - xm
    return get_scale_and_quant(x, scale, quant_dims=seq_dim, q_max=q_max, q_min=q_min)


class MojoPagedPrefillSageGQA(MojoOperator):
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
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - query_dtype (torch.dtype): the dtype for query, default torch.bfloat16 for non-quantized query.
        - context_dtype (torch.dtype): The stored KV-cache dtype before dequantization, default torch.int8.
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
        assert self.query_dtype == torch.int8, f"Unsupported query dtype {self.query_dtype}"
        assert self.context_dtype == torch.int8, f"Quant attention support int8 context only, but got {self.context_dtype}"
        assert self.compute_dtype == torch.int8, f"Unsupported compute dtype {self.compute_dtype}"
        if self.compute_dtype == torch.int8:
            bits = 8
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -(2 ** (bits - 1))

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill sage attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i; `cu_q_lens[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            query_scale (torch.Tensor): if using per token dynamic quant it should be None, otherwise its shape is (T, Hq) with dtype fp32.
            key_scale (torch.Tensor): Key scale for quant, should be None if using per token dynamic quant, otherwise shape is (N_blocks, Hkv, block_size) with dtype fp32.
            value_scale (torch.Tensor): Value scale for quant, should be None if using per channel dynamic quant, otherwise shape is [Hkv, D] with dtype fp32.
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

        # apply per_token_int8 quant, quant_dim = -1
        assert query_scale is None, "dynamic Q quant, query_scale should be None"
        assert key_scale is None, "dynamic K quant, key_scale should be None"
        # assert value_scale is not None, "static V quant, value should not be None"
        query, query_scale = per_token_int8(x=query, scale=query_scale, q_max=self.q_max, q_min=self.q_min)
        key_cache, key_scale = per_token_int8(x=key_cache, scale=key_scale, q_max=self.q_max, q_min=self.q_min)

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
            # k_scale_unpadded: [kv_seq_len, num_kv_heads, 1]
            k_scale_unpadded = torch.zeros(kv_seq_len, num_kv_heads, 1, dtype=key_scale.dtype, device=key_scale.device)
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

                k_scale_slice = key_scale[physical_block_id, :, :tokens_in_block, :]
                k_scale_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_scale_slice.permute(1, 0, 2)

                v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
                v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

            if num_q_heads != num_kv_heads:
                if self.gqa_layout == "AABB":
                    k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                    k_scale_expanded = k_scale_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                    v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                else:
                    k_expanded = k_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
                    k_scale_expanded = k_scale_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
                    v_expanded = v_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
            else:
                k_expanded = k_unpadded
                k_scale_expanded = k_scale_unpadded
                v_expanded = v_unpadded

            q_scale = query_scale[start_loc:end_loc]
            # npu don't support int8
            q, k_expanded = q.float(), k_expanded.float()
            attn_scores = torch.einsum("thd,khd->thk", q, k_expanded).float() * softmax_scale * q_scale * k_scale_expanded
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
            attn_probs_int8, attn_probs_scale = per_token_int8(x=attn_probs, q_max=self.q_max, q_min=self.q_min)
            v_expanded_int8, v_expanded_scale = per_channel_int8(x=v_expanded, seq_dim=-2, q_max=self.q_max, q_min=self.q_min)
            attn_probs_int8, v_expanded_int8 = attn_probs_int8.float(), v_expanded_int8.float()
            outputs[start_loc:end_loc] = torch.einsum("thk,khd->thd", attn_probs_int8, v_expanded_int8) * attn_probs_scale * v_expanded_scale
        return outputs

    def extra_repr(self) -> str:
        return f"{self.is_causal=}, {self.gqa_layout=}, {self.query_dtype=}, {self.context_dtype=}, {self.compute_dtype=}, {self.q_max=}, {self.q_min=}".replace("self.", "")
