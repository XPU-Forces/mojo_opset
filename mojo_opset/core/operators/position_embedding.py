from typing import Optional
from typing import Tuple, List

import torch

from ..operator import MojoOperator


class MojoRotaryEmbedding(MojoOperator):
    def __init__(self, rope_theta, rope_dim, attention_scaling: float = 1.0, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device = self.tensor_factory_kwargs.get("device")) / rope_dim)
        )
        self.attention_scaling = attention_scaling
        self.init_max_length = None
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if init_max_length is not None:
            self._rope_init(init_max_length)

        def load_state_dict_post_hook(module, incompatible_keys) -> None:
            key2ignore = []
            for miss in incompatible_keys.missing_keys:
                if miss.split('.')[-1] in ("inv_freq", "cos", "sin"):
                    key2ignore.append(miss)
            for key in key2ignore:
                incompatible_keys.missing_keys.remove(key)
        self.register_load_state_dict_post_hook(load_state_dict_post_hook)


    def _rope_init(self, max_length: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.init_max_length = max_length
        position_ids = torch.arange(max_length, device = self.tensor_factory_kwargs.get("device"))
        freqs = position_ids[..., None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos/sin for Rotary Position Embedding (RoPE).
        x is necessary for the kernel to determine the output shape.

        Scenario descriptions:
        1. Varlen prefill: input [T, H], cu_seqlens_q [T+1] or position_ids [T] -> cos/sin [T, D].
        2. Padded prefill: input [B, S, H], cu_seqlens_q None, position_ids None -> cos/sin [S, D].
        3. Decode: input [B, H], cu_seqlens_q None, position_ids [B] -> cos/sin [B, D].
        """
        if cu_seqlens_q is not None:
            assert cu_seqlens_q.dtype == torch.int32
        if seqlens_kv is not None:
            assert seqlens_kv.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        assert position_ids is None or cu_seqlens_q is None, "At most one of cu_seqlens_q or position_ids should be provided"

        if cu_seqlens_q is not None:
            assert x.dim() == 2, "x must be 2D: [T, D]"
            position_ids = torch.full((x.shape[0],), -1, device = x.device, dtype = torch.int32)
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            bsz = seqlens_q.size(0)
            for i in range(bsz):
                q_len = seqlens_q[i].item()
                context_len = 0 if seqlens_kv is None else seqlens_kv[i].item() - q_len
                position_ids[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = torch.arange(
                    context_len,
                    context_len + q_len, 
                    device = cu_seqlens_q.device,
                    dtype = torch.int32,
                )
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
        else:
            position_ids = torch.arange(x.shape[1], device = x.device, dtype = torch.int32)

        if self.init_max_length is None:
            freqs = position_ids[..., None] * self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        else:
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]
        
        return cos, sin


class MojoApplyRoPE(MojoOperator):

    def __init__(self, interleaved: bool = False):
        super().__init__()
        assert not interleaved, "interleaved impl is not supported yet."
        self.interleaved = interleaved

    def extra_repr(self) -> str:
        return f"{self.interleaved=}".replace("self.", "")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = cos.shape[-1]
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k = torch.split(k, [nope_dim, rope_dim], dim=-1)

        q_rot = (q * cos + self._rotate_half(q) * sin).to(q.dtype)
        k_rot = (k * cos + self._rotate_half(k) * sin).to(k.dtype)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embedding (RoPE).

        Scenario descriptions:
        1. Varlen prefill: q/k [T, N, D] or [N, T, D], cos/sin [T, d]
        2. Padded prefill: q/k [B, S, N, D] or [B, N, S, D], cos/sin [S, d] or [B, S, d]
        3. Decode: q/k [B, N, D] or [N, B, D], cos/sin [B, d]

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            unsqueeze_dim: Unsqueeze dimension for cos and sin for multi-heads

        Returns:
            (q_rot, k_rot) with same shape as input
        """
        assert q.ndim == k.ndim, "q and k must have the same dimension"
        assert q.ndim == 3 or q.ndim == 4, "q and k must be 3D or 4D"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        if q.ndim == 3:
            assert cos.ndim == 2, "rotary position embedding (cos/sin) must be of shape [num_tokens, rope_dim] for varlen prefill or decode"
        elif q.ndim == 4 and cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        if head_first:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        else:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        return self._apply_rope(q, k, cos, sin)


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass


class MojoGridRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply 3D grid rotary position embeddings (RoPE) over (F, H, W) axes using
        precomputed per-sample frequency tensors.

        Args:
            x (torch.Tensor): [B, L, N, D]; D must be even (paired into complex components).
            grid_sizes (torch.Tensor): [B, 3] per-sample (F, H, W); seq_len = F*H*W.
            freqs_list (List[torch.Tensor]): length-B list; each item is a complex unit-phase tensor
                of shape [seq_len, 1, D/2], broadcastable to [seq_len, N, D/2].

        Returns:
            torch.Tensor: Same shape as `x`. Per sample, the first F*H*W tokens are rotated;
                remaining padding tokens are preserved. Output dtype matches input.
        """
        assert x.dim() == 4, "x must be 4D: [B, L, N, D]"
        assert x.size(-1) % 2 == 0, "D must be even for complex pairing"
        assert grid_sizes.dim() == 2 and grid_sizes.size(1) == 3, "grid_sizes must be [B, 3]"

        n = x.size(2)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            freqs_i = freqs_list[i]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        y = torch.stack(output)
        return y.type_as(x)


class MojoMRoPE(MojoOperator):
    """Multimodal Rotary Position Embedding (MRoPE) for Qwen2-VL.

    Applies 3D rotary position embedding over temporal (T), height (H), and width (W)
    dimensions to query and key tensors. Supports both interleaved and non-interleaved modes.

    Reference: https://qwenlm.github.io/blog/qwen2-vl/
    """

    supported_platforms_list = ["npu"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        hidden_size = hidden_states.shape[-1]
        hidden_states_half = hidden_size // 2
        left = hidden_states[..., :hidden_states_half]
        right = hidden_states[..., hidden_states_half:]
        return torch.cat((-right, left), dim=-1)

    @staticmethod
    def _apply_interleaved_mrope(
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply interleaved MRoPE pattern to cos/sin tables."""
        cos_interleaved = cos_table[0].clone()
        cos_interleaved[..., 1::3] = cos_table[1, ..., 1::3]
        cos_interleaved[..., 2::3] = cos_table[2, ..., 2::3]

        sin_interleaved = sin_table[0].clone()
        sin_interleaved[..., 1::3] = sin_table[1, ..., 1::3]
        sin_interleaved[..., 2::3] = sin_table[2, ..., 2::3]

        return cos_interleaved, sin_interleaved

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Multimodal Rotary Position Embedding to query and key tensors.

        Args:
            query: ``(num_tokens, n_qh * head_dim)`` query tensor.
            key: ``(num_tokens, n_kh * head_dim)`` key tensor.
            cos_table: ``(3, num_tokens, rotary_dim // 2)`` cos values for T/H/W dimensions.
            sin_table: ``(3, num_tokens, rotary_dim // 2)`` sin values for T/H/W dimensions.
            mrope_section: ``[t_section, h_section, w_section]`` - how half rope_dim is split.
            is_interleaved: if True, T/H/W positions are interleaved.
            head_dim: head dimension. If None, inferred from cos_table.

        Returns:
            ``(query, key)`` with RoPE applied, same shape as input.
        """
        num_tokens, n_qh_head_dim = query.shape
        num_tokens_k, n_kh_head_dim = key.shape

        rope_dim = sum(mrope_section) * 2
        half_rope_dim = rope_dim // 2

        if head_dim is None:
            head_dim = cos_table.shape[-1] * 2
            if head_dim != rope_dim:
                raise ValueError(
                    f"head_dim ({head_dim}) inferred from cos_table does not match "
                    f"rope_dim ({rope_dim}). Please pass head_dim explicitly."
                )

        n_qh = n_qh_head_dim // head_dim
        n_kh = n_kh_head_dim // head_dim

        query = query.view(num_tokens, n_qh, head_dim)
        key = key.view(num_tokens_k, n_kh, head_dim)

        query_rot, query_pass = query.split([rope_dim, head_dim - rope_dim], dim=-1)
        key_rot, key_pass = key.split([rope_dim, head_dim - rope_dim], dim=-1)

        if cos_table.dim() == 3:
            if is_interleaved:
                cos_table, sin_table = self._apply_interleaved_mrope(cos_table, sin_table, mrope_section)
            else:
                cos_table = torch.cat([m[i] for i, m in enumerate(cos_table.split(mrope_section, dim=-1))], dim=-1)
                sin_table = torch.cat([m[i] for i, m in enumerate(sin_table.split(mrope_section, dim=-1))], dim=-1)

        cos_table = cos_table.view(num_tokens, half_rope_dim)
        sin_table = sin_table.view(num_tokens, half_rope_dim)

        query_rot_half1 = query_rot[..., :half_rope_dim]
        query_rot_half2 = query_rot[..., half_rope_dim:]
        key_rot_half1 = key_rot[..., :half_rope_dim]
        key_rot_half2 = key_rot[..., half_rope_dim:]

        cos_expanded = cos_table.unsqueeze(1)
        sin_expanded = sin_table.unsqueeze(1)

        query_rot_new_half1 = query_rot_half1 * cos_expanded - query_rot_half2 * sin_expanded
        query_rot_new_half2 = query_rot_half2 * cos_expanded + query_rot_half1 * sin_expanded
        key_rot_new_half1 = key_rot_half1 * cos_expanded - key_rot_half2 * sin_expanded
        key_rot_new_half2 = key_rot_half2 * cos_expanded + key_rot_half1 * sin_expanded

        query_rot = torch.cat([query_rot_new_half1, query_rot_new_half2], dim=-1)
        key_rot = torch.cat([key_rot_new_half1, key_rot_new_half2], dim=-1)

        query = torch.cat([query_rot, query_pass], dim=-1).view(num_tokens, -1)
        key = torch.cat([key_rot, key_pass], dim=-1).view(num_tokens_k, -1)

        return query, key
