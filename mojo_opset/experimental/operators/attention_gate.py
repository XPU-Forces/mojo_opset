import torch
from torch import nn

from ...core.operator import MojoOperator


class MojoFusedAttnOutputGate(MojoOperator):
    """Fused gated attention output for dual-path (full + SWA) attention.

    Holds two separate weight parameters (one per attention path) for
    checkpoint compatibility, but internally concatenates them to execute
    the gate computation in a single GEMM + sigmoid + broadcast-multiply pass.

    Accepts raw attention outputs (3D [T, N, D] or 2D [T, N*D]) from each
    path, performs reshape and concatenation internally.

    Computation (head mode, fp32 internal):
        cat_weight = cat([full_gate_weight, swa_gate_weight], dim=0)
        gate = sigmoid(hidden_states @ cat_weight.T)
        attn_cat = cat([full_attn.view(T, N_full, D), swa_attn.view(T, N_swa, D)], dim=1)
        output = (attn_cat * gate.unsqueeze(-1)).view(T, N_total * D)

    The concatenated weight is cached after the first forward call and
    invalidated on subsequent parameter updates (e.g., load_state_dict).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads_full: int,
        num_heads_swa: int,
        head_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert num_heads_full > 0 and num_heads_swa > 0
        self.hidden_size = hidden_size
        self.num_heads_full = num_heads_full
        self.num_heads_swa = num_heads_swa
        self.num_heads_total = num_heads_full + num_heads_swa
        self.head_dim = head_dim

        self.full_gate_weight = nn.Parameter(
            torch.empty(num_heads_full, hidden_size, **self.tensor_factory_kwargs)
        )
        self.swa_gate_weight = nn.Parameter(
            torch.empty(num_heads_swa, hidden_size, **self.tensor_factory_kwargs)
        )
        if bias:
            self.full_gate_bias = nn.Parameter(
                torch.empty(num_heads_full, **self.tensor_factory_kwargs)
            )
            self.swa_gate_bias = nn.Parameter(
                torch.empty(num_heads_swa, **self.tensor_factory_kwargs)
            )
        else:
            self.register_parameter("full_gate_bias", None)
            self.register_parameter("swa_gate_bias", None)

        self._cached_weight: torch.Tensor | None = None
        self._cached_bias: torch.Tensor | None = None

    def _get_fused_weight(self) -> torch.Tensor:
        if self._cached_weight is None:
            self._cached_weight = torch.cat(
                [self.full_gate_weight, self.swa_gate_weight], dim=0
            )
        return self._cached_weight

    def _get_fused_bias(self) -> torch.Tensor | None:
        if self.full_gate_bias is None:
            return None
        if self._cached_bias is None:
            self._cached_bias = torch.cat(
                [self.full_gate_bias, self.swa_gate_bias], dim=0
            )
        return self._cached_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_output: torch.Tensor,
        swa_attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:    [T, hidden_size] — gate input (pre-attn residual).
            full_attn_output: [T, N_full, D] or [T, N_full * D] — full attention output.
            swa_attn_output:  [T, N_swa, D] or [T, N_swa * D] — SWA attention output.

        Returns:
            [T, (N_full + N_swa) * D], same dtype as hidden_states.
        """
        T = hidden_states.shape[0]
        full_attn_output = full_attn_output.view(T, self.num_heads_full, self.head_dim)
        swa_attn_output = swa_attn_output.view(T, self.num_heads_swa, self.head_dim)

        weight = self._get_fused_weight()
        gate = torch.matmul(hidden_states.float(), weight.t().float())
        bias = self._get_fused_bias()
        if bias is not None:
            gate = gate + bias.float()
        gate = torch.sigmoid(gate)

        attn_cat = torch.cat([full_attn_output, swa_attn_output], dim=1)
        gated = attn_cat.float() * gate.unsqueeze(-1)
        return gated.view(T, self.num_heads_total * self.head_dim).to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads_full={self.num_heads_full}, "
            f"num_heads_swa={self.num_heads_swa}, "
            f"head_dim={self.head_dim}, "
            f"bias={self.full_gate_bias is not None}"
        )


class MojoFusedAttnGateConcat(MojoOperator):
    """Apply full/SWA attention gates and concatenate the gated outputs."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        full_attn_out: torch.Tensor,
        full_attn_gate_score: torch.Tensor,
        swa_attn_out: torch.Tensor,
        swa_attn_gate_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            full_attn_out: Full attention output with shape
                ``[total_seq, full_head_num, head_dim]``.
            full_attn_gate_score: Gate score for full attention with shape
                ``[total_seq, full_head_num]``.
            swa_attn_out: SWA attention output with shape
                ``[total_seq, swa_head_num, head_dim]``.
            swa_attn_gate_score: Gate score for SWA attention with shape
                ``[total_seq, swa_head_num]``.

        Returns:
            Gated concatenated attention output with shape
            ``[total_seq, full_head_num + swa_head_num, head_dim]`` and the
            same dtype as ``full_attn_out``.
        """
        if full_attn_gate_score is None:
            raise ValueError("full_attn_gate_score is required.")
        if swa_attn_gate_score is None:
            raise ValueError("swa_attn_gate_score is required.")
        if full_attn_out.dim() != 3:
            raise ValueError(f"full_attn_out must be 3D, got {tuple(full_attn_out.shape)}.")
        if swa_attn_out.dim() != 3:
            raise ValueError(f"swa_attn_out must be 3D, got {tuple(swa_attn_out.shape)}.")
        if full_attn_gate_score.dim() != 2:
            raise ValueError(f"full_attn_gate_score must be 2D, got {tuple(full_attn_gate_score.shape)}.")
        if swa_attn_gate_score.dim() != 2:
            raise ValueError(f"swa_attn_gate_score must be 2D, got {tuple(swa_attn_gate_score.shape)}.")

        total_seq, full_head_num, head_dim = full_attn_out.shape
        swa_total_seq, swa_head_num, swa_head_dim = swa_attn_out.shape
        if swa_total_seq != total_seq or swa_head_dim != head_dim:
            raise ValueError(
                "full_attn_out and swa_attn_out must have matching total_seq and head_dim, "
                f"got {tuple(full_attn_out.shape)} and {tuple(swa_attn_out.shape)}."
            )
        if full_attn_gate_score.shape != (total_seq, full_head_num):
            raise ValueError(
                f"full_attn_gate_score must have shape [{total_seq}, {full_head_num}], got {tuple(full_attn_gate_score.shape)}."
            )
        if swa_attn_gate_score.shape != (total_seq, swa_head_num):
            raise ValueError(
                f"swa_attn_gate_score must have shape [{total_seq}, {swa_head_num}], got {tuple(swa_attn_gate_score.shape)}."
            )

        full_out = full_attn_out.float() * torch.sigmoid(full_attn_gate_score.float()).unsqueeze(-1)
        swa_out = swa_attn_out.float() * torch.sigmoid(swa_attn_gate_score.float()).unsqueeze(-1)
        return torch.cat((full_out, swa_out), dim=1).to(full_attn_out.dtype)
