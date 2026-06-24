import torch
from torch import nn

from ...core.operator import MojoOperator


class MojoFusedAttnOutputGate(MojoOperator):
    """Fused gated attention output for a single attention path.

    Accepts a raw attention output (3D [T, N, D] or 2D [T, N*D]), reshapes
    internally, and applies a per-head sigmoid gate driven by ``hidden_states``.

    Computation (head mode, fp32 internal):
        gate = sigmoid(hidden_states @ gate_weight.T)
        output = (attn_output.view(T, N, D) * gate.unsqueeze(-1)).view(T, N * D)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert num_heads > 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.gate_weight = nn.Parameter(
            torch.empty(num_heads, hidden_size, **self.tensor_factory_kwargs)
        )
        if bias:
            self.gate_bias = nn.Parameter(
                torch.empty(num_heads, **self.tensor_factory_kwargs)
            )
        else:
            self.register_parameter("gate_bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [T, hidden_size] — gate input (pre-attn residual).
            attn_output:   [T, N, D] or [T, N * D] — attention output.

        Returns:
            [T, N * D], same dtype as hidden_states.
        """
        T = hidden_states.shape[0]
        attn_output = attn_output.view(T, self.num_heads, self.head_dim)

        gate = torch.matmul(hidden_states.float(), self.gate_weight.t().float())
        if self.gate_bias is not None:
            gate = gate + self.gate_bias.float()
        gate = torch.sigmoid(gate)

        gated = attn_output.float() * gate.unsqueeze(-1)
        return gated.view(T, self.num_heads * self.head_dim).to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"bias={self.gate_bias is not None}"
        )


class MojoFusedConcatAttnOutputGate(MojoOperator):
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

        self.register_buffer(
                "_cached_weight",
                None,
                persistent=False,
            )

        self.register_buffer(
                "_cached_bias",
                None,
                persistent=False,
            )

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
