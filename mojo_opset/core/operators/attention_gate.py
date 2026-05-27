import torch
from torch import nn

from ..operator import MojoOperator


class MojoAttnOutputGate(MojoOperator):
    """Gated attention output: sigmoid(hidden @ W^T [+ bias]) * attn_output.

    Supports two gating granularities:
      - "head":    one scalar gate per attention head, broadcast over head_dim
      - "element": one gate value per element (num_heads * head_dim)

    The gate projection and sigmoid are computed in fp32 for numerical
    stability (matching the original M13 modeling logic).  The final output
    is cast back to the input dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        method: str = "head",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if method not in ("head", "element"):
            raise ValueError(f"method must be 'head' or 'element', got '{method}'")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.method = method

        out_features = num_heads if method == "head" else num_heads * head_dim
        self.weight = nn.Parameter(
            torch.empty(out_features, hidden_size, **self.tensor_factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **self.tensor_factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, hidden_size] — gate input (typically pre-attn residual).
            attn_output:   [seq_len, num_heads * head_dim] — attention output to be gated.

        Returns:
            [seq_len, num_heads * head_dim], same dtype as hidden_states.
        """
        gate = torch.matmul(hidden_states.float(), self.weight.t().float())
        if self.bias is not None:
            gate = gate + self.bias.float()
        gate = torch.sigmoid(gate)

        if self.method == "head":
            # gate: [seq, num_heads] → [seq, num_heads, 1]
            # attn_output: [seq, num_heads * head_dim] → [seq, num_heads, head_dim]
            gated = attn_output.float().view(-1, self.num_heads, self.head_dim) * gate.unsqueeze(-1)
            return gated.view(-1, self.num_heads * self.head_dim).to(hidden_states.dtype)
        else:
            return (attn_output.float() * gate).to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, method='{self.method}', "
            f"bias={self.bias is not None}"
        )
