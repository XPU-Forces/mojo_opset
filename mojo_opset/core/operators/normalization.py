import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoLayerNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Initialize LayerNorm patch parameters.

        Args:
            norm_size (int): Size of 1-D affine scale and shift vector.
            eps (float, default=1e-5): Epsilon added to the variance for numerical stability; must be > 0.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.
        """
        super().__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm over the last dimension of the input.

        Args:
            hidden_state (torch.Tensor): Input tensor whose last dimension is the hidden size
                (e.g., shape (B, T, D) or (..., D)). The normalization is performed across D.

        Returns:
            torch.Tensor: Tensor of the same shape and dtype as `hidden_state`, normalized
                over the last dimension.
        """
        return F.layer_norm(
            hidden_state,
            [hidden_state.shape[-1]],
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )


class MojoRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Initialize RMSNorm patch parameters.

        Args:
            norm_size (int): Size of 1-D affine scale vector.
            eps (float, default=1e-5): Epsilon added for numerical stability; must be > 0.\
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.
        """
        super().__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm over the last dimension of the input.

        Args:
            hidden_state (torch.Tensor): Input tensor whose last dimension is the hidden size
                (e.g., shape (B, T, D) or (..., D)). The normalization is performed across D.

        Returns:
            torch.Tensor: Tensor of the same shape and dtype as `hidden_state`, normalized
            over the last dimension.
        """
        return F.rms_norm(
            hidden_state,
            [hidden_state.shape[-1]],
            weight=self.weight,
            eps=self.variance_epsilon,
        )


class MojoNormQuant(MojoOperator):
    pass


class MojoResidualAddRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "pre",
        **kwargs,
    ):
        """
        Initialize residual-add RMSNorm operator with position control.

        Args:
            norm_size (int): Size of  1-D affine scale of length D (hidden size).
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = rms_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = rms_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__(**kwargs)
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.variance_epsilon = float(eps)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))

        self.norm_pos = norm_pos

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            hidden_state = F.rms_norm(
                residual,
                (residual.size(-1),),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            hidden_state = F.rms_norm(
                hidden_state,
                (hidden_state.size(-1),),
                weight=self.weight,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        return hidden_state, residual


class MojoResidualAddLayerNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "pre",
        **kwargs,
    ):
        """
        Initialize residual-add LayerNorm operator with position control.

        Args:
            norm_size (int): Size of 1-D affine scale and shift vector.
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = layer_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = layer_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__(**kwargs)
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.variance_epsilon = float(eps)
        self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        self.norm_pos = norm_pos
        self.affine = self.weight is not None and self.bias is not None

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Residual-add LayerNorm with configurable position ("pre"/"post").

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (..., D), normalized over the last dim D.
            residual (torch.Tensor): Residual tensor to add; must be provided and shape-compatible.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized `hidden_state` and updated `residual`.
        """
        if self.norm_pos == "pre":
            residual = hidden_state + residual
            hidden_state = F.layer_norm(
                residual,
                [residual.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
        else:
            hidden_state = hidden_state + residual
            hidden_state = F.layer_norm(
                hidden_state,
                [hidden_state.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.variance_epsilon,
            )
            residual = hidden_state

        return hidden_state, residual


class MojoResidualAddNormQuant(MojoOperator):
    pass


class MojoResidualAddNormCast(MojoOperator):
    pass
