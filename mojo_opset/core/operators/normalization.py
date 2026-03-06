import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoLayerNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        """
        Initialize LayerNorm patch parameters.

        Args:
            norm_size (int): Size of 1-D affine scale and shift vector.
            eps (float, default=1e-5): Epsilon added to the variance for numerical stability; must be > 0.
            elementwise_affine (bool, default=True): Whether to apply elementwise affine transform.
            **kwargs: The keyword arguments of torch.empty, such as device, dtype and so on to create the weight and bias.
        """
        super().__init__(**kwargs)
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(norm_size, **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None
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


class MojoChannelRMSNorm(MojoOperator):
    def __init__(
        self,
        norm_size: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        """
        Initialize channel-wise RMS-like normalization operator.
        
        Args:
            norm_size (int): Number of channels to normalize over.
            channel_first (bool, default=True): If True, treat input as channel-first (e.g., NCHW/NCTHW).
            images (bool, default=True): Controls broadcast shape of parameters:
                - True  -> parameters shaped as (C, 1, 1) for 2D/broadcast to 3D
                - False -> parameters shaped as (C, 1, 1, 1) for explicit time dimension
            bias (bool, default=False): Whether to include learnable bias.
            **kwargs: Additional tensor factory kwargs (device, dtype, etc.).
        """
        super().__init__(**kwargs)
        b_dims = (1, 1) if images else (1, 1, 1)
        shape = (norm_size, *b_dims) if channel_first else (norm_size,)
        self.scale = norm_size**0.5
        self.weight = torch.nn.Parameter(torch.ones(shape, **self.tensor_factory_kwargs))
        self.bias = torch.nn.Parameter(torch.zeros(shape, **self.tensor_factory_kwargs)) if bias else None
        self.channel_first = channel_first

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise normalization by:
          1) L2-normalization along the channel axis (or last axis if channel_first=False)
          2) Scaling by sqrt(norm_size) to match RMS normalization semantics
          3) Applying affine transform with `weight` (and optional `bias`)
        
        Args:
            hidden_state (torch.Tensor): Input must include a channel dimension. Shapes must match constructor:
                - channel_first=True,  images=True  -> (N, C, H, W)
                - channel_first=True,  images=False -> (N, C, T, H, W)
                - channel_first=False, images=True  -> (N, H, W, C)
                - channel_first=False, images=False -> (N, T, H, W, C)
                Here, N is batch size, C is channels, and T/H/W are time/height/width. Normalization is applied along the channel dimension.
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape and dtype as `hidden_state`.
        """
        dim = 1 if self.channel_first else -1
        hidden_state = torch.nn.functional.normalize(hidden_state, dim=dim) * self.scale
        hidden_state = hidden_state * self.weight
        if self.bias is not None:
            hidden_state = hidden_state + self.bias
        return hidden_state


class MojoResidualAddNormQuant(MojoOperator):
    pass


class MojoResidualAddNormCast(MojoOperator):
    pass
