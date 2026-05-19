import torch
import torch.nn.functional as F

from mojo_opset.core.operator import MojoOperator


class MojoGroupLayerNorm(MojoOperator):
    def __init__(self, num_groups, norm_size, eps, elementwise_affine=True, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.norm_size = norm_size
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty((num_groups, norm_size), **self.tensor_factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty((num_groups, norm_size), **self.tensor_factory_kwargs))
        else:
            self.weight = None
            self.bias = None
        self.variance_epsilon = eps

    def forward(self, input_groups):
        # Note: input_groups is a list of tensors, each tensor has compatible shapes for norm
        output_groups = []
        for group_id in range(self.num_groups):
            output_groups.append(F.layer_norm(input_groups[group_id], (self.norm_size,), weight=self.weight[group_id], bias=self.bias[group_id], eps=self.variance_epsilon))
        return output_groups

    def extra_repr(self) -> str:
        return f"{self.num_groups=}, {self.norm_size=}, {self.variance_epsilon=} {self.elementwise_affine=}".replace("self.", "")


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
        self.norm_size = norm_size
        self.images = images
        self.has_bias = bias
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

    def extra_repr(self) -> str:
        return f"{self.norm_size=}, {self.channel_first=}, {self.images=}, {self.has_bias=}, {self.scale=}".replace(
            "self.", ""
        )


__all__ = [
    "MojoGroupLayerNorm",
    "MojoChannelRMSNorm",
]
