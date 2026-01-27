import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoLayerNorm(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        """
        Initialize LayerNorm patch parameters.

        Args:
            weight (torch.Tensor): 1-D affine scale vector of length D (hidden size).
            bias (torch.Tensor): 1-D affine shift vector of length D (hidden size).
            eps (float, default=1e-5): Epsilon added to the variance for numerical stability; must be > 0.

        Notes:
            - `weight` and `bias` must be 1-D and match the last dimension D of the input; mismatches
              will lead to errors when applying LayerNorm.
        """
        super().__init__()
        self.weight = weight
        self.bias = bias
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
        weight: torch.Tensor,
        eps: float = 1e-5,
    ):
        """
        Initialize RMSNorm patch parameters.

        Args:
            weight (torch.Tensor): 1-D affine scale vector of length D (hidden size).
            eps (float, default=1e-5): Epsilon added for numerical stability; must be > 0.

        Notes:
            - `weight` must be 1-D and match the last dimension D of the input; mismatches
              will lead to errors when applying RMSNorm.
        """
        super().__init__()
        self.weight = weight
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
        weight: torch.Tensor,
        eps: float = 1e-05,
        norm_pos: str = "pre",
    ):
        """
        Initialize residual-add RMSNorm operator with position control.

        Args:
            weight (torch.Tensor): Required 1-D affine scale of length D (hidden size).
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = rms_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = rms_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__()
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.variance_epsilon = float(eps)
        self.weight = weight
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
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-05,
        norm_pos: str = "pre",
    ):
        """
        Initialize residual-add LayerNorm operator with position control.

        Args:
            weight (torch.Tensor): Required 1-D affine scale vector of length D (hidden size).
            bias (torch.Tensor): Required 1-D affine shift vector of length D (hidden size).
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.

        Behavior:
            - norm_pos="pre": residual = hidden_state + residual; hidden_state = layer_norm(residual).
            - norm_pos="post": hidden_state = hidden_state + residual; hidden_state = layer_norm(hidden_state);
              residual = hidden_state.
        """
        super().__init__()
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.variance_epsilon = float(eps)
        self.weight = weight
        self.bias = bias
        self.norm_pos = norm_pos
        self.affine = weight is not None and bias is not None

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
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        eps: float = 1e-05,
        norm_type: str = "layernorm",
        norm_pos: str = "pre",
        cast_type: str = "int8",
    ):
        """
        Initialize residual-add normalization with type casting operator.

        Args:
            weight (torch.Tensor): Required 1-D affine scale vector of length D (hidden size).
            bias (torch.Tensor, optional): 1-D affine shift vector of length D (hidden size). 
                Required for LayerNorm, optional for RMSNorm.
            eps (float, default=1e-05): Epsilon for numerical stability; must be > 0.
            norm_type (str, default="layernorm"): Normalization type, one of {"layernorm", "rmsnorm"}.
            norm_pos (str, default="pre"): Normalization placement; one of {"pre", "post"}.
            cast_type (str, default="int8"): Type to cast the normalized output to,
                one of {"int8", "float16", "bfloat16", "float32"}.

        Behavior:
            - norm_pos="pre": 
                1. residual = hidden_state + residual
                2. hidden_state = norm(residual)  # norm in input dtype
                3. cast_out = hidden_state.to(cast_type)  # cast after norm as separate output
            - norm_pos="post": 
                1. hidden_state = hidden_state + residual
                2. hidden_state = norm(hidden_state)  # norm in input dtype
                3. cast_out = hidden_state.to(cast_type)  # cast after norm as separate output
                4. residual = hidden_state  # residual gets the un-casted output
        """
        super().__init__()
        
        if norm_type not in ["layernorm", "rmsnorm"]:
            raise ValueError("norm_type should be 'layernorm' or 'rmsnorm'")
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")
        if cast_type not in ["int8", "float16", "bfloat16", "float32"]:
            raise ValueError("cast_type should be 'int8', 'float16', 'bfloat16', or 'float32'")
        
        if norm_type == "layernorm" and bias is None:
            raise ValueError("bias is required for LayerNorm")
        if norm_type == "rmsnorm" and bias is not None:
            print("Warning: bias is provided but will be ignored for RMSNorm")
        
        self.norm_type = norm_type
        self.norm_pos = norm_pos
        self.cast_type = cast_type
        self.variance_epsilon = float(eps)
        self.weight = weight
        self.bias = bias
        
        # Map string to torch dtype
        self.cast_dtype = {
            "int8": torch.int8,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cast_type]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor):
        """
        Residual-add normalization with type casting as separate output.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (..., D), normalized over the last dim D.
            residual (torch.Tensor): Residual tensor to add; must be provided and shape-compatible.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - cast_out: Normalized `hidden_state` cast to `cast_type`
                - hidden_state: Normalized `hidden_state` in original dtype (unchanged)
                - residual: Updated `residual`
        """
        cast_out = None
        
        if self.norm_pos == "pre":
            # 1. Add residual (in input dtype)
            residual = hidden_state + residual
            
            # 2. Apply normalization (in input dtype)
            if self.norm_type == "rmsnorm":
                hidden_state = F.rms_norm(
                    residual,
                    [residual.shape[-1]],
                    weight=self.weight,
                    eps=self.variance_epsilon,
                )
            else:  # layernorm
                hidden_state = F.layer_norm(
                    residual,
                    [residual.shape[-1]],
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.variance_epsilon,
                )
            
            # 3. Create cast_out as separate output
            cast_out = hidden_state.to(self.cast_dtype)
            
            # hidden_state remains unchanged (in original dtype)
            # residual remains unchanged (in original dtype)
            
        else:  # norm_pos == "post"
            # 1. Add residual (in input dtype)
            hidden_state = hidden_state + residual
            
            # 2. Apply normalization (in input dtype)
            if self.norm_type == "rmsnorm":
                hidden_state = F.rms_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.weight,
                    eps=self.variance_epsilon,
                )
            else:  # layernorm
                hidden_state = F.layer_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.variance_epsilon,
                )
            
            # 3. Create cast_out as separate output
            cast_out = hidden_state.to(self.cast_dtype)
            
            # 4. Update residual to the un-casted hidden_state
            residual = hidden_state
        
        return cast_out, hidden_state, residual