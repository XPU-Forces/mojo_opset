import torch

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm

def _get_ixf_and_check_device(tensor: torch.Tensor, class_name: str):
    """Helper to import ixformer and check for CUDA device."""
    if not tensor.is_cuda:
        raise RuntimeError(f"{class_name} expects CUDA tensors on Iluvatar.")
    from ixformer import functions as ixf_f
    return ixf_f


class IxformerResidualAddRMSNorm(MojoResidualAddRMSNorm):
    """Fused residual + RMSNorm via ixformer ``residual_rms_norm`` (``is_post`` matches ``norm_pos``)."""

    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor):
        ixf_f = _get_ixf_and_check_device(hidden_state, self.__class__.__name__)

        is_post = self.norm_pos == "post"
        out, res_out = ixf_f.residual_rms_norm(
            hidden_state,
            self.weight,
            eps=self.variance_epsilon,
            residual=residual,
            residual_alpha=1.0,
            residual_bias=None,
            is_post=is_post,
        )
        return out, res_out


class IxformerResidualAddLayerNorm(MojoResidualAddLayerNorm):
    """Fused residual + LayerNorm via ixformer ``residual_layer_norm``.

    ``norm_pos=="pre"``: single fused call. ``norm_pos=="post"``: add then ``residual_layer_norm``
    with ``residual=None`` (Python binding does not expose ``is_post`` for LN).
    """

    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor):
        ixf_f = _get_ixf_and_check_device(hidden_state, self.__class__.__name__)

        if self.norm_pos == "pre":
            out, res_out = ixf_f.residual_layer_norm(
                hidden_state,
                self.weight,
                self.bias,
                residual=residual,
                residual_bias=None,
                eps=self.variance_epsilon,
            )
            return out, res_out

        summed = hidden_state + residual
        out, _ = ixf_f.residual_layer_norm(
            summed,
            self.weight,
            self.bias,
            residual=None,
            residual_bias=None,
            eps=self.variance_epsilon,
        )
        return out, out


class IxformerLayerNorm(MojoLayerNorm):
    """LayerNorm via ixformer inference ``layer_norm`` (``residual_layer_norm`` with ``residual=None``)."""

    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine or self.weight is None or self.bias is None:
            raise NotImplementedError(
                "IxformerLayerNorm requires elementwise_affine=True with weight and bias (ixformer infer kernel)."
            )

        ixf_f = _get_ixf_and_check_device(hidden_state, self.__class__.__name__)

        out, _ = ixf_f.residual_layer_norm(
            hidden_state,
            self.weight,
            self.bias,
            residual=None,
            residual_bias=None,
            eps=self.variance_epsilon,
        )
        return out


class IxformerRMSNorm(MojoRMSNorm):
    """RMSNorm via ixformer inference ``rms_norm`` (same path as ``residual_rms_norm`` without residual)."""

    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ixf_f = _get_ixf_and_check_device(hidden_state, self.__class__.__name__)

        # Aligns with ixformer tests: plain RMSNorm uses residual_rms_norm with residual=None
        # (dispatches to ops.infer.rms_norm); optional fused bias is unused by MojoRMSNorm.
        out, _ = ixf_f.residual_rms_norm(
            hidden_state,
            self.weight,
            eps=self.variance_epsilon,
            residual=None,
            residual_bias=None,
        )
        return out
