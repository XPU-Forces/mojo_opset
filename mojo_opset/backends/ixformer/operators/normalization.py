import torch

from ixformer import functions as ixf_f

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoGroupRMSNorm


class IxformerResidualAddRMSNorm(MojoResidualAddRMSNorm):
    """Fused residual + RMSNorm via ixformer ``residual_rms_norm`` (``is_post`` matches ``norm_pos``)."""

    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor):
        if not hidden_state.is_cuda:
            raise RuntimeError(f"{self.__class__.__name__} expects CUDA tensors on Iluvatar.")

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
        if not hidden_state.is_cuda:
            raise RuntimeError(f"{self.__class__.__name__} expects CUDA tensors on Iluvatar.")

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

        if not hidden_state.is_cuda:
            raise RuntimeError(f"{self.__class__.__name__} expects CUDA tensors on Iluvatar.")

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
        if not hidden_state.is_cuda:
            raise RuntimeError(f"{self.__class__.__name__} expects CUDA tensors on Iluvatar.")

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


class IxformerGroupRMSNorm(MojoGroupRMSNorm):

    supported_platforms_list = ["ilu"]

    def forward(self, input_groups):
        
        output_groups = []
        if self.norm_size == 128 and self.num_groups % 2 == 0:
            for i in range(0, self.num_groups, 2):
                expected_k_ptr = input_groups[i].data_ptr() + input_groups[i].size(1) * input_groups[i].stride(1) * input_groups[i].element_size()
                out_q, out_k = ixf_f.rms_norm_qk(input_groups[i], input_groups[i+1], 
                                                 self.weight[i], self.weight[i+1],
                                                 eps=self.variance_epsilon)
                output_groups.append(out_q)
                output_groups.append(out_k)
        else:
            for group_id in range(self.num_groups):
                out = ixf_f.rms_norm(input_groups[group_id], 
                                     self.weight[group_id], 
                                     eps=self.variance_epsilon)
                output_groups.append(out)
        return output_groups
