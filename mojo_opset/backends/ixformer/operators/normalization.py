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

    @staticmethod
    def _can_merge_adjacent_head_dim(input_q: torch.Tensor, input_k: torch.Tensor) -> bool:
        if input_q.dim() < 3 or input_q.dim() != input_k.dim():
            return False
        if input_q.shape[:-1] != input_k.shape[:-1]:
            return False
        if input_q.stride() != input_k.stride():
            return False
        if input_q.untyped_storage().data_ptr() != input_k.untyped_storage().data_ptr():
            return False

        expected_k_ptr = input_q.data_ptr() + input_q.size(-1) * input_q.stride(-1) * input_q.element_size()
        return expected_k_ptr == input_k.data_ptr()

    def _rms_norm_merged_head_dim(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
    ):
        merged_shape = list(input_q.shape)
        q_head_dim = input_q.size(-1)
        k_head_dim = input_k.size(-1)
        merged_shape[-1] = q_head_dim + k_head_dim
        merged = input_q.as_strided(merged_shape, input_q.stride())
        merged_weight = torch.cat([weight_q, weight_k], dim=0)

        normalized = ixf_f.rms_norm(merged, merged_weight, eps=self.variance_epsilon)
        return torch.split(normalized, [q_head_dim, k_head_dim], dim=-1)

    def forward(self, input_groups):
        
        output_groups = []
        if self.num_groups % 2 == 0:
            for i in range(0, self.num_groups, 2):
                if self._can_merge_adjacent_head_dim(input_groups[i], input_groups[i + 1]):
                    out_q, out_k = self._rms_norm_merged_head_dim(
                        input_groups[i],
                        input_groups[i + 1],
                        self.weight[i],
                        self.weight[i + 1],
                    )
                else:
                    out_q = ixf_f.rms_norm(input_groups[i],
                                           self.weight[i],
                                           eps=self.variance_epsilon)
                    out_k = ixf_f.rms_norm(input_groups[i + 1],
                                           self.weight[i + 1],
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
