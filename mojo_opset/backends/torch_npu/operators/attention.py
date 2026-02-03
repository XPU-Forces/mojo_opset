import torch
import torch_npu

from mojo_opset.core import MojoSdpa


class TorchNpuSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if self.scale is None:
            self.scale = 1.0 / (query.shape[-1] ** 0.5)

        output = torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            scale=self.scale,
            head_num=self.num_heads,
            input_layout=self.layout,
            sparse_mode=0
        )[0]

        return output
