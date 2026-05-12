import torch

from mojo_opset.core import MojoSilu


class AscendcSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor):
        raise NotImplementedError("AscendcSilu is a placeholder backend implementation and is not available yet.")
