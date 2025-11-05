import torch

from mojo_opset.core import MojoGelu
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class XOpsGelu(MojoGelu, default_priority=1):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        logger.info(f"XOpsGelu init, op_name: {op_name}")
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor):
        raise NotImplementedError
