from typing import Optional

from ..mojo_operator import MojoOperator


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - is_varlen (bool): When True, prioritize TND (per token) routing; when False, use BSND; default True.
        - op_name: Operator name placeholder.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__(op_name)
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.is_varlen = is_varlen


class MojoBigEPDispatch(MojoOperator):
    pass
