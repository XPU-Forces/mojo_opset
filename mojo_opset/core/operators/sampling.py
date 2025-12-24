from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from mojo_opset.utils.mode import get_mojo_exec_mode

from ..mojo_operator import MojoOperator


class MojoTopKSampling(MojoOperator):
    pass


class MojoTopPSampling(MojoOperator):
    def __init__(
        self,
        top_p: float = 0.75,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        rand_top_k: int = 1000,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.rand_top_k = rand_top_k

        mode_str = get_mojo_exec_mode(MojoTopPSampling.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)

    @abstractmethod
    def forward_std(self, logits: torch.Tensor) -> Tuple[Any]:
        raise NotImplementedError

    def forward_analysis(self, logits) -> Tuple[int, int, int]:
        raise NotImplementedError


class MojoTopPFilter(MojoOperator):
    def __init__(
        self,
        filter_value: float = -float("Inf"),
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        self.filter_value = filter_value

        mode_str = get_mojo_exec_mode(MojoTopPSampling.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)

    @abstractmethod
    def forward_std(self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int) -> Tuple[Any]:
        raise NotImplementedError

    def forward_analysis(
        self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int
    ) -> Tuple[int, int, int]:
        raise NotImplementedError


class MojoRejectSampling(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    @abstractmethod
    def forward_std(
        self,
        target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,   # [batch, spec_step]
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_analysis(
        self,
        target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,   # [batch, spec_step]
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    ) -> Tuple[int, int, int]:
        raise NotImplementedError


class MojoMagicRejectSampling(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    @abstractmethod
    def forward_std(
        self,
        target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,   # [batch, spec_step]
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_analysis(
        self,
        target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
        draft_tokens: torch.Tensor,  # [batch, spec_step]
        draft_probs: torch.Tensor,   # [batch, spec_step]
        spec_step: int,
        top_p: float,
        rand_top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1, 
    ) -> Tuple[int, int, int]:
        raise NotImplementedError

class MojoApplyPenaltiesTempurate(MojoOperator):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        mode_str = get_mojo_exec_mode(MojoTopPSampling.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)

    @abstractmethod
    def forward_std(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_analysis(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> Tuple[int, int, int]:
        raise NotImplementedError
