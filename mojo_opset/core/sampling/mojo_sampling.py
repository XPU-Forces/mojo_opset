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

    def forward_ref(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.to(torch.float32)
        top_k = min(self.rand_top_k, logits.size(-1))
        sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)

        cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, self.filter_value)

        final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1)

        select_index = torch.multinomial(final_probs_dist, num_samples=1)

        next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
        next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)

        return next_probs, next_tokens

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

    def forward_ref(
        self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = logits.dtype
        logits = logits.to(torch.float32)
        top_k = min(rand_top_k, logits.size(-1))
        sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)

        cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, self.filter_value)

        final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1).to(dtype)

        return final_probs_dist, sorted_topk_indices

    def forward_analysis(
        self, logits: torch.Tensor, top_p: float, min_tokens_to_keep: int, rand_top_k: int
    ) -> Tuple[int, int, int]:
        raise NotImplementedError


class MojoRejectSampling(MojoOperator):
    pass


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

    def forward_ref(
        self,
        logits: torch.Tensor,
        token_freqs: List[Union[None, torch.Tensor]],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        temps: Optional[List[Optional[float]]] = None,
    ) -> torch.Tensor:
        dtype = logits.dtype
        logits = logits.to(torch.float32)

        for i, freq_token in enumerate(token_freqs):
            if freq_token is not None:
                device_freq_token = freq_token.to(logits.device, non_blocking=True)
                if frequency_penalties[i] != 0.0:
                    logits[i] -= frequency_penalties[i] * device_freq_token
                if presence_penalties[i] != 0.0:
                    logits[i] -= presence_penalties[i] * (device_freq_token > 0)
                if repetition_penalties[i] != 1.0:
                    conds = logits[i] * device_freq_token
                    logits[i] = torch.where(
                        conds < 0,
                        logits[i] * repetition_penalties[i],
                        torch.where(conds > 0, logits[i] / repetition_penalties[i], logits[i]),
                    )
            if temps is not None and temps[i] is not None:
                logits[i] /= temps[i]
        return logits.to(dtype)

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
