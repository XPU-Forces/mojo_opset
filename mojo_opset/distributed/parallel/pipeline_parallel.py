from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineStageInfo:
    """Describes how layers are partitioned across pipeline-parallel stages.

    Each stage owns a contiguous range of layers.  When ``num_layers`` is not
    evenly divisible by ``pp_size``, the first ``num_layers % pp_size`` stages
    each get one extra layer (standard balanced split).
    """

    pp_size: int
    pp_rank: int
    num_layers: int

    @property
    def stage_layer_range(self) -> Tuple[int, int]:
        """Return ``[start, end)`` global layer indices for this stage."""
        base = self.num_layers // self.pp_size
        remainder = self.num_layers % self.pp_size
        start = self.pp_rank * base + min(self.pp_rank, remainder)
        end = start + base + (1 if self.pp_rank < remainder else 0)
        return (start, end)

    @property
    def num_stage_layers(self) -> int:
        s, e = self.stage_layer_range
        return e - s

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1
