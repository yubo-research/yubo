from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ExpParams:
    num_dim: int
    num_active: int
    num_rounds: int
    num_reps: int
    seed: int
    optimizer: Callable[[Any, Any, int], Any]
    controller: Callable[[int], Any]

    def __post_init__(self) -> None:
        assert isinstance(self.num_rounds, int) and self.num_rounds >= 0
        assert isinstance(self.num_reps, int) and self.num_reps >= 0
        assert isinstance(self.seed, int)
        assert callable(self.optimizer)
        assert callable(self.controller)
