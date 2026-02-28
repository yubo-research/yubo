"""Types for MNIST environment."""

from typing import NamedTuple


class StepResult(NamedTuple):
    """Result from a single environment step."""

    state: object
    reward: float
    done: bool
    info: object | None
