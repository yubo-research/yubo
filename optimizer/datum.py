from dataclasses import dataclass
from typing import Any

from .trajectory import Trajectory


@dataclass
class Datum:
    designer: Any
    policy: Any
    expected_acqf: float
    trajectory: Trajectory
