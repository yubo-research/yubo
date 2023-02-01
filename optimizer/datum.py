from dataclasses import dataclass
from typing import Any

from .trajectories import Trajectory


@dataclass
class Datum:
    policy: Any
    trajectory: Trajectory
