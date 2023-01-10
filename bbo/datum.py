from dataclasses import dataclass
from typing import Any

from bbo.trajectories import Trajectory


@dataclass
class Datum:
    policy: Any
    trajectory: Trajectory
