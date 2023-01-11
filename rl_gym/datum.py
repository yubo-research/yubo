from dataclasses import dataclass
from typing import Any

from rl_gym.trajectories import Trajectory


@dataclass
class Datum:
    policy: Any
    trajectory: Trajectory
