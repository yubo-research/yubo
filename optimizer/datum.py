from dataclasses import dataclass
from typing import Any

from .trajectories import Trajectory


@dataclass
class Datum:
    designer: Any
    policy: Any
    expected_acqf: float
    trajectory: Trajectory

    def e_acqf(self):
        if self.expected_acqf is not None:
            return self.expected_acqf
        return self.trajectory.rreturn
