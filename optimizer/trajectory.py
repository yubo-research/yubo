from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    rreturn: float

    states: np.ndarray
    actions: np.ndarray
    rreturn_se: float = None
    rreturn_est: float = None
    num_steps: int = 0

    def get_decision_rreturn(self) -> float:
        if self.rreturn_est is None:
            return float(self.rreturn)
        return float(self.rreturn_est)
