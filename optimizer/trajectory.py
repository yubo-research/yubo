from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    rreturn: float

    states: np.ndarray
    actions: np.ndarray
    rreturn_se: float | None = None
    rreturn_est: float | None = None
    num_steps: int = 0

    rewards: np.ndarray | None = None
    log_probs: np.ndarray | None = None
    values: np.ndarray | None = None
    dones: np.ndarray | None = None
    noise_seed: int | None = None
    iter_index: int | None = None

    def get_decision_rreturn(self) -> float:
        if self.rreturn_est is None:
            return float(self.rreturn)
        return float(self.rreturn_est)
