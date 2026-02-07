import math


class StepSizeAdapter:
    def __init__(
        self,
        sigma_0: float,
        dim: int,
        *,
        sigma_min: float = 1e-5,
        sigma_max: float = 0.2,
        success_tolerance: int = 3,
    ):
        self._sigma = sigma_0
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._success_tolerance = success_tolerance
        self._failure_tolerance = math.ceil(dim)
        self._success_count = 0
        self._failure_count = 0
        self._y_max: float | None = None

    @property
    def sigma(self) -> float:
        return self._sigma

    def update(self, y: float) -> bool:
        """Update step size based on observed y. Returns True if y improved."""
        if self._y_max is None or y > self._y_max:
            self._y_max = y
            self._success_count += 1
            self._failure_count = 0
            improved = True
        else:
            self._success_count = 0
            self._failure_count += 1
            improved = False

        if self._success_count >= self._success_tolerance:
            self._sigma = min(2.0 * self._sigma, self._sigma_max)
            self._success_count = 0
        elif self._failure_count >= self._failure_tolerance:
            self._sigma /= 2.0
            self._sigma = max(self._sigma, self._sigma_min)
            self._failure_count = 0

        return improved
