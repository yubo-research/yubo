class StepSizeAdapter:
    _SIGMA_DEFAULT = 1e-3

    def __init__(
        self,
        dim: int,
        *,
        sigma_0: float = _SIGMA_DEFAULT,
        sigma_min: float = 1e-8,
        sigma_max: float = 1e8,
        success_tolerance: int = 3,
    ):
        self._sigma = max(sigma_min, min(sigma_max, sigma_0))
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._success_tolerance = success_tolerance
        self._failure_tolerance = max(10, 5 * success_tolerance)
        self._success_count = 0
        self._failure_count = 0

    @property
    def sigma(self) -> float:
        return self._sigma

    def update(self, *, accepted: bool) -> None:
        """Update step size based on acceptance."""
        if accepted:
            self._success_count += 1
            self._failure_count = 0
        else:
            self._success_count = 0
            self._failure_count += 1

        if self._success_count >= self._success_tolerance:
            self._sigma = min(2.0 * self._sigma, self._sigma_max)
            self._success_count = 0
        elif self._failure_count >= self._failure_tolerance:
            self._sigma /= 2.0
            self._sigma = max(self._sigma, self._sigma_min)
            self._failure_count = 0
