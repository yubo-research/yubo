from optimizer.gaussian_perturbator import PerturbatorBase


class _P(PerturbatorBase):
    def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
        _ = (seed, sigma, chunk_size)
