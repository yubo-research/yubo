import torch

from model.enn_t import EpistemicNearestNeighborsT
from sampling.sobol_indices_t import calculate_sobol_indices_t


class ENNWeighterT:
    def __init__(self, *, weighting: str, k: int = 3):
        assert weighting in {"sobol_indices", "sigma_x", "sobol_over_sigma"}
        self._weighting = weighting
        self._enn = EpistemicNearestNeighborsT(k=k)
        self._weights_xdtype: torch.Tensor | None = None
        self._xy: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self._enn)

    @property
    def weights(self) -> torch.Tensor:
        self._set_weights()
        return self._weights_xdtype.to(dtype=torch.float64)

    def _calc_weights(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.view(-1)
        if self._weighting == "sobol_indices":
            w = calculate_sobol_indices_t(x, y)
            w = w / torch.clamp(w.sum(), min=torch.as_tensor(1e-12, dtype=w.dtype, device=w.device))
        elif self._weighting == "sigma_x":
            s = torch.std(x, dim=0, unbiased=False)
            s = torch.clamp_min(s, torch.as_tensor(1e-6, dtype=s.dtype, device=s.device))
            w = 1.0 / s
        elif self._weighting == "sobol_over_sigma":
            s = torch.std(x, dim=0, unbiased=False)
            s = torch.clamp_min(s, torch.as_tensor(1e-6, dtype=s.dtype, device=s.device))
            si = calculate_sobol_indices_t(x, y)
            w = si / s
            w = w / torch.clamp(w.sum(), min=torch.as_tensor(1e-12, dtype=w.dtype, device=w.device))
        else:
            assert False
        w = torch.clamp_min(w, torch.as_tensor(1e-6, dtype=w.dtype, device=w.device))
        return w.to(dtype=x.dtype)

    def _set_weights(self) -> None:
        if self._weights_xdtype is None:
            assert self._xy is not None
            x, y = self._xy
            self._xy = "done"  # type: ignore[assignment]
            w = self._calc_weights(x, y)
            self._weights_xdtype = w
            y_var = torch.zeros_like(y)
            self._enn.add(x * w, y, y_var)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert self._xy is None
        self._xy = (x, y)

    def __call__(self, x: torch.Tensor):
        return self.posterior(x)

    def posterior(self, x: torch.Tensor, *, k: int | None = None, exclude_nearest: bool = False):
        self._set_weights()
        return self._enn.posterior(x * self._weights_xdtype.to(device=x.device, dtype=x.dtype), k=k, exclude_nearest=exclude_nearest)
