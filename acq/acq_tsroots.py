import warnings

import numpy as np
import torch

from acq.acq_ts import AcqTS

try:
    from tsroots.optim import TSRoots
except Exception as exc:  # pragma: no cover - optional third-party dependency
    TSRoots = None
    _TSROOTS_IMPORT_ERROR = exc
else:
    _TSROOTS_IMPORT_ERROR = None


# See https://github.com/UQUH/TSRoots/tree/main
class AcqTSRoots:
    def __init__(self, model, num_fallback_candidates: int = 4096):
        # TODO: avoid fitting the model
        self._model = model
        X_0 = model.train_inputs[0].detach()
        self._dtype = X_0.dtype
        self._device = X_0.device
        self._X = X_0.cpu().numpy()
        num_obs = self._X.shape[0]
        assert num_obs > 0, num_obs
        self._num_dim = self._X.shape[-1]
        self._Y = model.train_targets.detach().cpu().numpy()
        self._fallback_ts = None
        if TSRoots is None:
            warnings.warn(
                f"TSRoots unavailable ({type(_TSROOTS_IMPORT_ERROR).__name__}: {_TSROOTS_IMPORT_ERROR}); falling back to internal Thompson sampling.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fallback_ts = AcqTS(model, num_candidates=num_fallback_candidates)

    def draw(self, num_arms):
        if self._fallback_ts is not None:
            return self._fallback_ts.draw(num_arms)

        tsr = TSRoots(
            self._X,
            self._Y,
            np.zeros(shape=(self._num_dim,)),
            np.ones(shape=(self._num_dim,)),
        )

        return torch.stack(
            [
                torch.as_tensor(
                    tsr.xnew_TSroots()[0],
                    dtype=self._dtype,
                    device=self._device,
                )
                for _ in range(num_arms)
            ]
        )
