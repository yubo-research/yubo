import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class TrustRegionConfig:
    length_min: float
    length_max: float
    length_init: float
    succtol: int
    failtol: int


class TrustRegionAdjustor:
    def __init__(
        self, *, dim: int, batch_size: int, config: Optional[TrustRegionConfig] = None
    ) -> None:
        assert isinstance(dim, int) and dim > 0
        assert isinstance(batch_size, int) and batch_size > 0
        if config is None:
            length_min = 0.5**7
            length_max = 1.6
            length_init = 0.8
            failtol = int(math.ceil(max(4.0 / batch_size, dim / batch_size)))

            succtol = 3
            config = TrustRegionConfig(
                length_min=length_min,
                length_max=length_max,
                length_init=length_init,
                succtol=succtol,
                failtol=failtol,
            )
        assert isinstance(config.length_min, (int, float)) and config.length_min > 0
        assert (
            isinstance(config.length_max, (int, float))
            and config.length_max > config.length_min
        )
        assert (
            isinstance(config.length_init, (int, float))
            and config.length_min <= config.length_init <= config.length_max
        )
        assert isinstance(config.succtol, int) and config.succtol > 0
        assert isinstance(config.failtol, int) and config.failtol > 0
        self._dim = dim
        self._batch_size = batch_size
        self._cfg = config
        self._length = float(config.length_init)
        self._succcount = 0
        self._failcount = 0
        self._best = None

    @property
    def length(self) -> float:
        return float(self._length)

    @property
    def config(self) -> TrustRegionConfig:
        return self._cfg

    def reset(self) -> None:
        self._length = float(self._cfg.length_init)
        self._succcount = 0
        self._failcount = 0
        self._best = None

    def update(self, y: Iterable[float]) -> None:
        vals = np.asarray(list(y), dtype=float).reshape(-1)
        assert vals.size == self._batch_size
        if self._best is None:
            self._best = float(np.max(vals))
            return
        rel = 1e-3 * math.fabs(self._best)
        improved = float(np.max(vals)) > self._best + rel
        if improved:
            self._succcount += 1
            self._failcount = 0
            self._best = float(max(self._best, float(np.max(vals))))
        else:
            self._succcount = 0
            self._failcount += 1
        print("FT:", self._failcount, self._cfg.failtol)
        if self._succcount == self._cfg.succtol:
            self._length = float(min(2.0 * self._length, self._cfg.length_max))
            self._succcount = 0
        elif self._failcount == self._cfg.failtol:
            self._length = float(max(self._length / 2.0, self._cfg.length_min))
            self._failcount = 0
