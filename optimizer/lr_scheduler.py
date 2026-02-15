import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class LRScheduler(Protocol):
    @property
    def lr(self) -> float: ...

    def step(self) -> None: ...


class ConstantLR:
    """Constant learning rate (no schedule)."""

    def __init__(self, lr: float):
        self._lr = lr

    @property
    def lr(self) -> float:
        return self._lr

    def step(self) -> None:
        pass


class LinearLRScheduler:
    """Linear warmup then linear decay to zero.

    lr(t) =
        lr_0 * t / warmup_steps              if t < warmup_steps
        lr_0 * (num_steps - t) / decay_steps  otherwise

    where decay_steps = num_steps - warmup_steps.
    """

    def __init__(
        self,
        lr_0: float,
        num_steps: int,
        *,
        warmup_steps: int = 0,
    ):
        assert num_steps > 0
        assert 0 <= warmup_steps < num_steps
        self._lr_0 = lr_0
        self._num_steps = num_steps
        self._warmup_steps = warmup_steps
        self._decay_steps = num_steps - warmup_steps
        self._t = 0

    @property
    def lr(self) -> float:
        if self._t >= self._num_steps:
            return 0.0
        if self._t < self._warmup_steps:
            return self._lr_0 * self._t / self._warmup_steps
        return self._lr_0 * (self._num_steps - self._t) / self._decay_steps

    def step(self) -> None:
        self._t += 1


class OneCycleLR:
    """One-cycle LR schedule (Smith 2018).

    Phase 1 (0 .. warmup_steps):
        Linear ramp from max_lr / div_factor  →  max_lr

    Phase 2 (warmup_steps .. num_steps):
        Cosine anneal from max_lr  →  max_lr / (div_factor * final_div_factor)

    Matches torch.optim.lr_scheduler.OneCycleLR semantics.
    """

    def __init__(
        self,
        max_lr: float,
        num_steps: int,
        *,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        assert num_steps > 0
        assert 0.0 < pct_start < 1.0
        self._max_lr = max_lr
        self._num_steps = num_steps
        self._warmup_steps = int(num_steps * pct_start)
        self._decay_steps = num_steps - self._warmup_steps
        self._initial_lr = max_lr / div_factor
        self._min_lr = self._initial_lr / final_div_factor
        self._t = 0

    @property
    def lr(self) -> float:
        if self._t >= self._num_steps:
            return self._min_lr
        if self._t < self._warmup_steps:
            # Linear warmup.
            frac = self._t / max(1, self._warmup_steps)
            return self._initial_lr + frac * (self._max_lr - self._initial_lr)
        # Cosine anneal.
        frac = (self._t - self._warmup_steps) / max(1, self._decay_steps)
        cosine = (1.0 + math.cos(math.pi * frac)) / 2.0
        return self._min_lr + cosine * (self._max_lr - self._min_lr)

    def step(self) -> None:
        self._t += 1
