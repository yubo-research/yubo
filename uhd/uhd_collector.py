import time
from typing import Optional

import torch
import torch.nn as nn

from common.collector import Collector


class UHDCollector:
    def __init__(self, name: str, opt_name: str, collector: Optional[Collector] = None) -> None:
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(opt_name, str) and len(opt_name) > 0
        self.name = name
        self.opt_name = opt_name
        self._collector = collector if collector is not None else Collector()
        self._t0_prop: Optional[float] = None
        self._t0_eval: Optional[float] = None
        self._dt_prop: Optional[float] = None
        self._dt_eval: Optional[float] = None
        self._y_best: Optional[float] = None
        self._i_iter: int = 0

    def start_prop(self) -> None:
        assert self._t0_prop is None, "start_prop() already called, call stop_prop() first"
        assert self._dt_prop is None, "stop_prop() must be called before starting a new prop timing"
        self._t0_prop = time.time()

    def stop_prop(self) -> float:
        assert self._dt_prop is None, "stop_prop() already called for this iteration"
        assert self._t0_prop is not None, "start_prop() must be called before stop_prop()"
        dt = time.time() - self._t0_prop
        self._t0_prop = None
        self._dt_prop = dt
        return dt

    def start_eval(self) -> None:
        assert self._t0_eval is None, "start_eval() already called, call stop_eval() first"
        assert self._dt_eval is None, "stop_eval() must be called before starting a new eval timing"
        self._t0_eval = time.time()

    def stop_eval(self) -> float:
        assert self._dt_eval is None, "stop_eval() already called for this iteration"
        assert self._t0_eval is not None, "start_eval() must be called before stop_eval()"
        dt = time.time() - self._t0_eval
        self._t0_eval = None
        self._dt_eval = dt
        return dt

    def params(self, controller: nn.Module) -> None:
        assert isinstance(controller, nn.Module)
        total = sum(int(p.numel()) for p in controller.parameters())
        if total == 0:
            return
        flat = torch.cat([p.view(-1).detach() for p in controller.parameters()])
        if flat.numel() == 0:
            return
        mn = float(flat.min().item())
        mx = float(flat.max().item())
        mu = float(flat.mean().item())
        sd = float(flat.std(unbiased=False).item())
        target_name = getattr(controller, "name", type(controller).__name__)
        self._collector(f"PARAMS: target = {target_name} min = {mn:.6f} max = {mx:.6f} mean = {mu:.6f} std = {sd:.6f}")

    def trace(self, y: float) -> None:
        assert isinstance(y, (int, float))
        assert self._dt_prop is not None, "stop_prop() must be called exactly once before trace()"
        assert self._dt_eval is not None, "stop_eval() must be called exactly once before trace()"
        assert self._t0_prop is None, "start_prop() called but stop_prop() not called"
        assert self._t0_eval is None, "start_eval() called but stop_eval() not called"
        self.update_best(float(y))
        self._collector(
            f"TRACE: name = {self.name} opt_name = {self.opt_name} i_iter = {self._i_iter} dt_prop = {self._dt_prop:.3e} dt_eval = {self._dt_eval:.3e} y = {float(y):.4f} y_best = {self._y_best:.4f}"
        )
        self._dt_prop = None
        self._dt_eval = None
        self._i_iter += 1

    def __call__(self, line: str) -> None:
        self._collector(line)

    def update_best(self, y: float) -> None:
        assert isinstance(y, (int, float))
        if self._y_best is None or float(y) > self._y_best:
            self._y_best = float(y)

    def best(self) -> Optional[float]:
        return self._y_best

    def reset_eval_timing(self) -> None:
        self._dt_eval = None
