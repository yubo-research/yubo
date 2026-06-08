from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _botorch_modules(monkeypatch, *, empty_front: bool):
    root = types.ModuleType("botorch")
    utils = types.ModuleType("botorch.utils")
    mo = types.ModuleType("botorch.utils.multi_objective")
    hv_mod = types.ModuleType("botorch.utils.multi_objective.hypervolume")
    pareto_mod = types.ModuleType("botorch.utils.multi_objective.pareto")

    class Hypervolume:
        def __init__(self, ref):
            self.ref = ref

        def compute(self, front):
            return 0.0 if empty_front else float(torch.sum(front).item())

    pareto_mod.is_non_dominated = lambda y: torch.zeros(y.shape[0], dtype=torch.bool) if empty_front else torch.ones(y.shape[0], dtype=torch.bool)
    hv_mod.Hypervolume = Hypervolume
    mo.hypervolume = hv_mod
    mo.pareto = pareto_mod
    utils.multi_objective = mo
    root.utils = utils
    for k, v in [
        ("botorch", root),
        ("botorch.utils", utils),
        ("botorch.utils.multi_objective", mo),
        ("botorch.utils.multi_objective.hypervolume", hv_mod),
        ("botorch.utils.multi_objective.pareto", pareto_mod),
    ]:
        monkeypatch.setitem(sys.modules, k, v)


def _mo_datum(pol):
    def datum(r, p=None):
        return SimpleNamespace(
            trajectory=SimpleNamespace(rreturn=np.asarray(r, dtype=np.float64)),
            policy=p or SimpleNamespace(clone=lambda: SimpleNamespace()),
        )

    return datum


def _run_mo_hv_scenarios(monkeypatch, MO, datum, pol):
    o = MO()
    o.y_best = np.zeros(2)
    o.r_best_est = 0.0
    o.best_datum = None
    o.best_policy = pol
    o._ref_point = np.array([2.0, 2.0], dtype=np.float64)
    o._data = [datum([1.0, 1.5]), datum([0.5, 2.0])]
    o._env_conf = SimpleNamespace(problem_seed=0, noise_seed_0=None)
    o._num_arms = 2
    o._num_denoise = 1
    o._collector = lambda _s: None
    assert isinstance(o._handle_hypervolume(2), float)
    o._ref_point = None
    monkeypatch.setattr(
        "analysis.ref_point.SobolRefPoint",
        lambda **kwargs: SimpleNamespace(compute=lambda *a, **kw: np.array([3.0, 3.0], dtype=np.float64)),
    )
    o2 = MO()
    o2.y_best = np.zeros(2)
    o2.r_best_est = 0.0
    o2.best_datum = None
    o2.best_policy = pol
    o2._ref_point = None
    o2._data = [datum([1.0, 1.0])]
    o2._env_conf = SimpleNamespace(problem_seed=1, noise_seed_0=2, env_tag="f:ackley-3d")
    o2._policy_tag = "pure-function"
    o2._num_arms = 2
    o2._num_denoise = 1
    o2._collector = lambda _s: None
    o2._handle_hypervolume(2)
    _botorch_modules(monkeypatch, empty_front=True)
    o3 = MO()
    o3.y_best = np.zeros(2)
    o3.r_best_est = 0.0
    o3.best_datum = datum([1.0, 1.0])
    o3.best_policy = pol
    o3._ref_point = np.array([1.0, 1.0], dtype=np.float64)
    o3._data = [datum([0.0, 0.0])]
    o3._env_conf = SimpleNamespace(problem_seed=0, noise_seed_0=None)
    o3._num_arms = 1
    o3._num_denoise = 1
    o3._collector = lambda _s: None
    o3._handle_hypervolume(2)


def _run_mo_returns_and_fit(MO, datum, pol):
    from optimizer.optimizer_types import ReturnSummary
    from optimizer.uhd_enn_fit_helpers import fit_enn_regressor_on_points

    o4 = MO()
    o4.y_best = np.array([0.0])
    o4.r_best_est = -1.0
    o4.best_datum = None
    o4.best_policy = None
    dlist = [datum([0.5], pol), datum([0.1], pol)]
    rb = np.array([[0.1], [0.5]], dtype=np.float64)
    o4._handle_first_objective(dlist, rb)
    assert isinstance(o4._handle_multi_objective_returns(dlist, rb, 1), ReturnSummary)
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(5) for _ in range(25)]
    ys = [float(rng.standard_normal()) for _ in range(25)]
    ym, _ys, model, params = fit_enn_regressor_on_points(xs, ys, k=3)
    assert isinstance(ym, float) and model is not None and params is not None


def run_optimizer_mo_and_fit(monkeypatch):
    from optimizer.optimizer_mo import OptimizerMultiObjectiveMixin

    _botorch_modules(monkeypatch, empty_front=False)
    MO = type("MO", (OptimizerMultiObjectiveMixin,), {})
    pol = SimpleNamespace(clone=lambda: SimpleNamespace())
    datum = _mo_datum(pol)
    _run_mo_hv_scenarios(monkeypatch, MO, datum, pol)
    _run_mo_returns_and_fit(MO, datum, pol)
