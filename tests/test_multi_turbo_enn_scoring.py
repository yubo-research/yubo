from types import SimpleNamespace

import numpy as np
import pytest


class _Posterior:
    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)


class _Surrogate:
    def __init__(self, mu, sigma):
        self._posterior = _Posterior(mu, sigma)

    def predict(self, _x):
        return self._posterior


class _BrokenSurrogate(_Surrogate):
    def predict(self, _x):
        raise RuntimeError("predict failed")

    def sample(self, _x, _n, _rng):
        raise RuntimeError("sample failed")


class _UCBAcqOptimizer:
    def __init__(self, beta: float):
        self._beta = float(beta)


class _ParetoAcqOptimizer:
    pass


class _ThompsonAcqOptimizer:
    pass


class _Turbo:
    def __init__(self, *, mu, sigma, acq, tr_state=None):
        self._surrogate = _Surrogate(mu, sigma)
        self._acq_optimizer = acq
        self._tr_state = tr_state


class _BrokenTurbo(_Turbo):
    def __init__(self, *, mu, sigma, acq, tr_state=None):
        self._surrogate = _BrokenSurrogate(mu, sigma)
        self._acq_optimizer = acq
        self._tr_state = tr_state


class _ChildDesigner:
    def __init__(self, turbo):
        self._turbo = turbo

    def best_datum(self):
        return None


class _FixedChild:
    def __init__(self, n_params: int, n_emit: int):
        self._n_params = int(n_params)
        self._n_emit = int(n_emit)

    def __call__(self, _data, _n):
        # Keep this intentionally different from allocation-test helper to avoid duplication gate.
        return [_Policy(np.full((self._n_params,), float(i), dtype=float)) for i in range(self._n_emit)]


class _Policy:
    def __init__(self, params):
        self._params = np.asarray(params, dtype=float)

    def get_params(self):
        return self._params


def test_score_multi_candidates_top_level_fills_nans(monkeypatch):
    from optimizer import multi_turbo_enn_scoring as scoring

    calls = {"n": 0}

    def _fake_score_region_candidates(_child, *, x_region, acq_type, allow_random, region_rng):
        _ = _child, acq_type, allow_random, region_rng
        calls["n"] += 1
        out = np.ones((x_region.shape[0],), dtype=float)
        if calls["n"] == 1:
            out[0] = np.nan
        return out

    monkeypatch.setattr(scoring, "_score_region_candidates", _fake_score_region_candidates)
    x_all = np.zeros((4, 2), dtype=float)
    scores = scoring.score_multi_candidates(
        x_all,
        [0, 1, 0, 1],
        child_designers=[object(), object()],
        region_data_lens=[0, 1],
        region_rngs=[np.random.default_rng(0), np.random.default_rng(1)],
        acq_type="ucb",
        rng=np.random.default_rng(2),
    )
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_call_multi_designer_split_allocated_and_errors(monkeypatch):
    from optimizer import multi_turbo_enn_utils as utils

    events: list[str] = []

    monkeypatch.setattr(utils, "assert_scalar_rreturn", lambda _data: events.append("assert"))
    monkeypatch.setattr(utils, "_tell_new_data_if_any", lambda _designer, _data: events.append("tell"))
    monkeypatch.setattr(utils, "_call_fixed_arms", lambda _designer, *, num_arms: [f"fixed-{num_arms}"])
    monkeypatch.setattr(utils, "_call_allocated", lambda _designer, *, num_arms: [f"alloc-{num_arms}"])

    class _Designer:
        def __init__(self, arm_mode: str):
            self._tr_type = "turbo"
            self._arm_mode = arm_mode

        def _init_regions(self, _data, _num_arms):
            events.append("init")

        def _set_telemetry(self, _telemetry):
            events.append("telemetry")

    out_fixed = utils.call_multi_designer(_Designer("split"), data=["x"], num_arms=2, telemetry=object())
    assert out_fixed == ["fixed-2"]
    assert events[:4] == ["assert", "init", "tell", "telemetry"]

    events.clear()
    out_alloc = utils.call_multi_designer(_Designer("allocated"), data=["x"], num_arms=3)
    assert out_alloc == ["alloc-3"]
    assert events == ["assert", "init", "tell"]

    with pytest.raises(ValueError):
        utils.call_multi_designer(_Designer("split"), data=["x"], num_arms=0)


def test_score_region_candidates_ucb_uses_region_beta():
    from optimizer.multi_turbo_enn_scoring import _score_region_candidates

    mu = np.array([[1.0], [1.0], [1.0]], dtype=float)
    sigma = np.array([[1.0], [2.0], [0.5]], dtype=float)
    child = _ChildDesigner(_Turbo(mu=mu, sigma=sigma, acq=_UCBAcqOptimizer(beta=2.0)))
    rng = np.random.default_rng(0)
    x_region = np.zeros((3, 2), dtype=float)
    scores = _score_region_candidates(
        child,
        x_region=x_region,
        acq_type="ucb",
        allow_random=False,
        region_rng=rng,
    )
    assert np.allclose(scores, np.array([3.0, 5.0, 2.0]))


def test_score_region_candidates_pareto_uses_front_ranks():
    from optimizer.multi_turbo_enn_scoring import _score_region_candidates

    mu = np.array([[2.0], [1.0], [0.0]], dtype=float)
    sigma = np.array([[0.0], [2.0], [1.0]], dtype=float)
    child = _ChildDesigner(_Turbo(mu=mu, sigma=sigma, acq=_ParetoAcqOptimizer()))
    rng = np.random.default_rng(1)
    x_region = np.zeros((3, 2), dtype=float)
    scores = _score_region_candidates(
        child,
        x_region=x_region,
        acq_type="pareto",
        allow_random=False,
        region_rng=rng,
    )
    assert float(scores[0]) == float(scores[1])
    assert float(scores[0]) > float(scores[2])


def test_score_region_candidates_falls_back_when_surrogate_predict_fails():
    from optimizer.multi_turbo_enn_scoring import _score_region_candidates

    mu = np.array([[2.0], [1.0], [0.0]], dtype=float)
    sigma = np.array([[0.0], [2.0], [1.0]], dtype=float)
    child = _ChildDesigner(_BrokenTurbo(mu=mu, sigma=sigma, acq=_ParetoAcqOptimizer()))
    rng = np.random.default_rng(1)
    x_region = np.zeros((3, 2), dtype=float)
    scores = _score_region_candidates(
        child,
        x_region=x_region,
        acq_type="pareto",
        allow_random=False,
        region_rng=rng,
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_score_region_candidates_falls_back_when_surrogate_sample_fails():
    from optimizer.multi_turbo_enn_scoring import _score_region_candidates

    mu = np.array([[2.0], [1.0], [0.0]], dtype=float)
    sigma = np.array([[0.0], [2.0], [1.0]], dtype=float)
    child = _ChildDesigner(_BrokenTurbo(mu=mu, sigma=sigma, acq=_ThompsonAcqOptimizer()))
    rng = np.random.default_rng(1)
    x_region = np.zeros((3, 2), dtype=float)
    scores = _score_region_candidates(
        child,
        x_region=x_region,
        acq_type="thompson",
        allow_random=False,
        region_rng=rng,
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_assign_new_data_consumes_partial_pending_indices():
    from optimizer.multi_turbo_enn_designer import (
        MultiTurboENNConfig,
        MultiTurboENNDesigner,
        MultiTurboHarnessConfig,
        TurboENNRegionConfig,
    )
    from optimizer.multi_turbo_enn_utils import _tell_new_data_if_any

    class _MockPolicy:
        problem_seed = 0

        def clone(self):
            return self

        def num_params(self):
            return 1

    designer = MultiTurboENNDesigner(
        _MockPolicy(),
        config=MultiTurboENNConfig(
            harness=MultiTurboHarnessConfig(num_regions=2),
            region=TurboENNRegionConfig(turbo_mode="turbo-enn"),
        ),
    )
    state = designer._state
    state.region_data = [[], []]
    state.region_assignments = []
    state.last_region_indices = [0, 1, 1]
    state.num_told_global = 0
    _tell_new_data_if_any(designer, ["a", "b"])
    assert designer._state.region_data == [["a"], ["b"]]
    assert designer._state.region_assignments == [0, 1]
    assert designer._state.last_region_indices == [1]
    _tell_new_data_if_any(designer, ["a", "b", "c"])
    assert designer._state.region_data == [["a"], ["b", "c"]]
    assert designer._state.region_assignments == [0, 1, 1]
    assert designer._state.last_region_indices is None


def test_call_fixed_arms_appends_pending_region_indices():
    from optimizer.multi_turbo_enn_utils import _call_fixed_arms

    designer = type("D", (), {})()
    designer._per_region_counts = lambda _num_arms: [1, 1]
    designer._designers = [_FixedChild(2, 1), _FixedChild(2, 1)]
    designer._state = type("S", (), {"region_data": [[], []], "last_region_indices": [0]})()
    policies = _call_fixed_arms(designer, num_arms=2)
    assert len(policies) == 2
    assert designer._state.last_region_indices == [0, 0, 1]


def test_extract_tolerance_targets_avoids_uninitialized_failure_tolerance_property():
    from optimizer.multi_turbo_enn_utils import _extract_tolerance_targets

    class _TRState:
        num_dim = 3

        @property
        def failure_tolerance(self):
            raise RuntimeError("failure_tolerance not initialized")

        @failure_tolerance.setter
        def failure_tolerance(self, value):
            self._failure_tolerance = int(value)

    tr_state = _TRState()
    child = SimpleNamespace(_turbo=SimpleNamespace(_tr_state=tr_state))
    targets = _extract_tolerance_targets([child])
    assert len(targets) == 1
    assert targets[0].num_dim == 3
    assert targets[0].set_failure_tolerance is not None
    targets[0].set_failure_tolerance(6)
    assert tr_state._failure_tolerance == 6
