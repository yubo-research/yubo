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


class _UCBAcqOptimizer:
    def __init__(self, beta: float):
        self._beta = float(beta)


class _ParetoAcqOptimizer:
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
        out = []
        for _ in range(self._n_emit):
            out.append(_Policy(np.zeros((self._n_params,), dtype=float)))
        return out


class _Policy:
    def __init__(self, params):
        self._params = np.asarray(params, dtype=float)

    def get_params(self):
        return self._params


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


def test_score_region_candidates_raises_on_surrogate_error():
    from optimizer.multi_turbo_enn_scoring import _score_region_candidates

    mu = np.array([[2.0], [1.0], [0.0]], dtype=float)
    sigma = np.array([[0.0], [2.0], [1.0]], dtype=float)
    child = _ChildDesigner(_BrokenTurbo(mu=mu, sigma=sigma, acq=_ParetoAcqOptimizer()))
    rng = np.random.default_rng(1)
    x_region = np.zeros((3, 2), dtype=float)
    with pytest.raises(RuntimeError, match="predict failed"):
        _score_region_candidates(
            child,
            x_region=x_region,
            acq_type="pareto",
            allow_random=False,
            region_rng=rng,
        )


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
