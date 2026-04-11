import types

import numpy as np

from optimizer.enn_turbo_optimizer import HnRAcquisitionOptimizer, TurboOptimizer


class _FakePosteriorENN:
    def __init__(self):
        self.calls = 0

    def posterior_function_draw(self, x, params, function_seeds):
        del params
        self.calls += 1
        x = np.asarray(x, dtype=float)
        function_seeds = np.asarray(function_seeds, dtype=np.int64)
        samples = x[None, :, :1] + function_seeds[:, None, None] * 1e-9
        return samples, None


class _FakeThompsonSurrogate:
    def __init__(self):
        self._enn = _FakePosteriorENN()
        self._params = object()
        self.sample_calls = 0

    def sample(self, x, num_samples, rng):
        self.sample_calls += 1
        base_seed = rng.integers(0, 2**31)
        function_seeds = np.arange(base_seed, base_seed + num_samples, dtype=np.int64)
        samples, _ = self._enn.posterior_function_draw(x, self._params, function_seeds=function_seeds)
        return samples


class _FakeUCBSurrogate:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, x):
        self.predict_calls += 1
        x = np.asarray(x, dtype=float)
        mu = np.sum(x, axis=1, keepdims=True)
        sigma = np.full_like(mu, 0.5)
        return types.SimpleNamespace(mu=mu, sigma=sigma)


def test_hnr_thompson_batch_scores_match_single_point_seeded_semantics():
    from enn.turbo.components.acquisition import ThompsonAcqOptimizer

    surrogate = _FakeThompsonSurrogate()
    optimizer = HnRAcquisitionOptimizer(ThompsonAcqOptimizer(), num_iterations=0)
    x_batch = np.array([[0.1, 0.2], [0.4, 0.3], [0.8, 0.5]], dtype=float)
    arm_seeds = np.array([7, 11, 13], dtype=np.int64)

    got = optimizer._score_thompson(x_batch, surrogate, arm_seeds)
    expected = []
    for x_pt, seed in zip(x_batch, arm_seeds, strict=False):
        fixed_rng = np.random.default_rng(int(seed))
        sample = surrogate.sample(x_pt.reshape(1, -1), 1, fixed_rng)
        expected.append(float(sample[0, 0, 0]))

    np.testing.assert_allclose(got, np.array(expected, dtype=float))


def test_hnr_ucb_batches_predict_calls_across_arms_and_iterations():
    from enn.turbo.components.acquisition import UCBAcqOptimizer

    surrogate = _FakeUCBSurrogate()
    optimizer = HnRAcquisitionOptimizer(UCBAcqOptimizer(beta=1.5), num_iterations=5)
    rng = np.random.default_rng(0)
    x_cand = rng.uniform(size=(32, 4))

    selected = optimizer.select(x_cand, 3, surrogate, rng)

    assert selected.shape == (3, 4)
    assert surrogate.predict_calls == 6


def test_hnr_thompson_select_uses_batched_posterior_draws():
    from enn.turbo.components.acquisition import ThompsonAcqOptimizer

    surrogate = _FakeThompsonSurrogate()
    optimizer = HnRAcquisitionOptimizer(ThompsonAcqOptimizer(), num_iterations=4)
    rng = np.random.default_rng(1)
    x_cand = rng.uniform(size=(24, 3))

    selected = optimizer.select(x_cand, 2, surrogate, rng)

    assert selected.shape == (2, 3)
    assert surrogate._enn.calls == 5
    assert surrogate.sample_calls == 0


def test_sobol_engine_is_reused_and_reset_for_same_state(monkeypatch):
    class _FakeSobol:
        init_calls = 0

        def __init__(self, d, scramble, seed):
            self.d = d
            self.scramble = scramble
            self.seed = seed
            self.reset_calls = 0
            _FakeSobol.init_calls += 1

        def reset(self):
            self.reset_calls += 1

    monkeypatch.setattr("scipy.stats.qmc.Sobol", _FakeSobol)
    monkeypatch.setattr(
        "optimizer.enn_turbo_optimizer.turbo_optimizer_utils.sobol_seed_for_state",
        lambda _seed_base, *, num_arms, **_kwargs: num_arms,
    )
    optimizer = TurboOptimizer.__new__(TurboOptimizer)
    optimizer._num_dim = 5
    optimizer._tr_state = object()
    optimizer._sobol_seed_base = 0
    optimizer._restart_generation = 3
    optimizer._x_obs = [None, None]
    optimizer._sobol_engine = None
    optimizer._sobol_engine_key = None

    first = optimizer._sobol_engine_for(2)
    second = optimizer._sobol_engine_for(2)
    third = optimizer._sobol_engine_for(4)

    assert first is second
    assert first.reset_calls == 1
    assert _FakeSobol.init_calls == 2
    assert third is not first
