import sys
import types

import numpy as np

from optimizer.bayesian_mars_fit import _fit_bayesian_mars
from optimizer.bayesian_mars_mcmc import (
    _bayesian_linear_log_marginal,
    _build_mcmc_basis_pool,
    _fit_bayesian_mars_mcmc,
    _mcmc_move_probs,
    _propose_mcmc_basis_state,
)
from optimizer.mars_basis import (
    _build_main_basis,
    _coerce_scalar_y,
    _HingeFactor,
    _MarsTerm,
    _ridge_solve,
    _safe_solve,
    _score_columns,
    _screen_features,
    _select_top,
    _standardize_y,
    _standardized_y_var,
)
from optimizer.mars_config import BayesianMarsSurrogateConfig, ENNMarsGeometrySurrogateConfig, MarsSurrogateConfig
from optimizer.mars_fit import _fit_single_mars
from optimizer.mars_geometry import _low_rank_factor_from_isotropic_spectrum
from optimizer.mars_surrogate import BayesianMarsSurrogate, ENNMarsGeometrySurrogate, MarsSurrogate


def _term_signature(term):
    return tuple((factor.feature, factor.knot, factor.side) for factor in term.factors)


def _reference_main_basis(x, features, quantiles):
    terms = []
    cols = []
    for feature in features:
        knots = np.unique(np.quantile(x[:, int(feature)], quantiles))
        for knot in knots:
            if np.isfinite(knot):
                _append_reference_terms(x, terms, cols, feature, knot)
    return terms, cols


def _append_reference_terms(x, terms, cols, feature, knot):
    for side in (1, -1):
        term = _MarsTerm((_HingeFactor(int(feature), float(knot), int(side)),))
        col = term.eval(x)
        if np.linalg.norm(col - float(np.mean(col))) > 1e-12:
            terms.append(term)
            cols.append(col)


def _linear_data(seed=0, n=56, d=4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, d))
    y = (1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2]).reshape(-1, 1)
    return rng, x, y


def _small_mars_config(**kwargs):
    params = dict(max_terms=12, num_bootstrap=3, active_rank=2, active_samples=32, feature_screen=8)
    params.update(kwargs)
    return MarsSurrogateConfig(**params)


def test_mars_main_basis_vectorized_matches_reference_loop():
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, size=(37, 9))
    x[:, 7] = 0.5
    features = np.array([4, 7, 1, 2], dtype=np.int64)
    quantiles = np.linspace(0.2, 0.8, 3)

    terms, cols = _build_main_basis(x, features, quantiles)
    ref_terms, ref_cols = _reference_main_basis(x, features, quantiles)

    assert [_term_signature(term) for term in terms] == [_term_signature(term) for term in ref_terms]
    assert len(cols) == len(ref_cols)
    for got, expected in zip(cols, ref_cols, strict=True):
        np.testing.assert_allclose(got, expected)


def test_mars_basis_helpers_handle_scalar_targets_and_ranked_columns():
    x = np.array([[0.0, 0.1], [0.5, 0.4], [1.0, 0.9]])
    y = np.array([[1.0], [2.0], [4.0]])
    y_std, y_mean, y_scale = _standardize_y(_coerce_scalar_y(y))
    cols = np.column_stack([x[:, 0], x[:, 1]])

    score = _score_columns(cols, y_std)
    selected = _select_top(score, 1)
    features = _screen_features(x, y_std, 1)
    coef = _ridge_solve(np.column_stack([np.ones(3), x[:, 0]]), y_std, 1e-8)
    solved = _safe_solve(np.eye(2), np.array([1.0, 2.0]))

    assert y_mean == np.mean(y[:, 0])
    assert y_scale > 0.0
    assert selected.shape == (1,)
    assert features.shape == (1,)
    assert coef.shape == (2,)
    np.testing.assert_allclose(solved, [1.0, 2.0])
    np.testing.assert_allclose(_standardized_y_var(np.full((3, 1), y_scale**2), y_scale), np.ones(3))


def test_fit_single_mars_predicts_signal_and_active_features():
    rng, x, y = _linear_data(seed=5)
    cfg = _small_mars_config(interaction_order=1)
    model = _fit_single_mars(x, y[:, 0], cfg)
    pred = model.predict(x[:10])
    low_rank = model.active_low_rank_factor(cfg, rng)

    assert pred.shape == (10,)
    assert np.corrcoef(pred, y[:10, 0])[0, 1] > 0.95
    assert set(model.active_features()).issubset(set(range(x.shape[1])))
    assert low_rank is not None
    assert low_rank.basis.shape[0] == x.shape[1]


def test_mars_surrogate_fit_predicts_mu_and_bootstrap_sigma():
    rng, x, y = _linear_data(seed=0, n=48)
    surrogate = MarsSurrogate(_small_mars_config(feature_screen=4))

    surrogate.fit(x, y, rng=rng)
    posterior = surrogate.predict(x[:7])
    samples = surrogate.sample(x[:7], 4, rng)

    assert posterior.mu.shape == (7, 1)
    assert posterior.sigma.shape == (7, 1)
    assert samples.shape == (4, 7, 1)
    assert np.all(np.isfinite(posterior.mu))
    assert np.all(posterior.sigma >= 0.0)


def test_mars_surrogate_builds_active_low_rank_factor_and_updates_trust_region():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(64, 5))
    y = (3.0 * x[:, 2] - 0.5 * x[:, 3]).reshape(-1, 1)
    surrogate = MarsSurrogate(_small_mars_config(max_terms=10, num_bootstrap=2, feature_screen=5))
    tr_state = types.SimpleNamespace(received=None)
    tr_state.set_low_rank_factor = lambda low_rank: setattr(tr_state, "received", low_rank)

    surrogate.fit(x, y, rng=rng)
    low_rank = surrogate.active_low_rank_factor(rng)
    surrogate.update_trust_region(tr_state, x[0], y, 0, rng)

    assert low_rank is not None
    assert tr_state.received is not None
    assert 1 <= low_rank.basis.shape[1] <= 2
    assert float(low_rank.sqrt_alpha) > 0.0


def test_low_rank_factor_rejects_bad_spectrum_and_shapes():
    basis = np.eye(3, 2)
    low_rank = _low_rank_factor_from_isotropic_spectrum(
        alpha_base=0.0,
        basis=basis,
        extra_eigvals=np.array([2.0, 1.0]),
        dim=3,
        lam_min=1e-4,
        lam_max=1e4,
        eps=1e-6,
        rank_cap=1,
    )

    assert low_rank is not None
    assert low_rank.basis.shape == (3, 1)
    assert (
        _low_rank_factor_from_isotropic_spectrum(
            alpha_base=0.0,
            basis=basis,
            extra_eigvals=np.array([np.nan]),
            dim=3,
            lam_min=1e-4,
            lam_max=1e4,
            eps=1e-6,
            rank_cap=1,
        )
        is None
    )


def test_bayesian_mars_fit_predicts_posterior_and_uses_observation_variance():
    rng, x, y = _linear_data(seed=3)
    y_var = np.full_like(y, 0.05 * float(np.std(y[:, 0])) ** 2)
    cfg = BayesianMarsSurrogateConfig(
        basis=_small_mars_config(interaction_order=1, feature_screen=4),
        prior_precision=1.0,
        min_noise_variance=1e-6,
    )

    model = _fit_bayesian_mars(x, y[:, 0], cfg, y_var=y_var)
    mu, sigma = model.predict(x[:8])
    samples = model.sample(x[:8], 5, rng)

    np.testing.assert_allclose(model.noise_variance, 0.05)
    assert mu.shape == (8,)
    assert sigma.shape == (8,)
    assert samples.shape == (5, 8)
    assert np.all(sigma >= 0.0)


def test_bayesian_mars_surrogate_fit_predicts_posterior_and_samples():
    rng, x, y = _linear_data(seed=4)
    surrogate = BayesianMarsSurrogate(
        BayesianMarsSurrogateConfig(
            basis=_small_mars_config(interaction_order=1, feature_screen=4),
            min_noise_variance=1e-6,
        )
    )

    surrogate.fit(x, y, rng=rng)
    posterior = surrogate.predict(x[:8])
    samples = surrogate.sample(x[:8], 5, rng)

    assert posterior.mu.shape == (8, 1)
    assert posterior.sigma.shape == (8, 1)
    assert samples.shape == (5, 8, 1)
    assert np.std(samples[:, 0, 0]) > 0.0


def test_bayesian_mars_mcmc_averages_basis_structures_and_post_burn_in_steps():
    rng = np.random.default_rng(33)
    x = rng.uniform(0.0, 1.0, size=(64, 5))
    y = (1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.4 * x[:, 2] * x[:, 3]).reshape(-1, 1)
    cfg = BayesianMarsSurrogateConfig(
        basis=_small_mars_config(max_terms=8, interaction_order=2, active_samples=24, feature_screen=5),
        basis_sampler="mcmc",
        mcmc_steps=32,
        mcmc_burn_in=32,
        mcmc_thin=4,
        mcmc_num_models=8,
        mcmc_pool_size=16,
        mcmc_term_prior=0.2,
        min_noise_variance=1e-6,
    )

    result = _fit_bayesian_mars_mcmc(x, y[:, 0], cfg, rng=rng)
    surrogate = BayesianMarsSurrogate(cfg)
    surrogate.fit(x, y, rng=rng)
    posterior = surrogate.predict(x[:6])

    assert 1 < len(result.models) <= 8
    np.testing.assert_allclose(np.sum(result.weights), 1.0)
    assert 1 < len(surrogate._models) <= 8
    assert posterior.mu.shape == (6, 1)
    assert np.all(posterior.sigma >= 0.0)


def test_bayesian_mars_mcmc_helpers_score_and_propose_basis_states():
    rng, x, y = _linear_data(seed=6, n=40, d=4)
    cfg = BayesianMarsSurrogateConfig(basis=_small_mars_config(max_terms=6, feature_screen=4), mcmc_pool_size=8)
    y_std, _, _ = _standardize_y(y[:, 0])
    terms, cols = _build_mcmc_basis_pool(x, y_std, cfg)
    phi_full = np.column_stack([np.ones(x.shape[0]), cols[:, : min(2, cols.shape[1])]])
    phi_intercept = np.ones((x.shape[0], 1))
    proposed, log_q_forward, log_q_reverse = _propose_mcmc_basis_state((0,), pool_size=max(len(terms), 2), max_terms=3, rng=rng)

    assert terms
    assert cols.shape[0] == x.shape[0]
    assert _bayesian_linear_log_marginal(phi_full, y_std, prior_precision=1.0, intercept_prior_precision=1e-8, noise_variance=0.1) > (
        _bayesian_linear_log_marginal(phi_intercept, y_std, prior_precision=1.0, intercept_prior_precision=1e-8, noise_variance=0.1)
    )
    assert _mcmc_move_probs(0, max_terms=3, pool_size=4) == (1.0, 0.0)
    assert isinstance(proposed, tuple)
    assert np.isfinite(log_q_forward)
    assert np.isfinite(log_q_reverse)


class _DummyENNSurrogate:
    def __init__(self, config):
        self.config = config
        self.fit_calls = 0

    @property
    def lengthscales(self):
        return None

    def fit(self, x_obs, y_obs, y_var=None, *, num_steps=0, rng=None):
        from enn.turbo.python_fallback.components import SurrogateResult

        self.fit_calls += 1
        return SurrogateResult(model=object(), lengthscales=None)

    def get_incumbent_candidate_indices(self, y_obs):
        return np.arange(np.asarray(y_obs).shape[0], dtype=np.int64)

    def find_x_center(self, x_obs, y_obs, tr_state, rng):
        return np.asarray(x_obs)[0]

    def predict(self, x):
        from enn.turbo.python_fallback.components import PosteriorResult

        x = np.asarray(x, dtype=float)
        return PosteriorResult(mu=np.zeros((x.shape[0], 1)), sigma=np.ones((x.shape[0], 1)))

    def sample(self, x, num_samples, rng):
        return np.zeros((int(num_samples), np.asarray(x).shape[0], 1), dtype=float)


def test_enn_mars_geometry_uses_enn_posterior_and_mars_geometry(monkeypatch):
    module = types.ModuleType("optimizer.enn_surrogate_ext")
    module.GeometryENNSurrogate = _DummyENNSurrogate
    monkeypatch.setitem(sys.modules, "optimizer.enn_surrogate_ext", module)
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, size=(40, 4))
    y = (x[:, 0] - x[:, 1]).reshape(-1, 1)
    surrogate = ENNMarsGeometrySurrogate(
        ENNMarsGeometrySurrogateConfig(
            enn=object(),
            mars=_small_mars_config(max_terms=8, num_bootstrap=2, active_samples=16, feature_screen=4),
        )
    )

    surrogate.fit(x, y, rng=rng)
    posterior = surrogate.predict(x[:3])

    assert posterior.mu.shape == (3, 1)
    assert np.all(posterior.sigma == 1.0)
    assert len(surrogate._mars._ensemble) == 1
