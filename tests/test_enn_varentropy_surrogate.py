from __future__ import annotations

import numpy as np


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, size=(24, 4))
    y = (1.4 * x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2]).reshape(-1, 1)
    q = rng.uniform(0.0, 1.0, size=(7, 4))
    return x, y, q


def test_enn_varentropy_surrogate_zero_scale_matches_enn_posterior():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    from optimizer.enn_varentropy_config import ENNVarentropySurrogateConfig
    from optimizer.enn_varentropy_surrogate import ENNVarentropySurrogate

    x, y, q = _training_data()
    surrogate = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=5, scale_x=False, varentropy_scale=0.0))
    surrogate.fit(x, y)

    posterior = surrogate.predict(q)
    enn = EpistemicNearestNeighbors(x, y, scale_x=False)
    expected = enn.posterior(
        q,
        params=ENNParams(
            k_num_neighbors=5,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        ),
    )

    np.testing.assert_allclose(posterior.mu[:, 0], np.asarray(expected.mu).reshape(-1), atol=1e-7)
    np.testing.assert_allclose(posterior.sigma[:, 0], np.asarray(expected.se).reshape(-1), atol=1e-7)


def test_enn_varentropy_surrogate_inflates_sigma_without_changing_mean():
    from optimizer.enn_varentropy_config import ENNVarentropySurrogateConfig
    from optimizer.enn_varentropy_surrogate import ENNVarentropySurrogate

    x, y, q = _training_data()
    base = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=5, scale_x=False, varentropy_scale=0.0))
    inflated = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=5, scale_x=False, varentropy_scale=1.0))
    base.fit(x, y)
    inflated.fit(x, y)

    base_post = base.predict(q)
    inflated_post = inflated.predict(q)

    np.testing.assert_allclose(inflated_post.mu, base_post.mu, atol=1e-10)
    assert np.all(inflated_post.sigma >= base_post.sigma)
    assert np.any(inflated_post.sigma > base_post.sigma)


def test_enn_varentropy_surrogate_appends_incremental_fit_prefix():
    from optimizer.enn_varentropy_config import ENNVarentropySurrogateConfig
    from optimizer.enn_varentropy_surrogate import ENNVarentropySurrogate

    x, y, q = _training_data()
    surrogate = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=4, scale_x=False, varentropy_scale=0.5))
    surrogate.fit(x[:10], y[:10])
    assert surrogate._model is not None
    enn_model = surrogate._model.enn_model

    surrogate.fit(x[:14], y[:14])
    posterior = surrogate.predict(q[:3])

    assert surrogate._model is not None
    assert surrogate._model.enn_model is enn_model
    assert surrogate._model.train_x.shape == (14, x.shape[1])
    assert posterior.mu.shape == (3, 1)
    assert np.all(np.isfinite(posterior.sigma))


def test_enn_varentropy_leave_one_out_does_not_self_predict_training_value():
    from optimizer.enn_varentropy_config import ENNVarentropySurrogateConfig
    from optimizer.enn_varentropy_surrogate import ENNVarentropySurrogate

    x, y, _q = _training_data()
    surrogate = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=4, scale_x=False, varentropy_scale=0.0))
    surrogate.fit(x[:12], y[:12])

    in_sample = surrogate.predict(x[:3])
    loo = surrogate.predict_leave_one_out(x[:3])

    np.testing.assert_allclose(in_sample.mu, y[:3], atol=1e-7)
    assert np.max(np.abs(loo.mu - y[:3])) > 1e-4
    assert np.all(np.isfinite(loo.sigma))


def test_enn_varentropy_surrogate_sample_shape():
    from optimizer.enn_varentropy_config import ENNVarentropySurrogateConfig
    from optimizer.enn_varentropy_surrogate import ENNVarentropySurrogate

    rng = np.random.default_rng(13)
    x, y, q = _training_data()
    surrogate = ENNVarentropySurrogate(ENNVarentropySurrogateConfig(k=3))
    surrogate.fit(x, y, rng=rng)

    samples = surrogate.sample(q, 4, rng)

    assert samples.shape == (4, q.shape[0], 1)
    assert np.all(np.isfinite(samples))
