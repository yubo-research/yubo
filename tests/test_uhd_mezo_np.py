"""Tests for UHDMeZONp and UHDMeZOBENp (numpy-based MeZO optimizers)."""

import numpy as np
import pytest


class _MockPolicy:
    """Mock policy for testing."""

    def __init__(self, dim=12):
        self._params = np.zeros(dim, dtype=np.float64)
        self._last_params = None

    def get_params(self):
        return self._params.copy()

    def set_params(self, x):
        self._last_params = np.asarray(x, dtype=np.float64).copy()
        self._params = self._last_params


class _MockEmbedder:
    """Mock behavioral embedder for testing."""

    def __init__(self, embed_dim=8):
        self._embed_dim = embed_dim

    def embed_policy(self, policy, x):
        # Deterministic embedding based on params
        params = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(int(np.sum(params) * 1000) % 2**31)
        return rng.standard_normal(self._embed_dim)


def test_uhd_mezo_np_basic():
    from optimizer.uhd_mezo_np import UHDMeZONp

    policy = _MockPolicy(dim=12)
    uhd = UHDMeZONp(policy, sigma=0.001, lr=0.001, beta=0.9, param_clip=(-1.0, 1.0))

    assert uhd.eval_seed == 0
    assert uhd.sigma == 0.001
    assert uhd.y_best is None
    assert uhd.positive_phase is True

    # First ask (positive phase)
    uhd.ask()
    assert policy._last_params is not None
    assert uhd.positive_phase is True

    # Tell with some reward
    uhd.tell(-10.0, 0.0)
    assert uhd.y_best == -10.0
    assert uhd.positive_phase is False  # Now in negative phase

    # Second ask (negative phase)
    uhd.ask()
    assert uhd.positive_phase is False

    # Tell with negative phase reward
    uhd.tell(-12.0, 0.0)
    assert uhd.y_best == -10.0  # Should keep best
    assert uhd.positive_phase is True  # Back to positive
    assert uhd.eval_seed == 1  # Seed incremented


def test_uhd_mezo_np_improves():
    from optimizer.uhd_mezo_np import UHDMeZONp

    policy = _MockPolicy(dim=12)
    uhd = UHDMeZONp(policy, sigma=0.001, lr=0.001, beta=0.9, param_clip=(-1.0, 1.0))

    # Run a few iterations with improving rewards
    rewards = [-20.0, -18.0, -15.0, -10.0]
    for i, r in enumerate(rewards):
        uhd.ask()
        uhd.tell(r, 0.0)
        uhd.ask()
        uhd.tell(r - 1.0, 0.0)  # Worse negative phase

    assert uhd.y_best == -10.0


def test_uhd_mezo_be_np_basic():
    from optimizer.uhd_mezo_np import UHDMeZOBENp

    policy = _MockPolicy(dim=12)
    embedder = _MockEmbedder(embed_dim=8)

    uhd = UHDMeZOBENp(
        policy,
        embedder,
        sigma=0.001,
        lr=0.001,
        beta=0.9,
        param_clip=(-1.0, 1.0),
        num_candidates=5,
        warmup=2,
        fit_interval=1,
        enn_k=3,
    )

    assert uhd.eval_seed == 0
    assert uhd.sigma == 0.001
    assert uhd.y_best is None
    assert uhd.positive_phase is True

    # Warmup phase (no ENN selection yet)
    for i in range(4):
        uhd.ask()
        assert uhd.positive_phase is True
        uhd.tell(-10.0 + i, 0.0)  # Improving
        uhd.ask()
        uhd.tell(-11.0 + i, 0.0)

    assert uhd.y_best == -7.0


def test_uhd_mezo_np_set_next_seed():
    from optimizer.uhd_mezo_np import UHDMeZONp

    policy = _MockPolicy(dim=12)
    uhd = UHDMeZONp(policy, sigma=0.001, lr=0.001)

    # Can set seed during positive phase
    uhd.set_next_seed(42)
    assert uhd.eval_seed == 42

    # After ask/tell, should be in negative phase
    uhd.ask()
    uhd.tell(-10.0, 0.0)
    assert uhd.positive_phase is False

    # Cannot set seed during negative phase
    with pytest.raises(RuntimeError):
        uhd.set_next_seed(100)


def test_uhd_mezo_np_properties():
    from optimizer.uhd_mezo_np import UHDMeZONp

    policy = _MockPolicy(dim=12)
    uhd = UHDMeZONp(policy, sigma=0.001, lr=0.001)

    # Check properties
    assert isinstance(uhd.eval_seed, int)
    assert isinstance(uhd.sigma, float)
    assert uhd.mu_avg == 0.0  # Initially
    assert uhd.se_avg == 0.0  # Initially

    uhd.ask()
    uhd.tell(-5.0, 0.5)

    assert uhd.mu_avg == -5.0
    assert uhd.se_avg == 0.5
    assert uhd.y_best == -5.0


def test_uhd_mezo_be_np_warmup():
    from optimizer.uhd_mezo_np import UHDMeZOBENp

    policy = _MockPolicy(dim=12)
    embedder = _MockEmbedder(embed_dim=8)

    uhd = UHDMeZOBENp(
        policy,
        embedder,
        sigma=0.001,
        lr=0.001,
        num_candidates=3,
        warmup=3,
        fit_interval=10,
        enn_k=2,
    )

    # During warmup, ENN doesn't select (not enough data)
    for i in range(3):
        uhd.ask()
        uhd.tell(float(-10 - i), 0.0)
        uhd.ask()
        uhd.tell(float(-11 - i), 0.0)

    # Should have collected some observations
    assert len(uhd._zs) >= 3
    assert len(uhd._ys) >= 3
