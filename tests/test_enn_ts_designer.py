"""Behavioral tests for EnnTSDesigner."""

from __future__ import annotations

import numpy as np

from optimizer.datum import Datum
from optimizer.trajectory import Trajectory


def _policy_and_data(n_obs: int = 0, dim: int = 2, seed: int = 0):
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=seed, noise_seed_0=seed + 1)
    policy = default_policy(env_conf)
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_obs):
        p = policy.clone()
        x = rng.uniform(-1.0, 1.0, size=dim)
        p.set_params(x)
        y = float(-np.sum(x**2))
        data.append(
            Datum(
                designer="test",
                policy=p,
                expected_acqf=0.0,
                trajectory=Trajectory(rreturn=y, states=np.array([]), actions=np.array([])),
            )
        )
    return policy, data


def test_registry_resolves_enn_ts():
    from optimizer.designers import Designers
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)
    d = Designers(policy, num_arms=2).create("enn-ts/num_ts_iters=4")
    assert d._num_ts_iters == 4
    assert d._k == 10
    assert d._num_fit_samples == 100


def test_empty_enn_first_call_returns_arms_in_bounds():
    from common import all_bounds as ab
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, _ = _policy_and_data(n_obs=0)
    designer = EnnTSDesigner(policy, num_ts_iters=2, num_init=4, k=3, num_fit_samples=4)
    out = designer([], num_arms=2)
    assert len(out) == 2
    assert len(designer._enn) == 0
    for p in out:
        x = np.asarray(p.get_params(), dtype=np.float64)
        assert np.all(np.isfinite(x))
        assert np.all(x >= ab.x_low - 1e-9)
        assert np.all(x <= ab.x_high + 1e-9)


def test_second_call_appends_to_enn():
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=3)
    designer = EnnTSDesigner(policy, num_ts_iters=1, num_init=4, k=2, num_fit_samples=4)
    out1 = designer(data[:2], num_arms=1)
    assert len(out1) == 1
    assert len(designer._enn) == 2
    out2 = designer(data, num_arms=1)
    assert len(out2) == 1
    assert len(designer._enn) == 3
    x = np.asarray(out2[0].get_params(), dtype=np.float64)
    assert np.all(np.isfinite(x))


def test_num_arms_two_invokes_maximize_twice(monkeypatch):
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=2)
    designer = EnnTSDesigner(policy, num_ts_iters=1, num_init=4, k=2, num_fit_samples=4)
    calls = []

    def _fake_maximize(seed_ts):
        calls.append(int(seed_ts))
        return np.zeros(policy.num_params(), dtype=np.float64)

    monkeypatch.setattr(designer, "_maximize_draw", _fake_maximize)
    out = designer(data, num_arms=2)
    assert len(out) == 2
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_init_tell_batch_lhd_plus_incumbents(monkeypatch):
    from optimizer import enn_ts_designer_impl as impl
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=3)
    designer = EnnTSDesigner(policy, num_ts_iters=0, num_init=4, k=2, num_fit_samples=4)
    designer(data, num_arms=1)  # populate ENN + params

    configs = []
    tells = []

    class FakeOpt:
        def tell(self, x, y):
            tells.append(int(np.asarray(x).shape[0]))

        def ask(self, n):
            raise AssertionError("should not ask when num_ts_iters=0")

    real_im = impl._im

    def fake_im(name):
        if name == "enn":
            return type(
                "enn",
                (),
                {"create_optimizer": staticmethod(lambda **kw: FakeOpt())},
            )()
        if name == "enn.turbo.config.factory":
            return type(
                "factory",
                (),
                {"turbo_zero_config": staticmethod(lambda **kw: configs.append(dict(kw)) or object())},
            )()
        return real_im(name)

    monkeypatch.setattr(impl, "_im", fake_im)
    x = designer._maximize_draw(seed_ts=11)
    assert x.shape == (policy.num_params(),)
    assert len(configs) == 1
    assert configs[0]["num_init"] == tells[0]
    from enn.turbo.config.candidate_rv import CandidateRV

    assert configs[0]["candidate_rv"] is CandidateRV.RAASP
    assert "num_candidates" not in configs[0]
    # num_init=4 → 2 LHD + min(3,2)=2 incumbents → 4
    assert tells[0] == 4


def test_draw_values_never_added_to_enn(monkeypatch):
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=2)
    designer = EnnTSDesigner(policy, num_ts_iters=1, num_init=4, k=2, num_fit_samples=4)
    designer(data, num_arms=1)
    add_calls = []
    real_add = designer._enn.add

    def spy_add(x, y, yvar=None):
        add_calls.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return real_add(x, y, yvar)

    monkeypatch.setattr(designer._enn, "add", spy_add)
    n_before = len(designer._enn)
    designer._maximize_draw(seed_ts=3)
    assert add_calls == []
    assert len(designer._enn) == n_before


def test_empty_enn_init_progress_clears_after_external_tell():
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, _ = _policy_and_data(n_obs=0)
    designer = EnnTSDesigner(policy, num_ts_iters=1, num_init=4, k=2, num_fit_samples=4)
    designer._ensure_enn()
    designer._fit_params()

    # Instrument maximize to check init_progress after external tell.
    import enn
    from enn.turbo.config.factory import turbo_zero_config

    seed_ts = 5
    init_X = designer._build_init_X(seed_ts, num_init=4)
    assert init_X.shape[0] == 2  # LHD only when empty
    from enn.turbo.config.candidate_rv import CandidateRV

    cfg = turbo_zero_config(num_init=len(init_X), candidate_rv=CandidateRV.RAASP)
    opt = enn.create_optimizer(bounds=designer._bounds, config=cfg, rng=designer._rng)
    assert opt.init_progress == (0, len(init_X))
    y = designer._draw_values(init_X, seed_ts)
    opt.tell(init_X, y)
    assert opt.init_progress is None
    x1 = opt.ask(1)
    assert np.asarray(x1).shape[0] == 1


def test_inner_iter_tells_draw_argmax_of_candidate_pool(monkeypatch):
    from optimizer import enn_ts_designer_impl as impl
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=2)
    designer = EnnTSDesigner(policy, num_ts_iters=1, num_init=4, k=2, num_fit_samples=4)
    designer(data, num_arms=1)

    asks = []
    tells = []
    pool = np.array([[0.0, 0.0], [0.5, 0.5], [0.9, -0.9]], dtype=np.float64)
    # Draw scores will be set so index 1 wins.
    draw_scores = {0: 0.1, 1: 9.0, 2: 2.0}

    class FakeOpt:
        def __init__(self):
            self._told_init = False

        def tell(self, x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            tells.append((x.copy(), y.copy()))
            if not self._told_init:
                self._told_init = True
                return
            assert x.shape[0] == 1

        def ask(self, n):
            asks.append(int(n))
            assert n == pool.shape[0]
            return pool.copy()

    class FakeCandCfg:
        def resolve_num_candidates(self, *, num_dim, num_arms):
            return pool.shape[0]

    real_im = impl._im
    real_draw = designer._draw_values

    def fake_draw(x, seed_ts):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == pool.shape[0] and np.allclose(x, pool):
            return np.array([[draw_scores[i]] for i in range(pool.shape[0])], dtype=np.float64)
        return real_draw(x, seed_ts)

    def fake_im(name):
        if name == "enn":
            return type(
                "enn",
                (),
                {"create_optimizer": staticmethod(lambda **kw: FakeOpt())},
            )()
        if name == "enn.turbo.config.factory":
            return type(
                "factory",
                (),
                {
                    "turbo_zero_config": staticmethod(
                        lambda **kw: type(
                            "cfg",
                            (),
                            {"candidates": FakeCandCfg()},
                        )()
                    )
                },
            )()
        return real_im(name)

    monkeypatch.setattr(impl, "_im", fake_im)
    monkeypatch.setattr(designer, "_draw_values", fake_draw)
    x = designer._maximize_draw(seed_ts=7)
    assert len(asks) == 1
    assert asks[0] == pool.shape[0]
    # last tell is the argmax of the pool (index 1)
    x_told, y_told = tells[-1]
    assert np.allclose(x_told, pool[1:2])
    assert float(y_told.reshape(-1)[0]) == 9.0
    assert np.allclose(x, pool[1])


def test_fewer_obs_than_incumbent_half():
    from optimizer.enn_ts_designer import EnnTSDesigner

    policy, data = _policy_and_data(n_obs=1)
    designer = EnnTSDesigner(policy, num_ts_iters=0, num_init=6, k=2, num_fit_samples=4)
    designer(data, num_arms=1)
    init_X = designer._build_init_X(seed_ts=9, num_init=6)
    # LHD half=3, incumbents=min(1,3)=1 → 4
    assert init_X.shape[0] == 4
