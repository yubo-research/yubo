import sys
from types import SimpleNamespace

import numpy as np


def test_eggroll_solver_accepts_muon_when_optax_exposes_it():
    from optimizer.eggroll_designer import _solver, _solver_kwargs

    def dummy_muon(*args, **kwargs):
        return args, kwargs

    optax_mod = SimpleNamespace(
        adam=object(),
        adamw=object(),
        sgd=object(),
        contrib=SimpleNamespace(muon=dummy_muon),
    )

    assert _solver(optax_mod, "muon", b1=0.9, b2=0.999, weight_decay=0.1) is dummy_muon
    assert _solver_kwargs("muon", b1=0.9, b2=0.999, weight_decay=0.1) == {"weight_decay": 0.1}


def test_eggroll_solver_rejects_muon_when_optax_does_not_expose_it():
    import pytest

    from optimizer.designer_errors import NoSuchDesignerError
    from optimizer.eggroll_designer import _solver

    optax_mod = SimpleNamespace(adam=object(), adamw=object(), sgd=object())

    with pytest.raises(NoSuchDesignerError, match="adam, adamw, sgd"):
        _solver(optax_mod, "muon", b1=0.9, b2=0.999, weight_decay=0.1)


class _FakeNanoEggObjective:
    def __init__(self, dim: int):
        self.x0 = np.zeros((dim,), dtype=np.float64)
        self.closed = False
        self.jnp = np
        self.jax = SimpleNamespace(block_until_ready=lambda value: value)

    def sample_eggroll_noiser_noise(self, _x, *, seed, noiser_name, rank, group_size, freeze_nonlora):
        assert noiser_name == "eggroll"
        assert rank == 1
        assert group_size == 0
        assert freeze_nonlora is False
        direction = np.zeros_like(self.x0)
        direction[int(seed) % direction.size] = 1.0
        return direction

    def evaluate(self, x, *, seed):
        _ = seed
        return float(np.sum(np.asarray(x, dtype=np.float64))), 0.0

    def evaluate_many(self, x_batch, *, seed):
        mus, ses = [], []
        for i, x in enumerate(np.asarray(x_batch, dtype=np.float64)):
            mu, se = self.evaluate(x, seed=int(seed) + i)
            mus.append(mu)
            ses.append(se)
        return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)

    def make_policy(self, x):
        return SimpleNamespace(x=np.asarray(x, dtype=np.float64).copy(), params={"x": np.asarray(x).copy()})

    def close(self):
        self.closed = True


class _FakeNanoEggPolicy:
    is_nanoegg_pretrain_policy = True
    problem_seed = 0

    def __init__(self):
        self.objective = _FakeNanoEggObjective(dim=4)
        self.x = np.zeros((4,), dtype=np.float64)

    def make_objective(self, **kwargs):
        assert kwargs["search_dim"] == 4
        assert kwargs["delta_scale"] == 1.0
        return self.objective

    def with_snapshot(self, snapshot):
        policy = _FakeNanoEggPolicy()
        policy.objective = self.objective
        policy.x = np.asarray(snapshot.x, dtype=np.float64).copy()
        return policy

    def clone(self):
        policy = _FakeNanoEggPolicy()
        policy.objective = self.objective
        policy.x = self.x.copy()
        return policy

    def num_params(self):
        return int(self.x.size)


class _FakeOptaxTx:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def init(self, _params):
        return {}

    def update(self, grad, state, _params):
        return -self.learning_rate * np.asarray(grad), state


def _fake_optax():
    def factory(learning_rate, **_kwargs):
        return _FakeOptaxTx(learning_rate)

    return SimpleNamespace(
        adam=factory,
        adamw=factory,
        sgd=factory,
        apply_updates=lambda params, updates: np.asarray(params) + np.asarray(updates),
    )


def test_eggroll_designer_handles_nanoegg_policy_without_separate_designer(monkeypatch):
    from optimizer.eggroll_designer import EggRollDesigner

    monkeypatch.setitem(sys.modules, "optax", _fake_optax())
    policy = _FakeNanoEggPolicy()
    designer = EggRollDesigner(
        policy,
        env_conf=SimpleNamespace(env_name="pretrain:nanoegg:synthetic"),
        noiser="eggroll",
        sigma=0.1,
        lr=0.01,
        rank=1,
        steps=1,
        num_envs=1,
        search_dim=4,
        delta_scale=1.0,
    )

    result = designer.iterate([], 2)

    assert result.data[0].policy.x.shape == (4,)
    assert result.data[0].trajectory.rreturn > 0.0
    assert designer.best_datum() is result.data[0]
    assert designer.__class__.__name__ == "EggRollDesigner"
    designer.stop()
    assert policy.objective.closed is True
