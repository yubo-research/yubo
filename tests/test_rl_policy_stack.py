from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from gymnasium.spaces import Box

from problems.gaussian_policy import GaussianPolicyFactory
from problems.mlp_policy import MLPPolicyFactory
from problems.rl_policy_factory import RLPolicyFactory, gaussian_policy_factory, project
from rl.core import eval as rl_eval


def _env_conf():
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=None,
        state_space=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        ensure_spaces=lambda: None,
    )


def test_gaussian_policy_factory_builds_policy_and_schema():
    factory = GaussianPolicyFactory(variant="rl-gauss-small", init_log_std=-0.7)
    schema = factory.to_rl_schema()
    policy = factory(_env_conf())

    action = policy(np.zeros(3, dtype=np.float32))

    assert schema["family"] == "gaussian_backbone"
    assert schema["variant"] == "rl-gauss-small"
    assert schema["log_std_init"] == -0.7
    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_rl_policy_factory_projects_mlp_schema_for_ppo():
    factory = RLPolicyFactory(
        MLPPolicyFactory((8, 4), activation="relu", use_layer_norm=True),
        critic=MLPPolicyFactory((16,), activation="tanh"),
        share_backbone=False,
        log_std_init=-0.3,
        ppo_overrides={"extra_flag": True},
    )

    model = project(factory.to_rl_schema(), "ppo")

    assert model["backbone_hidden_sizes"] == (8, 4)
    assert model["backbone_activation"] == "relu"
    assert model["backbone_layer_norm"] is True
    assert model["critic_backbone_hidden_sizes"] == (16,)
    assert model["critic_backbone_activation"] == "tanh"
    assert model["share_backbone"] is False
    assert model["log_std_init"] == -0.3
    assert model["extra_flag"] is True


def test_gaussian_policy_factory_wrapper(monkeypatch):
    import rl.policy_backbone as policy_backbone

    class FakeFactory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(policy_backbone, "GaussianActorBackbonePolicyFactory", FakeFactory)

    factory = gaussian_policy_factory("rl-gauss", extra_flag=True)

    assert isinstance(factory, FakeFactory)
    assert factory.kwargs["variant"] == "rl-gauss"
    assert factory.kwargs["deterministic_eval"] is True
    assert factory.kwargs["squash_mode"] == "clip"
    assert factory.kwargs["init_log_std"] == -0.5
    assert factory.kwargs["extra_flag"] is True


def test_eval_run_updates_best_and_heldout():
    state = SimpleNamespace(
        best_return=1.0,
        best_actor_state=None,
        last_eval_return=None,
        last_heldout_return=None,
    )
    seen = {}

    ep = rl_eval.run(
        current=10,
        interval=5,
        seed=7,
        eval_seed_base=100,
        eval_noise_mode="frozen",
        state=state,
        evaluate_actor=lambda *, eval_seed: float(eval_seed),
        capture_actor_state=lambda: {"actor": 1},
        evaluate_heldout=lambda *, best_actor_state, heldout_i_noise: (seen.update(state=best_actor_state, noise=heldout_i_noise) or 3.5),
    )

    assert ep.eval_seed == 100
    assert state.best_return == 100.0
    assert state.best_actor_state == {"actor": 1}
    assert state.last_heldout_return == 3.5
    assert seen == {"state": {"actor": 1}, "noise": ep.heldout_i_noise}


def test_eval_heldout_uses_actor_state_context():
    seen = []

    @contextmanager
    def with_actor_state(actor_state):
        seen.append(("enter", actor_state))
        try:
            yield
        finally:
            seen.append(("exit", actor_state))

    value = rl_eval.heldout(
        best_actor_state={"best": 1},
        num_denoise_passive=2,
        heldout_i_noise=4,
        with_actor_state=with_actor_state,
        best=lambda _env_conf, _eval_policy, num_denoise, i_noise: float(num_denoise + i_noise),
        eval_env_conf=object(),
        eval_policy=object(),
    )

    assert value == 6.0
    assert seen == [("enter", {"best": 1}), ("exit", {"best": 1})]
