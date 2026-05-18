from __future__ import annotations

import numpy as np
from gymnasium import spaces
from isaaclab_score_fakes import FakeIsaacEnv, FakeVectorIsaacEnv, make_fake_runtime, make_fake_vector_runtime


def test_isaaclab_score_evaluates_flat_policy_batch():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy, episodes=2)
    try:
        assert score.dim == policy.num_params()
        mu, se = score.evaluate(score.x0, seed=7)
        assert np.isfinite(mu)
        assert np.isfinite(se)
        assert score.last_num_steps == 6
        means, ses = score.evaluate_many(np.stack([score.x0, score.x0]), seed=11)
        assert means.shape == (2,)
        assert ses.shape == (2,)
    finally:
        score.close()


def test_isaaclab_score_uses_vector_env_for_batches():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_vector_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy, episodes=2)
    try:
        means, ses = score.evaluate_many(np.stack([score.x0, score.x0]), seed=11)
        assert runtime.vector_slots == [4]
        assert means.shape == (2,)
        assert ses.shape == (2,)
        assert score.last_num_steps == 12
    finally:
        score.close()


def test_isaaclab_score_switches_vector_env_to_single_env():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_runtime()
    active = {"value": False}
    closed = []

    def _claim_context() -> None:
        if active["value"]:
            raise RuntimeError("Simulation context already exists")
        active["value"] = True

    def _release_context(label: str) -> None:
        active["value"] = False
        closed.append(label)

    class GuardedSingleEnv(FakeIsaacEnv):
        def __init__(self) -> None:
            _claim_context()
            super().__init__()

        def close(self):
            _release_context("single")

    class GuardedVectorEnv(FakeVectorIsaacEnv):
        def __init__(self, num_envs: int) -> None:
            _claim_context()
            super().__init__(num_envs)

        def close(self):
            _release_context(f"vector:{self.num_envs}")

    def _make(**kwargs):
        if kwargs.get("batched"):
            return GuardedVectorEnv(int(kwargs["num_envs"]))
        return GuardedSingleEnv()

    runtime.make = _make
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy, episodes=2)
    try:
        score.evaluate_many(np.stack([score.x0, score.x0]), seed=11)
        score.evaluate(score.x0, seed=17)
        assert closed == ["vector:4"]
    finally:
        score.close()


def test_eggroll_policy_tag_builds_torch_policy_for_isaaclab():
    from policies.eggroll_policy import EggRollActorCriticMLPPolicyFactory, EggRollActorCriticMLPSpec
    from policies.mlp_policy import MLPPolicy

    policy = EggRollActorCriticMLPPolicyFactory(EggRollActorCriticMLPSpec(hidden_dim=4, layers=2))(make_fake_runtime())
    assert isinstance(policy, MLPPolicy)


def test_eggroll_external_designer_iterates_without_jax_env_adapter():
    from optimizer.eggroll_designer import EggRollDesigner
    from policies.mlp_policy import MLPPolicyFactory

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    designer = EggRollDesigner(
        policy,
        runtime,
        sigma=0.01,
        lr=0.01,
        num_envs=1,
        steps=3,
        batch_size=2,
        optax="adam",
    )
    result = designer.iterate([], 2)
    assert len(result.data) == 1
    assert np.isfinite(float(result.data[0].trajectory.rreturn))
    assert result.data[0].trajectory.num_steps > 0


def test_uhd_supports_isaaclab_vector_objective_tag():
    from problems.uhd_obj import supports_uhd_vector_objective

    assert supports_uhd_vector_objective("isaaclab:Fake-v0")


def test_isaaclab_adapter_unwraps_vector_env_action_space():
    from problems.isaaclab_env_adapters import IsaacLabGymEnvAdapter

    class Env:
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 3), dtype=np.float32)
        action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 12), dtype=np.float32)

        def reset(self, *args, **kwargs):
            return np.zeros((1, 3), dtype=np.float32), {}

        def step(self, action):
            assert tuple(action.shape) == (1, 12)
            return np.zeros((1, 3), dtype=np.float32), np.asarray([0.0]), np.asarray([True]), np.asarray([False]), {}

        def close(self):
            return None

    adapter = IsaacLabGymEnvAdapter(Env(), num_envs=1)
    assert adapter.action_space.shape == (12,)
    adapter.step(np.zeros((1, 12), dtype=np.float32))


def test_unbounded_action_space_is_not_rescaled_to_nan():
    from optimizer.trajectories import _scale_action_to_space

    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    action = _scale_action_to_space(np.zeros((12,), dtype=np.float32), action_space)
    assert np.all(np.isfinite(action))
    assert action.shape == (12,)


def test_video_action_scaling_handles_unbounded_spaces():
    from common.video_spaces import scale_action_to_space

    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    action = scale_action_to_space(np.zeros((12,), dtype=np.float32), action_space)
    assert np.all(np.isfinite(action))
    assert action.shape == (12,)
