from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from isaaclab_score_fakes import FakeIsaacEnv, FakeVectorIsaacEnv, make_fake_dm_control_runtime, make_fake_runtime, make_fake_vector_runtime


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


def test_functional_policy_actions_match_clone_path():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore
    from problems.torch_policy_batch import try_functional_policy_actions

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy)
    xs = np.stack([score.x0, score.x0 + 0.01])
    candidate_idx = np.asarray([0, 0, 1, 1], dtype=np.int64)
    obs = np.asarray([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    active = np.asarray([True, False, True, True])
    zero_action = np.zeros((1,), dtype=np.float32)
    try:
        actions = try_functional_policy_actions(policy, score._codec, xs, candidate_idx, obs, active, zero_action)
        expected = _clone_reference_actions(score, xs, candidate_idx, obs, active, zero_action)
        assert actions is not None
        np.testing.assert_allclose(actions, expected, atol=1e-6)
    finally:
        score.close()


def _clone_reference_actions(score, xs, candidate_idx, obs, active, zero_action):
    raw = np.zeros((int(candidate_idx.size), *tuple(zero_action.shape)), dtype=np.float32)
    for cand_idx in range(int(xs.shape[0])):
        slots = np.flatnonzero(active & (candidate_idx == int(cand_idx)))
        if slots.size == 0:
            continue
        policy = score._make_loaded_policy(xs[cand_idx])
        raw[slots] = np.asarray(policy(obs[slots]), dtype=np.float32).reshape((int(slots.size), *tuple(zero_action.shape)))
    return raw


def test_isaaclab_vector_eval_prefers_functional_policy_path():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_vector_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy, episodes=2)
    score._make_loaded_policy = lambda _x: (_ for _ in ()).throw(AssertionError("clone fallback used"))
    try:
        means, ses = score.evaluate_many(np.stack([score.x0, score.x0]), seed=11)
        assert means.shape == (2,)
        assert ses.shape == (2,)
    finally:
        score.close()


def test_tensor_vectorized_scoring_path_returns_scores():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore
    from problems.isaaclab_tensor_score import try_evaluate_many_tensor_vectorized

    runtime = make_fake_vector_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy, episodes=2)
    env = runtime.make(batched=True, num_envs=4)
    try:
        result = try_evaluate_many_tensor_vectorized(score, np.stack([score.x0, score.x0]), env, seed=11)
        assert result is not None
        means, ses, num_steps = result
        assert means.shape == (2,)
        assert ses.shape == (2,)
        assert num_steps == 12
    finally:
        env.close()
        score.close()


def test_isaaclab_eggroll_noiser_uses_ranked_matrix_noise():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy)
    try:
        rank1 = score.sample_eggroll_noiser_noise(score.x0, seed=3, rank=1)
        rank1_again = score.sample_eggroll_noiser_noise(score.x0, seed=3, rank=1)
        rank4 = score.sample_eggroll_noiser_noise(score.x0, seed=3, rank=4)

        assert rank1.shape == (score.dim,)
        np.testing.assert_allclose(rank1, rank1_again)
        assert not np.allclose(rank1, rank4)
    finally:
        score.close()


def test_isaaclab_eggroll_noiser_freezes_nonlora_params():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy)
    nonlora = np.zeros(score.dim, dtype=bool)
    for start, size, shape in zip(score._codec.offsets, score._codec.sizes, score._codec.shapes, strict=True):
        if len(shape) < 2:
            nonlora[int(start) : int(start) + int(size)] = True
    try:
        dense = score.sample_eggroll_noiser_noise(score.x0, seed=5, rank=2, freeze_nonlora=False)
        frozen = score.sample_eggroll_noiser_noise(score.x0, seed=5, rank=2, freeze_nonlora=True)

        assert np.any(nonlora)
        assert np.any(np.abs(dense[nonlora]) > 0.0)
        assert np.allclose(frozen[nonlora], 0.0)
        np.testing.assert_allclose(frozen[~nonlora], dense[~nonlora])
        assert np.any(np.abs(frozen[~nonlora]) > 0.0)
    finally:
        score.close()


def test_isaaclab_eggroll_noiser_rejects_invalid_options():
    from policies.mlp_policy import MLPPolicyFactory
    from problems.isaaclab_score import IsaacLabScore

    runtime = make_fake_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    score = IsaacLabScore(runtime, policy)
    try:
        with pytest.raises(ValueError, match="rank"):
            score.sample_eggroll_noiser_noise(score.x0, seed=3, rank=0)
        with pytest.raises(ValueError, match="group_size"):
            score.sample_eggroll_noiser_noise(score.x0, seed=3, group_size=-1)
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


def test_eggroll_policy_tag_builds_torch_policy_for_dm_control():
    from policies.eggroll_policy import EggRollActorCriticMLPPolicyFactory, EggRollActorCriticMLPSpec
    from policies.mlp_policy import MLPPolicy

    policy = EggRollActorCriticMLPPolicyFactory(EggRollActorCriticMLPSpec(hidden_dim=4, layers=2))(make_fake_dm_control_runtime())
    assert isinstance(policy, MLPPolicy)


def _run_external_eggroll_once(runtime):
    from optimizer.eggroll_designer import EggRollDesigner
    from policies.mlp_policy import MLPPolicyFactory

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
    try:
        return designer, designer.iterate([], 2)
    except Exception:
        designer.stop()
        raise


def test_eggroll_external_designer_iterates_without_jax_env_adapter():
    designer, result = _run_external_eggroll_once(make_fake_runtime())
    try:
        assert len(result.data) == 1
        assert np.isfinite(float(result.data[0].trajectory.rreturn))
        assert result.data[0].trajectory.num_steps > 0
    finally:
        designer.stop()


def test_eggroll_external_designer_supports_dm_control():
    designer, result = _run_external_eggroll_once(make_fake_dm_control_runtime())
    try:
        assert len(result.data) == 1
        assert np.isfinite(float(result.data[0].trajectory.rreturn))
    finally:
        designer.stop()


def test_eggroll_external_reuses_training_vector_env_for_current_eval():
    from optimizer.eggroll_designer import EggRollDesigner
    from policies.mlp_policy import MLPPolicyFactory

    runtime = make_fake_vector_runtime()
    policy = MLPPolicyFactory((4,))(runtime)
    designer = EggRollDesigner(
        policy,
        runtime,
        sigma=0.01,
        lr=0.01,
        num_envs=2,
        steps=3,
        batch_size=2,
        optax="adam",
    )
    try:
        result = designer.iterate([], 2)
        assert len(result.data) == 1
        assert runtime.vector_slots == [4]
    finally:
        designer.stop()


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
    from video.spaces import scale_action_to_space

    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    action = scale_action_to_space(np.zeros((12,), dtype=np.float32), action_space)
    assert np.all(np.isfinite(action))
    assert action.shape == (12,)
