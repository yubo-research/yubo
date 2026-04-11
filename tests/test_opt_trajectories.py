from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from optimizer import opt_trajectories, trajectories


class MockTrajectory:
    def __init__(self, rreturn):
        self.rreturn = rreturn


class MockEnvConf:
    def __init__(self):
        self.noise_seed_0 = 0
        self.frozen_noise = False


class MockPolicy:
    pass


class _DeterministicEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._seed = 0
        self._step = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        self._seed = 0 if seed is None else int(seed)
        self._step = 0
        return np.array([float(self._seed), 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step += 1
        reward = float(self._seed + self._step)
        terminated = self._step >= 3
        observation = np.array([float(self._seed), float(self._step)], dtype=np.float32)
        return observation, reward, terminated, False, {}

    def close(self):
        return None


class _OneStepDeterministicEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._seed = 0
        self._step = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        self._seed = 0 if seed is None else int(seed)
        self._step = 0
        return np.array([float(self._seed), 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step += 1
        reward = float(self._seed + 1)
        return np.array([float(self._seed), 1.0], dtype=np.float32), reward, True, False, {}

    def close(self):
        return None


class _ReuseEnvConf:
    def __init__(self):
        self.noise_seed_0 = 11
        self.frozen_noise = False
        self.gym_conf = SimpleNamespace(max_steps=3, num_frames_skip=1, transform_state=False)
        self.make_calls = 0
        self.last_make_kwargs = None

    def make(self, **kwargs):
        self.make_calls += 1
        self.last_make_kwargs = kwargs
        return _DeterministicEnv()


class _DeterministicPolicy:
    def __init__(self):
        self.reset_calls = 0
        self.call_count = 0

    def reset_state(self):
        self.reset_calls += 1

    def __call__(self, obs):
        _ = obs
        self.call_count += 1
        return np.zeros((1,), dtype=np.float32)


class _TaggedDeterministicPolicy(_DeterministicPolicy):
    def __init__(self):
        super().__init__()
        self._turbo_enn_eval_reuse_ok = True


class _DesignerPolicy:
    def __init__(self):
        self._params = np.zeros(2, dtype=np.float32)
        self.clone_calls = 0

    def num_params(self):
        return 2

    def clone(self):
        self.clone_calls += 1
        cloned = _DesignerPolicy()
        cloned._params = self._params.copy()
        return cloned

    def set_params(self, params):
        self._params = np.asarray(params, dtype=np.float32)


class _CountingBatchActor(torch.nn.Module):
    forward_calls = 0

    def forward(self, x):
        type(self).forward_calls += 1
        return torch.zeros((x.shape[0], 1), dtype=x.dtype)


class _BatchedDeterministicPolicy:
    def __init__(self):
        self.actor = _CountingBatchActor()
        self._deterministic_eval = True
        self._turbo_enn_eval_reuse_ok = True

    def clone(self):
        return type(self)()

    def reset_state(self):
        return None

    def _normalize(self, state):
        return np.asarray(state, dtype=np.float32)

    def _postprocess_action(self, action_t):
        return action_t

    def __call__(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        with torch.inference_mode():
            action_t = self.actor.forward(state_t)
        return action_t.squeeze(0).detach().cpu().numpy()


def test_collect_trajectory_with_noise(monkeypatch):
    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=1.5 + noise_seed * 0.01)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, noise_seed = opt_trajectories.collect_trajectory_with_noise(env_conf, policy, i_noise=1, denoise_seed=0)
    assert traj.rreturn > 0
    assert noise_seed == 1


def test_mean_return_over_runs(monkeypatch):
    call_count = [0]

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        call_count[0] += 1
        return MockTrajectory(rreturn=1.0 + call_count[0] * 0.1)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    mean_ret, se_ret, all_same, num_steps_total = opt_trajectories.mean_return_over_runs(env_conf, policy, num_denoise=3, i_noise=0)
    assert np.isfinite(mean_ret)
    assert np.isfinite(se_ret)
    assert num_steps_total >= 0


def test_collect_denoised_trajectory_single(monkeypatch):
    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=2.0)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, _ = opt_trajectories.collect_denoised_trajectory(env_conf, policy, num_denoise=1)
    assert traj.rreturn == 2.0


def test_collect_denoised_trajectory_multiple(monkeypatch):
    counter = [0]

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        counter[0] += 1
        return MockTrajectory(rreturn=1.0 + counter[0] * 0.1)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, _ = opt_trajectories.collect_denoised_trajectory(env_conf, policy, num_denoise=3)
    assert np.isfinite(traj.rreturn)


def test_evaluate_for_best(monkeypatch):
    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=5.0)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    ret = opt_trajectories.evaluate_for_best(env_conf, policy, num_denoise_passiveuation=2)
    assert ret == 5.0


def test_reuse_env_preserves_objective_and_reduces_env_creation(monkeypatch):
    monkeypatch.delenv("YUBO_TURBO_ENN_REUSE_ENV", raising=False)
    trajectories._clear_cached_eval_envs()

    fresh_env_conf = _ReuseEnvConf()
    fresh_policy = _TaggedDeterministicPolicy()
    fresh_mean = opt_trajectories.mean_return_over_runs(fresh_env_conf, fresh_policy, num_denoise=3, i_noise=4)
    assert fresh_env_conf.make_calls == 3

    monkeypatch.setenv("YUBO_TURBO_ENN_REUSE_ENV", "1")
    trajectories._clear_cached_eval_envs()

    reuse_env_conf = _ReuseEnvConf()
    reuse_policy = _TaggedDeterministicPolicy()
    reuse_mean = opt_trajectories.mean_return_over_runs(reuse_env_conf, reuse_policy, num_denoise=3, i_noise=4)
    assert reuse_env_conf.make_calls == 1

    assert fresh_mean.all_same == reuse_mean.all_same
    assert fresh_mean.num_steps_total == reuse_mean.num_steps_total == 9
    np.testing.assert_allclose(fresh_mean.mean, reuse_mean.mean)
    np.testing.assert_allclose(fresh_mean.se, reuse_mean.se)


def test_reuse_env_requires_policy_opt_in(monkeypatch):
    monkeypatch.setenv("YUBO_TURBO_ENN_REUSE_ENV", "1")
    trajectories._clear_cached_eval_envs()

    env_conf = _ReuseEnvConf()
    policy = _DeterministicPolicy()
    _ = opt_trajectories.mean_return_over_runs(env_conf, policy, num_denoise=2, i_noise=4)
    assert env_conf.make_calls == 2


def test_vectorized_eval_reuses_cached_vector_env_and_preserves_returns(monkeypatch):
    from optimizer.vectorized_eval import clear_cached_vectorized_policy_evaluators

    monkeypatch.setenv("YUBO_TURBO_ENN_VECTORIZE_EVAL", "1")
    trajectories._clear_cached_eval_envs()
    clear_cached_vectorized_policy_evaluators()

    seq_env_conf = _ReuseEnvConf()
    seq_policy = _TaggedDeterministicPolicy()
    seq_mean = opt_trajectories.mean_return_over_runs(seq_env_conf, seq_policy, num_denoise=3, i_noise=4)
    assert seq_env_conf.make_calls == 3

    clear_cached_vectorized_policy_evaluators()
    vect_env_conf = _ReuseEnvConf()
    vect_policy = _TaggedDeterministicPolicy()
    vect_mean_1 = opt_trajectories.mean_return_over_runs(vect_env_conf, vect_policy, num_denoise=3, i_noise=4)
    vect_mean_2 = opt_trajectories.mean_return_over_runs(vect_env_conf, vect_policy, num_denoise=3, i_noise=4)
    assert vect_env_conf.make_calls == 3
    assert seq_mean.all_same == vect_mean_1.all_same == vect_mean_2.all_same
    assert seq_mean.num_steps_total == vect_mean_1.num_steps_total == vect_mean_2.num_steps_total == 9
    np.testing.assert_allclose(seq_mean.mean, vect_mean_1.mean)
    np.testing.assert_allclose(seq_mean.se, vect_mean_1.se)
    np.testing.assert_allclose(vect_mean_1.mean, vect_mean_2.mean)
    np.testing.assert_allclose(vect_mean_1.se, vect_mean_2.se)


def test_vectorized_eval_batches_actor_forward(monkeypatch):
    from optimizer.vectorized_eval import clear_cached_vectorized_policy_evaluators

    monkeypatch.setenv("YUBO_TURBO_ENN_VECTORIZE_EVAL", "1")
    trajectories._clear_cached_eval_envs()
    clear_cached_vectorized_policy_evaluators()
    _CountingBatchActor.forward_calls = 0

    class _BatchedEnvConf(_ReuseEnvConf):
        def make(self, **kwargs):
            self.make_calls += 1
            self.last_make_kwargs = kwargs
            return _OneStepDeterministicEnv()

    env_conf = _BatchedEnvConf()
    policy = _BatchedDeterministicPolicy()
    mean_result = opt_trajectories.mean_return_over_runs(env_conf, policy, num_denoise=3, i_noise=4)

    assert env_conf.make_calls == 3
    assert _CountingBatchActor.forward_calls == 1
    assert mean_result.num_steps_total == 3


def test_turbo_enn_designer_marks_policies_for_reuse():
    from optimizer.turbo_enn_designer import TurboENNDesigner

    policy = _DesignerPolicy()
    designer = TurboENNDesigner(policy, turbo_mode="turbo-enn", k=1)
    tagged = designer._make_policy(np.zeros(2, dtype=np.float32))
    assert getattr(tagged, "_turbo_enn_eval_reuse_ok", False) is True
