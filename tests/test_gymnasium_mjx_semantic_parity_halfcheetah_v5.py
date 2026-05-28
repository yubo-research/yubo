from __future__ import annotations

from contextlib import contextmanager

import numpy as np


@contextmanager
def _gym_env(env_id: str):
    import gymnasium as gym

    env = gym.make(env_id)
    try:
        yield env
    finally:
        env.close()


def _fixed_action(action_space) -> np.ndarray:
    return np.linspace(-0.75, 0.75, action_space.shape[0], dtype=np.float32)


def _mjx_data_from_gymnasium(unwrapped, *, jnp, mjx, qpos, qvel, time, ctrl):
    mjx_model = mjx.put_model(unwrapped.model)
    data = mjx.make_data(mjx_model)
    data = data.replace(
        qpos=jnp.asarray(qpos, dtype=jnp.float32),
        qvel=jnp.asarray(qvel, dtype=jnp.float32),
        time=jnp.asarray(time, dtype=jnp.float32),
        ctrl=jnp.asarray(ctrl, dtype=jnp.float32),
    )
    return mjx.forward(mjx_model, data), mjx_model


def _mjx_rollout_from_data(*, jax, jnp, mjx, mjx_model, data, action, frame_skip):
    data = data.replace(ctrl=jnp.asarray(action, dtype=jnp.float32))

    def step_once(carry, _):
        return mjx.step(mjx_model, carry), None

    next_data, _ = jax.lax.scan(step_once, data, xs=None, length=int(frame_skip))
    return next_data


def test_halfcheetah_v5_semantic_values_match_after_adapter_float32_normalization() -> None:
    import jax.numpy as jnp
    from mujoco import mjx

    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    with _gym_env("HalfCheetah-v5") as env:
        unwrapped = env.unwrapped
        spec = resolve_gymnasium_mujoco_spec(env)
        assert spec is not None

        obs0, _info0 = env.reset(seed=0)
        gym_obs0 = np.asarray(obs0, dtype=np.float32)
        assert gym_obs0.shape == (spec.obs_dim(unwrapped.model),)

        gym_before_qpos = np.asarray(unwrapped.data.qpos, dtype=np.float32)
        gym_before_qvel = np.asarray(unwrapped.data.qvel, dtype=np.float32)
        gym_before_time = float(unwrapped.data.time)

        action = _fixed_action(env.action_space)
        obs1, reward1, terminated1, truncated1, info1 = env.step(action)
        del truncated1
        gym_obs1 = np.asarray(obs1, dtype=np.float32)

        gym_after_qpos = np.asarray(unwrapped.data.qpos, dtype=np.float32)
        gym_after_qvel = np.asarray(unwrapped.data.qvel, dtype=np.float32)
        gym_after_time = float(unwrapped.data.time)

        # Reconstruct MJX "before/after" data from Gymnasium states so we can
        # validate semantic parity (obs/reward/terminated) without requiring
        # rollout parity across physics engines.
        mjx_before, _mjx_model = _mjx_data_from_gymnasium(
            unwrapped,
            jnp=jnp,
            mjx=mjx,
            qpos=gym_before_qpos,
            qvel=gym_before_qvel,
            time=gym_before_time,
            ctrl=np.asarray(action, dtype=np.float32),
        )
        mjx_after, _mjx_model = _mjx_data_from_gymnasium(
            unwrapped,
            jnp=jnp,
            mjx=mjx,
            qpos=gym_after_qpos,
            qvel=gym_after_qvel,
            time=gym_after_time,
            ctrl=np.asarray(action, dtype=np.float32),
        )

        spec_obs0 = np.asarray(spec.obs(mjx_before, unwrapped.model, jnp))
        spec_obs1 = np.asarray(spec.obs(mjx_after, unwrapped.model, jnp))
        np.testing.assert_array_equal(spec_obs0, gym_obs0)
        np.testing.assert_array_equal(spec_obs1, gym_obs1)

        spec_reward, spec_info = spec.reward_info(
            mjx_before,
            mjx_after,
            jnp.asarray(action, dtype=jnp.float32),
            unwrapped.model,
            jnp,
        )
        # NOTE: On different backends (local vs Modal) we may see 1-ULP float32
        # differences due to evaluation order / fused ops when reconstructing
        # the reward from injected state. Treat this as semantic parity, not
        # strict bitwise parity.
        np.testing.assert_allclose(
            np.asarray(spec_reward, dtype=np.float32),
            np.asarray(reward1, dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )

        # We only assert a small stable surface of keys/values.
        for key in ("x_position", "x_velocity", "reward_forward", "reward_ctrl"):
            assert key in info1
            assert key in spec_info
            np.testing.assert_allclose(
                np.asarray(spec_info[key], dtype=np.float32),
                np.asarray(info1[key], dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )

        spec_terminated = bool(spec.terminated(mjx_after, jnp))
        assert spec_terminated == bool(terminated1)


def test_halfcheetah_v5_default_runtime_is_not_bitwise_gymnasium_parity() -> None:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    with _gym_env("HalfCheetah-v5") as env:
        unwrapped = env.unwrapped
        spec = resolve_gymnasium_mujoco_spec(env)
        assert spec is not None

        gym_obs0, _info0 = env.reset(seed=0)
        gym_before_qpos = np.asarray(unwrapped.data.qpos)
        gym_before_qvel = np.asarray(unwrapped.data.qvel)
        gym_before_time = float(unwrapped.data.time)
        action = _fixed_action(env.action_space)

        mjx_before, mjx_model = _mjx_data_from_gymnasium(
            unwrapped,
            jnp=jnp,
            mjx=mjx,
            qpos=gym_before_qpos,
            qvel=gym_before_qvel,
            time=gym_before_time,
            ctrl=action,
        )
        spec_obs0 = np.asarray(spec.obs(mjx_before, unwrapped.model, jnp))

        assert gym_obs0.dtype == np.float64
        assert spec_obs0.dtype == np.float32
        np.testing.assert_array_equal(spec_obs0, gym_obs0.astype(np.float32))

        _gym_obs1, _reward1, _terminated1, _truncated1, _info1 = env.step(action)
        gym_after_qpos = np.asarray(unwrapped.data.qpos)
        gym_after_qvel = np.asarray(unwrapped.data.qvel)

        mjx_after = _mjx_rollout_from_data(
            jax=jax,
            jnp=jnp,
            mjx=mjx,
            mjx_model=mjx_model,
            data=mjx_before,
            action=action,
            frame_skip=spec.frame_skip,
        )
        mjx_after_qpos = np.asarray(mjx_after.qpos)
        mjx_after_qvel = np.asarray(mjx_after.qvel)

        qpos_diff = float(np.max(np.abs(mjx_after_qpos.astype(np.float64) - gym_after_qpos)))
        qvel_diff = float(np.max(np.abs(mjx_after_qvel.astype(np.float64) - gym_after_qvel)))
        assert qpos_diff > 0.0 or qvel_diff > 0.0


def test_gymnasium_mjx_adapter_jit_step_uses_gymnasium_oracle_callbacks() -> None:
    import jax
    import jax.numpy as jnp

    from problems.mjx_env import GymnasiumMJXAdapter

    adapter = GymnasiumMJXAdapter("gymnasium:HalfCheetah-v5", jax=jax, jnp=jnp)
    try:
        _obs, state = adapter.reset(jax.random.key(0))
        action = jnp.zeros(adapter.action_space.shape, dtype=jnp.float32)

        @jax.jit
        def run_step(key, carry, env_action):
            return adapter.step(key, carry, env_action)

        out = run_step(jax.random.key(1), state, action)
        out.obs.block_until_ready()

        assert out.obs.shape == adapter.observation_space.shape
        assert out.reward.shape == ()
        assert out.terminated.shape == ()
        assert out.truncated.shape == ()
    finally:
        adapter.close()
