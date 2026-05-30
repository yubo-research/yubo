from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _fake_halfcheetah_env(*, model=None, max_episode_steps=1000, frame_skip=5):
    if model is None:
        model = SimpleNamespace(nq=2, nv=1)
    data = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0], dtype=np.float64),
        qvel=np.asarray([0.0], dtype=np.float64),
    )
    unwrapped = SimpleNamespace(
        model=model,
        data=data,
        spec=SimpleNamespace(id="HalfCheetah-v5", max_episode_steps=max_episode_steps),
        _reset_noise_scale=0.1,
        frame_skip=frame_skip,
        dt=0.05,
    )
    unwrapped._get_obs = lambda: np.concatenate([np.ravel(unwrapped.data.qpos[1:]), np.ravel(unwrapped.data.qvel)])

    def _get_rew(x_velocity, action):
        ctrl_cost = np.float32(0.1) * np.sum(np.asarray(action, dtype=np.float32) ** 2)
        reward_forward = np.asarray(x_velocity)
        return reward_forward - ctrl_cost, {
            "reward_forward": reward_forward,
            "reward_ctrl": -ctrl_cost,
        }

    unwrapped._get_rew = _get_rew

    def do_simulation(_action, _frame_skip):
        return None

    def step(action):
        x_position_before = unwrapped.data.qpos[0]
        unwrapped.do_simulation(action, unwrapped.frame_skip)
        x_position_after = unwrapped.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / unwrapped.dt
        observation = unwrapped._get_obs()
        reward, reward_info = unwrapped._get_rew(x_velocity, action)
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
        }
        return observation, reward, False, False, info

    unwrapped.do_simulation = do_simulation
    unwrapped.step = step
    return SimpleNamespace(
        unwrapped=unwrapped,
        observation_space=SimpleNamespace(shape=(max(int(model.nq) - 1, 0) + int(model.nv),)),
        close=lambda: None,
    )


def test_gymnasium_tag_parse_and_registry_surface() -> None:
    from problems.jax_env_core import supported_jax_env_tags, supports_jax_env_tag
    from problems.mjx_env import is_gymnasium_env_tag, parse_gymnasium_env_id

    assert parse_gymnasium_env_id("gymnasium:HalfCheetah-v5") == "HalfCheetah-v5"
    assert parse_gymnasium_env_id("gymnasium_fast:HalfCheetah-v5") == "HalfCheetah-v5"
    assert is_gymnasium_env_tag("gymnasium:Ant-v5")
    assert is_gymnasium_env_tag("gymnasium_fast:Ant-v5")
    assert supports_jax_env_tag("gymnasium:HalfCheetah-v5")
    assert supports_jax_env_tag("gymnasium_fast:HalfCheetah-v5")
    assert "gymnasium:HalfCheetah-v5" in supported_jax_env_tags()
    assert "gymnasium_fast:HalfCheetah-v5" in supported_jax_env_tags()
    assert "mjx:xml:/path/to/model.xml" not in supported_jax_env_tags()


def test_gymnasium_tag_parse_rejects_empty_id() -> None:
    from problems.mjx_env import parse_gymnasium_env_id

    with pytest.raises(ValueError, match="missing an env id"):
        parse_gymnasium_env_id("gymnasium:")


def test_jax_env_factory_routes_gymnasium_to_mjx(monkeypatch) -> None:
    import problems.jax_env_factory as factory

    calls = []

    def fake_gymnasium_mjx_adapter(env_name, *, jax, jnp):
        calls.append((env_name, jax, jnp))
        return SimpleNamespace()

    monkeypatch.setattr(factory, "GymnasiumMJXAdapter", fake_gymnasium_mjx_adapter)

    adapter = factory.make_jax_env_adapter("gymnasium:HalfCheetah-v5", jax="jax", jnp="jnp")

    assert isinstance(adapter, SimpleNamespace)
    assert calls == [("gymnasium:HalfCheetah-v5", "jax", "jnp")]


@pytest.mark.parametrize(
    ("fast", "semantics_owner", "oracle_is_none"),
    [
        (False, "gymnasium", False),
        (True, "yubo_jax_fast", True),
    ],
)
def test_mjx_adapter_loads_verified_gymnasium_model(monkeypatch, fast, semantics_owner, oracle_is_none) -> None:
    import problems.mjx_env as mjx_env

    model = SimpleNamespace(nq=2, nv=1)
    env = _fake_halfcheetah_env(model=model)

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    loaded_model, spec = mjx_env._load_gymnasium_env_spec("HalfCheetah-v5", fast=fast)
    assert loaded_model is model
    assert spec.env_id == "HalfCheetah-v5"
    assert spec.frame_skip == 5
    assert (spec.oracle is None) is oracle_is_none
    assert spec.bindings["semantics_owner"] == semantics_owner


def test_mjx_adapter_rejects_unverified_gymnasium_semantics(monkeypatch) -> None:
    import problems.mjx_env as mjx_env

    closed = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(model=SimpleNamespace(), spec=SimpleNamespace(id="Other-v0")),
        close=lambda: closed.append(True),
    )

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    with pytest.raises(ValueError, match="no verified MuJoCo semantics"):
        mjx_env._load_gymnasium_env_spec("Other-v0")
    assert closed == [True]


def test_halfcheetah_spec_matches_gymnasium_v5_defaults() -> None:
    from problems.gymnasium_mujoco_specs import (
        GymnasiumMujocoSpec,
        resolve_gymnasium_mujoco_spec,
    )

    env = _fake_halfcheetah_env()

    spec = resolve_gymnasium_mujoco_spec(env)

    assert spec is not None
    assert isinstance(spec, GymnasiumMujocoSpec)
    assert spec.bindings["semantics_owner"] == "gymnasium"
    assert spec.bindings["obs_owner"].endswith("._get_obs")
    assert spec.bindings["reward_owner"].endswith("._get_rew")
    assert spec.reset_noise_scale == 0.1
    assert spec.frame_skip == 5


def test_halfcheetah_mjx_reward_delegates_to_gymnasium_oracle() -> None:
    from problems.gymnasium_mujoco_specs import (
        resolve_gymnasium_mujoco_spec,
        supported_gymnasium_mujoco_specs,
    )

    env = _fake_halfcheetah_env()
    calls = []

    def fake_get_rew(x_velocity, action):
        calls.append((float(x_velocity), np.asarray(action, dtype=np.float32).copy()))
        return np.asarray(123.0, dtype=np.float32), {
            "reward_forward": np.asarray(124.0, dtype=np.float32),
            "reward_ctrl": np.asarray(-1.0, dtype=np.float32),
        }

    env.unwrapped._get_rew = fake_get_rew
    spec = resolve_gymnasium_mujoco_spec(env)
    state = SimpleNamespace(
        qpos=np.array([1.0, 0.0], dtype=np.float32),
        qvel=np.array([0.0], dtype=np.float32),
    )
    next_state = SimpleNamespace(
        qpos=np.array([1.5, 0.0], dtype=np.float32),
        qvel=np.array([0.0], dtype=np.float32),
    )
    action = np.array([1.0, -2.0], dtype=np.float32)

    reward, info = spec.reward_info(state, next_state, action, SimpleNamespace(), np)

    assert reward == pytest.approx(123.0)
    assert info["x_position"] == pytest.approx(1.5)
    assert info["x_velocity"] == pytest.approx(10.0)
    assert info["reward_forward"] == pytest.approx(124.0)
    assert info["reward_ctrl"] == pytest.approx(-1.0)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(10.0)
    np.testing.assert_array_equal(calls[0][1], action)
    assert supported_gymnasium_mujoco_specs() == ("HalfCheetah-v5",)


def test_halfcheetah_fast_spec_has_no_gymnasium_oracle_callback() -> None:
    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = _fake_halfcheetah_env()
    spec = resolve_gymnasium_mujoco_spec(env, fast=True)

    assert spec.oracle is None
    assert spec.obs_dim(SimpleNamespace()) == 2


def test_halfcheetah_fast_spec_computes_jax_native_semantics() -> None:
    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = _fake_halfcheetah_env()
    spec = resolve_gymnasium_mujoco_spec(env, fast=True)
    state = SimpleNamespace(
        qpos=np.array([1.0, 2.0], dtype=np.float32),
        qvel=np.array([3.0], dtype=np.float32),
    )
    next_state = SimpleNamespace(
        qpos=np.array([1.5, 2.5], dtype=np.float32),
        qvel=np.array([3.5], dtype=np.float32),
    )
    action = np.array([1.0, -2.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = spec.step_semantics(state, next_state, action, SimpleNamespace(), np)

    np.testing.assert_array_equal(obs, np.asarray([2.5, 3.5], dtype=np.float32))
    assert reward == pytest.approx(9.5)
    assert terminated == np.asarray(False, dtype=bool)
    assert truncated == np.asarray(False, dtype=bool)
    assert info["x_position"] == pytest.approx(1.5)
    assert info["x_velocity"] == pytest.approx(10.0)
    assert info["reward_forward"] == pytest.approx(10.0)
    assert info["reward_ctrl"] == pytest.approx(-0.5)
    assert spec.terminated(next_state, np) == np.asarray(False, dtype=bool)


def test_halfcheetah_oracle_spec_requires_gymnasium_owned_methods() -> None:
    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = _fake_halfcheetah_env()
    del env.unwrapped._get_rew

    with pytest.raises(ValueError, match="missing: _get_rew"):
        resolve_gymnasium_mujoco_spec(env)


def test_gymnasium_mujoco_spec_wrapper_methods() -> None:
    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = _fake_halfcheetah_env()
    spec = resolve_gymnasium_mujoco_spec(env)
    model = SimpleNamespace(nq=2, nv=1)
    data = SimpleNamespace(
        qpos=np.asarray([1.0, 2.0], dtype=np.float32),
        qvel=np.asarray([3.0], dtype=np.float32),
    )

    assert spec.obs_dim(model) == 2
    np.testing.assert_array_equal(
        spec.obs(data, model, np),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    assert spec.terminated(data, np) == np.asarray(False, dtype=bool)


def test_gymnasium_mjx_state_is_named_tuple() -> None:
    from problems.mjx_env import GymnasiumMJXState

    state = GymnasiumMJXState(data="data", steps=3)

    assert state.data == "data"
    assert state.steps == 3
    assert tuple(state) == ("data", 3)


def test_gymnasium_mjx_adapter_uses_specs_frame_skip_and_time_limit(
    monkeypatch,
) -> None:
    import numpy as np

    from problems.mjx_env import GymnasiumMJXAdapter

    def make_fake_data(*, qpos=None, qvel=None, ctrl=None, time=0.0):
        data = SimpleNamespace(
            qpos=np.asarray([0.0, 0.0] if qpos is None else qpos, dtype=np.float32),
            qvel=np.asarray([0.0] if qvel is None else qvel, dtype=np.float32),
            ctrl=np.asarray([0.0] if ctrl is None else ctrl, dtype=np.float32),
            time=np.float32(time),
        )
        data.replace = lambda **kwargs: make_fake_data(
            qpos=kwargs.get("qpos", data.qpos),
            qvel=kwargs.get("qvel", data.qvel),
            ctrl=kwargs.get("ctrl", data.ctrl),
            time=kwargs.get("time", data.time),
        )
        return data

    def fake_scan(fn, carry, xs=None, length=1):
        del xs
        for _ in range(int(length)):
            carry, _out = fn(carry, None)
        return carry, None

    def fake_tree_map(fn, left, right):
        if hasattr(left, "qpos"):
            return make_fake_data(
                qpos=fn(left.qpos, right.qpos),
                qvel=fn(left.qvel, right.qvel),
                ctrl=fn(left.ctrl, right.ctrl),
                time=fn(left.time, right.time),
            )
        return fn(left, right)

    model = SimpleNamespace(
        nq=2,
        nv=1,
        nu=1,
        qpos0=np.asarray([0.0, 0.0], dtype=np.float32),
        actuator_ctrlrange=np.asarray([[-2.0, 2.0]], dtype=np.float32),
        actuator_ctrllimited=np.asarray([True]),
    )
    env = _fake_halfcheetah_env(model=model, max_episode_steps=1, frame_skip=5)
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    spaces_mod = ModuleType("gymnax.environments.spaces")
    spaces_mod.Box = lambda *, low, high, shape, dtype: SimpleNamespace(low=low, high=high, shape=shape, dtype=dtype)
    environments_mod = ModuleType("gymnax.environments")
    environments_mod.spaces = spaces_mod
    gymnax_mod = ModuleType("gymnax")
    gymnax_mod.environments = environments_mod
    mujoco_mod = ModuleType("mujoco")
    mujoco_mod.mjx = SimpleNamespace(
        put_model=lambda model, **kwargs: model,
        make_data=lambda _model: make_fake_data(),
        forward=lambda _model, data: data,
        step=lambda _model, data: data.replace(
            qpos=data.qpos + np.asarray([0.1, 0.0], dtype=np.float32),
            time=data.time + np.float32(0.01),
        ),
    )
    fake_cpu_device = SimpleNamespace()
    fake_jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda _key: ("left", "right"),
            uniform=lambda _key, shape, *, minval, maxval, dtype: np.zeros(shape, dtype=dtype),
            normal=lambda _key, shape, *, dtype: np.zeros(shape, dtype=dtype),
        ),
        lax=SimpleNamespace(scan=fake_scan),
        tree_util=SimpleNamespace(tree_map=fake_tree_map),
        devices=lambda backend: [] if backend == "cuda" else [fake_cpu_device],
    )
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gymnax", gymnax_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments.spaces", spaces_mod)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)

    adapter = GymnasiumMJXAdapter("gymnasium:HalfCheetah-v5", jax=fake_jax, jnp=np)
    obs, state = adapter.reset("reset-key")
    action = adapter.clip_action(np.asarray([4.0], dtype=np.float32))
    next_obs, next_state, reward, terminated, truncated, info = adapter.step("step-key", state, action)

    np.testing.assert_array_equal(obs, np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(next_obs, np.asarray([0.0, 0.0], dtype=np.float32))
    assert next_state.steps == np.asarray(0, dtype=np.int32)
    assert terminated == pytest.approx(0.0)
    assert truncated == pytest.approx(1.0)
    assert reward == pytest.approx(9.6)
    assert info["x_velocity"] == pytest.approx(10.0)


def test_mujoco_playground_impl_uses_valid_mjx_impl(monkeypatch) -> None:
    import problems.mujoco_playground_env as playground

    monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    assert playground._mjx_impl() == "jax"

    monkeypatch.setitem(sys.modules, "mujoco_warp", ModuleType("mujoco_warp"))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(types=SimpleNamespace()))
    assert playground._mjx_impl() == "jax"

    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(types=SimpleNamespace(warp_type_to_np_dtype={})),
    )
    assert playground._mjx_impl() == "warp"


def test_mujoco_playground_adapter_passes_valid_impl(monkeypatch) -> None:
    import numpy as np

    import problems.mujoco_playground_env as playground

    calls = []

    def make_state(obs, *, reward=0.0, done=0.0):
        return SimpleNamespace(
            obs={
                "state": np.asarray(obs, dtype=np.float32),
                "privileged_state": np.asarray([9.0, 9.0], dtype=np.float32),
            },
            reward=np.asarray(reward, dtype=np.float32),
            done=np.asarray(done, dtype=np.float32),
            metrics={"metric": np.asarray(1.0, dtype=np.float32)},
        )

    fake_env = SimpleNamespace(
        _mj_model=SimpleNamespace(nu=2),
        reset=lambda _key: make_state([1.0, 2.0, 3.0]),
        step=lambda _state, _action: make_state([4.0, 5.0, 6.0], reward=7.0, done=1.0),
    )

    def fake_load(env_name, config_overrides):
        calls.append((env_name, config_overrides))
        return fake_env

    registry_mod = SimpleNamespace(load=fake_load)
    playground_mod = ModuleType("mujoco_playground")
    playground_mod.registry = registry_mod
    spaces_mod = ModuleType("gymnax.environments.spaces")
    spaces_mod.Box = lambda *, low, high, shape, dtype: SimpleNamespace(low=low, high=high, shape=shape, dtype=dtype)
    environments_mod = ModuleType("gymnax.environments")
    environments_mod.spaces = spaces_mod
    gymnax_mod = ModuleType("gymnax")
    gymnax_mod.environments = environments_mod
    fake_jax = SimpleNamespace(
        random=SimpleNamespace(
            key=lambda seed: ("key", seed),
            split=lambda key: ((key, "step"), (key, "reset")),
        ),
        lax=SimpleNamespace(
            cond=lambda pred, true_fn, false_fn, operand: (true_fn(operand) if pred else false_fn(operand)),
        ),
        tree_util=SimpleNamespace(tree_leaves=lambda value: [value]),
    )
    monkeypatch.setitem(sys.modules, "mujoco_playground", playground_mod)
    monkeypatch.setitem(sys.modules, "gymnax", gymnax_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments.spaces", spaces_mod)
    monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
    monkeypatch.delitem(sys.modules, "warp", raising=False)

    adapter = playground.MujocoPlaygroundAdapter("mujoco_playground:G1JoystickRoughTerrain", jax=fake_jax, jnp=np)
    obs, state = adapter.reset("reset-key")
    action = adapter.clip_action(np.asarray([2.0, -3.0], dtype=np.float32))
    next_obs, next_state, reward, terminated, truncated, metrics = adapter.step("step-key", state, action)

    assert calls == [("G1JoystickRoughTerrain", {"impl": "jax"})]
    assert adapter.observation_space.shape == (3,)
    assert adapter.action_space.shape == (2,)
    np.testing.assert_array_equal(obs, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(action, np.asarray([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(next_obs, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert next_state is not state
    assert reward == pytest.approx(7.0)
    assert terminated == pytest.approx(1.0)
    assert truncated == pytest.approx(0.0)
    assert metrics["metric"] == pytest.approx(1.0)
